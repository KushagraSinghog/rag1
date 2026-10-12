### Prompt

from langchain_core.prompts import ChatPromptTemplate 



prompt = ChatPromptTemplate.from_template(
    """ 
You are answering questions about a machine learning book.

Use the provided context to answer the question.

If the answer is partially available in the context,
provide the best answer you can.

Context:
{context}

Question:
{question}

Answer:
"""
) 



### LLM 



from langchain_ollama import ChatOllama 

llm = ChatOllama(
    model="qwen3:0.6b", temperature=0.1
) 



### Build chain 
import time
from retriever import retriever
from reranker import rerank_docs



def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs) 

chain = prompt | llm 

def answer_question(question: str, top_k: int = 5): 
    start_time = time.time()
    
    docs = retriever.invoke(question) 

    docs = rerank_docs( 
        question, docs, top_n=5
    ) 

    context = format_docs(docs) 

    response = chain.invoke({ 
        "context": context, 
        "question": question
    }) 

    pages = sorted(
        set( 
            int(doc.metadata.get("page")) 
            for doc in docs 
            if doc.metadata.get("page") is not None
        )
    )

    elapsed = round(time.time() - start_time, 3)

    return {
        "answer": response.content, 
        "sources": pages,
        "latency_seconds": elapsed
    }



