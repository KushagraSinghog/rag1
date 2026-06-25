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
    model="qwen3:0.6b", temperature=0
) 



### Build chain 
from retriever import retrieve_context

retriever = retrieve_context()

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs) 

chain = prompt | llm 

def answer_question(question): 
    docs = retriever.invoke(question) 
    context = format_docs(docs) 

    response = chain.invoke( { 
        "context": context, 
        "question": question
    } ) 

    pages = sorted(
        set( 
            doc.metadata.get("page") 
            for doc in docs 
            if "page" in doc.metadata 
                        
        )
    )

    return{
        "answer": response.content, 
        "sources": pages
    }



