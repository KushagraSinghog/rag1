### BM25 retriever 

import pickle 
from langchain_community.retrievers import BM25Retriever 

CHUNKS_PATH = "vectorstore/chunks.pkl"

with open(CHUNKS_PATH, "rb") as f: 
    documents = pickle.load(f) 

bm25_retriever = BM25Retriever.from_documents(documents) 

bm25_retriever.k = 5



### Building retriever 

from langchain_community.vectorstores import FAISS 
from langchain_ollama import OllamaEmbeddings 

VECTORSTORE_PATH = "vectorstore" 

def retrieve_context(): 
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 

    vectorstore = FAISS.load_local(
                        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
                    ) 
    
    faiss_retriever = vectorstore.as_retriever(
                        search_type = "mmr", search_kwargs = {"k": 5, "fetch_k": 20} 
                    ) 
    
    return faiss_retriever 

faiss_retriever = retrieve_context() 



### Hybrid Retriever (BM25 + FAISS) 

from langchain.retrievers import EnsembleRetriever

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], 
    weights=[0.7, 0.3]
) 



