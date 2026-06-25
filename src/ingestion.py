import os 
from dotenv import load_dotenv 

load_dotenv() 



### loading pdf

from langchain_community.document_loaders import PyPDFLoader 

data_path = r"D:\Honey\Fun\sample_rag\Data\hands on machine learning with scikit-learn, keras and tensorflow.pdf" 


def load_pdf(data_path: str): 
    """ 
        Loads pdfs as langchain docs
    """ 

    pdf_loader = PyPDFLoader(data_path) 
    documents = pdf_loader.load() 

    print(f"Loaded {len(documents)} pages") 
    
    return documents 

documents = load_pdf(data_path) 



### PDF preprocessing

import re

def preprocess_text(text):
    text = re.sub(r'\n+', '\n', text)

    # Remove figure references
    text = re.sub(r'Figure\s+\d+[-.]?\d*', '', text)

    # Remove page numbers
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)

    lines = text.split("\n")

    filtered = []

    # Remove code
    for line in lines:

        if ">>>" in line:
            continue

        if line.strip().startswith("import "):
            continue

        if line.strip().startswith("from "):
            continue

        filtered.append(line)

    return "\n".join(filtered) 

for doc in documents:
    doc.page_content = preprocess_text(doc.page_content) 

# Verify preprocessing 
print(documents[20].page_content[:1000]) 



### chunking 

from langchain_text_splitters import RecursiveCharacterTextSplitter 

def split_docs(documents): 
    """ 
        This function is splits the doc into chunks
    """ 

    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1100, chunk_overlap = 250, length_function=len
                        ) 
    
    chunks = text_splitter.split_documents(documents) 

    print(f"Created {len(chunks)} chunks") 

    return chunks 

chunks = split_docs(documents) 

### saving chunks 

import pickle 

CHUNKS_PATH = "vectorstore/chunks.pkl" 

with open(CHUNKS_PATH, "wb") as f: 
    pickle.dump(chunks, f) 

print("chunks saved") 



### Chunk cleaning 

def clean_chunk(text): 
    return text.encode("utf-8", errors="ignore").decode("utf-8") 

for chunk in chunks: 
    chunk.page_content = clean_chunk(chunk.page_content)



### VectorDB

VECTORSTORE_PATH = "vectorstore" 

from langchain_community.vectorstores import FAISS 
from langchain_ollama import OllamaEmbeddings

def create_vector_store(chunks):
    """ 
        generate embeddings and create faiss indexes
    """

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore = FAISS.from_documents(
                        documents=chunks, embedding=embeddings
                    ) 
    
    return vectorstore 

vectorstore = create_vector_store(chunks) 


def save_vector_store(vectorstore): 
    """ 
        persist faiss index locally
    """ 

    vectorstore.save_local(VECTORSTORE_PATH) 

    print(f"vector store created at {VECTORSTORE_PATH}") 

saved_vectorstore = save_vector_store(vectorstore)
