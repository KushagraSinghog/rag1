from sentence_transformers import CrossEncoder 

model_name = "BAAI/bge-reranker-base" 

reranker = CrossEncoder(model_name) 



def rerank_docs(query, documents, top_n=5): 
    if not documents: 
        return [] 

    pairs = [
        (query, doc.page_content) for doc in documents
    ] 

    scores = reranker.predict(pairs) 

    docs_ranked = sorted( 
        zip(documents, scores), key= lambda x: x[1], reverse=True
    ) 

    return [doc for doc, score in docs_ranked[:top_n]]