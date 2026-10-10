import json 

from langchain_core.tools import tool 
from src.rag_chain import answer_question

@tool 
def book_search(query: str) -> str: 
    """ 
    Answer questions using the hands on machine learning with scikit-learn, keras and tensorflow book. 
    Use this tool for questions about concepts, explanations and information related to that the book. 
    Return the answer, book page sources and latency. 

    """ 

    try: 
        result = answer_question(query) 

        return json.dumps(
            {
                "answer": result.get("answer", ""), 
                "sources": result.get("sources", []), 
                "latency": result.get("latency_seconds"),
            }, 
            ensure_ascii=False
        ) 

    except Exception as exc: 
        return json.dumps(
            {
                "error": f"book search failed: {exc}"
            }
        )

