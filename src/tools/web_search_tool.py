import json 

from ddgs import DDGS 
from langchain_core.tools import tool 

@tool 
def web_search(query: str, max_results: int=5) -> str: 
    """
    Search the web for recent information, research, developements 
    and topics not covered in the machine learning book. 
    Return result titles, snippets and URLs.

    """ 

    try: 
        max_results = max(1, min(max_results, 0)) 

        with DDGS() as ddgs: 
            results = list(
                ddgs.text(query, max_results=max_results)
            ) 

        if not results: 
            return json.dumps(
                {
                    "query": query, 
                    "results": [], 
                    "message": "no results found",
                }
            ) 

        cleaned_results = [
            {
                "title": item.get("title", ""), 
                "snippet": item.get("body", ""), 
                "url": item.get("href", ""),
            } 
            for item in results
        ] 

        return json.dumps(
            {
                "query": query, 
                "results": cleaned_results,
            }, 
            ensure_ascii=False
        )

    except Exception as exc: 
        return json.dumps(
            {
                "query": query, 
                "error": f"web search failed: {exc}",
            }
        )