from tools.rag_tool import book_search 
from tools.web_search_tool import web_search 

def main(): 
    print("\n Testing Book Rag Tool ") 

    rag_result = book_search.invoke(
        {
            "query": "define supervised learning"
        }
    ) 
    print(rag_result) 

    print("\n Testing Web Search Tool ") 

    web_results = web_search.invoke(
        {
            "query": "what are the recent developments in agentic ai", 
            "max_results": 5,
        }
    ) 

    print(web_results) 
    