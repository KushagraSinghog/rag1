# defines the information passed between the graph nodes and the tools, including the web search tool and the book search tool. 

from typing import Any, Literal, TypedDict 

Route = Literal["greeting", "book_search", "web_search", "both"] 

class AgentState(TypedDict, total = False): 

    query: str 
    route: Route 
    
    book_result: str 
    web_result: str 

    answer: str 
    sources: list[int] 
    latency: float 

    routing_reason: str 