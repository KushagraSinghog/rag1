# defines the information passed between the graph nodes and the tools, including the web search tool and the book search tool. 

from typing import Annotated, Literal, TypedDict 
from langchain_core.messages import AnyMessage 
from langgraph.graph.message import add_messages 

Route = Literal["greeting", "book_search", "web_search", "both"] 

class AgentState(TypedDict, total = False): 

    query: str 
    search_query: str 
    
    messages = Annotated[
        list[AnyMessage], add_messages
    ] 
    
    route: Route 
    book_result: str 
    web_result: str 
    answer: str 
    sources: list[int] 
    latency: float 
    routing_reason: str 