#How to run: cd src 
#            $env:USE_TF = "0" 
#            python -m agent.graph

from langgraph.graph import START, END, StateGraph 

from agent.state import AgentState 
from agent.nodes import route_query, greeting_node, book_node, web_node, both_node, synthesis_node

def select_route(state: AgentState) -> str: 
    valid_routes = { "greeting", "book_search", "web_search", "both" } 

    route = state.get("route", "web_search") 

    if route not in valid_routes: 
        return "web_search" 

    return route 

builder = StateGraph(AgentState) 

builder.add_node("router", route_query) 
builder.add_node("greeting", greeting_node) 
builder.add_node("book_search", book_node) 
builder.add_node("web_search", web_node) 
builder.add_node("both", both_node) 
builder.add_node("synthesis", synthesis_node)

builder.add_edge(START, "router") 

builder.add_conditional_edges(
    "router", select_route, 
    {
        "greeting": "greeting", 
        "book_search": "book_search", 
        "web_search": "web_search", 
        "both": "both"
    }
) 

builder.add_edge("greeting", END) 
builder.add_edge("book_search", END) 
builder.add_edge("web_search", "synthesis") 
builder.add_edge("both", "synthesis") 
builder.add_edge("synthesis", END)

agent = builder.compile() 

def ask_agent(query: str) -> dict: 
    """ 
        Entry point for agent
    """ 

    result = agent.invoke({"query": query}) 

    return {
        "query": result.get("query", query), 
        "route": result.get("route", "book_search"), 
        "answer": result.get("answer", ""), 
        "sources": result.get("sources", []), 
        "latency": result.get("latency", 0.0), 
        "routing_reason": result.get("routing_reason", "") 
    } 
