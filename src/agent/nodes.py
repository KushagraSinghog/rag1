import json 
import re 
import time 

from langchain_ollama import ChatOllama 

from agent.state import AgentState 
from agent.prompt import ROUTER_PROMPT, SYNTHESIS_PROMPT 
from tools.rag_tool import book_search 
from tools.web_search_tool import web_search

llm = ChatOllama(
    model = "qwen3:0.6b", temperature = 0.1
) 

def content_to_text(value) -> str: 
    """ 
    Normalize the tool results into text
    """ 

    if isinstance(value, str): 
        return value 

    if isinstance(value, (dict, list)): 
        return json.dumps(value, ensure_asicii = False, default = str) 

    return str(value) 

def parse_json(value: str) -> dict: 
    """
    Parses json tool output
    """ 

    try: 
        parsed = json.loads(value) 

        if isinstance(parsed, dict): 
            return parsed 

    except(json.JSONDecodeError, TypeError): 
        pass 

    return {} 

def route_query(state: AgentState) -> dict: 
    """ 
    chooses which route the query goes to
    """ 

    query = state["query"].strip() 

    normalized = re.sub(r"[!.,?]+$", "", query.lower()).strip() 

    greetings = { 
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "wassup", "sup", 
        "howdy", "yo", "salutations", "bonjour", "hola", "ciao", "namaste", "shalom", "how are you", "what's up",
        "how's it going", "how are you doing", "how's everything", "how's life",
    }    

    if normalized in greetings: 
        return { 
            "route": "greeting", 
            "routing reason": "Simple greeting/message"
        } 

    response = llm.invoke(
        ROUTER_PROMPT.format(query = query)
    ) 

    raw_route = str(response.content).strip().lower() 

    match = re.search(r"\b(booksearch|websearch|both|greeting)\b", raw_route) 

    if match: 
        route = match.group(1) 

    else: 
        route = "websearch" 

    return {
        "route": route, 
        "routing_reason": raw_route
    } 

def greeting_node(state: AgentState) -> dict: 
    """
    Handle basic conversations and greetings. 
    """ 

    query = state["query"].strip().lower() 

    if "thank" in query: 
        answer = "You're welcome! Feel free to ask me anything about machine learning." 

    else: 
        answer = (
            "Hey! I'm your AI/ML knowledge assistant. "
            "I can answer questions using the ML book, "
            "search the web, or combine both sources."
        ) 

    return {
        "answer": answer, 
        "sources": [], 
        "latency": 0.0
    } 

def book_node(state: AgentState) -> dict: 
    """ 
    Executes the RAG using the book_search tool.
    """ 

    start = time.perf_counter() 

    result = book_search.invoke(
        {
            "query": state["query"]
        }
    ) 

    raw_result = content_to_text(result) 
    parsed_result = parse_json(raw_result) 

    answer = parsed_result.get("answer", raw_result) 
    sources = parsed_result.get("sources", []) 
    latency = parsed_result.get("latency", round(time.perf_counter() - start, 3)) 

    return {
        "book_result": raw_result, 
        "answer": str(answer), 
        "sources": sources if isinstance(sources, list) else [], 
        "latency": latency
    } 

def web_node(state: AgentState) -> dict: 
    """ 
    Performs web search using the web_search tool
    """ 

    result = web_search.invoke(
        {
            "query": state["query"], 
            "max_reuslts": 5, 
        }
    ) 

    return {
        "web_result": content_to_text(result)
    } 

def both_node(state: AgentState) -> dict: 
    """ 
    Uses both web_search and book_search tools to get results. 
    """ 

    start = time.perf_counter() 

    book_result = book_search.invoke(
        {
            "query": state["query"]
        }
    ) 

    web_result = web_search.invoke(
        {
            "query": state["query"], 
            "max_results": 5
        }
    ) 

    raw_book = content_to_text(book_result) 
    raw_web = content_to_text(web_result) 
    parsed_book_result = parse_json(raw_book) 

    return {
        "book_result": raw_book, 
        "web_result": raw_web, 
        "sources": parsed_book_result.get("sources", []),
        "latency": round(time.perf_counter() - start, 3)
    } 

def synthesis_node(state: AgentState) -> dict: 
    """ 
    Gives a structured output based on the collected results.
    """ 

    response = llm.invoke(
        SYNTHESIS_PROMPT.format(
            query = state["query"], 
            book_result = state.get("book_result", "not used"), 
            web_result = state.get("web_result", "not used")
        )
    ) 

    return {
        "answer": str(response.content).strip()
    }