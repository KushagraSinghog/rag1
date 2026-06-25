from router import handle_greeting, is_greeting 
from rag_chain import answer_question 

### Creating a chat function

def chat(query: str): 
    if is_greeting(query): 
        return{ 
            "answer": handle_greeting(), 
            "sources": []
        } 
    
    return answer_question(query)