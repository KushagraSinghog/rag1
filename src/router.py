GREETINGS = { 
    "hi", "hey", "how you doing", "how are you", "how you feeling", 
    "good morning", "good afternoon", "good evening"
} 

def is_greeting(query: str) -> bool : 
    return query.lower().strip() in GREETINGS 

def handle_greeting() -> str : 
    return(
        "Hello. " \
        "my name is mama " \
        "I can answer any question related to the book, Hands On Machine Learning ... " 
    )