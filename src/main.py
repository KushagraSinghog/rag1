from chat import chat 

def main(): 
    print("Book RAG agent") 
    print("type 'exit' to quit\n")

    while True: 
        query = input("You: ") 
         
        if query.lower() == "exit": 
            break 

        results = chat(query) 

        print("\nAssistant: ") 
        print(results["answer"]) 

        if results["sources"]: 
            print(f"\nSources: {results['sources']}") 

        print() 

if __name__ == "__main__": 
    main()