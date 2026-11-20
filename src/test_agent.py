#How to run: cd src 
#            $env:USE_TF = "0" 
#            python test_agent.py

from pprint import pprint
import traceback
from agent.graph import ask_agent


def main():
    test_queries = [
        "Hey!",
        "Define supervised learning.",
        "What are the latest developments in agentic AI?",
        (
            "Explain the machine learning concept from the book "
            "and compare it with current industry practices."
        ),
    ]

    for question in test_queries:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        try:
            result = ask_agent(question)
            pprint(result)

        except Exception:
            traceback.print_exc() 


if __name__ == "__main__":
    main()
