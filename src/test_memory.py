#How to run: cd src 
#            $env:USE_TF = "0" 
#            python test_memory.py 

from agent.graph import ask_agent 

thread_id = "mem_test_2" 

response_1 = ask_agent("What is supervised learning", thread_id) 
print("\n Response 1: ", response_1["answer"] ) 

response_2 = ask_agent("Give me an example of it", thread_id) 
print("\n Response 2: ", response_2["answer"] ) 

