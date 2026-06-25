                                                                 Hybrid RAG Assistant for Technical Book Question Answering

## Overview:

This project is a Retrieval-Augmented Generation (RAG) system designed to answer questions from a technical machine learning textbook. The application combines semantic search, lexical search, and local large language models to provide grounded responses with source attribution.

Unlike basic RAG implementations that rely solely on vector search, this project incorporates a hybrid retrieval architecture using both FAISS and BM25 to improve retrieval quality for definition-based and concept-oriented queries.

The system is built entirely with local components, enabling privacy-preserving inference without relying on external APIs.



## Key Features:

* PDF-based knowledge ingestion
* Text preprocessing and cleaning pipeline
* Recursive chunking strategy for long-form technical content
* Semantic retrieval using FAISS and Nomic embeddings
* Lexical retrieval using BM25
* Hybrid retrieval with EnsembleRetriever
* Local LLM inference using Qwen3
* Source page attribution for generated answers
* Greeting and query routing
* Retrieval evaluation and debugging workflow



## System Architecture:


```text
PDF Book
    │
    ▼
Document Loading (PyPDFLoader)
    │
    ▼
Text Cleaning
    │
    ▼
Chunking
    │
    ▼
Embedding Generation
(nomic-embed-text)
    │
    ▼
FAISS Vector Store
    │
    ├──────────────┐
    │              │
    ▼              ▼
FAISS         BM25 Retriever
Retriever
    │              │
    └──────┬───────┘
           ▼
   Ensemble Retriever
           │
           ▼
      Retrieved Context
           │
           ▼
       Qwen3-0.6B
           │
           ▼
 Answer + Source Pages
```



## Technology Stack:

### Retrieval & NLP:

	1. LangChain
	2. FAISS
	3. BM25
	4. Nomic Embed Text

### LLM:

	1. Qwen3-0.6B
	2. Ollama

### Data Processing:

	1. PyPDFLoader
	2. RecursiveCharacterTextSplitter

### Language:

	1. Python



## Project Structure:


```text
sample_rag/
│
├── src/
│   ├── ingestion.py
│   ├── retriever.py
│   ├── rag_chain.py
│   ├── router.py
│   └── main.py
│
├── vectorstore/
│   ├── index.faiss
│   ├── index.pkl
│   └── chunks.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```



## Retrieval Strategy

### Semantic Retrieval:

	The system uses:

		* nomic-embed-text
		* FAISS vector indexing

	This allows retrieval based on semantic similarity and concept matching.

### Lexical Retrieval:

	A BM25 retriever is added to improve retrieval for:

		* Definitions
		* Acronyms
		* Exact terminology
	* Keyword-heavy queries

### Hybrid Retrieval:

	Both retrievers are combined using an EnsembleRetriever.

	```python
	FAISS + BM25 → EnsembleRetriever → Context
	```

	This approach improves robustness across different query types.



## Challenges Encountered:

### 1. Retrieval Quality

Initial retrieval often returned semantically related content rather than the exact concept definition.

Example:

	Query:

	```text
	What is Machine Learning?
	```

	Retrieved pages frequently contained references to machine learning concepts rather than the actual definition.

### Solution:

	Implemented:

	* BM25 retrieval
	* Hybrid retrieval
	* Query reformulation experiments
	* Retrieval debugging with source inspection

---

### 2. PDF Noise:

The textbook contained:

	* Figure references
	* Code snippets
	* Page artifacts

These reduced embedding quality.

### Solution:

	Created custom preprocessing functions to:

		* Remove figure references
		* Remove page numbering artifacts
		* Filter code-heavy sections



### 3. Small Model Constraints:

The project uses Qwen3-0.6B for local inference.

While lightweight and efficient, the model occasionally struggled with:

	* Comparative reasoning
	* Multi-hop synthesis

This was mitigated through prompt engineering and retrieval improvements.



## Example Queries

### Definition Query

**Input**

```text
Define supervised learning
```

**Output**

```text
Supervised learning is a type of machine learning where
the training data includes labeled examples...
```

**Sources**

```text
[36, 37, 380, 579, 777]
```

---

### Comparison Query

**Input**

```text
Compare supervised learning and reinforcement learning
```

**Output**

```text
Supervised learning focuses on learning from labeled data,
while reinforcement learning learns through interaction
with an environment and reward signals.
```



## Possible Future Improvements:

	* Cross-Encoder Reranking
	* Query Rewriting Pipeline
	* Streamlit Web Interface
	* Conversational Memory
	* LangGraph-based Agent Workflow
	* Automated Retrieval Evaluation



## Key Learnings:

Through this project I gained practical experience with:

	* Retrieval-Augmented Generation (RAG)
	* Embedding models
	* Vector databases
	* Hybrid search systems
	* Retrieval evaluation
	* Prompt engineering
	* Local LLM deployment
	* LangChain pipelines

This project focuses not only on building a RAG system, but also on understanding and improving retrieval quality through experimentation and evaluation.
