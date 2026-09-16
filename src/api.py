import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

# Ensure src directory is in python path for clean imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chat import chat

# Initialize FastAPI App
app = FastAPI(
    title="Technical Book RAG API",
    description="FastAPI deployment for Hands-On Machine Learning RAG Agent powered by FAISS + BM25 hybrid search & local LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Data Schemas ---

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        example="What is supervised learning and what are its main algorithms?",
        description="User query about the technical book content",
    )
    top_k: Optional[int] = Field(
        5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve for context",
    )
    include_sources: Optional[bool] = Field(
        True,
        description="Whether to include source page numbers in response",
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[int]
    latency_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    llm_model: str
    embeddings_model: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    detail: str


# --- API Routes ---

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root path to interactive Web UI."""
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health, vector store availability, and LLM readiness."""
    try:
        from retriever import VECTORSTORE_PATH
        vstore_exists = Path(VECTORSTORE_PATH).exists()
        
        return HealthResponse(
            status="healthy" if vstore_exists else "degraded",
            vectorstore_loaded=vstore_exists,
            llm_model="qwen3:0.6b",
            embeddings_model="nomic-embed-text",
            version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check error: {str(e)}",
        )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["RAG Agent"],
    summary="Query the technical book RAG agent",
)
async def query_rag(request: QueryRequest):
    """
    Process user query through query router, hybrid retrieval (BM25 + FAISS),
    and local LLM answer synthesis with source page attribution.
    """
    try:
        start_t = time.time()
        
        # Execute chat function
        result = chat(request.query)
        
        latency = result.get("latency_seconds")
        if latency is None:
            latency = round(time.time() - start_t, 3)
            
        sources = result.get("sources", []) if request.include_sources else []
        
        return QueryResponse(
            query=request.query,
            answer=result.get("answer", "No answer generated."),
            sources=sources,
            latency_seconds=latency,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG chain: {str(e)}",
        )


@app.post(
    "/api/v1/chat",
    response_model=QueryResponse,
    tags=["RAG Agent"],
    summary="Chat endpoint (alias for /api/v1/query)",
)
async def chat_alias(request: QueryRequest):
    return await query_rag(request)


@app.get("/ui", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_ui():
    """Serve modern interactive Web UI for querying the technical book RAG agent."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technical Book RAG Agent</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --bg-card-border: rgba(255, 255, 255, 0.08);
            --accent-glow: rgba(99, 102, 241, 0.25);
            --accent-primary: #6366f1;
            --accent-hover: #4f46e5;
            --accent-cyan: #06b6d4;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --user-bubble: #312e81;
            --assistant-bubble: #1e293b;
            --badge-bg: rgba(99, 102, 241, 0.15);
            --badge-text: #818cf8;
            --radius-lg: 16px;
            --radius-md: 10px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 10% 10%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 90%, rgba(6, 182, 212, 0.12) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--bg-card-border);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-cyan));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.2rem;
            color: #fff;
            box-shadow: 0 4px 14px var(--accent-glow);
        }

        .logo-text h1 {
            font-size: 1.2rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .logo-text p {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .header-actions {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(34, 197, 94, 0.1);
            color: #4ade80;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid rgba(34, 197, 94, 0.2);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.9); }
            100% { opacity: 1; transform: scale(1); }
        }

        .doc-link {
            color: var(--text-secondary);
            text-decoration: none;
            font-size: 0.85rem;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid var(--bg-card-border);
            transition: all 0.2s;
        }

        .doc-link:hover {
            color: var(--text-primary);
            border-color: var(--accent-primary);
        }

        main {
            flex: 1;
            max-width: 1000px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .chat-container {
            flex: 1;
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--bg-card-border);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            min-height: 480px;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            overflow-y: auto;
            max-height: calc(100vh - 280px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .message {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-width: 85%;
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            align-self: flex-end;
        }

        .message.assistant {
            align-self: flex-start;
        }

        .bubble {
            padding: 1rem 1.25rem;
            border-radius: 14px;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .user .bubble {
            background: var(--user-bubble);
            border: 1px solid rgba(129, 140, 248, 0.3);
            color: #ffffff;
            border-bottom-right-radius: 4px;
        }

        .assistant .bubble {
            background: var(--assistant-bubble);
            border: 1px solid var(--bg-card-border);
            color: var(--text-primary);
            border-bottom-left-radius: 4px;
            white-space: pre-wrap;
        }

        .meta-row {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.75rem;
            color: var(--text-muted);
            padding: 0 4px;
        }

        .sources-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
        }

        .source-pill {
            background: var(--badge-bg);
            color: var(--badge-text);
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            font-family: 'Fira Code', monospace;
            border: 1px solid rgba(99, 102, 241, 0.2);
        }

        .latency-tag {
            font-family: 'Fira Code', monospace;
            color: var(--text-muted);
        }

        .samples-container {
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding-bottom: 4px;
        }

        .sample-chip {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid var(--bg-card-border);
            color: var(--text-secondary);
            padding: 8px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s;
        }

        .sample-chip:hover {
            background: rgba(99, 102, 241, 0.15);
            border-color: var(--accent-primary);
            color: var(--text-primary);
        }

        .input-form {
            display: flex;
            gap: 12px;
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--bg-card-border);
            border-radius: 14px;
            padding: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        .input-form input {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: inherit;
            padding: 8px 12px;
        }

        .input-form input::placeholder {
            color: var(--text-muted);
        }

        .send-btn {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-hover));
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
            box-shadow: 0 4px 12px var(--accent-glow);
        }

        .send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(99, 102, 241, 0.4);
        }

        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .loading-dots {
            display: flex;
            gap: 4px;
            align-items: center;
        }

        .loading-dot {
            width: 6px;
            height: 6px;
            background: white;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }

        .loading-dot:nth-child(1) { animation-delay: -0.32s; }
        .loading-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-group">
            <div class="logo-icon">RAG</div>
            <div class="logo-text">
                <h1>Hands-On Machine Learning AI Assistant</h1>
                <p>Hybrid Search (FAISS + BM25) • Local Qwen3-0.6B</p>
            </div>
        </div>
        <div class="header-actions">
            <div class="status-badge" id="statusBadge">
                <div class="status-dot"></div>
                <span>FastAPI Active</span>
            </div>
            <a href="/docs" target="_blank" class="doc-link">API Specs (/docs)</a>
        </div>
    </header>

    <main>
        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="bubble">Hello! I am your Technical Book RAG assistant. Ask me any question from <i>Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow</i>.</div>
                <div class="meta-row">
                    <span>System Ready</span>
                </div>
            </div>
        </div>

        <div class="samples-container">
            <div class="sample-chip" onclick="fillSample('What is supervised learning?')">What is supervised learning?</div>
            <div class="sample-chip" onclick="fillSample('Explain decision trees vs random forests')">Decision Trees vs Random Forests</div>
            <div class="sample-chip" onclick="fillSample('What is cross-validation?')">What is cross-validation?</div>
            <div class="sample-chip" onclick="fillSample('How does gradient descent work?')">How does gradient descent work?</div>
        </div>

        <form class="input-form" id="chatForm">
            <input type="text" id="queryInput" placeholder="Type your question about the machine learning book..." autocomplete="off" required />
            <button type="submit" class="send-btn" id="sendBtn">
                <span>Ask Agent</span>
            </button>
        </form>
    </main>

    <script>
        const chatContainer = document.getElementById('chatContainer');
        const chatForm = document.getElementById('chatForm');
        const queryInput = document.getElementById('queryInput');
        const sendBtn = document.getElementById('sendBtn');

        function fillSample(text) {
            queryInput.value = text;
            queryInput.focus();
        }

        function appendMessage(sender, text, sources = [], latency = null) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${sender}`;

            const bubbleDiv = document.createElement('div');
            bubbleDiv.className = 'bubble';
            bubbleDiv.textContent = text;
            msgDiv.appendChild(bubbleDiv);

            if (sender === 'assistant' && (sources.length > 0 || latency !== null)) {
                const metaRow = document.createElement('div');
                metaRow.className = 'meta-row';

                if (sources.length > 0) {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'sources-list';
                    sourcesDiv.innerHTML = '<span>Sources (Pages):</span> ' + 
                        sources.map(p => `<span class="source-pill">p. ${p}</span>`).join(' ');
                    metaRow.appendChild(sourcesDiv);
                }

                if (latency !== null) {
                    const latencySpan = document.createElement('span');
                    latencySpan.className = 'latency-tag';
                    latencySpan.textContent = `⚡ ${latency}s`;
                    metaRow.appendChild(latencySpan);
                }

                msgDiv.appendChild(metaRow);
            }

            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const query = queryInput.value.trim();
            if (!query) return;

            // User message
            appendMessage('user', query);
            queryInput.value = '';

            // Disable input
            sendBtn.disabled = true;
            sendBtn.innerHTML = `
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                <span>Thinking...</span>
            `;

            try {
                const response = await fetch('/api/v1/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: 5, include_sources: true })
                });

                if (!response.ok) {
                    const errData = await response.json();
                    throw new Error(errData.detail || 'Server error');
                }

                const data = await response.json();
                appendMessage('assistant', data.answer, data.sources, data.latency_seconds);
            } catch (err) {
                appendMessage('assistant', `⚠️ Error: ${err.message}`);
            } finally {
                sendBtn.disabled = false;
                sendBtn.innerHTML = '<span>Ask Agent</span>';
            }
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
