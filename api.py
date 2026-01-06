from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from agent.graph import build_graph
from langchain_core.messages import HumanMessage

# Initialize FastAPI app
app = FastAPI(
    title="Daddy's AI - Financial Agent API",
    description="Multi-layer financial agent for Indian stock market analysis and LTP Calculator",
    version="1.0.0"
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js dev server
        "http://127.0.0.1:3000",
        "https://your-frontend.vercel.app",  # Add your production URL
        "*"  # Allow all during development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the agent graph once at startup
print("[INFO] Building agent graph...")
agent = build_graph()
print("[INFO] Agent ready!")

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: Optional[str] = None
    language: Optional[str] = None
    error: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "healthy", "service": "Daddy's AI Financial Agent"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main endpoint for chatting with the financial agent.
    
    Examples:
    - "compare tcs and itc" → Professional fundamental analysis
    - "wtb kya hota hai" → Hindi LTP Calculator response
    - "what is wtb" → English LTP Calculator response
    - "hi" → Casual greeting with system knowledge
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Invoke the agent
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
        }
        
        result = agent.invoke(initial_state)
        
        # Extract response
        final_response = result.get("final_response", "")
        analysis = result.get("analysis_result", {})
        error = result.get("error")
        
        if error:
            return ChatResponse(
                response=f"Error: {error}",
                error=error
            )
        
        return ChatResponse(
            response=final_response,
            intent=analysis.get("intent") if isinstance(analysis, dict) else None,
            language=analysis.get("language") if isinstance(analysis, dict) else None
        )
        
    except Exception as e:
        print(f"[ERROR] Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming endpoint (for real-time responses)
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming endpoint for real-time responses.
    For now, returns the same as /chat. 
    Implement SSE/WebSocket for true streaming.
    """
    return await chat(request)

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable hot reload for development
        log_level="info"
    )
