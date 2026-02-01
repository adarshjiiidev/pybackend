"""
FastAPI application - Main entry point.
Provides streaming chat endpoint and session management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import logging
import uuid
import asyncio
import json
from datetime import datetime

from .config import settings, init_db, close_db
from .models import ChatRequest, SessionResponse, ConversationHistoryResponse
from .graph import run_agent_workflow
from .database import ConversationRepository, SessionRepository
from .utils import stream_groq_response

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Daaddys AI backend...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue anyway for development
    
    yield
    
    # Shutdown
    logger.info("Shutting down Daaddys AI backend...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="Daaddys AI",
    description="Autonomous Financial AI Agent for Indian Stock Markets and Crypto",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Daaddys AI",
        "version": "0.1.0",
        "status": "operational",
        "agents": [
            "market_research",
            "realtime_analysis",
            "portfolio",
            "explainer",
            "crypto"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with database status."""
    from .config.database import Database
    
    db_status = "unknown"
    try:
        if await Database.ping():
            db_status = "connected"
        else:
            db_status = "disconnected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "environment": settings.environment
    }


@app.post("/chat/session")
async def create_session() -> SessionResponse:
    """Create a new chat session."""
    try:
        session = await SessionRepository.create_session()
        return SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@app.get("/chat/history/{session_id}")
async def get_conversation_history(
    session_id: str,
    limit: int = 50
) -> ConversationHistoryResponse:
    """Retrieve conversation history for a session."""
    try:
        messages = await ConversationRepository.get_conversation_history(
            session_id=session_id,
            limit=limit
        )
        
        formatted_messages = [
            {
                "role": "user",
                "content": msg.user_query,
                "timestamp": msg.created_at.isoformat()
            }
            for msg in messages
        ] + [
            {
                "role": "assistant",
                "content": msg.agent_response,
                "mode": msg.agent_mode,
                "timestamp": msg.created_at.isoformat()
            }
            for msg in messages
        ]
        
        # Sort by timestamp
        formatted_messages.sort(key=lambda x: x["timestamp"])
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=formatted_messages,
            total_messages=len(messages)
        )
    except Exception as e:
        logger.error(f"Failed to retrieve conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@app.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its conversation history."""
    try:
        deleted_count = await ConversationRepository.delete_session(session_id)
        await SessionRepository.deactivate_session(session_id)
        
        return {
            "message": "Session deleted successfully",
            "deleted_messages": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint.
    Executes agent workflow and streams response via Server-Sent Events.
    """
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    async def generate_stream():
        try:
            # Retrieve conversation history
            history_messages = await ConversationRepository.get_conversation_history(
                session_id=session_id,
                limit=10
            )
            
            # Format for LLM context
            conversation_history = []
            for msg in history_messages:
                conversation_history.append({"role": "user", "content": msg.user_query})
                conversation_history.append({"role": "assistant", "content": msg.agent_response})
            
            # Run agent workflow
            logger.info(f"Processing query: {request.query[:100]}...")
            final_state = await run_agent_workflow(
                query=request.query,
                mode=request.mode,
                session_id=session_id,
                conversation_history=conversation_history
            )
            
            response_text = final_state.get("final_response", "I apologize, but I couldn't generate a response.")
            selected_mode = final_state.get("selected_mode", request.mode)
            metadata = final_state.get("execution_metadata", {})
            
            # Stream response
            async for chunk in stream_groq_response(response_text):
                yield chunk
            
            # Send final metadata chunk
            yield f"data: {json.dumps({'done': True, 'metadata': metadata})}\n\n"
            
            # Save conversation asynchronously (don't block streaming)
            asyncio.create_task(
                ConversationRepository.create_message(
                    session_id=session_id,
                    user_query=request.query,
                    agent_mode=selected_mode or request.mode or "auto",
                    agent_response=response_text,
                    metadata=metadata
                )
            )
            
        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            error_message = f"I encountered an error: {str(e)}"
            yield f"data: {json.dumps({'content': error_message, 'done': True, 'error': True})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
