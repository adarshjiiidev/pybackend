"""
FastAPI application - Main entry point.
Provides chat endpoint and session management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
import logging
import uuid
from datetime import datetime

from .config import settings, init_db, close_db
from .auth.cache import user_cache, blacklist_cache, otp_cache
import asyncio
from .models import ChatRequest, SessionResponse, ConversationHistoryResponse
from .graph import run_agent_workflow
from .database import ConversationRepository, SessionRepository
from .api.auth import router as auth_router
from .api.chat import router as chat_router
from .api.transcribe import router as transcribe_router
from .api.share import router as share_router  # New share endpoints

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def cleanup_caches_periodic():
    """Background task to cleanup expired cache entries."""
    while True:
        try:
            await asyncio.sleep(600)  # Cleanup every 10 minutes
            user_cache.cleanup()
            blacklist_cache.cleanup()
            otp_cache.cleanup()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Daddy's AI backend...")
    
    # Start cache cleanup task
    cleanup_task = asyncio.create_task(cleanup_caches_periodic())

    # Initialize API key rotation
    try:
        from .config.key_rotator import initialize_rotator
        api_keys = settings.get_all_api_keys()
        initialize_rotator(api_keys)
        logger.info(f"✅ API key rotation initialized with {len(api_keys)} keys")
    except Exception as e:
        logger.error(f"Failed to initialize key rotator: {e}")
        logger.warning("⚠️ Continuing with single key mode")
    
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue anyway for development
    
    yield
    
    # Shutdown
    logger.info("Shutting down Daddy's AI backend...")

    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Cleanup Groq clients
    try:
        from .config.key_rotator import get_rotator, close_fallback_client
        try:
            rotator = get_rotator()
            await rotator.aclose_all()
        except RuntimeError:
            # Rotator might not have been initialized
            pass
        await close_fallback_client()
    except Exception as e:
        logger.error(f"Error closing Groq clients: {e}")

    # Cleanup scraper connections
    try:
        from .tools.nse_scraper import get_nse_scraper
        scraper = get_nse_scraper()
        await scraper.close()
    except Exception as e:
        logger.error(f"Error closing NSE scraper: {e}")

    await close_db()


# Create FastAPI app
app = FastAPI(
    title="Daddy's AI Backend",
    description="Multi-agent financial intelligence system for Indian markets",
    version="1.0.0",
    lifespan=lifespan
)

# Session middleware (required for OAuth)
app.add_middleware(
    SessionMiddleware,
    secret_key="your-secret-key-min-32-chars-long-for-session-data-encryption"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(transcribe_router)
app.include_router(share_router)  # Public sharing endpoints


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
    Chat endpoint (legacy - returns JSON, no SSE).
    Executes agent workflow and returns full response.
    Use /chat/send for the primary chat API with auth and conversations.
    """
    session_id = request.session_id or str(uuid.uuid4())

    try:
        history_messages = await ConversationRepository.get_conversation_history(
            session_id=session_id,
            limit=10
        )

        conversation_history = []
        for msg in history_messages:
            conversation_history.append({"role": "user", "content": msg.user_query})
            conversation_history.append({"role": "assistant", "content": msg.agent_response})

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

        await ConversationRepository.create_message(
            session_id=session_id,
            user_query=request.query,
            agent_mode=selected_mode or request.mode or "auto",
            agent_response=response_text,
            metadata=metadata
        )

        return {
            "session_id": session_id,
            "content": response_text,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
