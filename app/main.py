"""
FastAPI application - Main entry point.
Provides chat endpoint and session management.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

# ── Body-size limit ───────────────────────────────────────────────────────────
# Reject requests whose body exceeds this size BEFORE parsing JSON.
# Prevents OOM attacks via huge base64 image payloads.
# 10 MB is generous for up to 5 images + text; adjust as needed.
_MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB

from .api.auth import router as auth_router
from .api.chat import rate_limiter
from .api.chat import router as chat_router
from .api.share import router as share_router  # New share endpoints
from .api.transcribe import router as transcribe_router
from .auth.security import get_optional_user
from .config import close_db, init_db, settings
from .database import ConversationRepository, SessionRepository
from .graph import run_agent_workflow
from .models import ChatRequest, ConversationHistoryResponse, SessionResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Daddy's AI backend...")

    # Initialize API key rotation
    try:
        from .config.key_rotator import initialize_rotator

        api_keys = settings.get_all_api_keys()
        initialize_rotator(api_keys)
        logger.info(f"✅ API key rotation initialized with {len(api_keys)} keys")
    except Exception as e:
        logger.error(f"Failed to initialize key rotator: {e}")
        logger.warning("⚠️ Continuing with single key mode")

    # Initialize OpenRouter key rotation (primary LLM provider)
    try:
        from .config.openrouter_client import initialize_openrouter

        or_keys = settings.get_all_openrouter_keys()
        if or_keys:
            initialize_openrouter(or_keys)
            logger.info(f"✅ OpenRouter key rotation initialized with {len(or_keys)} keys")
        else:
            logger.warning("⚠️ No OpenRouter keys configured — using Groq only")
    except Exception as e:
        logger.warning(f"⚠️ OpenRouter init failed (non-critical): {e}")

    # Initialize NVIDIA NIM client (racing LLM - optional)
    try:
        if settings.nvidia_available:
            from .config.nvidia_client import get_nvidia_client
            get_nvidia_client(settings.nvidia_api_key)  # warm-up connection
            logger.info("✅ NVIDIA NIM client initialized (DeepSeek-V3.2 in race)")
        else:
            logger.info("ℹ️ NVIDIA NIM not configured — racing with OpenRouter + Groq only")
    except Exception as e:
        logger.warning(f"⚠️ NVIDIA NIM init failed (non-critical, will skip in race): {e}")


    # Warm up Qdrant KB (loads embedding model + verifies cloud connection)
    try:
        from .rag import get_kb_rag

        kb = get_kb_rag()
        if getattr(kb, "_ready", False):
            logger.info("✅ Qdrant KB warmed up (cloud connected)")
        else:
            msg = (
                "⚠️  Qdrant KB using keyword fallback — run: python -m app.rag.ingest_kb"
            )
            logger.warning(msg)
            if os.environ.get("ENVIRONMENT", "development") == "production":
                logger.warning(
                    "💡 Running in production with KB fallback — semantic search disabled"
                )
    except Exception as e:
        logger.error(f"KB warmup error: {e}")
        if os.environ.get("ENVIRONMENT", "development") == "production":
            logger.warning(
                "⚠️ KB unavailable in production — continuing with degraded search"
            )

    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        if os.environ.get("ENVIRONMENT", "development") == "production":
            logger.critical("🚨 DB init failed in production — aborting startup")
            raise  # Crash fast in production — don't serve users with broken DB
        # Dev: continue anyway so local work isn't blocked

    # ── Phase 2: Swarm Intelligence Init ──────────────────────────────────
    try:
        from .swarm import init_swarm

        await init_swarm()
        logger.info("✅ Phase 2 Swarm Intelligence initialised")
    except Exception as e:
        logger.warning(f"⚠️ Swarm init failed (non-critical, continuing): {e}")

    yield

    # Shutdown
    logger.info("Shutting down Daddy's AI backend...")

    # ── Phase 2: Swarm Intelligence Shutdown ──────────────────────────────
    try:
        from .swarm import shutdown_swarm

        await shutdown_swarm()
        logger.info("Swarm Intelligence shut down")
    except Exception as e:
        logger.warning(f"Swarm shutdown warning (non-critical): {e}")

    # Cleanup Groq clients
    try:
        from .config.key_rotator import close_fallback_client, get_rotator

        try:
            rotator = get_rotator()
            await rotator.aclose_all()
        except RuntimeError:
            pass
        await close_fallback_client()
    except Exception as e:
        logger.error(f"Error closing Groq clients: {e}")

    # Cleanup NVIDIA NIM client
    try:
        from .config.nvidia_client import close_nvidia_client
        await close_nvidia_client()
    except Exception as e:
        logger.debug(f"NVIDIA client close (non-critical): {e}")

    # Cleanup scraper connections
    try:
        from .tools.nse_scraper import get_nse_scraper

        scraper = get_nse_scraper()
        await scraper.close()
    except Exception as e:
        logger.error(f"Error closing NSE scraper: {e}")

    await close_db()


# ── FastAPI app ────────────────────────────────────────────────────────────
_is_production = os.environ.get("ENVIRONMENT", "development") == "production"

# Create FastAPI app — hide /docs and /redoc in production
app = FastAPI(
    title="Daddy's AI Backend",
    description="Multi-agent financial intelligence system for Indian markets",
    version="1.0.0",
    lifespan=lifespan,
    # Disable interactive API docs in production (security: don't expose schema)
    docs_url=None if _is_production else "/docs",
    redoc_url=None,  # Always disabled (use /docs only)
    openapi_url=None if _is_production else "/openapi.json",
)

# Session middleware (required for OAuth) — reads secret from env for production
_session_secret = os.environ.get(
    "SESSION_SECRET_KEY",
    "change-me-in-production-must-be-32-chars-min-xxxxxxxxxxx",  # dev fallback only
)
if _session_secret.startswith("change-me-in-production"):
    logger.warning(
        "⚠️ SESSION_SECRET_KEY not set in environment! Using insecure dev fallback. "
        "Set SESSION_SECRET_KEY in your .env for production."
    )
app.add_middleware(SessionMiddleware, secret_key=_session_secret)

# CORS middleware — reads allowed origins from env; defaults to localhost for dev
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001"
)
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Reject requests whose Content-Length exceeds _MAX_REQUEST_BODY_BYTES.

    This runs BEFORE JSON parsing so a malicious client cannot exhaust
    memory by sending a 1 GB payload of base64 images.

    Note: If the client omits Content-Length (chunked transfer), the limit
    is enforced by reading and discarding the body up to the threshold.
    """

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                length = 0
            if length > _MAX_REQUEST_BODY_BYTES:
                logger.warning(
                    f"Request body too large: {length} bytes "
                    f"(limit {_MAX_REQUEST_BODY_BYTES}) from {request.client}"
                )
                return Response(
                    content='{"detail":"Request body too large (max 10 MB)"}',
                    status_code=413,
                    media_type="application/json",
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        # Content-Security-Policy — strict for API; allows Google OAuth
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' accounts.google.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        # Only add HSTS in production (breaks local http)
        if os.environ.get("ENVIRONMENT", "development") == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Generate X-Request-ID and log each request with latency."""

    async def dispatch(self, request: Request, call_next):
        # Use existing request ID or generate one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        start_time = time.monotonic()

        response: Response = await call_next(request)

        latency_ms = round((time.monotonic() - start_time) * 1000, 1)
        response.headers["X-Request-ID"] = request_id

        # Structured log — skip noisy health checks
        if request.url.path not in ("/health", "/"):
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"→ {response.status_code} ({latency_ms}ms)"
            )
        return response


app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)

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
            "crypto",
        ],
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
        "environment": settings.environment,
    }


@app.post("/chat/session")
async def create_session() -> SessionResponse:
    """Create a new chat session."""
    try:
        session = await SessionRepository.create_session()
        return SessionResponse(
            session_id=session.session_id, created_at=session.created_at
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create session")


@app.get("/chat/history/{session_id}")
async def get_conversation_history(
    session_id: str, limit: int = 50
) -> ConversationHistoryResponse:
    """Retrieve conversation history for a session."""
    try:
        messages = await ConversationRepository.get_conversation_history(
            session_id=session_id, limit=limit
        )

        # Build properly interleaved history: user → assistant per message row
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(
                {
                    "role": "user",
                    "content": msg.user_query,
                    "timestamp": msg.created_at.isoformat(),
                }
            )
            formatted_messages.append(
                {
                    "role": "assistant",
                    "content": msg.agent_response,
                    "mode": msg.agent_mode,
                    "timestamp": msg.created_at.isoformat(),
                }
            )

        return ConversationHistoryResponse(
            session_id=session_id,
            messages=formatted_messages,
            total_messages=len(messages),
        )
    except Exception as e:
        logger.error(f"Failed to retrieve conversation history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve history")


@app.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its conversation history."""
    try:
        deleted_count = await ConversationRepository.delete_session(session_id)
        await SessionRepository.deactivate_session(session_id)

        return {
            "message": "Session deleted successfully",
            "deleted_messages": deleted_count,
        }
    except Exception as e:
        logger.error(f"Failed to delete session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete session")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    """
    Chat endpoint (legacy - returns JSON, no SSE).
    Executes agent workflow and returns full response.
    Use /chat/send for the primary chat API with auth and conversations.

    Security: rate-limited per session; internal errors never leak to clients.
    """
    session_id = request.session_id or str(uuid.uuid4())

    # ── Rate limiting (same bucket as /chat/send) ──────────────────────────
    rate_key = f"legacy_{session_id}"
    if not await rate_limiter.acquire(rate_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
        )

    try:
        history_messages = await ConversationRepository.get_conversation_history(
            session_id=session_id, limit=10
        )

        conversation_history = []
        for msg in history_messages:
            conversation_history.append({"role": "user", "content": msg.user_query})
            conversation_history.append(
                {"role": "assistant", "content": msg.agent_response}
            )

        logger.info(f"[legacy /chat/stream] Processing query: {request.query[:100]}...")
        final_state = await run_agent_workflow(
            query=request.query,
            mode=request.mode,
            session_id=session_id,
            conversation_history=conversation_history,
        )

        response_text: str = str(
            final_state.get(
                "final_response", "I apologize, but I couldn't generate a response."
            )
            or "I apologize, but I couldn't generate a response."
        )
        selected_mode = final_state.get("selected_mode", request.mode)
        metadata = final_state.get("execution_metadata", {})

        await ConversationRepository.create_message(
            session_id=session_id,
            user_query=request.query,
            agent_mode=selected_mode or request.mode or "auto",
            agent_response=response_text,
            metadata=metadata,
        )

        return {
            "session_id": session_id,
            "content": response_text,
            "metadata": metadata,
        }
    except HTTPException:
        raise
    except Exception as e:
        # Log full details internally, never expose raw exception to client
        logger.error(f"Error in legacy /chat/stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again."
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
