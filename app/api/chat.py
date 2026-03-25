"""
Chat API endpoints — SSE streaming architecture.
Phase 1: POST /chat/send → instant init (<200ms), returns conversation_id
Phase 2: GET /chat/send/stream/{id} → SSE stream with tokens, status, title

Phase 2 Swarm Mode: mode=swarm triggers MasterOrchestrator deep research pipeline
  - Plans approach with LLM
  - Dispatches N parallel specialist agents
  - Streams real-time agent status events
  - Synthesises final report
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Annotated, Dict, List, Optional, Tuple

from app.auth.security import get_current_user, get_optional_user
from app.graph.workflow import create_agent_graph
from app.models.agent_state import AgentMode, AgentState
from app.models.chat_models import Conversation, Message
from app.models.db_models import User
from app.utils.rate_limiter import TokenBucket
from app.utils.sanitizer import SanitizationError, sanitize_user_message
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize rate limiter: 100 tokens/bucket, refill 10 per second
rate_limiter = TokenBucket(capacity=100, refill_rate=10.0, tokens_per_request=1)

# ── In-process SSE job store ───────────────────────────────────────────────
# Maps conversation_id → (Queue, Task, created_at)
# • Queue  — SSE events written by AI workflow, read by SSE generator
# • Task   — asyncio.Task handle so we can cancel on client disconnect
# • ts     — unix timestamp for TTL-based eviction (prevents memory leak)
_pending_jobs: Dict[str, asyncio.Queue] = {}
_active_tasks: Dict[str, asyncio.Task] = {}  # conversation_id → running AI Task
_job_timestamps: Dict[str, float] = {}  # conversation_id → created_at (unix)
JOB_TTL_SECONDS = 300  # 5 min — evict stale jobs that were never consumed

# ── Anonymous IP rate limit ───────────────────────────────────────────
# TTLCache auto-expires entries after the window (no memory leak, memory-bounded to 50K IPs).
from cachetools import TTLCache as _TTLCache
ANON_MAX_CONVOS = 10   # max new conversations per IP per window
ANON_WINDOW_SECONDS = 86400  # 24-hour window
_anon_ip_counts: _TTLCache = _TTLCache(maxsize=50_000, ttl=ANON_WINDOW_SECONDS)



def _cleanup_stale_jobs():
    """Evict SSE job entries older than JOB_TTL_SECONDS. Called on each new /send request."""
    now = time.monotonic()
    stale = [cid for cid, ts in _job_timestamps.items() if now - ts > JOB_TTL_SECONDS]
    for cid in stale:
        _pending_jobs.pop(cid, None)
        _job_timestamps.pop(cid, None)
        task = _active_tasks.pop(cid, None)
        if task and not task.done():
            task.cancel()
            logger.debug(f"Cancelled stale job for conversation {cid}")


def _get_client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For when behind a proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_anon_ip_limit(ip: str) -> bool:
    """Return True if IP is within anonymous conversation limit, False if exceeded."""
    now = time.monotonic()
    entry = _anon_ip_counts.get(ip)
    if entry:
        count, window_start = entry
        if now - window_start < ANON_WINDOW_SECONDS:
            if count >= ANON_MAX_CONVOS:
                return False  # Limit exceeded
            _anon_ip_counts[ip] = (count + 1, window_start)
        else:
            # Window expired — reset
            _anon_ip_counts[ip] = (1, now)
    else:
        _anon_ip_counts[ip] = (1, now)
    return True


# --- Pydantic models ---

# Max base64 chars per image ≈ 1.5 MB raw (2 MB * 4/3 base64 overhead)
_MAX_IMAGE_B64_CHARS = 2_000_000
_MAX_IMAGES_PER_REQUEST = 5


class SendMessageRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=10000)
    images: Optional[List[str]] = Field(
        default=None,
        description=f"Base64-encoded images (max {_MAX_IMAGES_PER_REQUEST}, each ≤ ~1.5 MB)",
    )
    stream: bool = True
    mode: Optional[str] = Field(
        default=None,
        description="Agent mode: auto, swarm (deep research), market_research, realtime_analysis, etc.",
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        if len(v) > _MAX_IMAGES_PER_REQUEST:
            raise ValueError(
                f"Too many images: got {len(v)}, maximum is {_MAX_IMAGES_PER_REQUEST}"
            )
        for i, img in enumerate(v):
            if not isinstance(img, str):
                raise ValueError(f"Image at index {i} must be a base64 string")
            if len(img) > _MAX_IMAGE_B64_CHARS:
                raise ValueError(
                    f"Image at index {i} is too large (max ~1.5 MB per image)"
                )
        return v


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    images: Optional[list[str]] = None
    created_at: datetime


class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


# --- Background AI processing ---


async def _run_swarm_workflow(
    conversation_id: str,
    user_message_content: str,
    images: Optional[List[str]],
    conversation_history: list,
    queue: asyncio.Queue,
    user: Optional[User] = None,
):
    """
    Background task: runs the MasterOrchestrator swarm pipeline and pushes SSE events.

    Flow (mirrors the architecture diagram):
      01 planning    → LLM plans which agents to spawn
      02 entities    → extracts symbols / topics
      03 dispatching → spawns N agents in parallel
      04 collecting  → gathers all agent results
      05 analyzing   → cross-validates signals
      06 reporting   → ReportAgent generates final markdown
      07 done        → full response delivered

    SSE events emitted:
      status  {phase, message, progress_pct, agents?, plan?}
      swarm_agent {agent_id, agent_type, task_name, status, signal}
      done    {message_id, content, created_at, agent, swarm_metadata}
      title   {title, conversation_id}
      error   {error}
    """
    try:
        # ── Import swarm lazily (avoids circular imports) ──────────────
        try:
            from app.swarm import get_orchestrator
        except ImportError as ie:
            logger.error(f"Swarm module import failed: {ie}")
            # Fallback to regular workflow
            await _run_ai_workflow(
                conversation_id,
                user_message_content,
                images,
                conversation_history,
                queue,
                user,
            )
            return

        # ── Status callback → pushes to SSE queue ────────────────────
        async def on_status(event: dict):
            await queue.put({"event": "status", "data": event})

        # Also push per-agent updates as they come in
        def on_status_sync(event: dict):
            try:
                asyncio.get_event_loop().call_soon_threadsafe(
                    lambda: asyncio.ensure_future(
                        queue.put({"event": "status", "data": event})
                    )
                )
            except Exception:
                pass

        # ── Initial status ────────────────────────────────────────────
        await queue.put(
            {
                "event": "status",
                "data": {
                    "phase": "planning",
                    "message": "🧠 Orchestrator analysing your query and planning approach...",
                    "progress_pct": 3,
                },
            }
        )

        # ── Run orchestrator ──────────────────────────────────────────
        orchestrator = get_orchestrator()
        result = await orchestrator.run(
            query=user_message_content,
            session_id=conversation_id,
            mode="swarm",
            conversation_history=conversation_history,
            on_status=lambda evt: (
                None
            ),  # status pushed via queue in _run_swarm_workflow directly
            images=images,
        )

        final_response = (
            result.final_response
            or result.report_md
            or (
                "I encountered an issue generating the deep research report. Please try again."
            )
        )

        # Truncate only if absurdly long (>30000 chars ≈ ~22000 words)
        MAX_LEN = 30000
        if len(final_response) > MAX_LEN:
            final_response = (
                final_response[:MAX_LEN]
                + "\n\n*[Report truncated — full analysis available on request.]*"
            )

        # ── Save AI message to DB ─────────────────────────────────────
        ai_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=final_response,
            has_code="```" in final_response,
        )
        await ai_message.insert()

        # Update conversation metadata
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id
        )
        if conversation:
            messages = await Message.find(
                Message.conversation_id == conversation_id
            ).count()
            conversation.message_count = messages // 2
            conversation.updated_at = datetime.utcnow()
            await conversation.save()

        # ── Push done event ───────────────────────────────────────────
        await queue.put(
            {
                "event": "done",
                "data": {
                    "message_id": ai_message.message_id,
                    "content": final_response,
                    "created_at": ai_message.created_at.isoformat(),
                    "agent": "swarm_orchestrator",
                    "swarm_metadata": {
                        "agents_used": result.agents_used,
                        "agents_succeeded": result.agents_succeeded,
                        "agents_failed": result.agents_failed,
                        "total_duration_s": round(result.total_duration_s, 2),
                        "signal": result.signal,
                        "confidence": round(result.confidence, 3),
                        "intent": result.metadata.get("intent", ""),
                        "complexity": result.metadata.get("complexity", ""),
                        "plan_reasoning": result.metadata.get("plan_reasoning", ""),
                        "key_findings": result.key_findings[:5],
                    },
                },
            }
        )

        # ── Async title generation ────────────────────────────────────
        msg_count = await Message.find(
            Message.conversation_id == conversation_id
        ).count()
        if msg_count <= 2 and conversation:
            try:
                from app.config.key_rotator import get_rotator as _get_rotator
                from groq import AsyncGroq as _AsyncGroq

                _title_prompt = (
                    f"Generate a short title (3-6 words) for this deep research query.\n"
                    f"Be specific. Use Title Case. No quotes, no periods.\n"
                    f"Examples: 'RELIANCE Deep Dive Analysis', 'Nifty Options OI Report', "
                    f"'IT Sector Prediction Report'\n\n"
                    f"Query: {user_message_content[:200]}\n\nTitle:"
                )
                # Use PRODUCTION models only (preview models cause 401)
                # Snapshot all keys to avoid shared-rotator race conditions
                try:
                    _all_keys = list(_get_rotator().api_keys)
                except RuntimeError:
                    from app.config.settings import settings as _s
                    _all_keys = [_s.groq_api_key]

                generated_title = None
                for _t_model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                    for _key in _all_keys:
                        try:
                            _title_client = _AsyncGroq(api_key=_key)
                            _title_resp = await _title_client.chat.completions.create(
                                model=_t_model,
                                messages=[{"role": "user", "content": _title_prompt}],
                                temperature=0.2,
                                max_tokens=20,
                            )
                            generated_title = (
                                (_title_resp.choices[0].message.content or "")
                                .strip()
                                .strip('"')
                                .strip("'")
                                .rstrip(".")
                            )
                            if generated_title:
                                break
                        except Exception:
                            continue
                    if generated_title:
                        break

                if not generated_title:
                    generated_title = " ".join(user_message_content.split()[:5])

                if len(generated_title) > 50:
                    generated_title = generated_title[:50].rsplit(" ", 1)[0]

                # Prefix with 🔬 to mark as deep research
                generated_title = f"🔬 {generated_title}"
                conversation.title = generated_title
                await conversation.save()

                await queue.put(
                    {
                        "event": "title",
                        "data": {
                            "title": generated_title,
                            "conversation_id": conversation_id,
                        },
                    }
                )
            except Exception as te:
                logger.error(f"Swarm title generation error: {te}")
                fallback = f"🔬 {' '.join(user_message_content.split()[:5])}"
                if conversation:
                    conversation.title = fallback
                    await conversation.save()
                await queue.put(
                    {
                        "event": "title",
                        "data": {"title": fallback, "conversation_id": conversation_id},
                    }
                )

    except Exception as e:
        logger.error(f"Swarm workflow error: {e}", exc_info=True)
        await queue.put({"event": "error", "data": {"error": str(e)}})
    finally:
        await queue.put(None)  # Signal stream end


async def _run_ai_workflow(
    conversation_id: str,
    user_message_content: str,
    images: Optional[List[str]],
    conversation_history: list,
    queue: asyncio.Queue,
    user: Optional[User] = None,
):
    """
    Background task: runs the LangGraph workflow and pushes SSE events to the queue.
    Events: status, token, done, title, error
    """
    try:
        # Push initial status
        await queue.put(
            {
                "event": "status",
                "data": {"stage": "routing", "message": "Routing your query..."},
            }
        )

        # Build agent state
        state: AgentState = {
            "query": user_message_content,
            "images": images,
            "mode": AgentMode.AUTO,
            "session_id": conversation_id,
            "selected_mode": None,
            "extracted_entities": None,
            "conversation_history": conversation_history,
            "tool_results": None,
            "internal_reasoning": None,
            "final_response": None,
            "execution_metadata": None,
            "error": None,
            "needs_web_search": None,
            "enable_research_loop": None,
            "has_vision_content": None,
        }

        # Create and run workflow (compile returns a CompiledGraph with ainvoke)
        compiled_workflow = create_agent_graph()

        # Push thinking status
        await queue.put(
            {"event": "status", "data": {"stage": "thinking", "message": "Thinking..."}}
        )

        final_state = await compiled_workflow.ainvoke(state)

        if final_state is None:
            await queue.put(
                {"event": "error", "data": {"error": "Workflow returned no response."}}
            )
            await queue.put(None)  # Signal end
            return

        final_response = (
            final_state.get("final_response")
            or "I apologize, but I couldn't generate a response."
        )
        selected_mode = final_state.get("selected_mode", "unknown")

        # Truncate extremely long responses
        MAX_RESPONSE_LENGTH = 8000
        if len(final_response) > MAX_RESPONSE_LENGTH:
            final_response = (
                final_response[:MAX_RESPONSE_LENGTH]
                + "\n\n[Response truncated due to length.]"
            )

        # Save AI message to DB
        ai_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=final_response,
            has_code="```" in final_response,
        )
        await ai_message.insert()

        # Update conversation metadata
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id
        )
        if conversation:
            messages = await Message.find(
                Message.conversation_id == conversation_id
            ).count()
            conversation.message_count = messages // 2
            conversation.updated_at = datetime.utcnow()
            await conversation.save()

        # Push the full response as a single "done" event
        await queue.put(
            {
                "event": "done",
                "data": {
                    "message_id": ai_message.message_id,
                    "content": final_response,
                    "created_at": ai_message.created_at.isoformat(),
                    "agent": selected_mode,
                },
            }
        )

        # --- Async title generation (fire & forget within this task) ---
        # Only generate title for first exchange
        msg_count = await Message.find(
            Message.conversation_id == conversation_id
        ).count()
        if msg_count <= 2 and conversation:  # 1 user + 1 assistant = first exchange
            try:
                # Quick check: if it's a greeting, use a simple title without LLM
                greeting_words = {"hi", "hello", "hey", "namaste", "hola", "yo", "sup"}
                msg_lower = user_message_content.strip().lower().rstrip("!. ")
                if msg_lower in greeting_words or msg_lower.startswith(
                    ("good morning", "good afternoon", "good evening")
                ):
                    generated_title = "Welcome Chat"
                else:
                    from app.config.key_rotator import get_rotator as _get_rotator2
                    from groq import AsyncGroq as _AsyncGroq2

                    _ai_title_prompt = f"""Generate a short title (3-6 words) that tells the user WHAT THIS CHAT IS ABOUT at a glance.

RULES:
- Describe the TOPIC, not the action. "Reliance Stock Analysis" not "Analyzing Reliance Stock"
- Be specific: "PE Ratio Explained" > "Financial Concept"
- Use Title Case. No quotes, no periods.
- If about a stock: include the stock name. "TCS Price Check" or "HDFC Bank Outlook"
- If about a concept: name it. "Options Trading Basics" or "SIP vs Lump Sum"
- If comparing: "TCS vs Infosys Comparison"
- If about market: "Market Sentiment Today" or "Nifty Analysis"

User message: {user_message_content[:200]}

Output ONLY the title, nothing else:"""
                    # Use PRODUCTION models only, snapshot keys
                    try:
                        _all_keys2 = list(_get_rotator2().api_keys)
                    except RuntimeError:
                        from app.config.settings import settings as _s2
                        _all_keys2 = [_s2.groq_api_key]

                    generated_title = None
                    for _t_model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                        for _key in _all_keys2:
                            try:
                                _ai_title_client = _AsyncGroq2(api_key=_key)
                                _ai_title_resp = await _ai_title_client.chat.completions.create(
                                    model=_t_model,
                                    messages=[{"role": "user", "content": _ai_title_prompt}],
                                    temperature=0.2,
                                    max_tokens=20,
                                )
                                generated_title = (
                                    (_ai_title_resp.choices[0].message.content or "")
                                    .strip()
                                    .strip('"')
                                    .strip("'")
                                    .rstrip(".")
                                )
                                if generated_title:
                                    break
                            except Exception:
                                continue
                        if generated_title:
                            break

                    if not generated_title:
                        generated_title = " ".join(user_message_content.split()[:5])
                    if len(generated_title) > 50:
                        generated_title = generated_title[:50].rsplit(" ", 1)[0]

                conversation.title = generated_title
                await conversation.save()

                # Push title event to stream
                await queue.put(
                    {
                        "event": "title",
                        "data": {
                            "title": generated_title,
                            "conversation_id": conversation_id,
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Title generation error: {e}")
                # Use fallback title from first few words
                words = user_message_content.split()[:6]
                fallback_title = " ".join(words) + ("..." if len(words) == 6 else "")
                if conversation:
                    conversation.title = fallback_title
                    await conversation.save()
                await queue.put(
                    {
                        "event": "title",
                        "data": {
                            "title": fallback_title,
                            "conversation_id": conversation_id,
                        },
                    }
                )

    except Exception as e:
        logger.error(f"AI workflow error: {e}")
        await queue.put({"event": "error", "data": {"error": str(e)}})
    finally:
        await queue.put(None)  # Signal stream end


# --- SSE helper ---


async def _event_stream_generator(queue: asyncio.Queue, conversation_id: str):  # noqa: E501
    """Async generator that yields SSE events from the queue with keepalive pings."""
    KEEPALIVE_INTERVAL = 20.0   # send a comment ping every 20s of silence
    MAX_IDLE_TIME = 300.0       # stop the stream if nothing happens for 5 min
    idle_since = 0.0

    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                # Send an SSE comment (invisible to JS) to keep the TCP connection alive
                yield ": keepalive\n\n"
                idle_since += KEEPALIVE_INTERVAL
                if idle_since >= MAX_IDLE_TIME:
                    yield f"event: error\ndata: {json.dumps({'error': 'Stream timeout after 5 min idle'})}\n\n"
                    break
                continue

            idle_since = 0.0  # reset on any real event

            if event is None:
                # Stream complete
                break

            event_type = event.get("event", "message")
            event_data = json.dumps(event.get("data", {}))
            yield f"event: {event_type}\ndata: {event_data}\n\n"
    finally:
        # ── Cleanup: cancel background task if client disconnected early ──
        task = _active_tasks.pop(conversation_id, None)
        if task and not task.done():
            task.cancel()
            logger.info(
                f"🛑 Cancelled AI task for conversation {conversation_id} (client disconnected)"
            )
        _pending_jobs.pop(conversation_id, None)
        _job_timestamps.pop(conversation_id, None)


# --- API ENDPOINTS ---


@router.post("/send")
async def send_message(
    request: SendMessageRequest,
    http_request: Request,
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Phase 1: Instant init (<200ms).
    Creates conversation, saves user message, starts background AI task.
    Returns conversation_id immediately so frontend can redirect + open SSE.
    """
    try:
        # Rate limiting check
        user_id = (
            str(user.user_id) if user else f"anon_{request.conversation_id or 'new'}"
        )

        if not await rate_limiter.acquire(user_id):
            status = await rate_limiter.get_status(user_id)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please slow down.",
                    "tokens_available": status["tokens"],
                    "capacity": status["capacity"],
                    "retry_after_seconds": 10,
                },
            )

        # ── Anonymous IP rate limit: max 10 new conversations per IP per 24h ──
        client_ip = _get_client_ip(http_request)
        if not user and not request.conversation_id:
            # Only enforce on new conversation creation (not continuation)
            if not _check_anon_ip_limit(client_ip):
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Conversation limit reached",
                        "message": (
                            f"Anonymous users can start up to {ANON_MAX_CONVOS} conversations per day. "
                            "Please sign in to continue."
                        ),
                        "retry_after_seconds": ANON_WINDOW_SECONDS,
                    },
                )

        # Evict stale jobs from previous sessions (lightweight, runs each request)
        _cleanup_stale_jobs()

        # ── Input sanitization ─────────────────────────────────────
        try:
            clean_message = sanitize_user_message(request.message, field="message")
        except SanitizationError as se:
            logger.warning(
                f"🚨 Sanitization blocked message from {user_id}: {se.reason}"
            )
            raise HTTPException(
                status_code=400,
                detail={"error": "Invalid input", "reason": se.reason},
            )
        # replace with cleaned version
        request.message = clean_message

        # Check message limit for non-authenticated users
        if not user and request.conversation_id:
            message_count = await Message.find(
                Message.conversation_id == request.conversation_id
            ).count()
            if message_count >= 10:
                raise HTTPException(
                    status_code=403,
                    detail="Message limit reached. Please sign in to continue chatting.",
                )

        # Get or create conversation
        if request.conversation_id:
            conversation = await Conversation.find_one(
                Conversation.conversation_id == request.conversation_id
            )
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # ── IDOR guard: verify the requesting user owns this conversation ──
            # An authenticated user can only continue their own conversations.
            # Anonymous users can only continue conversations with no owner (user_id=None).
            if conversation.user_id is not None:
                # Conversation belongs to a specific user — require matching auth
                if not user or str(user.user_id) != conversation.user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="You do not have permission to access this conversation",
                    )
            else:
                # Anonymous conversation — only allow anonymous access
                # (prevents authenticated users from claiming orphaned convos)
                pass
        else:
            conversation = Conversation(
                user_id=str(user.user_id) if user else None,
                title="New Chat",
            )
            try:
                await conversation.insert()
            except Exception as db_err:
                logger.warning(f"⚠️ MongoDB unavailable (insert conversation): {db_err} — running in-memory mode")
                # Give it a synthetic ID so the rest of the code works
                import uuid as _uuid
                if not conversation.conversation_id:
                    conversation.conversation_id = str(_uuid.uuid4())

        # Save user message
        user_message = Message(
            conversation_id=conversation.conversation_id,
            role="user",
            content=request.message,
            images=request.images,
            has_images=bool(request.images),
        )
        try:
            await user_message.insert()
        except Exception as db_err:
            logger.warning(f"⚠️ MongoDB unavailable (insert message): {db_err} — skipping persistence")

        # Fetch conversation history (truncated to last 20 messages)
        try:
            messages = (
                await Message.find(Message.conversation_id == conversation.conversation_id)
                .sort("+created_at")
                .to_list()
            )
        except Exception:
            messages = [user_message]  # graceful fallback: treat as new conversation

        raw_history = [
            {
                "role": msg.role,
                "content": msg.content,
                "images": msg.images if hasattr(msg, "images") else None,
            }
            for msg in messages[:-1]  # Exclude current message
        ]
        conversation_history = (
            raw_history[-20:] if len(raw_history) > 20 else raw_history
        )

        # Create event queue, store task handle for lifecycle management
        queue = asyncio.Queue()
        _pending_jobs[conversation.conversation_id] = queue
        _job_timestamps[conversation.conversation_id] = time.monotonic()

        # ── Route to swarm orchestrator or standard workflow ──────────
        use_swarm = request.mode == "swarm" or request.mode == "deep_research"

        if use_swarm:
            logger.info(
                f"🕷️  Routing to MasterOrchestrator (swarm) for "
                f"conversation {conversation.conversation_id}"
            )
            task = asyncio.create_task(
                _run_swarm_workflow(
                    conversation_id=conversation.conversation_id,
                    user_message_content=request.message,
                    images=request.images,
                    conversation_history=conversation_history,
                    queue=queue,
                    user=user,
                )
            )
        else:
            task = asyncio.create_task(
                _run_ai_workflow(
                    conversation_id=conversation.conversation_id,
                    user_message_content=request.message,
                    images=request.images,
                    conversation_history=conversation_history,
                    queue=queue,
                    user=user,
                )
            )
        _active_tasks[conversation.conversation_id] = task

        # Return immediately — frontend redirects and opens SSE stream
        return {
            "conversation_id": conversation.conversation_id,
            "user_message": {
                "message_id": user_message.message_id,
                "role": "user",
                "content": user_message.content,
                "images": user_message.images,
                "created_at": user_message.created_at.isoformat(),
            },
        }

    except HTTPException:
        raise
    except ValueError as e:
        # Pydantic validation errors (e.g. too many images) — safe to surface
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Log the full error internally, never expose internals to the client
        logger.error(f"Error in send_message init: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        )


@router.get("/send/stream/{conversation_id}")
async def stream_response(
    conversation_id: str,
    token: Optional[str] = Query(
        None, description="JWT token for auth (EventSource API cannot set headers)"
    ),
):
    """
    Phase 2: SSE stream.
    Client connects after POST /send returns. Streams events as AI processes.
    Events: status, done, title, error

    Auth: if the conversation belongs to a user, a valid matching JWT must be
    supplied via ?token= (EventSource API cannot send Authorization headers).
    """
    # ── Ownership check ───────────────────────────────────────────────
    conversation = await Conversation.find_one(
        Conversation.conversation_id == conversation_id
    )
    if conversation and conversation.user_id:
        # Owned conversation — require a valid access token matching the owner
        from app.auth.security import verify_token, TOKEN_TYPE_ACCESS
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Authentication required to stream this conversation",
            )
        payload = verify_token(token)
        if (
            not payload
            or payload.get("token_type") != TOKEN_TYPE_ACCESS
            or payload.get("sub") != conversation.user_id
        ):
            raise HTTPException(
                status_code=403,
                detail="Token does not match conversation owner",
            )

    queue = _pending_jobs.get(conversation_id)
    if queue is None:
        raise HTTPException(
            status_code=404,
            detail="No pending AI response for this conversation. Send a message first.",
        )

    return StreamingResponse(
        _event_stream_generator(queue, conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Existing CRUD endpoints (unchanged) ---


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    limit: int = Query(50, le=100),
    user: User = Depends(get_current_user),
):
    """Get user's conversations (authenticated only)."""
    try:
        conversations = (
            await Conversation.find(Conversation.user_id == str(user.user_id))
            .sort("-updated_at")
            .limit(limit)
            .to_list()
        )  # type: ignore[arg-type]

        return [
            ConversationResponse(
                conversation_id=conv.conversation_id,
                title=conv.title,
                message_count=conv.message_count,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversations")


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    user: Optional[User] = Depends(get_optional_user),
):
    """Get conversation details by ID."""
    try:
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.user_id and (
            not user or str(user.user_id) != conversation.user_id
        ):
            raise HTTPException(status_code=403, detail="Access denied")

        return ConversationResponse(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            message_count=conversation.message_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch conversation")


@router.get(
    "/conversations/{conversation_id}/messages", response_model=List[MessageResponse]
)
async def get_messages(
    conversation_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
    before: Optional[str] = Query(
        None, description="Return messages created before this message_id (cursor pagination)"
    ),
    user: Optional[User] = Depends(get_optional_user),
):
    """Get messages for a conversation with cursor-based pagination."""
    try:
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.user_id and (
            not user or str(user.user_id) != conversation.user_id
        ):
            raise HTTPException(status_code=403, detail="Access denied")

        query = Message.find(Message.conversation_id == conversation_id)

        # Cursor: if 'before' message_id is given, fetch messages created before that message
        if before:
            cursor_msg = await Message.find_one(Message.message_id == before)
            if cursor_msg:
                query = Message.find(
                    Message.conversation_id == conversation_id,
                    Message.created_at < cursor_msg.created_at,
                )

        messages = await query.sort("+created_at").limit(limit).to_list()

        return [
            MessageResponse(
                message_id=msg.message_id,
                role=msg.role,
                content=msg.content,
                images=msg.images if hasattr(msg, "images") else None,
                created_at=msg.created_at,
            )
            for msg in messages
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching messages: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch messages")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
):
    """Delete a conversation and all its messages completely."""
    try:
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == str(user.user_id),
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        deleted_messages = await Message.find(
            Message.conversation_id == conversation_id
        ).delete()

        logger.info(
            f"Deleted {deleted_messages.deleted_count if deleted_messages else 0} messages for conversation {conversation_id}"
        )

        await conversation.delete()
        logger.info(f"Successfully deleted conversation {conversation_id}")

        return {
            "message": "Conversation deleted successfully",
            "deleted_messages": deleted_messages.deleted_count
            if deleted_messages
            else 0,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error deleting conversation {conversation_id}: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
