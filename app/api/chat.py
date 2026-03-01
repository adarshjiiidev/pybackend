"""
Chat API endpoints — SSE streaming architecture.
Phase 1: POST /chat/send → instant init (<200ms), returns conversation_id
Phase 2: GET /chat/send/stream/{id} → SSE stream with tokens, status, title
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import logging
import asyncio
import json
import uuid

from app.models.chat_models import Conversation, Message
from app.models.db_models import User
from app.models.agent_state import AgentState, AgentMode
from app.auth.security import get_current_user, get_optional_user
from app.graph.workflow import agent_graph
from app.utils.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize rate limiter: 100 tokens/bucket, refill 10 per second
rate_limiter = TokenBucket(capacity=100, refill_rate=10.0, tokens_per_request=1)

# In-memory store for pending AI jobs: conversation_id -> asyncio.Queue
# The SSE endpoint reads from this queue; the background task writes to it.
_pending_jobs: Dict[str, asyncio.Queue] = {}

# Lazy-loaded singleton for title generation to avoid re-instantiation overhead
_title_llm = None


def get_title_llm():
    """Get or create a singleton ChatGroq instance for title generation."""
    global _title_llm
    if _title_llm is None:
        from langchain_groq import ChatGroq
        from app.config import settings
        _title_llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            api_key=settings.groq_api_key,
        )
    return _title_llm


# --- Pydantic models ---

class SendMessageRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=10000)
    images: Optional[List[str]] = Field(default=None, description="List of base64 encoded images")
    stream: bool = True


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
        await queue.put({"event": "status", "data": {"stage": "routing", "message": "Routing your query..."}})

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

        # Push thinking status
        await queue.put({"event": "status", "data": {"stage": "thinking", "message": "Thinking..."}})

        # Run pre-compiled global workflow instance (Performance: avoid rebuilding graph)
        final_state = await agent_graph.ainvoke(state)

        if final_state is None:
            await queue.put({"event": "error", "data": {"error": "Workflow returned no response."}})
            await queue.put(None)  # Signal end
            return

        final_response = final_state.get("final_response") or "I apologize, but I couldn't generate a response."
        selected_mode = final_state.get("selected_mode", "unknown")

        # Truncate extremely long responses
        MAX_RESPONSE_LENGTH = 8000
        if len(final_response) > MAX_RESPONSE_LENGTH:
            final_response = final_response[:MAX_RESPONSE_LENGTH] + "\n\n[Response truncated due to length.]"

        # Save AI message to DB
        ai_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=final_response,
            has_code="```" in final_response,
        )
        await ai_message.insert()

        # Update conversation metadata
        conversation = await Conversation.find_one(Conversation.conversation_id == conversation_id)
        if conversation:
            messages = await Message.find(
                Message.conversation_id == conversation_id
            ).count()
            conversation.message_count = messages // 2
            conversation.updated_at = datetime.utcnow()
            await conversation.save()

        # Push the full response as a single "done" event
        await queue.put({
            "event": "done",
            "data": {
                "message_id": ai_message.message_id,
                "content": final_response,
                "created_at": ai_message.created_at.isoformat(),
                "agent": selected_mode,
            },
        })

        # --- Async title generation (fire & forget within this task) ---
        # Only generate title for first exchange
        msg_count = await Message.find(Message.conversation_id == conversation_id).count()
        if msg_count <= 2 and conversation:  # 1 user + 1 assistant = first exchange
            try:
                # Quick check: if it's a greeting, use a simple title without LLM
                greeting_words = {'hi', 'hello', 'hey', 'namaste', 'hola', 'yo', 'sup'}
                msg_lower = user_message_content.strip().lower().rstrip('!. ')
                if msg_lower in greeting_words or msg_lower.startswith(('good morning', 'good afternoon', 'good evening')):
                    generated_title = "Welcome Chat"
                else:
                    # Use lazy-loaded singleton LLM
                    llm = get_title_llm()

                    title_prompt = f"""Generate a short title (3-6 words) that tells the user WHAT THIS CHAT IS ABOUT at a glance.

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

                    title_response = await llm.ainvoke(title_prompt)
                    generated_title = title_response.content.strip().strip('"').strip("'").rstrip('.')
                    if len(generated_title) > 50:
                        generated_title = generated_title[:50].rsplit(" ", 1)[0]

                conversation.title = generated_title
                await conversation.save()

                # Push title event to stream
                await queue.put({
                    "event": "title",
                    "data": {"title": generated_title, "conversation_id": conversation_id},
                })
            except Exception as e:
                logger.error(f"Title generation error: {e}")
                # Use fallback title from first few words
                words = user_message_content.split()[:6]
                fallback_title = " ".join(words) + ("..." if len(words) == 6 else "")
                if conversation:
                    conversation.title = fallback_title
                    await conversation.save()
                await queue.put({
                    "event": "title",
                    "data": {"title": fallback_title, "conversation_id": conversation_id},
                })

    except Exception as e:
        logger.error(f"AI workflow error: {e}")
        await queue.put({"event": "error", "data": {"error": str(e)}})
    finally:
        await queue.put(None)  # Signal stream end


# --- SSE helper ---

async def _event_stream_generator(queue: asyncio.Queue, conversation_id: str):
    """Async generator that yields SSE events from the queue."""
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120.0)  # 2 min timeout
            except asyncio.TimeoutError:
                # Send keepalive and end
                yield f"event: error\ndata: {json.dumps({'error': 'Stream timeout'})}\n\n"
                break

            if event is None:
                # Stream complete
                break

            event_type = event.get("event", "message")
            event_data = json.dumps(event.get("data", {}))
            yield f"event: {event_type}\ndata: {event_data}\n\n"
    finally:
        # Clean up pending job
        _pending_jobs.pop(conversation_id, None)


# --- API ENDPOINTS ---

@router.post("/send")
async def send_message(
    request: SendMessageRequest,
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Phase 1: Instant init (<200ms).
    Creates conversation, saves user message, starts background AI task.
    Returns conversation_id immediately so frontend can redirect + open SSE.
    """
    try:
        # Rate limiting check
        user_id = str(user.user_id) if user else f"anon_{request.conversation_id or 'new'}"

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
        else:
            conversation = Conversation(
                user_id=str(user.user_id) if user else None,
                title="New Chat",
            )
            await conversation.insert()

        # Save user message
        user_message = Message(
            conversation_id=conversation.conversation_id,
            role="user",
            content=request.message,
            images=request.images,
            has_images=bool(request.images),
        )
        await user_message.insert()

        # Fetch conversation history (Performance: fetch only last 21 messages from DB)
        # We fetch 21 to get the last 20 and the current message if it exists (though it's handled separately below)
        # Fetching directly from DB reduces I/O and memory overhead for long conversations.
        messages = await Message.find(
            Message.conversation_id == conversation.conversation_id
        ).sort("-created_at").limit(21).to_list()

        # Reverse to maintain chronological order for the AI model
        messages.reverse()

        raw_history = [
            {
                "role": msg.role,
                "content": msg.content,
                "images": msg.images if hasattr(msg, "images") else None,
            }
            for msg in messages[:-1]  # Exclude current message
        ]
        conversation_history = raw_history

        # Create event queue and start background AI task
        queue = asyncio.Queue()
        _pending_jobs[conversation.conversation_id] = queue

        asyncio.create_task(
            _run_ai_workflow(
                conversation_id=conversation.conversation_id,
                user_message_content=request.message,
                images=request.images,
                conversation_history=conversation_history,
                queue=queue,
                user=user,
            )
        )

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
    except Exception as e:
        logger.error(f"Error in send_message init: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/send/stream/{conversation_id}")
async def stream_response(
    conversation_id: str,
    token: Optional[str] = Query(None, description="JWT token for auth (EventSource can't send headers)"),
):
    """
    Phase 2: SSE stream.
    Client connects after POST /send returns. Streams events as AI processes.
    Events: status, done, title, error
    Note: Uses query param token since EventSource API can't set Authorization headers.
    """
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
            "X-Accel-Buffering": "no",  # Disable nginx buffering
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
        conversations = await Conversation.find(
            Conversation.user_id == str(user.user_id)
        ).sort("-updated_at").limit(limit).to_list()

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
        logger.error(f"Error fetching conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
        if conversation.user_id and (not user or str(user.user_id) != conversation.user_id):
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
        logger.error(f"Error fetching conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: str,
    user: Optional[User] = Depends(get_optional_user),
):
    """Get messages for a conversation."""
    try:
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.user_id and (not user or str(user.user_id) != conversation.user_id):
            raise HTTPException(status_code=403, detail="Access denied")

        messages = await Message.find(
            Message.conversation_id == conversation_id
        ).sort("+created_at").to_list()

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
        logger.error(f"Error fetching messages: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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

        logger.info(f"Deleted {deleted_messages.deleted_count if deleted_messages else 0} messages for conversation {conversation_id}")

        await conversation.delete()
        logger.info(f"Successfully deleted conversation {conversation_id}")

        return {
            "message": "Conversation deleted successfully",
            "deleted_messages": deleted_messages.deleted_count if deleted_messages else 0,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
