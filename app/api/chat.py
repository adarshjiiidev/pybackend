"""
Chat API endpoints using the existing multi-agent LangGraph workflow.
Includes rate limiting for protection against abuse.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import logging
import json

from app.models.chat_models import Conversation, Message
from app.models.db_models import User
from app.models.agent_state import AgentState, AgentMode
from app.auth.security import get_current_user, get_optional_user
from app.graph.workflow import create_agent_graph
from app.utils.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize rate limiter: 100 tokens/bucket, refill 10 per second
# Allows bursts of 100 requests, sustained rate of 10 requests/second per user
rate_limiter = TokenBucket(capacity=100, refill_rate=10.0, tokens_per_request=1)



# Pydantic models
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


# Initialize agent workflow
workflow = create_agent_graph()


@router.post("/send")
async def send_message(
    request: SendMessageRequest,
    user: Optional[User] = Depends(get_optional_user)
):
    """
    Send a message and get AI response using multi-agent workflow.
    Supports streaming responses.
    Includes rate limiting protection.
    """
    try:
        # Rate limiting check
        user_id = str(user.user_id) if user else f"anon_{request.conversation_id or 'new'}"
        
        if not await rate_limiter.acquire(user_id):
            # Get rate limit status for helpful error message
            status = await rate_limiter.get_status(user_id)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please slow down.",
                    "tokens_available": status["tokens"],
                    "capacity": status["capacity"],
                    "retry_after_seconds": 10
                }
            )
        
        # Check message limit for non-authenticated users
        if not user and request.conversation_id:
            message_count = await Message.find(
                Message.conversation_id == request.conversation_id
            ).count()
            
            if message_count >= 10:
                raise HTTPException(
                    status_code=403,
                    detail="Message limit reached. Please sign in to continue chatting."
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
                title="New Chat"
            )
            await conversation.insert()
        
        # Save user message with images
        user_message = Message(
            conversation_id=conversation.conversation_id,
            role="user",
            content=request.message,
            images=request.images,
            has_images=bool(request.images)
        )
        await user_message.insert()
        
        # Get conversation history
        messages = await Message.find(
            Message.conversation_id == conversation.conversation_id
        ).sort("+created_at").to_list()
        
        # Build conversation history for agent
        conversation_history = [
            {
                "role": msg.role, 
                "content": msg.content,
                "images": msg.images if hasattr(msg, "images") else None
            }
            for msg in messages[:-1]  # Exclude current message
        ]
        
        # Prepare agent state
        state: AgentState = {
            "query": request.message,
            "images": request.images,  # Pass images to workflow
            "mode": AgentMode.AUTO,
            "session_id": conversation.conversation_id,
            "selected_mode": None,
            "extracted_entities": None,
            "conversation_history": conversation_history,
            "tool_results": None,
            "internal_reasoning": None,
            "final_response": None,
            "execution_metadata": None,
            "error": None
        }
        
        # Stream AI response
        if request.stream:
            async def generate():
                try:
                    full_response = ""
                    
                    # Execute workflow and get final state
                    final_state = await workflow.ainvoke(state)
                    final_response = final_state.get("final_response", "I apologize, but I couldn't generate a response.")
                    
                    # Stream the response word by word
                    words = final_response.split()
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        yield f"data: {chunk}\n\n"
                    
                    # Save AI response
                    ai_message = Message(
                        conversation_id=conversation.conversation_id,
                        role="assistant",
                        content=final_response,
                        has_code="```" in final_response
                    )
                    await ai_message.insert()
                    
                    # Update conversation - count exchanges (user-AI pairs) not individual messages
                    total_messages = len(messages) + 2  # +2 for current user msg and AI response
                    conversation.message_count = total_messages // 2  # Divide by 2 to get exchange count
                    conversation.updated_at = datetime.utcnow()
                    
                    # Debug logging
                    logger.info(f"💬 Conversation {conversation.conversation_id} - exchanges: {conversation.message_count}")
                    
                    # Generate AI-powered title if first exchange (2 messages total: 1 user + 1 assistant)
                    if len(messages) == 1:  # messages list excludes current message, so 1 means first exchange
                        logger.info("🎯 Triggering title generation for first exchange")
                        try:
                            # Use Groq Llama for title generation
                            from langchain_groq import ChatGroq
                            from app.config import settings
                            
                            llm = ChatGroq(
                                model="meta-llama/llama-4-scout-17b-16e-instruct",
                                temperature=0.3,
                                api_key=settings.groq_api_key
                            )
                            
                            title_prompt = f"""You are a title generator. Create a short, descriptive title for this conversation.

Rules:
- Use 3-6 words maximum
- Be clear and specific about the topic
- Use proper capitalization (title case)
- No quotes, no punctuation at the end
- Keep it professional and relevant

Conversation:
User: {request.message[:200]}
Assistant: {final_response[:200]}

Generate only the title, nothing else:"""
                            
                            logger.info("🤖 Calling Groq for title generation...")
                            title_response = await llm.ainvoke(title_prompt)
                            generated_title = title_response.content.strip().strip('"').strip("'")
                            
                            # Clean up the title
                            if len(generated_title) > 50:
                                generated_title = generated_title[:50].rsplit(' ', 1)[0] + "..."
                            
                            conversation.title = generated_title
                            logger.info(f"✅ Generated title: {generated_title}")
                        except Exception as e:
                            logger.error(f"❌ Title generation error: {e}")
                            # Fallback to truncation
                            title_words = request.message.split()[:6]
                            conversation.title = " ".join(title_words) + ("..." if len(title_words) == 6 else "")
                            logger.info(f"📝 Fallback title: {conversation.title}")
                    
                    await conversation.save()
                    
                    # Send final metadata with conversation ID
                    yield f"data: [DONE]\n\n"
                    yield f"event: metadata\n"
                    yield f"data: {conversation.conversation_id}\n\n"
                    
                except Exception as e:
                    logger.error(f"Workflow execution error: {e}")
                    yield f"data: I apologize, but I encountered an error while processing your request.\n\n"
                    yield f"data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"Workflow execution error: {e}")
                    yield f"data: I apologize, but I encountered an error while processing your request.\n\n"
                    yield f"data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response
            final_state = await workflow.ainvoke(state)
            response_text = final_state.get("final_response", "I apologize, but I couldn't generate a response.")
            
            # Save AI response
            ai_message = Message(
                conversation_id=conversation.conversation_id,
                role="assistant",
                content=response_text,
                has_code="```" in response_text
            )
            await ai_message.insert()
            
            # Update conversation
            conversation.message_count = len(messages) + 2
            conversation.updated_at = datetime.utcnow()
            await conversation.save()
            
            return {
                "conversation_id": conversation.conversation_id,
                "message": MessageResponse(
                    message_id=ai_message.message_id,
                    role=ai_message.role,
                    content=ai_message.content,
                    created_at=ai_message.created_at
                )
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    limit: int = Query(50, le=100),
    user: User = Depends(get_current_user)
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
                updated_at=conv.updated_at
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Error fetching conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    user: Optional[User] = Depends(get_optional_user)
):
    """Get conversation details by ID."""
    try:
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check access for authenticated conversations
        if conversation.user_id and (not user or str(user.user_id) != conversation.user_id):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return ConversationResponse(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            message_count=conversation.message_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_messages(
    conversation_id: str,
    user: Optional[User] = Depends(get_optional_user)
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
                images=msg.images if hasattr(msg, 'images') else None,
                created_at=msg.created_at
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
    user: User = Depends(get_current_user)
):
    """Delete a conversation and all its messages completely."""
    try:
        # Find conversation and verify ownership
        conversation = await Conversation.find_one(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == str(user.user_id)
        )
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete ALL messages associated with this conversation
        deleted_messages = await Message.find(
            Message.conversation_id == conversation_id
        ).delete()
        
        logger.info(f"Deleted {deleted_messages.deleted_count if deleted_messages else 0} messages for conversation {conversation_id}")
        
        # Delete the conversation record
        await conversation.delete()
        
        logger.info(f"Successfully deleted conversation {conversation_id} and all associated data")
        
        return {
            "message": "Conversation deleted successfully",
            "deleted_messages": deleted_messages.deleted_count if deleted_messages else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
