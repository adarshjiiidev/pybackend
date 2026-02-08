"""
Share API endpoints for making conversations public and retrieving shared conversations.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid
import logging

from ..models.chat_models import Conversation, Message
from ..models import User  # Correct import for User model
from ..auth.security import get_current_user  # Use existing auth dependency

router = APIRouter(prefix="/share", tags=["share"])
logger = logging.getLogger(__name__)


class ShareResponse(BaseModel):
    """Response when making a conversation public."""
    success: bool
    share_id: str
    share_url: str
    is_public: bool


class PublicConversationResponse(BaseModel):
    """Public conversation data."""
    conversation_id: str
    title: str
    created_at: datetime
    message_count: int
    messages: List[Dict[str, Any]]


@router.post("/conversation/{conversation_id}/public", response_model=ShareResponse)
async def make_conversation_public(
    conversation_id: str,
    user: User = Depends(get_current_user)
):
    """Make a conversation public and generate a share ID."""
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Find the conversation
    conversation = await Conversation.find_one(
        Conversation.conversation_id == conversation_id
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Verify ownership
    if conversation.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to share this conversation")
    
    # Generate share ID if not already public
    if not conversation.share_id:
        conversation.share_id = str(uuid.uuid4())[:12]  # Short, unique ID
    
    conversation.is_public = True
    conversation.shared_at = datetime.utcnow()
    conversation.updated_at = datetime.utcnow()
    
    await conversation.save()
    
    logger.info(f"Conversation {conversation_id} made public with share_id: {conversation.share_id}")
    
    return ShareResponse(
        success=True,
        share_id=conversation.share_id,
        share_url=f"/share/{conversation.share_id}",
        is_public=True
    )


@router.post("/conversation/{conversation_id}/private")
async def make_conversation_private(
    conversation_id: str,
    user: User = Depends(get_current_user)
):
    """Make a conversation private (remove public access)."""
    
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Find the conversation
    conversation = await Conversation.find_one(
        Conversation.conversation_id == conversation_id
    )
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Verify ownership
    if conversation.user_id != user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this conversation")
    
    conversation.is_public = False
    conversation.updated_at = datetime.utcnow()
    # Keep share_id for potential re-sharing
    
    await conversation.save()
    
    logger.info(f"Conversation {conversation_id} made private")
    
    return {"success": True, "is_public": False}


@router.get("/{share_id}", response_model=PublicConversationResponse)
async def get_public_conversation(share_id: str):
    """Get a public conversation by its share ID. No authentication required."""
    
    # Find conversation by share_id
    conversation = await Conversation.find_one(
        Conversation.share_id == share_id
    )
    
    if not conversation or not conversation.is_public:
        raise HTTPException(
            status_code=404,
            detail="Shared conversation not found or no longer public"
        )
    
    # Get all messages for this conversation
    messages = await Message.find(
        Message.conversation_id == conversation.conversation_id
    ).sort("+created_at").to_list()
    
    # Format messages
    formatted_messages = [
        {
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat(),
            "images": msg.images if msg.images else []
        }
        for msg in messages
    ]
    
    logger.info(f"Public conversation accessed: {share_id}")
    
    return PublicConversationResponse(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        created_at=conversation.created_at,
        message_count=len(formatted_messages),
        messages=formatted_messages
    )
