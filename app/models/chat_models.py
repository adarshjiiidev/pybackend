"""
Chat-related database models for conversations and messages.
"""

from beanie import Document, Indexed, Link
from pymongo import IndexModel, ASCENDING, DESCENDING
from pydantic import Field
from datetime import datetime
from typing import Optional, List
import uuid


class Conversation(Document):
    """Represents a chat conversation."""
    
    conversation_id: Indexed(str, unique=True) = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique conversation identifier"
    )
    user_id: Optional[Indexed(str)] = Field(
        default=None,
        description="User ID (null for non-authenticated users)"
    )
    
    # Conversation metadata
    title: str = Field(default="New Chat", description="Conversation title")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Message count for quick access
    message_count: int = Field(default=0)
    
    # Public sharing
    is_public: bool = Field(default=False, description="Whether conversation is publicly shareable")
    share_id: Optional[Indexed(str, unique=True)] = Field(
        default=None,
        description="Unique share ID for public URL (generated when made public)"
    )
    shared_at: Optional[datetime] = Field(default=None, description="When conversation was made public")
    
    class Settings:
        name = "conversations"
        indexes = [
            "conversation_id",
            IndexModel([("user_id", ASCENDING), ("updated_at", DESCENDING)]),
            "created_at",
            "share_id"  # Index for fast public conversation lookups
        ]


class Message(Document):
    """Represents a single message in a conversation."""
    
    message_id: Indexed(str, unique=True) = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message identifier"
    )
    conversation_id: Indexed(str) = Field(
        ...,
        description="Associated conversation ID"
    )
    
    # Message content
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Multimedia support
    images: Optional[List[str]] = Field(default=None, description="List of base64 encoded images or URLs")
    
    # Formatting flags
    has_code: bool = Field(default=False)
    has_images: bool = Field(default=False)
    
    class Settings:
        name = "messages"
        indexes = [
            "message_id",
            IndexModel([("conversation_id", ASCENDING), ("created_at", ASCENDING)]),
        ]
