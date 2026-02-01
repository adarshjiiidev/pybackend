"""
MongoDB document models using Beanie ODM.
Defines database schemas for conversation persistence and caching.
"""

from beanie import Document, Indexed
from pydantic import Field
from datetime import datetime, timedelta
from typing import Optional, Any
import uuid


class ConversationMessage(Document):
    """Stores individual chat messages."""
    
    session_id: Indexed(str) = Field(..., description="Session identifier")
    user_query: str = Field(..., description="User's input query")
    agent_mode: str = Field(..., description="Agent mode used for this response")
    agent_response: str = Field(..., description="Agent's complete response")
    
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata (time, model, etc.)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "conversation_messages"
        indexes = [
            "session_id",
            "created_at"
        ]


class UserSession(Document):
    """Stores user session information."""
    
    session_id: Indexed(str, unique=True) = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier"
    )
    user_id: Optional[str] = Field(default=None, description="User ID for future authentication")
    active: bool = Field(default=True, description="Whether session is active")
    
    preferences: dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences (default_mode, language, etc.)"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "user_sessions"
        indexes = [
            "session_id",
            "user_id"
        ]


class MarketDataCache(Document):
    """Caches market data to reduce API calls."""
    
    symbol: Indexed(str) = Field(..., description="Stock/crypto symbol")
    data_type: str = Field(..., description="Type of data: info, historical, crypto, indices")
    data: dict[str, Any] = Field(..., description="Cached market data")
    
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = Field(default=300, description="Time to live in seconds")
    
    class Settings:
        name = "market_data_cache"
        indexes = [
            [("symbol", 1), ("data_type", 1)],  # Compound index
            "cached_at"
        ]
    
    def is_expired(self) -> bool:
        """Check if cached data has expired."""
        expiry_time = self.cached_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time
