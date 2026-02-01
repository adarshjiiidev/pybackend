"""
FastAPI request/response models.
Defines the API contract for client-server communication.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    mode: str = Field(default="auto", description="Agent mode: auto, market_research, realtime_analysis, portfolio, explainer, crypto")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    enable_deep_search: bool = Field(default=False, description="Enable autonomous research loop for deep analysis")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Analyze Reliance Industries stock",
                "mode": "auto",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "enable_deep_search": False
            }
        }


class StreamChunk(BaseModel):
    """Streaming response chunk model."""
    content: str = Field(..., description="Partial or complete response text")
    done: bool = Field(default=False, description="Whether this is the final chunk")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional metadata about the response")


class ChatResponse(BaseModel):
    """Complete chat response model (non-streaming)."""
    response: str
    mode: str
    session_id: str
    metadata: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionResponse(BaseModel):
    """Session creation response."""
    session_id: str
    created_at: datetime
    

class ConversationHistoryResponse(BaseModel):
    """Conversation history response."""
    session_id: str
    messages: list[dict[str, Any]]
    total_messages: int
