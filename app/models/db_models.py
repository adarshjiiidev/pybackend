"""
Database models using Beanie ODM (async MongoDB ODM for Pydantic).
"""

from beanie import Document, Indexed
from pydantic import Field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import uuid


class ConversationMessage(Document):
    """Represents a single conversation message in the AI chat system."""
    
    session_id: Indexed(str) = Field(..., description="Unique session identifier")
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Message content
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    
    class Settings:
        name = "conversation_messages"
        indexes = [
            "session_id",
            "timestamp"
        ]


class UserSession(Document):
    """Represents a user's chat session."""
    
    session_id: Indexed(str, unique=True) = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    
    # Session metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    
    # Session context
    context: Optional[Dict[str, Any]] = Field(default=None)
    
    class Settings:
        name = "user_sessions"
        indexes = [
            "session_id",
            "user_id",
            "last_activity"
        ]


class MarketDataCache(Document):
    """Cache for market data to reduce API calls."""
    
    cache_key: Indexed(str, unique=True) = Field(..., description="Unique cache identifier")
    symbol: Indexed(str) = Field(..., description="Stock/crypto symbol")
    data_type: str = Field(..., description="Type of data: stock_price, crypto_price, etc.")
    
    # Cached data
    data: Dict[str, Any] = Field(..., description="Cached data payload")
    
    # Cache metadata
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = Field(default=300, description="Time to live in seconds")
    
    class Settings:
        name = "market_data_cache"
        indexes = [
            "cache_key",
            "symbol",
            "data_type",
            "cached_at"
        ]
    
    def is_expired(self) -> bool:
        """Check if cached data has expired."""
        expiry_time = self.cached_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time


class User(Document):
    """User account information for authentication."""
    
    user_id: Indexed(str, unique=True) = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique user identifier"
    )
    email: Indexed(str, unique=True) = Field(..., description="User email address")
    password_hash: Optional[str] = Field(default=None, description="Hashed password (optional for OAuth users)")
    full_name: Optional[str] = Field(default=None, description="User full name")
    
    is_verified: bool = Field(default=False, description="Email verification status")
    is_active: bool = Field(default=True, description="Account active status")
    is_premium: bool = Field(default=False, description="Premium subscription status")
    
    # Storage tracking
    storage_used_bytes: int = Field(default=0, description="Total storage used in bytes")
    storage_limit_bytes: int = Field(default=20 * 1024 * 1024, description="Storage limit in bytes (20MB for free users)")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    
    class Settings:
        name = "users"
    
    def get_storage_used_mb(self) -> float:
        """Get storage used in MB."""
        return self.storage_used_bytes / (1024 * 1024)
    
    def get_storage_limit_mb(self) -> float:
        """Get storage limit in MB."""
        return self.storage_limit_bytes / (1024 * 1024)
    
    def get_storage_percentage(self) -> float:
        """Get storage usage as percentage."""
        if self.storage_limit_bytes == 0:
            return 0
        return (self.storage_used_bytes / self.storage_limit_bytes) * 100
    
    def has_storage_available(self, required_bytes: int) -> bool:
        """Check if user has enough storage for new content."""
        if self.is_premium:
            return True  # Premium users have unlimited storage
        return (self.storage_used_bytes + required_bytes) <= self.storage_limit_bytes


class VerificationToken(Document):
    """Email verification and magic link tokens."""
    
    token: Indexed(str, unique=True) = Field(..., description="Verification token")
    user_id: Indexed(str) = Field(..., description="Associated user ID")
    token_type: str = Field(..., description="Type: email_verification or magic_link")
    
    expires_at: datetime = Field(..., description="Token expiration time")
    used: bool = Field(default=False, description="Whether token has been used")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "verification_tokens"
        indexes = [
            "token",
            "user_id",
            "expires_at"
        ]
    
    def is_expired(self) -> bool:
        """Check if token has expired."""
        return datetime.utcnow() > self.expires_at


class TokenBlacklist(Document):
    """Blacklisted JWT tokens for logout."""
    
    jti: Indexed(str, unique=True) = Field(..., description="JWT ID")
    expires_at: datetime = Field(..., description="Token expiration time")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "token_blacklist"
        indexes = [
            "jti",
            "expires_at"
        ]


class OTP(Document):
    """One-Time Password for authentication."""
    
    email: Indexed(str) = Field(..., description="User email address")
    code: str = Field(..., description="6-digit OTP code")
    purpose: str = Field(..., description="Purpose: registration, login, password_reset")
    expires_at: datetime = Field(..., description="OTP expiration time")
    attempts: int = Field(default=0, description="Number of verification attempts")
    is_used: bool = Field(default=False, description="Whether OTP has been used")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "otps"
        indexes = [
            "email",
            "expires_at",
            "is_used"
        ]
    
    def is_expired(self) -> bool:
        """Check if OTP has expired."""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if OTP is valid (not expired, not used, attempts < 5)."""
        return not self.is_expired() and not self.is_used and self.attempts < 5
