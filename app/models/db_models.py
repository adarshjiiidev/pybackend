"""
Database models using Beanie ODM (async MongoDB ODM for Pydantic).
"""

import hashlib
import hmac
import uuid
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, Optional

from beanie import Document, Indexed
from pydantic import Field


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
        indexes = ["session_id", "timestamp"]


class UserSession(Document):
    """Represents a user's chat session."""

    session_id: Indexed(str, unique=True) = Field(
        ..., description="Unique session identifier"
    )
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")

    # Session metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

    # Session context
    context: Optional[Dict[str, Any]] = Field(default=None)

    class Settings:
        name = "user_sessions"
        indexes = ["session_id", "user_id", "last_activity"]


class MarketDataCache(Document):
    """Cache for market data to reduce API calls."""

    cache_key: Indexed(str, unique=True) = Field(
        ..., description="Unique cache identifier"
    )
    symbol: Indexed(str) = Field(..., description="Stock/crypto symbol")
    data_type: str = Field(
        ..., description="Type of data: stock_price, crypto_price, etc."
    )

    # Cached data
    data: Dict[str, Any] = Field(..., description="Cached data payload")

    # Cache metadata
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = Field(default=300, description="Time to live in seconds")

    class Settings:
        name = "market_data_cache"
        indexes = ["cache_key", "symbol", "data_type", "cached_at"]

    def is_expired(self) -> bool:
        """Check if cached data has expired."""
        expiry_time = self.cached_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time


class User(Document):
    """User account information for authentication."""

    user_id: Indexed(str, unique=True) = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique user identifier"
    )
    email: Indexed(str, unique=True) = Field(..., description="User email address")
    password_hash: Optional[str] = Field(
        default=None, description="Hashed password (optional for OAuth users)"
    )
    full_name: Optional[str] = Field(default=None, description="User full name")

    is_verified: bool = Field(default=False, description="Email verification status")
    is_active: bool = Field(default=True, description="Account active status")
    is_premium: bool = Field(default=False, description="Premium subscription status")

    # Storage tracking
    storage_used_bytes: int = Field(
        default=0, description="Total storage used in bytes"
    )
    storage_limit_bytes: int = Field(
        default=20 * 1024 * 1024,
        description="Storage limit in bytes (20MB for free users)",
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = Field(
        default=None, description="Last login timestamp"
    )

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
            # expires_at TTL index is managed in database.py
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
            # expires_at TTL index is managed in database.py
        ]


class OTPRecord(Document):
    """
    OTP records stored in MongoDB — multi-worker safe, survives restarts.
    Replaces the old in-memory dict in OTPService.
    The code is stored as an HMAC-SHA256 hash so plaintext is never at rest.
    """

    email: Indexed(str) = Field(..., description="Target email address")
    purpose: str = Field(
        ..., description="OTP purpose: registration | login | password_reset"
    )
    code_hash: str = Field(..., description="HMAC-SHA256 hash of the OTP code")
    expires_at: datetime = Field(
        ..., description="Expiry timestamp (auto-deleted by TTL index)"
    )
    attempts: int = Field(
        default=0, description="Number of failed verification attempts"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "otp_records"
        indexes = [
            # Compound index for fast lookup by email + purpose (one OTP per purpose at a time)
            [("email", 1), ("purpose", 1)],
            # TTL index — MongoDB deletes documents automatically after expires_at
            # (created in database.py via expireAfterSeconds: 0)
        ]

    @staticmethod
    def _get_secret() -> bytes:
        """Return the HMAC secret from settings, falling back to a loud dev placeholder."""
        from ..config import settings
        secret = settings.otp_secret
        if not secret:
            import logging, os
            if os.environ.get("ENVIRONMENT", "development") == "production":
                raise RuntimeError(
                    "OTP_SECRET is not set! Cannot hash OTP codes in production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            logging.getLogger(__name__).warning(
                "⚠️  OTP_SECRET not set — using insecure dev fallback. Set OTP_SECRET in .env."
            )
            secret = "dev-otp-fallback-change-me-in-production"
        return secret.encode()

    @staticmethod
    def hash_code(code: str) -> str:
        """Return HMAC-SHA256 hex digest of the OTP code using the configured secret."""
        secret = OTPRecord._get_secret()
        return hmac.new(secret, code.encode(), hashlib.sha256).hexdigest()

    def verify_code(self, code: str) -> bool:
        """Constant-time comparison to verify a supplied code against stored hash."""
        secret = OTPRecord._get_secret()
        expected = hmac.new(secret, code.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.code_hash, expected)

    def is_expired(self) -> bool:
        """Check if this OTP record has expired."""
        return datetime.utcnow() > self.expires_at


class LoginAttempt(Document):
    """
    Track failed login attempts per email for brute-force protection.
    MongoDB TTL index automatically purges records after WINDOW_MINUTES.
    Stored in MongoDB so protection works across all workers/processes.
    """

    WINDOW_MINUTES: ClassVar[int] = 15  # Class-level constant — sliding window duration
    MAX_ATTEMPTS: ClassVar[int] = 5  # Max failures before lockout within the window
    LOCKOUT_MINUTES: ClassVar[int] = 15  # How long to lock out after MAX_ATTEMPTS

    email: Indexed(str) = Field(..., description="Email that failed login")
    ip_address: Optional[str] = Field(
        default=None, description="Client IP (informational)"
    )
    attempt_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the failed attempt occurred (used by TTL index)",
    )

    class Settings:
        name = "login_attempts"
        indexes = [
            [("email", 1), ("attempt_at", -1)],  # Fast recent-attempt lookup
            # TTL index on attempt_at (expireAfterSeconds = WINDOW_MINUTES * 60)
            # created in database.py
        ]
