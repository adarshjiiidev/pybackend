"""
Database models using Beanie ODM (async MongoDB ODM for Pydantic).
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

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


# ─────────────────────────────────────────────────────────────
#  Subscription & Payment Models
# ─────────────────────────────────────────────────────────────

class PlanType(str, Enum):
    FREE = "free"
    PRO = "pro"
    PREMIUM = "premium"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PENDING = "pending"


class Subscription(Document):
    """Tracks a user's active/historical subscription plan."""

    user_id: Indexed(str, unique=True) = Field(
        ..., description="User this subscription belongs to"
    )
    plan: PlanType = Field(default=PlanType.FREE, description="Current plan")
    status: SubscriptionStatus = Field(
        default=SubscriptionStatus.ACTIVE, description="Subscription status"
    )

    # Billing period
    started_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(
        default=None, description="None = lifetime / free plan"
    )
    is_yearly: bool = Field(default=False, description="Monthly or yearly billing")

    # Razorpay identifiers
    razorpay_subscription_id: Optional[str] = Field(default=None)
    latest_payment_id: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "subscriptions"
        indexes = ["user_id", "plan", "status", "expires_at"]

    def is_active_plan(self) -> bool:
        """Return True if subscription is active and not expired."""
        if self.status != SubscriptionStatus.ACTIVE:
            return False
        if self.plan == PlanType.FREE:
            return True
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def days_remaining(self) -> Optional[int]:
        """Return days remaining in subscription, None if no expiry."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.days)


class PaymentTransaction(Document):
    """Records every Razorpay payment attempt for audit and idempotency."""

    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Internal transaction UUID",
    )
    user_id: Indexed(str) = Field(..., description="Paying user")

    # Plan info
    plan: PlanType = Field(...)
    is_yearly: bool = Field(default=False)
    amount_paise: int = Field(..., description="Amount in paise (INR smallest unit)")
    currency: str = Field(default="INR")

    # Razorpay identifiers
    razorpay_order_id: Indexed(str, unique=True) = Field(
        ..., description="Razorpay order_id"
    )
    razorpay_payment_id: Optional[str] = Field(default=None)
    razorpay_signature: Optional[str] = Field(default=None)

    # Status
    status: str = Field(
        default="created",
        description="created | captured | failed | suspicious",
    )
    failure_reason: Optional[str] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "payment_transactions"
        indexes = [
            "user_id",
            "razorpay_order_id",
            "status",
            "created_at",
        ]
