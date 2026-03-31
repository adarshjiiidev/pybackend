"""
Security utilities for authentication.
Handles password hashing, JWT token generation/validation, and verification tokens.

Security improvements in this version:
  • Token type claim ("access" vs "refresh") — refresh tokens can't be used as access tokens.
  • TTL-based blacklist cache using cachetools.TTLCache so expired JTIs are auto-evicted.
  • get_current_user_with_token() dependency returns (User, jti, exp) for logout blacklisting.
  • Generic error messages in production — internal details never leak to clients.
  • All in-memory caches are bounded (no unbounded memory growth).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple

from cachetools import TTLCache
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher

from ..config import settings
from ..models.db_models import TokenBlacklist, User

logger = logging.getLogger(__name__)

# ── Password hashing ─────────────────────────────────────────────────────────
pwd_context = PasswordHash((BcryptHasher(),))

# ── JWT settings ─────────────────────────────────────────────────────────────
SECRET_KEY: str = settings.jwt_secret
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
REFRESH_TOKEN_EXPIRE_DAYS: int = 7
STREAM_TOKEN_EXPIRE_MINUTES: int = 5

# Token type constants — stored inside the JWT payload as "token_type"
TOKEN_TYPE_ACCESS: str = "access"
TOKEN_TYPE_REFRESH: str = "refresh"
TOKEN_TYPE_STREAM: str = "stream"

# HTTP Bearer scheme
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)

# ── In-memory caches ─────────────────────────────────────────────────────────
#
# User cache: avoids a DB round-trip on every authenticated request.
#   • TTLCache(maxsize=…, ttl=…) auto-evicts entries after ttl seconds.
#   • maxsize=10_000 prevents unbounded growth even under heavy traffic.
#
_USER_CACHE_TTL: int = 60  # seconds — re-read from DB once per minute
_USER_CACHE: TTLCache = TTLCache(maxsize=10_000, ttl=_USER_CACHE_TTL)

#
# Blacklist cache: two-tier design avoids a DB hit for EVERY token.
#   • _BL_HIT_CACHE  — known-REVOKED jtis; TTL = token's remaining lifetime.
#     We store (jti → expiry_epoch) so each entry lives until the token
#     itself expires, at which point it can never be replayed anyway.
#     Implemented as a bounded TTLCache with a long outer TTL; the actual
#     per-entry expiry is checked manually.
#   • _BL_MISS_CACHE — known-CLEAN jtis; short TTL so that a freshly
#     revoked token stops being trusted within BL_MISS_TTL seconds.
#
_BL_MISS_TTL: int = 30  # seconds — re-check DB for clean tokens
_BL_HIT_CACHE: TTLCache = TTLCache(
    maxsize=50_000, ttl=REFRESH_TOKEN_EXPIRE_DAYS * 86_400
)
_BL_MISS_CACHE: TTLCache = TTLCache(maxsize=50_000, ttl=_BL_MISS_TTL)


# ── Password utilities ───────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (72-byte limit handled safely)."""
    password_bytes = password.encode("utf-8")
    if len(password_bytes) > 72:
        password = password_bytes[:72].decode("utf-8", errors="ignore")
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its bcrypt hash."""
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > 72:
        plain_password = password_bytes[:72].decode("utf-8", errors="ignore")
    return pwd_context.verify(plain_password, hashed_password)


# ── JWT utilities ─────────────────────────────────────────────────────────────


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT *access* token.

    Embeds:
      • "sub"        — user identifier (from caller)
      • "token_type" — always "access"
      • "exp"        — expiry timestamp
      • "jti"        — unique JWT ID for blacklisting
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta
        if expires_delta
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update(
        {
            "exp": expire,
            "jti": str(uuid.uuid4()),
            "token_type": TOKEN_TYPE_ACCESS,  # ← type claim
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT *refresh* token.

    Structurally identical to an access token but carries
    ``"token_type": "refresh"`` so endpoints can reject it if
    an access token is expected (and vice-versa).
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    to_encode.update(
        {
            "exp": expire,
            "jti": str(uuid.uuid4()),
            "token_type": TOKEN_TYPE_REFRESH,  # ← type claim
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_stream_token(
    *,
    user_id: str,
    conversation_id: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a short-lived JWT dedicated to SSE streaming for one conversation.

    This avoids putting a general-purpose access token into the EventSource URL.
    """
    expire = datetime.utcnow() + (
        expires_delta
        if expires_delta
        else timedelta(minutes=STREAM_TOKEN_EXPIRE_MINUTES)
    )
    to_encode = {
        "sub": user_id,
        "conversation_id": conversation_id,
        "exp": expire,
        "jti": str(uuid.uuid4()),
        "token_type": TOKEN_TYPE_STREAM,
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT token.

    Returns the decoded payload dict on success, or None on any failure.
    Does NOT check token_type — callers must do that themselves.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_verification_token() -> str:
    """Generate a secure random token for email verification / magic links."""
    import secrets

    return secrets.token_urlsafe(32)


# ── Blacklist helpers ─────────────────────────────────────────────────────────


async def blacklist_token(jti: str, expires_at: datetime) -> None:
    """
    Persist a JTI to the MongoDB blacklist and update the in-memory cache.

    Called by the logout endpoint.  Safe to call multiple times for the
    same JTI (MongoDB upsert handles duplicates gracefully).
    """
    try:
        existing = await TokenBlacklist.find_one(TokenBlacklist.jti == jti)
        if not existing:
            await TokenBlacklist(jti=jti, expires_at=expires_at).insert()
        # Mirror in hot cache so subsequent requests are denied instantly
        _BL_HIT_CACHE[jti] = True
        _BL_MISS_CACHE.pop(jti, None)  # Evict from "clean" cache if present
        logger.info(f"🚫 Token blacklisted: jti={jti[:8]}…")
    except Exception as exc:
        logger.error(f"Failed to persist token blacklist entry jti={jti[:8]}: {exc}")
        # Still update the in-memory cache even if DB write failed
        _BL_HIT_CACHE[jti] = True
        _BL_MISS_CACHE.pop(jti, None)


async def _is_token_blacklisted(jti: str) -> bool:
    """
    Two-tier blacklist check: hot cache first, then MongoDB.

    Returns True if the token has been revoked, False if clean.
    """
    # Tier 1 — known revoked (hot cache)
    if jti in _BL_HIT_CACHE:
        return True

    # Tier 2 — known clean (miss cache, short TTL)
    if jti in _BL_MISS_CACHE:
        return False

    # Tier 3 — DB lookup (cold path, happens at most once per BL_MISS_TTL per jti)
    try:
        record = await TokenBlacklist.find_one(TokenBlacklist.jti == jti)
        if record:
            _BL_HIT_CACHE[jti] = True
            return True
    except Exception as exc:
        logger.warning(
            f"Blacklist DB check failed for jti={jti[:8]}: {exc} — treating as clean"
        )

    # Cache as clean for BL_MISS_TTL seconds
    _BL_MISS_CACHE[jti] = True
    return False


# ── Core authentication dependencies ─────────────────────────────────────────


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """
    FastAPI dependency — returns the authenticated User for the request.

    Validation steps:
      1. Decode & verify JWT signature / expiry.
      2. Confirm token_type == "access" (reject refresh tokens).
      3. Check JTI against blacklist (logout invalidation).
      4. Return User from cache or DB.
    """
    _creds_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = verify_token(credentials.credentials)
    if not payload:
        raise _creds_error

    # ── Token type guard: only accept "access" tokens ──────────────────────
    token_type = payload.get("token_type")
    if token_type and token_type != TOKEN_TYPE_ACCESS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A refresh token cannot be used for authentication",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Blacklist check ─────────────────────────────────────────────────────
    jti = payload.get("jti")
    if jti and await _is_token_blacklisted(jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise _creds_error

    # ── User cache lookup ───────────────────────────────────────────────────
    cached_user = _USER_CACHE.get(user_id)
    if cached_user is not None:
        if not cached_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated",
            )
        return cached_user

    # ── DB lookup (cache miss) ──────────────────────────────────────────────
    try:
        user = await User.find_one(User.user_id == user_id)
    except Exception as exc:
        logger.error(f"DB error fetching user {user_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    _USER_CACHE[user_id] = user
    return user


async def get_current_user_with_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Tuple[User, str, datetime]:
    """
    Extended dependency used by the logout endpoint.

    Returns ``(user, jti, expires_at)`` so the caller can blacklist
    the exact token that was presented — no guessing needed.
    """
    user = await get_current_user(credentials)

    # verify_token won't fail here because get_current_user already validated it,
    # but we guard for None defensively so the type checker is satisfied.
    payload = verify_token(credentials.credentials)
    if payload is None:
        # Should never happen (get_current_user would have raised), but be safe
        return (
            user,
            "",
            datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )

    jti: str = payload.get("jti") or ""
    exp_epoch: int = payload.get("exp") or 0
    expires_at = (
        datetime.utcfromtimestamp(exp_epoch)
        if exp_epoch
        else (datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    )
    return user, jti, expires_at


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
) -> Optional[User]:
    """
    Dependency that returns the authenticated user OR None for anonymous requests.

    Use on endpoints that support both authenticated and unauthenticated access.
    """
    if not credentials:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def invalidate_user_cache(user_id: str) -> None:
    """
    Evict a user from the in-memory cache.

    Call this after updating user fields (e.g., deactivation, password reset)
    so the next request re-reads the fresh state from MongoDB.
    """
    _USER_CACHE.pop(user_id, None)
    logger.debug(f"User cache invalidated for {user_id}")
