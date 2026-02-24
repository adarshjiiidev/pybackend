"""
Security utilities for authentication.
Handles password hashing, JWT token generation/validation, and verification tokens.
"""

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import uuid

from ..config import settings
from ..models import User, TokenBlacklist
import time

# ── In-memory caches ────────────────────────────────────────────────────────
# User cache: avoids DB lookup on every authenticated request
_user_cache: dict[str, tuple] = {}          # user_id → (User, timestamp)
_USER_CACHE_TTL = 60                         # seconds

# Blacklist cache: avoids DB lookup for non-revoked tokens
_blacklist_cache: set[str] = set()           # set of known-revoked jtis
_blacklist_miss_cache: set[str] = set()      # set of known-clean jtis (not revoked)
_BLACKLIST_MISS_TTL_MAX = 5000               # evict oldest miss entries when set gets this big

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.jwt_secret  # Use production JWT secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# HTTP Bearer for authorization header
security = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token (usually {"sub": user_id})
        expires_delta: Optional expiration time override
    
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "jti": str(uuid.uuid4())  # JWT ID for blacklisting
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
    
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_verification_token() -> str:
    """Generate a secure random token for email verification or magic links."""
    return secrets.token_urlsafe(32)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Dependency to get the current authenticated user.
    Uses two in-memory caches to avoid repeated DB hits:
    - _user_cache: user object cached for 60s by user_id
    - _blacklist_miss_cache: known-clean jtis skip DB blacklist check
    """
    token = credentials.credentials
    payload = verify_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check token blacklist — cache-first
    jti = payload.get("jti")
    if jti:
        if jti in _blacklist_cache:
            # Known revoked
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if jti not in _blacklist_miss_cache:
            # Unknown — check DB once
            blacklisted = await TokenBlacklist.find_one(TokenBlacklist.jti == jti)
            if blacklisted:
                _blacklist_cache.add(jti)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # Cache as clean; evict if set too large
            if len(_blacklist_miss_cache) > _BLACKLIST_MISS_TTL_MAX:
                _blacklist_miss_cache.clear()
            _blacklist_miss_cache.add(jti)

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # User cache lookup
    cached = _user_cache.get(user_id)
    if cached:
        user_obj, ts = cached
        if time.time() - ts < _USER_CACHE_TTL:
            if not user_obj.is_active:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is deactivated")
            return user_obj

    # DB lookup (cache miss)
    user = await User.find_one(User.user_id == user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is deactivated")

    _user_cache[user_id] = (user, time.time())
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    Use this for endpoints that support both authenticated and non-authenticated users.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
