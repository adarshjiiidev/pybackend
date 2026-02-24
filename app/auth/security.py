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

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = settings.jwt_secret  # Use production JWT secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

# HTTP Bearer for authorization header
security = HTTPBearer()

# In-memory cache for ultra-fast auth (Bolt: ⚡ Speed boost)
# Stores user_id -> (User, expiry) and jti -> (is_blacklisted, expiry)
_user_cache: dict[str, tuple[User, datetime]] = {}
_blacklist_cache: dict[str, datetime] = {}
AUTH_CACHE_TTL_SECONDS = 300  # 5 minutes
MAX_CACHE_SIZE = 1000  # Prevent memory leaks


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
    Dependency to get the current authenticated user from JWT token.
    Optimized with in-memory caching to avoid redundant DB calls on every request.
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if token is blacklisted (logged out)
    jti = payload.get("jti")
    if jti:
        # Check cache first
        now = datetime.utcnow()
        if jti in _blacklist_cache:
            if _blacklist_cache[jti] > now:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                del _blacklist_cache[jti]

        blacklisted = await TokenBlacklist.find_one(TokenBlacklist.jti == jti)
        if blacklisted:
            # Cache blacklist status until token would have expired anyway
            if len(_blacklist_cache) >= MAX_CACHE_SIZE:
                _blacklist_cache.clear()
            _blacklist_cache[jti] = blacklisted.expires_at
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Get user_id from token
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Try cache first (Bolt: ⚡ Avoid DB hit on every request)
    now = datetime.utcnow()
    if user_id in _user_cache:
        cached_user, expiry = _user_cache[user_id]
        if expiry > now:
            # logger.debug(f"Auth cache HIT for {user_id}")
            return cached_user
        else:
            del _user_cache[user_id]

    # Cache miss - Get user from database
    user = await User.find_one(User.user_id == user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    # Cache user for future requests (with simple size limit)
    if len(_user_cache) >= MAX_CACHE_SIZE:
        # Clear oldest (very simple approach)
        _user_cache.clear()

    _user_cache[user_id] = (user, now + timedelta(seconds=AUTH_CACHE_TTL_SECONDS))

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
