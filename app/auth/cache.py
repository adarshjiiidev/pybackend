"""
In-memory caching for authentication to reduce database load.
"""

import time
import logging
from typing import Dict, Any, Optional, TypeVar, Generic
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar("T")

class AuthCache(Generic[T]):
    """
    Simple thread-safe (mostly) in-memory cache with TTL.
    Used for user profiles and token blacklists.
    """
    def __init__(self, default_ttl_seconds: int = 60):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl_seconds

    def get(self, key: str) -> Optional[T]:
        """Get item from cache if not expired."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            return None

        return entry["data"]

    def set(self, key: str, data: T, ttl_seconds: Optional[int] = None):
        """Set item in cache with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        self._cache[key] = {
            "data": data,
            "expires_at": time.time() + ttl
        }

    def delete(self, key: str):
        """Remove item from cache."""
        if key in self._cache:
            del self._cache[key]

    def clear(self):
        """Clear all items from cache."""
        self._cache.clear()

    def cleanup(self):
        """Remove all expired items from cache to prevent memory leaks."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now > v["expires_at"]]
        for k in expired_keys:
            del self._cache[k]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries from cache")


# Specialized caches
# User cache: stores User objects. TTL 60 seconds.
user_cache = AuthCache(default_ttl_seconds=60)

# Blacklist cache: stores JTI strings. TTL matches token expiry (default 1 hour).
blacklist_cache = AuthCache(default_ttl_seconds=3600)

# OTP Cache: can be used to speed up OTP verification if needed
otp_cache = AuthCache(default_ttl_seconds=600)

def get_user_cache():
    return user_cache

def get_blacklist_cache():
    return blacklist_cache
