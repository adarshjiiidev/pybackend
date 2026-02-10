"""
Rate limiter using Token Bucket algorithm.
Controls request rate per user to prevent abuse and API overload.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict
import asyncio
import logging

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Each user has a bucket that holds tokens.
    - Tokens refill at a constant rate
    - Each request consumes tokens
    - If bucket is empty, request is denied
    
    Example:
        limiter = TokenBucket(capacity=100, refill_rate=10)
        
        if await limiter.acquire(user_id="user123"):
            # Process request
            pass
        else:
            # Rate limit exceeded
            raise HTTPException(429, "Too many requests")
    """
    
    def __init__(
        self,
        capacity: int = 100,
        refill_rate: float = 10.0,
        tokens_per_request: int = 1
    ):
        """
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            tokens_per_request: Tokens consumed per request
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens_per_request = tokens_per_request
        
        self.buckets: Dict[str, dict] = defaultdict(lambda: {
            "tokens": float(capacity),
            "last_refill": datetime.now()
        })
        self._lock = asyncio.Lock()
    
    async def acquire(self, user_id: str, tokens_needed: int = None) -> bool:
        """
        Try to acquire tokens for a request.
        
        Args:
            user_id: Unique user identifier
            tokens_needed: Number of tokens to consume (default: tokens_per_request)
        
        Returns:
            True if tokens acquired, False if rate limit exceeded
        """
        if tokens_needed is None:
            tokens_needed = self.tokens_per_request
        
        async with self._lock:
            bucket = self.buckets[user_id]
            
            # Refill tokens based on time elapsed
            now = datetime.now()
            elapsed = (now - bucket["last_refill"]).total_seconds()
            refill_amount = elapsed * self.refill_rate
            
            # Update tokens (capped at capacity)
            bucket["tokens"] = min(self.capacity, bucket["tokens"] + refill_amount)
            bucket["last_refill"] = now
            
            # Check if enough tokens available
            if bucket["tokens"] >= tokens_needed:
                bucket["tokens"] -= tokens_needed
                logger.debug(
                    f"Rate limit OK for {user_id}: {bucket['tokens']:.1f}/{self.capacity} tokens"
                )
                return True
            else:
                logger.warning(
                    f"Rate limit EXCEEDED for {user_id}: "
                    f"{bucket['tokens']:.1f}/{self.capacity} tokens available, "
                    f"{tokens_needed} needed"
                )
                return False
    
    async def get_status(self, user_id: str) -> dict:
        """Get current token bucket status for a user"""
        async with self._lock:
            bucket = self.buckets.get(user_id)
            
            if not bucket:
                return {
                    "tokens": self.capacity,
                    "capacity": self.capacity,
                    "percentage": 100.0
                }
            
            # Calculate current tokens with refill
            now = datetime.now()
            elapsed = (now - bucket["last_refill"]).total_seconds()
            refill_amount = elapsed * self.refill_rate
            current_tokens = min(self.capacity, bucket["tokens"] + refill_amount)
            
            return {
                "tokens": round(current_tokens, 2),
                "capacity": self.capacity,
                "percentage": round((current_tokens / self.capacity) * 100, 1),
                "refill_rate": self.refill_rate
            }
    
    async def reset(self, user_id: str):
        """Reset token bucket for a user (fill to capacity)"""
        async with self._lock:
            if user_id in self.buckets:
                self.buckets[user_id] = {
                    "tokens": float(self.capacity),
                    "last_refill": datetime.now()
                }
                logger.info(f"Reset rate limiter for {user_id}")


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter (alternative to token bucket).
    Tracks requests in a time window.
    
    Example:
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        
        if await limiter.allow(user_id="user123"):
            # Process request
            pass
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def allow(self, user_id: str) -> bool:
        """Check if request is allowed"""
        async with self._lock:
            now = datetime.now()
            user_requests = self.requests[user_id]
            
            # Remove old requests outside window
            cutoff = now.timestamp() - self.window_seconds
            user_requests[:] = [
                req_time for req_time in user_requests
                if req_time > cutoff
            ]
            
            # Check if under limit
            if len(user_requests) < self.max_requests:
                user_requests.append(now.timestamp())
                logger.debug(
                    f"Rate limit OK for {user_id}: "
                    f"{len(user_requests)}/{self.max_requests} requests in window"
                )
                return True
            else:
                logger.warning(
                    f"Rate limit EXCEEDED for {user_id}: "
                    f"{len(user_requests)}/{self.max_requests} requests in {self.window_seconds}s window"
                )
                return False
    
    async def get_count(self, user_id: str) -> int:
        """Get current request count in window"""
        async with self._lock:
            now = datetime.now()
            cutoff = now.timestamp() - self.window_seconds
            user_requests = self.requests.get(user_id, [])
            
            # Count requests in window
            return sum(1 for req_time in user_requests if req_time > cutoff)
