"""
High-level market data caching utilities.
Provides convenience methods for caching market data purely in-memory.
"""

from typing import Optional, Any, Callable, Awaitable, Dict
from datetime import datetime, timedelta
import logging

from ..config import settings

logger = logging.getLogger(__name__)


class MarketDataCacheManager:
    """High-level in-memory cache manager."""
    
    # In-memory store: Dict[f"{symbol}:{data_type}", Dict{"data": Any, "expires_at": datetime}]
    _cache: Dict[str, Dict[str, Any]] = {}
    
    @staticmethod
    async def get_or_fetch(
        symbol: str,
        data_type: str,
        fetch_func: Callable[[], Awaitable[dict[str, Any]]],
        ttl_seconds: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Get data from cache or fetch if missing/expired.
        
        Args:
            symbol: Stock/crypto symbol
            data_type: Type of data (info, historical, etc.)
            fetch_func: Async function to fetch fresh data
            ttl_seconds: Cache TTL (defaults to settings.cache_ttl_seconds)
        
        Returns:
            Market data dictionary
        """
        ttl = ttl_seconds or settings.cache_ttl_seconds
        cache_key = f"{symbol}:{data_type}"
        
        # Try to get from cache
        cache_entry = MarketDataCacheManager._cache.get(cache_key)
        if cache_entry and datetime.utcnow() <= cache_entry["expires_at"]:
            return cache_entry["data"]
        
        # Cache miss or expired - fetch fresh data
        try:
            fresh_data = await fetch_func()
            
            # Store in cache
            MarketDataCacheManager._cache[cache_key] = {
                "data": fresh_data,
                "expires_at": datetime.utcnow() + timedelta(seconds=ttl)
            }
            
            return fresh_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol} ({data_type}): {e}")
            raise
    
    @staticmethod
    async def invalidate(symbol: str) -> int:
        """Invalidate all cache entries for a symbol."""
        keys_to_delete = [
            k for k in MarketDataCacheManager._cache.keys()
            if k.startswith(f"{symbol}:")
        ]
        for k in keys_to_delete:
            del MarketDataCacheManager._cache[k]
        return len(keys_to_delete)
    
    @staticmethod
    async def cleanup_expired() -> int:
        """Clean up all expired cache entries."""
        now = datetime.utcnow()
        keys_to_delete = [
            k for k, v in MarketDataCacheManager._cache.items()
            if now > v["expires_at"]
        ]
        for k in keys_to_delete:
            del MarketDataCacheManager._cache[k]
        return len(keys_to_delete)
