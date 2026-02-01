"""
High-level market data caching utilities.
Provides convenience methods for caching market data with TTL management.
"""

from typing import Optional, Any, Callable, Awaitable
import logging

from .repositories import CacheRepository
from ..config import settings

logger = logging.getLogger(__name__)


class MarketDataCacheManager:
    """High-level cache manager with automatic cache refresh."""
    
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
        
        # Try to get from cache
        cached_data = await CacheRepository.get_cached_data(symbol, data_type)
        if cached_data:
            return cached_data
        
        # Cache miss - fetch fresh data
        try:
            fresh_data = await fetch_func()
            
            # Store in cache
            await CacheRepository.set_cached_data(
                symbol=symbol,
                data_type=data_type,
                data=fresh_data,
                ttl_seconds=ttl
            )
            
            return fresh_data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol} ({data_type}): {e}")
            raise
    
    @staticmethod
    async def invalidate(symbol: str) -> int:
        """Invalidate all cache entries for a symbol."""
        return await CacheRepository.invalidate_cache(symbol)
    
    @staticmethod
    async def cleanup_expired() -> int:
        """Clean up all expired cache entries."""
        return await CacheRepository.clear_expired_cache()
