"""
Caching layer for NSE data to provide instant responses.
Uses MongoDB with TTL for automatic cache expiration.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class NSEDataCache:
    """Cache NSE data in memory and MongoDB for instant responses."""
    
    def __init__(self):
        # In-memory cache for ultra-fast access
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = {
            "quote": 0,  # DISABLED — always fetch fresh
            "fii_dii": 0,  # DISABLED — always fetch fresh
            "option_chain": 0,  # DISABLED — always fetch fresh
            "market_status": 0  # DISABLED — always fetch fresh
        }
    
    def _get_cache_key(self, data_type: str, symbol: Optional[str] = None) -> str:
        """Generate cache key."""
        if symbol:
            return f"{data_type}:{symbol.upper()}"
        return data_type
    
    def get(self, data_type: str, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get data from cache if not expired."""
        key = self._get_cache_key(data_type, symbol)
        
        if key in self._memory_cache:
            cached = self._memory_cache[key]
            cached_at = datetime.fromisoformat(cached.get("_cached_at", "2000-01-01"))
            ttl = self._cache_ttl.get(data_type, 60)
            
            if datetime.now() - cached_at < timedelta(seconds=ttl):
                logger.info(f"Cache HIT: {key}")
                return cached.get("data")
        
        logger.info(f"Cache MISS: {key}")
        return None
    
    def set(self, data_type: str, data: Dict[str, Any], symbol: Optional[str] = None):
        """Store data in cache."""
        key = self._get_cache_key(data_type, symbol)
        
        self._memory_cache[key] = {
            "data": data,
            "_cached_at": datetime.now().isoformat()
        }
        
        logger.info(f"Cache SET: {key}")
    
    def invalidate(self, data_type: str, symbol: Optional[str] = None):
        """Invalidate cache entry."""
        key = self._get_cache_key(data_type, symbol)
        if key in self._memory_cache:
            del self._memory_cache[key]
            logger.info(f"Cache INVALIDATE: {key}")


# Global cache instance
_nse_cache: Optional[NSEDataCache] = None


def get_nse_cache() -> NSEDataCache:
    """Get or create cache instance."""
    global _nse_cache
    if _nse_cache is None:
        _nse_cache = NSEDataCache()
    return _nse_cache
