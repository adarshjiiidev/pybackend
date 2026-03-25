"""
Async-safe TTL Cache for expensive API/scraper calls.

Usage:
    from app.utils.ttl_cache import TTLCache

    # Module-level cache instance
    nse_quote_cache = TTLCache(ttl=60)  # 60-second TTL

    # In your tool:
    cached = nse_quote_cache.get("RELIANCE")
    if cached is not None:
        return cached
    result = await fetch_live_data("RELIANCE")
    nse_quote_cache.set("RELIANCE", result)
    return result
"""

import time
import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TTLCache:
    """
    Thread-safe (asyncio-safe) in-memory cache with per-entry TTL expiry.

    Args:
        ttl: Time-to-live in seconds for each cached entry (default: 60s)
        max_size: Maximum number of cached entries (LRU eviction when full, default: 500)
    """

    def __init__(self, ttl: float = 60.0, max_size: int = 500):
        self._ttl = ttl
        self._max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._lock = asyncio.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and not expired, else None. Thread-safe (sync)."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            # Expired — evict lazily
            self._cache.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with TTL expiry. Evicts oldest entry if at max_size. Thread-safe (sync)."""
        if len(self._cache) >= self._max_size:
            # Evict the entry with the soonest expiry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            self._cache.pop(oldest_key, None)
        self._cache[key] = (value, time.monotonic() + self._ttl)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    def size(self) -> int:
        """Return current number of entries (including potentially expired ones)."""
        return len(self._cache)

    async def get_or_fetch(self, key: str, fetch_fn, *args, **kwargs) -> Any:
        """
        Async helper: return cached value or call fetch_fn(*args, **kwargs) and cache result.

        Example:
            result = await cache.get_or_fetch("NIFTY50", fetch_nse_index, "NIFTY 50")
        """
        cached = self.get(key)
        if cached is not None:
            logger.debug(f"TTLCache hit: {key}")
            return cached

        logger.debug(f"TTLCache miss: {key} — fetching fresh data")
        result = await fetch_fn(*args, **kwargs)
        if result is not None:
            self.set(key, result)
        return result


# ── Shared cache instances (import these in tool files) ─────────────────────
# 60s TTL: Prices and quotes change every minute, faster is wasteful API calls
nse_quote_cache = TTLCache(ttl=60, max_size=200)

# 5min TTL: Sector performance is slower-moving aggregated data
sector_cache = TTLCache(ttl=300, max_size=50)

# 10min TTL: Market-wide indices (Nifty, Sensex) — updated every few minutes on NSE
index_cache = TTLCache(ttl=600, max_size=20)
