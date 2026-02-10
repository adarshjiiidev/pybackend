"""
Fallback chain for graceful degradation.
Try multiple data sources in order until one succeeds.
"""

from typing import List, Tuple, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


class AllSourcesFailedError(Exception):
    """Raised when all fallback sources fail"""
    pass


class FallbackChain:
    """
    Try multiple data sources in order.
    Falls back to next source if current fails.
    
    Example:
        stock_data_chain = FallbackChain(
            ("NSE_Scraper", fetch_nse_quote),
            ("Compound_AI", get_stock_price_compound),
            ("Cache", get_cached_price)
        )
        
        try:
            data = await stock_data_chain.fetch("RELIANCE")
        except AllSourcesFailedError:
            # All sources failed
            return default_response
    """
    
    def __init__(self, *sources: Tuple[str, Callable]):
        """
        Args:
            sources: Tuples of (source_name, async_function)
        """
        self.sources = sources
    
    async def fetch(self, *args, **kwargs) -> Any:
        """
        Try each source in order until one succeeds.
        
        Returns:
            Result from first successful source
        
        Raises:
            AllSourcesFailedError: If all sources fail
        """
        errors = []
        
        for source_name, source_func in self.sources:
            try:
                logger.info(f"Trying source: {source_name}")
                result = await source_func(*args, **kwargs)
                
                # Check if result indicates success
                if result and not self._is_error_result(result):
                    logger.info(f"✅ {source_name} succeeded")
                    return result
                else:
                    logger.warning(f"⚠️ {source_name} returned error result: {result}")
                    errors.append((source_name, "Error result returned"))
                    
            except Exception as e:
                logger.warning(f"❌ {source_name} failed: {e}")
                errors.append((source_name, str(e)))
                continue
        
        # All sources failed
        error_summary = ", ".join([f"{name}: {error}" for name, error in errors])
        raise AllSourcesFailedError(f"All sources failed. Errors: {error_summary}")
    
    @staticmethod
    def _is_error_result(result: Any) -> bool:
        """Check if result indicates an error"""
        if isinstance(result, dict):
            return "error" in result or result.get("status") == "error"
        return False


class PriorityFallback:
    """
    Fallback with configurable priority and timeout.
    
    Example:
        fallback = PriorityFallback()
        fallback.add_source("primary", fetch_primary, priority=1, timeout=2.0)
        fallback.add_source("secondary", fetch_secondary, priority=2, timeout=5.0)
        fallback.add_source("cache", fetch_cache, priority=3, timeout=1.0)
        
        result = await fallback.fetch("RELIANCE")
    """
    
    def __init__(self):
        self.sources: List[Tuple[int, str, Callable, float]] = []
    
    def add_source(
        self,
        name: str,
        func: Callable,
        priority: int = 1,
        timeout: Optional[float] = None
    ):
        """Add a data source with priority"""
        self.sources.append((priority, name, func, timeout))
        # Sort by priority (lower number = higher priority)
        self.sources.sort(key=lambda x: x[0])
    
    async def fetch(self, *args, **kwargs) -> Any:
        """Fetch from sources in priority order"""
        import asyncio
        
        errors = []
        
        for priority, name, func, timeout in self.sources:
            try:
                logger.info(f"Trying {name} (priority {priority})")
                
                if timeout:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                else:
                    result = await func(*args, **kwargs)
                
                if result and not FallbackChain._is_error_result(result):
                    logger.info(f"✅ {name} succeeded")
                    return result
                    
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ {name} timed out after {timeout}s")
                errors.append((name, f"Timeout after {timeout}s"))
            except Exception as e:
                logger.warning(f"❌ {name} failed: {e}")
                errors.append((name, str(e)))
        
        error_summary = ", ".join([f"{name}: {error}" for name, error in errors])
        raise AllSourcesFailedError(f"All sources failed. Errors: {error_summary}")


class CachedFallback:
    """
    Fallback with automatic caching of successful results.
    
    Example:
        fallback = CachedFallback(cache_ttl=300)  # 5 minute cache
        fallback.add_source("api", fetch_api)
        fallback.add_source("webscrape", scrape_web)
        
        # First call fetches from API, caches result
        result1 = await fallback.fetch("RELIANCE")
        
        # Second call within 5 minutes returns cached result
        result2 = await fallback.fetch("RELIANCE")
    """
    
    def __init__(self, cache_ttl: int = 300):
        """
        Args:
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_ttl = cache_ttl
        self.sources: List[Tuple[str, Callable]] = []
        self._cache = {}
        self._cache_times = {}
    
    def add_source(self, name: str, func: Callable):
        """Add a data source"""
        self.sources.append((name, func))
    
    async def fetch(self, cache_key: str, *args, **kwargs) -> Any:
        """Fetch with caching"""
        import time
        
        # Check cache first
        if cache_key in self._cache:
            cache_age = time.time() - self._cache_times[cache_key]
            if cache_age < self.cache_ttl:
                logger.info(f"Cache HIT for {cache_key} (age: {cache_age:.1f}s)")
                return self._cache[cache_key]
        
        # Try sources
        errors = []
        for name, func in self.sources:
            try:
                result = await func(*args, **kwargs)
                
                if result and not FallbackChain._is_error_result(result):
                    # Cache successful result
                    self._cache[cache_key] = result
                    self._cache_times[cache_key] = time.time()
                    logger.info(f"✅ {name} succeeded, result cached")
                    return result
                    
            except Exception as e:
                logger.warning(f"❌ {name} failed: {e}")
                errors.append((name, str(e)))
        
        error_summary = ", ".join([f"{name}: {error}" for name, error in errors])
        raise AllSourcesFailedError(f"All sources failed. Errors: {error_summary}")
