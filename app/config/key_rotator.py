"""
API Key Rotation Manager for Groq
Intelligently rotates through multiple API keys to distribute load and avoid rate limits.
"""

import threading
import logging
import asyncio
from typing import Optional, Dict
from groq import AsyncGroq

logger = logging.getLogger(__name__)


class GroqKeyRotator:
    """
    Thread-safe round-robin API key rotator.
    Cycles through available keys to distribute requests evenly.
    Caches AsyncGroq instances to enable connection pooling across requests.
    """
    
    def __init__(self, api_keys: list[str]):
        """
        Initialize with list of API keys.
        
        Args:
            api_keys: List of Groq API keys to rotate through
        """
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        self.api_keys = api_keys
        self.current_index = 0
        self.lock = threading.Lock()
        self.request_counts = {key: 0 for key in api_keys}
        
        # Performance Optimization: Cache clients to reuse httpx connection pools
        self._clients: Dict[str, AsyncGroq] = {}

        logger.info(f"🔑 Initialized key rotator with {len(api_keys)} API keys")
    
    def get_next_key(self) -> str:
        """
        Get the next API key in rotation (round-robin).
        Thread-safe operation.
        
        Returns:
            str: Next API key to use
        """
        with self.lock:
            key = self.api_keys[self.current_index]
            self.request_counts[key] += 1
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            logger.debug(
                f"🔄 Using key #{self.current_index} "
                f"(Total requests on this key: {self.request_counts[key]})"
            )
            
            return key
    
    def get_client(self) -> AsyncGroq:
        """
        Get a Groq client with the next rotated API key.
        Reuses cached clients to benefit from connection pooling.
        
        Returns:
            AsyncGroq: Groq client configured with next key
        """
        api_key = self.get_next_key()

        with self.lock:
            if api_key not in self._clients:
                logger.debug(f"Creating new AsyncGroq client for key index {self.api_keys.index(api_key)}")
                self._clients[api_key] = AsyncGroq(api_key=api_key)
            return self._clients[api_key]
    
    async def aclose_all(self):
        """
        Asynchronously close all cached Groq clients.
        Should be called during application shutdown.
        """
        with self.lock:
            clients_to_close = list(self._clients.values())
            self._clients.clear()

        if clients_to_close:
            logger.info(f"Closing {len(clients_to_close)} cached Groq clients...")
            # Close all clients in parallel
            await asyncio.gather(*[client.close() for client in clients_to_close], return_exceptions=True)
            logger.info("✅ All cached Groq clients closed")

    def get_stats(self) -> dict:
        """
        Get rotation statistics.
        
        Returns:
            dict: Request counts per key
        """
        with self.lock:
            total_requests = sum(self.request_counts.values())
            return {
                "total_keys": len(self.api_keys),
                "total_requests": total_requests,
                "requests_per_key": dict(self.request_counts),
                "average_per_key": total_requests / len(self.api_keys) if self.api_keys else 0
            }
    
    def reset_stats(self):
        """Reset request counters."""
        with self.lock:
            self.request_counts = {key: 0 for key in self.api_keys}
            logger.info("📊 Reset rotation statistics")


# Global rotator instance (will be initialized in main.py)
_global_rotator: Optional[GroqKeyRotator] = None
# Cache for fallback client when rotator is not initialized
_fallback_client: Optional[AsyncGroq] = None
_fallback_lock = threading.Lock()


def initialize_rotator(api_keys: list[str]):
    """
    Initialize the global key rotator.
    Should be called once during application startup.
    
    Args:
        api_keys: List of Groq API keys
    """
    global _global_rotator
    _global_rotator = GroqKeyRotator(api_keys)
    logger.info(f"✅ Global API key rotator initialized with {len(api_keys)} keys")


def get_rotator() -> GroqKeyRotator:
    """
    Get the global key rotator instance.
    
    Returns:
        GroqKeyRotator: Global rotator
        
    Raises:
        RuntimeError: If rotator hasn't been initialized
    """
    if _global_rotator is None:
        raise RuntimeError(
            "Key rotator not initialized. "
            "Call initialize_rotator() during app startup."
        )
    return _global_rotator


def get_groq_client() -> AsyncGroq:
    """
    Convenience function to get a Groq client with rotated key.
    Falls back to primary API key if rotator not yet initialized.
    Reuses client instances for performance.
    
    Returns:
        AsyncGroq: Groq client with next available key
    """
    if _global_rotator is not None:
        return _global_rotator.get_client()
    
    # Fallback: rotator not initialized yet (happens during module imports)
    global _fallback_client
    with _fallback_lock:
        if _fallback_client is None:
            from .settings import settings
            logger.debug("⚠️ Rotator not initialized yet, creating/reusing fallback primary API key client")
            _fallback_client = AsyncGroq(api_key=settings.groq_api_key)
        return _fallback_client


def get_next_api_key() -> str:
    """
    Convenience function to get the next API key.
    Falls back to primary API key if rotator not yet initialized.
    
    Returns:
        str: Next API key in rotation
    """
    if _global_rotator is None:
        from .settings import settings
        logger.debug("⚠️ Rotator not initialized yet, using primary API key")
        return settings.groq_api_key
    
    return _global_rotator.get_next_key()


async def close_fallback_client():
    """Close the fallback client if it exists."""
    global _fallback_client
    client_to_close = None
    with _fallback_lock:
        if _fallback_client:
            client_to_close = _fallback_client
            _fallback_client = None

    if client_to_close:
        await client_to_close.close()
        logger.debug("Closed fallback Groq client")
