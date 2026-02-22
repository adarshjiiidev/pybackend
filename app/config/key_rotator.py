"""
API Key Rotation Manager for Groq
Intelligently rotates through multiple API keys to distribute load and avoid rate limits.
"""

import threading
import logging
from typing import Optional
from groq import AsyncGroq

logger = logging.getLogger(__name__)


class GroqKeyRotator:
    """
    Thread-safe round-robin API key rotator.
    Cycles through available keys to distribute requests evenly.
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
        
        # Performance: Pre-initialize and reuse clients for connection pooling
        self.clients = {key: AsyncGroq(api_key=key) for key in api_keys}

        logger.info(f"🔑 Initialized key rotator with {len(api_keys)} API keys and persistent clients")
    
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
        Reuses persistent clients to benefit from connection pooling.
        
        Returns:
            AsyncGroq: Cached Groq client configured with next key
        """
        api_key = self.get_next_key()
        return self.clients[api_key]

    async def close_all(self):
        """
        Close all persistent Groq clients.
        Should be called during application shutdown.
        """
        logger.info(f"🔌 Closing {len(self.clients)} persistent Groq clients...")
        for i, (key, client) in enumerate(self.clients.items()):
            try:
                # Groq SDK uses .close() as an async method
                await client.close()
            except Exception as e:
                logger.error(f"Error closing Groq client #{i}: {e}")
    
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
    
    Returns:
        AsyncGroq: Groq client with next available key
    """
    if _global_rotator is None:
        # Fallback: rotator not initialized yet (happens during module imports)
        # Use primary key from settings
        from .settings import settings
        logger.debug("⚠️ Rotator not initialized yet, using primary API key")
        return AsyncGroq(api_key=settings.groq_api_key)
    
    return _global_rotator.get_client()


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
