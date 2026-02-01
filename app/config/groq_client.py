"""
API client with key rotation, retry logic, and rate limit handling.
Ensures robust Groq API access with automatic fallback.
"""

import asyncio
from typing import Optional, List, Any
from groq import AsyncGroq
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class GroqClientManager:
    """
    Manages multiple Groq API keys with automatic rotation and retry logic.
    Handles rate limits and provides fallback mechanisms.
    """
    
    def __init__(self, api_keys: List[str], max_retries: int = 3):
        """
        Initialize with multiple API keys for rotation.
        
        Args:
            api_keys: List of Groq API keys
            max_retries: Maximum retry attempts per request
        """
        self.api_keys = api_keys if isinstance(api_keys, list) else [api_keys]
        self.current_key_index = 0
        self.max_retries = max_retries
        self.clients = {key: AsyncGroq(api_key=key) for key in self.api_keys}
        
        # Track rate limits per key
        self.rate_limit_reset = {}
        self.request_counts = {key: 0 for key in self.api_keys}
        
        logger.info(f"Initialized GroqClientManager with {len(self.api_keys)} API key(s)")
    
    def get_current_client(self) -> AsyncGroq:
        """Get the current active Groq client."""
        current_key = self.api_keys[self.current_key_index]
        return self.clients[current_key]
    
    def rotate_key(self):
        """Rotate to the next API key."""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.info(f"Rotated to API key index {self.current_key_index}")
    
    def is_rate_limited(self, api_key: str) -> bool:
        """Check if a specific API key is rate limited."""
        if api_key in self.rate_limit_reset:
            reset_time = self.rate_limit_reset[api_key]
            if datetime.now() < reset_time:
                return True
            else:
                # Reset expired, clear it
                del self.rate_limit_reset[api_key]
        return False
    
    def mark_rate_limited(self, api_key: str, retry_after: int = 60):
        """Mark an API key as rate limited."""
        reset_time = datetime.now() + timedelta(seconds=retry_after)
        self.rate_limit_reset[api_key] = reset_time
        logger.warning(f"API key marked as rate limited until {reset_time}")
    
    async def create_completion_with_retry(
        self,
        model: str,
        messages: List[dict],
        **kwargs
    ) -> Any:
        """
        Create chat completion with automatic retry and key rotation.
        
        Handles:
        - Rate limit errors (429)
        - Network errors
        - API errors
        - Automatic key rotation on failure
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            current_key = self.api_keys[self.current_key_index]
            
            # Skip rate-limited keys
            if self.is_rate_limited(current_key):
                logger.warning(f"Key {self.current_key_index} is rate limited, rotating...")
                self.rotate_key()
                continue
            
            client = self.get_current_client()
            
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries} with key index {self.current_key_index}")
                
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                
                # Success! Track the request
                self.request_counts[current_key] += 1
                return response
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Handle rate limit (429)
                if "429" in error_str or "rate limit" in error_str:
                    logger.warning(f"Rate limit hit on key {self.current_key_index}")
                    
                    # Try to extract retry-after from error
                    retry_after = 60  # Default 1 minute
                    self.mark_rate_limited(current_key, retry_after)
                    self.rotate_key()
                    
                    # Exponential backoff
                    wait_time = min(2 ** attempt, 30)
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    
                # Handle other errors
                elif "api key" in error_str or "unauthorized" in error_str:
                    logger.error(f"API key error on key {self.current_key_index}, rotating...")
                    self.rotate_key()
                    await asyncio.sleep(1)
                    
                elif "timeout" in error_str or "network" in error_str:
                    logger.warning(f"Network error: {e}, retrying...")
                    await asyncio.sleep(2 ** attempt)
                    
                else:
                    # Unknown error, log and rotate
                    logger.error(f"Unexpected error: {e}")
                    if attempt < self.max_retries - 1:
                        self.rotate_key()
                        await asyncio.sleep(2)
        
        # All retries exhausted
        logger.error(f"All {self.max_retries} retry attempts failed")
        raise last_exception if last_exception else Exception("All retries failed")
    
    def get_status(self) -> dict:
        """Get current status of all API keys."""
        return {
            "total_keys": len(self.api_keys),
            "current_key_index": self.current_key_index,
            "rate_limited_keys": len(self.rate_limit_reset),
            "request_counts": self.request_counts
        }


# Global client manager instance
_client_manager: Optional[GroqClientManager] = None


def initialize_groq_client(api_keys: List[str] | str) -> GroqClientManager:
    """Initialize the global Groq client manager."""
    global _client_manager
    keys = api_keys if isinstance(api_keys, list) else [api_keys]
    _client_manager = GroqClientManager(keys, max_retries=3)
    return _client_manager


def get_groq_client() -> GroqClientManager:
    """Get the global Groq client manager."""
    if _client_manager is None:
        raise RuntimeError("Groq client not initialized. Call initialize_groq_client() first.")
    return _client_manager
