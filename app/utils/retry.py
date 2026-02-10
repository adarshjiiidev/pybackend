"""
Retry decorator with exponential backoff.
Automatically retries failed async operations with configurable backoff.
"""

import asyncio
from functools import wraps
from typing import Callable, Type, Tuple
import logging

logger = logging.getLogger(__name__)


def async_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] = None
):
    """
    Decorator for automatic retries with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (including initial)
        backoff_factor: Multiplier for delay between retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function(exception, attempt) called on each retry
    
    Example:
        @async_retry(max_attempts=3, backoff_factor=2, exceptions=(aiohttp.ClientError,))
        async def fetch_data(url):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()
    
    Retry delays:
    - Attempt 1: immediate
    - Attempt 2: initial_delay (e.g., 1s)
    - Attempt 3: initial_delay * backoff_factor (e.g., 2s)
    - Attempt 4: initial_delay * backoff_factor^2 (e.g., 4s)
    - ...
    - Capped at max_delay
    """
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            last_exception = None
            
            while attempt < max_attempts:
                try:
                    # Attempt the function call
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    attempt += 1
                    last_exception = e
                    
                    # If this was the last attempt, raise the exception
                    if attempt >= max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (backoff_factor ** (attempt - 1)),
                        max_delay
                    )
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt)
                        except Exception as callback_error:
                            logger.error(f"Retry callback failed: {callback_error}")
                    
                    # Wait before retrying
                    await asyncio.sleep(delay)
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retry logic (alternative to decorator).
    
    Example:
        async with RetryContext(max_attempts=3) as retry:
            while retry.should_retry():
                try:
                    result = await some_operation()
                    retry.success()
                    return result
                except Exception as e:
                    retry.failure(e)
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        
        self.attempt = 0
        self.last_exception = None
        self._succeeded = False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and not self._succeeded:
            logger.error(f"Retry context failed after {self.attempt} attempts: {exc_val}")
        return False  # Don't suppress exceptions
    
    def should_retry(self) -> bool:
        """Check if we should attempt/retry"""
        return self.attempt < self.max_attempts and not self._succeeded
    
    async def failure(self, exception: Exception):
        """Record a failure and wait before next retry"""
        self.attempt += 1
        self.last_exception = exception
        
        if self.attempt < self.max_attempts:
            delay = min(
                self.initial_delay * (self.backoff_factor ** (self.attempt - 1)),
                self.max_delay
            )
            logger.warning(
                f"Retry attempt {self.attempt}/{self.max_attempts} failed: {exception}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
        else:
            logger.error(f"All {self.max_attempts} retry attempts exhausted")
    
    def success(self):
        """Mark operation as successful"""
        self._succeeded = True
