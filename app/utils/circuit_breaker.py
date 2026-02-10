"""
Circuit Breaker Pattern Implementation.
Prevents cascading failures by temporarily blocking calls to failing services.
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation - requests pass through
    OPEN = "open"          # Failing - reject requests immediately
    HALF_OPEN = "half_open"  # Testing recovery - allow limited requests


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, reject all requests
    - HALF_OPEN: Testing if service recovered, allow one request
    
    Example:
        groq_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
        result = await groq_breaker.call(groq_client.chat, prompt)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting recovery (OPEN → HALF_OPEN)
            success_threshold: Consecutive successes needed to close circuit from HALF_OPEN
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit breaker entering HALF_OPEN state")
                    self.state = CircuitState.HALF_OPEN
                else:
                    time_until_reset = self._time_until_reset()
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. Service unavailable. "
                        f"Retry in {time_until_reset}s"
                    )
        
        try:
            # Call the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success
            await self._on_success()
            return result
            
        except Exception as e:
            # Failure
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.info(f"Circuit breaker success count: {self.success_count}/{self.success_threshold}")
                
                if self.success_count >= self.success_threshold:
                    logger.info("Circuit breaker CLOSED - service recovered")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = 0
    
    async def _on_failure(self, exception: Exception):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = datetime.now()
            
            logger.warning(
                f"Circuit breaker failure {self.failure_count}/{self.failure_threshold}: {exception}"
            )
            
            if self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker reopened during HALF_OPEN test")
                self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
    
    def _time_until_reset(self) -> int:
        """Calculate seconds until circuit attempts reset"""
        if self.last_failure_time is None:
            return 0
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return max(0, int(self.timeout - elapsed))
    
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_until_reset": self._time_until_reset() if self.state == CircuitState.OPEN else None
        }
    
    async def reset(self):
        """Manually reset circuit breaker to CLOSED"""
        async with self._lock:
            logger.info("Circuit breaker manually reset to CLOSED")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
