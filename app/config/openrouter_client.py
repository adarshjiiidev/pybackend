"""
OpenRouter API Key Rotation Manager
Mirrors key_rotator.py but uses the openai Python SDK with OpenRouter's base URL.

OpenRouter is 100% OpenAI-compatible:
  base_url = "https://openrouter.ai/api/v1"
  Uses the same chat.completions.create() interface.

Lock strategy: same as key_rotator.py — threading.Lock for nanosecond
critical sections (safe from async code).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterKeyRotator:
    """
    Thread-safe round-robin API key rotator for OpenRouter.
    Caches AsyncOpenAI instances for connection pooling.
    """

    def __init__(self, api_keys: List[str]) -> None:
        if not api_keys:
            raise ValueError("At least one OpenRouter API key is required")

        self.api_keys = api_keys
        self.current_index = 0
        self._sync_lock = threading.Lock()
        self.request_counts: Dict[str, int] = {key: 0 for key in api_keys}
        self._clients: Dict[str, AsyncOpenAI] = {}

        logger.info(f"🔑 OpenRouter rotator initialized with {len(api_keys)} keys")

    def get_next_key(self) -> str:
        """Round-robin key selection (thread-safe)."""
        with self._sync_lock:
            key = self.api_keys[self.current_index]
            self.request_counts[key] += 1
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            return key

    def get_client(self) -> AsyncOpenAI:
        """Get an AsyncOpenAI client configured for OpenRouter with rotated key."""
        api_key = self.get_next_key()
        with self._sync_lock:
            if api_key not in self._clients:
                self._clients[api_key] = AsyncOpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=api_key,
                    default_headers={
                        "HTTP-Referer": "https://daddysai.com",
                        "X-Title": "DaddysAI",
                    },
                )
            return self._clients[api_key]

    async def aclose_all(self) -> None:
        """Close all cached clients during shutdown."""
        with self._sync_lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            try:
                await client.close()
            except Exception:
                pass
        if clients:
            logger.info(f"✅ Closed {len(clients)} OpenRouter clients")

    def get_stats(self) -> dict:
        with self._sync_lock:
            total = sum(self.request_counts.values())
            return {
                "total_keys": len(self.api_keys),
                "total_requests": total,
                "requests_per_key": dict(self.request_counts),
            }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_rotator: Optional[OpenRouterKeyRotator] = None
_fallback_client: Optional[AsyncOpenAI] = None
_fallback_lock = threading.Lock()


def initialize_openrouter(api_keys: List[str]) -> None:
    """Initialize the global OpenRouter rotator. Call once at startup."""
    global _global_rotator
    _global_rotator = OpenRouterKeyRotator(api_keys)
    logger.info(f"✅ Global OpenRouter rotator initialized with {len(api_keys)} keys")


def get_openrouter_rotator() -> OpenRouterKeyRotator:
    """Get the global rotator. Raises RuntimeError if not initialized."""
    if _global_rotator is None:
        raise RuntimeError("OpenRouter rotator not initialized. Call initialize_openrouter() at startup.")
    return _global_rotator


def get_openrouter_client() -> AsyncOpenAI:
    """
    Get an OpenRouter client with rotated key.
    Falls back to primary key if rotator not yet initialized.
    """
    if _global_rotator is not None:
        return _global_rotator.get_client()

    global _fallback_client
    with _fallback_lock:
        if _fallback_client is None:
            from .settings import settings
            key = settings.openrouter_api_key
            if not key:
                raise RuntimeError("No OpenRouter API key configured")
            logger.debug("⚠️ OpenRouter rotator not initialized, using fallback primary key")
            _fallback_client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=key,
                default_headers={
                    "HTTP-Referer": "https://daddysai.com",
                    "X-Title": "DaddysAI",
                },
            )
        return _fallback_client


async def call_openrouter(model: str, messages: list, **kwargs) -> Any:
    """
    Call OpenRouter LLM with automatic key rotation on 401/429.
    Tries ALL configured keys before falling back to Groq.
    Uses cached clients from the rotator for connection pooling.
    """
    # Collect all keys to try
    if _global_rotator is not None:
        all_keys = list(_global_rotator.api_keys)
    else:
        from .settings import settings
        all_keys = settings.get_all_openrouter_keys()
        if not all_keys:
            raise RuntimeError("No OpenRouter API key configured")

    last_error: Optional[Exception] = None

    for key_idx, api_key in enumerate(all_keys):
        try:
            # Reuse cached client from rotator; create ad-hoc if rotator not up
            if _global_rotator is not None:
                with _global_rotator._sync_lock:
                    if api_key not in _global_rotator._clients:
                        _global_rotator._clients[api_key] = AsyncOpenAI(
                            base_url=OPENROUTER_BASE_URL,
                            api_key=api_key,
                            default_headers={
                                "HTTP-Referer": "https://daddysai.com",
                                "X-Title": "DaddysAI",
                            },
                        )
                    client = _global_rotator._clients[api_key]
            else:
                client = AsyncOpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=api_key,
                    default_headers={
                        "HTTP-Referer": "https://daddysai.com",
                        "X-Title": "DaddysAI",
                    },
                )

            response = await client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            # Update rotator stats if available
            if _global_rotator is not None:
                with _global_rotator._sync_lock:
                    _global_rotator.request_counts[api_key] = (
                        _global_rotator.request_counts.get(api_key, 0) + 1
                    )
            return response

        except Exception as e:
            last_error = e
            err_str = str(e)
            if "401" in err_str or "invalid" in err_str.lower():
                logger.warning(f"OpenRouter key {key_idx+1}/{len(all_keys)} invalid (401) — trying next")
                continue
            elif "429" in err_str or "rate" in err_str.lower() or "rate_limit" in err_str.lower():
                wait = 1.0 + key_idx * 0.5  # 1s, 1.5s, 2s ...
                logger.warning(
                    f"OpenRouter key {key_idx+1}/{len(all_keys)} rate-limited (429) on {model} "
                    f"— waiting {wait:.1f}s then trying next key"
                )
                await asyncio.sleep(wait)
                continue
            else:
                # Unknown error — don't bother trying other keys
                raise

    # All OpenRouter keys exhausted — fall back to Groq
    logger.warning(
        f"All {len(all_keys)} OpenRouter key(s) rate-limited for {model}. "
        "Falling back to Groq."
    )
    try:
        from .key_rotator import get_groq_client
        from .settings import settings
        groq_client = get_groq_client()
        # Map OR model → closest Groq model
        groq_model = settings.model_analysis  # llama-3.3-70b-versatile
        # Strip tool_choice / tools from kwargs (Groq API is compatible but
        # some OR-only params may not be; passing them is safe, they're ignored)
        response = await groq_client.chat.completions.create(
            model=groq_model, messages=messages, **kwargs
        )
        logger.info(f"✅ Groq fallback succeeded using {groq_model}")
        return response
    except Exception as groq_err:
        logger.error(f"Groq fallback also failed: {groq_err}")
        # Re-raise original OpenRouter error so caller sees the real problem
        raise last_error


async def close_openrouter_fallback() -> None:
    """Close the fallback client if it exists."""
    global _fallback_client
    client = None
    with _fallback_lock:
        if _fallback_client:
            client = _fallback_client
            _fallback_client = None
    if client:
        await client.close()
