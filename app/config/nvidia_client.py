"""
NVIDIA NIM API Client
=====================
OpenAI-compatible client for NVIDIA's Inference Microservices (NIM) platform.
Supports DeepSeek-V3.2 with extended reasoning (thinking tokens).

Base URL: https://integrate.api.nvidia.com/v1
Auth: Bearer nvapi-* key

Production-ready: thread-safe, async, connection-pooled.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Default model — DeepSeek-V3.2 with extended thinking
NVIDIA_DEFAULT_MODEL = "deepseek-ai/deepseek-v3.2"

_client: Optional[AsyncOpenAI] = None
_lock = threading.Lock()


def get_nvidia_client(api_key: Optional[str] = None) -> AsyncOpenAI:
    """Get (or create) the cached NVIDIA NIM AsyncOpenAI client."""
    global _client
    with _lock:
        if _client is not None:
            return _client
        if api_key is None:
            from .settings import settings
            api_key = getattr(settings, "nvidia_api_key", None)
        if not api_key:
            raise RuntimeError(
                "NVIDIA API key not configured. "
                "Set NVIDIA_API_KEY in your .env file."
            )
        _client = AsyncOpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
        )
        logger.info("✅ NVIDIA NIM client initialised (DeepSeek-V3.2 ready)")
        return _client


async def call_nvidia(
    messages: list,
    model: str = NVIDIA_DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    thinking: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Call NVIDIA NIM API (non-streaming).
    Returns an openai.types.chat.ChatCompletion object.

    Args:
        messages:    Standard OpenAI messages list.
        model:       NIM model name. Default: deepseek-ai/deepseek-v3.2
        temperature: Sampling temperature.
        max_tokens:  Max output tokens (NIM supports up to 8192 for DeepSeek-V3.2).
        thinking:    Enable extended reasoning / thinking tokens.
    """
    client = get_nvidia_client()

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **kwargs,
    }

    if thinking:
        # NVIDIA NIM DeepSeek extended thinking
        create_kwargs["extra_body"] = {
            "chat_template_kwargs": {"thinking": True}
        }

    try:
        response = await client.chat.completions.create(**create_kwargs)
        logger.debug(f"🟢 NVIDIA NIM responded | model={model}")
        return response
    except Exception as e:
        logger.warning(f"NVIDIA NIM call failed: {e}")
        raise


async def close_nvidia_client() -> None:
    """Gracefully close the NVIDIA client on shutdown."""
    global _client
    with _lock:
        client, _client = _client, None
    if client:
        try:
            await client.close()
            logger.info("✅ NVIDIA NIM client closed")
        except Exception:
            pass
