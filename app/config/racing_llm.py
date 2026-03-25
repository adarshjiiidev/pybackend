"""
Racing LLM — Multi-Provider Parallel Inference with NVIDIA Priority + Refinement
==================================================================================
Strategy:
  1. Fire OpenRouter + NVIDIA NIM (DeepSeek-V3.2) + Groq simultaneously
  2. NVIDIA DeepSeek gets PRIORITY — if it responds within a grace window (NVIDIA_GRACE_S),
     it is always chosen even if Groq was faster.
  3. After a winner is chosen, its raw response is REFINED by a fast model that:
       • Strips thinking-trace artifacts / XML tags from DeepSeek output
       • Fixes markdown formatting
       • Enforces proper section structure and removes bullet points
  4. The refined, polished response is returned.

Priority tiers:
  🟣 NVIDIA DeepSeek-V3.2   — highest quality reasoning  (grace window: 8s)
  🟢 OpenRouter              — good quality, free models  (fallback)
  🔵 Groq llama 70B          — fastest, lowest quality    (emergency fallback)

Modes:
  "fast"      → 8B models, no thinking, no refine (latency-critical)
  "synthesis" → DeepSeek-V3.2 preferred, balanced refine
  "deep"      → DeepSeek-V3.2 + thinking tokens, full refine
  "report"    → DeepSeek-V3.2 + thinking, heavy formatting refine
  "plan"      → 8B models, no thinking, no refine (JSON output)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# How long to wait for NVIDIA before falling back to whoever already won
NVIDIA_GRACE_S = 8.0

# Modes that skip the refinement pass (JSON outputs, fast greeter responses)
_NO_REFINE_MODES = {"fast", "plan"}

# Modes where NVIDIA thinking is enabled
_THINKING_MODES = {"deep", "report"}


# ── Provider result ───────────────────────────────────────────────────────────


@dataclass
class RaceResult:
    text: str
    provider: str      # "openrouter" | "nvidia" | "groq"
    model: str
    latency_s: float
    tokens_used: Optional[int] = None
    is_refined: bool = False


# ── Model configuration per mode ─────────────────────────────────────────────


def _get_race_config(mode: str, max_tokens: int, temperature: float) -> dict:
    from .settings import settings

    configs: dict[str, dict[str, Any]] = {
        "fast": {
            "openrouter_model": settings.or_model_fast,
            "nvidia_model":     "meta/llama-3.1-8b-instruct",
            "groq_model":       "llama-3.1-8b-instant",
            "temperature":      0.2,
            "max_tokens":       min(max_tokens, 800),
            "nvidia_thinking":  False,
            "nvidia_priority":  False,   # no priority for fast mode
            "refine":           False,
        },
        "synthesis": {
            "openrouter_model": settings.or_model_analysis,
            "nvidia_model":     "deepseek-ai/deepseek-v3.2",
            "groq_model":       "llama-3.3-70b-versatile",
            "temperature":      temperature,
            "max_tokens":       min(max_tokens, 3500),
            "nvidia_thinking":  False,
            "nvidia_priority":  True,    # prefer DeepSeek if it wins within grace
            "refine":           True,
        },
        "deep": {
            "openrouter_model": settings.or_model_deep,
            "nvidia_model":     "deepseek-ai/deepseek-v3.2",
            "groq_model":       "llama-3.3-70b-versatile",
            "temperature":      temperature,
            "max_tokens":       min(max_tokens, 4096),
            "nvidia_thinking":  True,
            "nvidia_priority":  True,
            "refine":           True,
        },
        "report": {
            "openrouter_model": settings.or_model_analysis,
            "nvidia_model":     "deepseek-ai/deepseek-v3.2",
            "groq_model":       "llama-3.3-70b-versatile",
            "temperature":      temperature,
            "max_tokens":       min(max_tokens, 4096),
            "nvidia_thinking":  True,
            "nvidia_priority":  True,
            "refine":           True,
        },
        "plan": {
            "openrouter_model": settings.or_model_fast,
            "nvidia_model":     "meta/llama-3.1-8b-instruct",
            "groq_model":       "llama-3.1-8b-instant",
            "temperature":      0.1,
            "max_tokens":       min(max_tokens, 2500),
            "nvidia_thinking":  False,
            "nvidia_priority":  False,
            "refine":           False,
        },
    }
    return configs.get(mode, configs["synthesis"])


# ── Individual provider coroutines ────────────────────────────────────────────


async def _call_openrouter_provider(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> RaceResult:
    from .openrouter_client import call_openrouter
    t0 = time.monotonic()
    response = await call_openrouter(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, **kwargs,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("OpenRouter returned empty content")
    latency = round(time.monotonic() - t0, 2)
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "completion_tokens", None) if usage else None
    logger.info(f"🟢 OpenRouter responded in {latency}s | model={model}")
    return RaceResult(text=text, provider="openrouter", model=model,
                      latency_s=latency, tokens_used=tokens)


async def _call_nvidia_provider(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking: bool = False,
    **kwargs: Any,
) -> RaceResult:
    from .nvidia_client import call_nvidia
    t0 = time.monotonic()
    response = await call_nvidia(
        messages=messages, model=model,
        temperature=temperature, max_tokens=max_tokens,
        thinking=thinking,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("NVIDIA NIM returned empty content")
    latency = round(time.monotonic() - t0, 2)
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "completion_tokens", None) if usage else None
    logger.info(f"🟣 NVIDIA DeepSeek responded in {latency}s | model={model}")
    return RaceResult(text=text, provider="nvidia", model=model,
                      latency_s=latency, tokens_used=tokens)


async def _call_groq_provider(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> RaceResult:
    from .key_rotator import get_groq_client
    t0 = time.monotonic()
    client = get_groq_client()
    safe_kwargs = {k: v for k, v in kwargs.items()
                   if k not in ("extra_body", "extra_headers")}
    response = await client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, **safe_kwargs,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("Groq returned empty content")
    latency = round(time.monotonic() - t0, 2)
    usage = getattr(response, "usage", None)
    tokens = getattr(usage, "completion_tokens", None) if usage else None
    logger.info(f"🔵 Groq responded in {latency}s | model={model}")
    return RaceResult(text=text, provider="groq", model=model,
                      latency_s=latency, tokens_used=tokens)


# ── Refinement pass ───────────────────────────────────────────────────────────


async def _refine_response(
    raw: str,
    original_messages: list,
    mode: str,
    source_provider: str,
) -> str:
    """
    Polish the winning provider's raw output using a fast model.

    Handles:
      - DeepSeek thinking traces  (<think>...</think> XML blocks)
      - Broken markdown headers
      - Bullet-point enforcement → flowing paragraphs
      - Section structure alignment
      - Redundant preamble removal ("Certainly! Here is...")

    Uses a fast, cheap model (8B class) so the overhead is minimal (~1-2s).
    """
    import re as _re

    # ── Pre-clean: strip DeepSeek thinking artifacts ──────────────────────────
    # DeepSeek-V3.2 with thinking=True may emit <think>...</think> blocks
    cleaned = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
    # Also strip ```think ... ``` fenced blocks
    cleaned = _re.sub(r"```think\s*.*?```", "", cleaned, flags=_re.DOTALL).strip()

    # If already clean and short skip the LLM refine call
    if len(cleaned) < 200:
        return cleaned or raw

    # Build a minimal refinement prompt
    refine_prompt = f"""You are a financial content editor. Polish the following AI-generated financial analysis:

**Rules (STRICT - follow exactly):**
- Output ONLY the polished content, no preamble like "Here is", "Certainly" etc.
- Use flowing paragraphs under descriptive headings. NO bullet points except in data tables.
- Keep ALL numbers, signals, and factual data exactly as they are.
- Fix any broken markdown: ensure ## for sections, **bold** for key terms, proper tables.
- Remove any remaining <think> tags, XML tags, or internal reasoning traces.
- Do NOT add new content. Only clean and reformat what exists.

---
{cleaned[:6000]}
---

Output the polished version now:"""

    try:
        from .key_rotator import get_groq_client
        client = get_groq_client()
        # Use fast 8B model for refinement — low latency
        refine_response = await asyncio.wait_for(
            client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.15,
                max_tokens=4096,
            ),
            timeout=20.0,
        )
        refined = (refine_response.choices[0].message.content or "").strip()
        if refined and len(refined) > 100:
            logger.info(
                f"✨ Refined [{source_provider}→groq-8b] "
                f"{len(raw)}→{len(refined)} chars"
            )
            return refined
    except Exception as e:
        logger.warning(f"⚠️ Refine pass failed ({e}) — using pre-cleaned output")

    return cleaned


# ── Priority-aware race ───────────────────────────────────────────────────────


async def _priority_race(
    nvidia_task: Optional[asyncio.Task],
    other_tasks: list[asyncio.Task],
    nvidia_priority: bool,
    overall_deadline: float,
) -> tuple[Optional[RaceResult], list[asyncio.Task]]:
    """
    Run the race with NVIDIA priority logic.

    Phase A (grace window): wait up to NVIDIA_GRACE_S for NVIDIA.
      - If NVIDIA wins in grace window → return it immediately.
      - If another provider wins first in grace window → hold their result.
      - After grace window expires → return the best available result.

    If nvidia_priority is False, just use pure FIRST_COMPLETED.
    """
    all_tasks: set[asyncio.Task] = set(other_tasks)
    if nvidia_task:
        all_tasks.add(nvidia_task)

    if not nvidia_priority or nvidia_task is None:
        # Pure speed race — no priority
        winner: Optional[RaceResult] = None
        errors: list = []
        pending = all_tasks
        while pending and winner is None:
            remaining = overall_deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                exc = task.exception()
                if exc:
                    errors.append(exc)
                    logger.warning(f"⚠️ {task.get_name()} failed: {exc}")
                else:
                    r = task.result()
                    if r and r.text:
                        winner = r
                        break
        return winner, list(pending)

    # ── Phase A: NVIDIA grace window ──────────────────────────────────────────
    grace_deadline = asyncio.get_event_loop().time() + NVIDIA_GRACE_S
    fallback_winner: Optional[RaceResult] = None
    pending = all_tasks.copy()

    while pending:
        remaining_grace = grace_deadline - asyncio.get_event_loop().time()
        remaining_total = overall_deadline - asyncio.get_event_loop().time()
        remaining = min(remaining_grace, remaining_total)
        if remaining <= 0:
            break

        done, pending = await asyncio.wait(
            pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            exc = task.exception()
            if exc:
                logger.warning(f"⚠️ {task.get_name()} failed in grace window: {exc}")
                continue
            r = task.result()
            if not r or not r.text:
                continue

            if task is nvidia_task:
                # NVIDIA won inside grace window — use it immediately 🏆
                logger.info(f"🏆 NVIDIA DeepSeek chosen (grace window) in {r.latency_s}s")
                return r, list(pending)
            else:
                # Another provider won before NVIDIA — hold it as fallback
                if fallback_winner is None:
                    fallback_winner = r
                    logger.info(
                        f"⏳ {r.provider} finished first ({r.latency_s}s), "
                        f"waiting {remaining_grace:.1f}s more for NVIDIA..."
                    )

        # If grace window is up, stop waiting for NVIDIA specifically
        if asyncio.get_event_loop().time() >= grace_deadline:
            break

    # ── Phase B: grace window expired ────────────────────────────────────────
    if fallback_winner:
        logger.info(
            f"⌛ NVIDIA grace window expired — using {fallback_winner.provider} "
            f"({fallback_winner.latency_s}s)"
        )
        return fallback_winner, list(pending)

    # ── Phase C: nothing won yet — wait for any remaining provider ────────────
    while pending:
        remaining = overall_deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        done, pending = await asyncio.wait(
            pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            exc = task.exception()
            if exc:
                logger.warning(f"⚠️ {task.get_name()} failed: {exc}")
                continue
            r = task.result()
            if r and r.text:
                logger.info(f"✅ Late winner: {r.provider} ({r.latency_s}s)")
                return r, list(pending)

    return None, list(pending)


# ── Main racing function ──────────────────────────────────────────────────────


async def call_racing_llm(
    messages: list,
    mode: str = "synthesis",
    max_tokens: int = 3500,
    temperature: float = 0.3,
    timeout: float = 60.0,
    refine: Optional[bool] = None,   # None = auto (based on mode config)
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Fire all configured LLM providers in parallel.
    NVIDIA DeepSeek-V3.2 is given priority via a grace window.
    The winning response is refined before delivery.

    Returns:
        (polished_response_text, winning_provider_name)
    """
    from .settings import settings

    cfg = _get_race_config(mode, max_tokens, temperature)
    should_refine = refine if refine is not None else cfg.get("refine", True)

    nvidia_task: Optional[asyncio.Task] = None
    other_tasks: list[asyncio.Task] = []

    # --- NVIDIA NIM (DeepSeek-V3.2) ---
    if settings.nvidia_available:
        nvidia_task = asyncio.create_task(
            _call_nvidia_provider(
                messages=messages,
                model=cfg["nvidia_model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                thinking=cfg.get("nvidia_thinking", False),
            ),
            name="nvidia",
        )

    # --- OpenRouter ---
    if settings.openrouter_available:
        other_tasks.append(asyncio.create_task(
            _call_openrouter_provider(
                messages=messages,
                model=cfg["openrouter_model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                **kwargs,
            ),
            name="openrouter",
        ))

    # --- Groq ---
    try:
        from .key_rotator import get_groq_client
        get_groq_client()
        other_tasks.append(asyncio.create_task(
            _call_groq_provider(
                messages=messages,
                model=cfg["groq_model"],
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
            ),
            name="groq",
        ))
    except Exception:
        pass

    total_tasks = len(other_tasks) + (1 if nvidia_task else 0)
    if total_tasks == 0:
        raise RuntimeError("No LLM providers configured. Check API keys in .env")

    logger.info(
        f"🏁 Racing {total_tasks} providers | mode={mode} "
        f"nvidia_priority={cfg['nvidia_priority']} "
        f"max_tokens={cfg['max_tokens']} refine={should_refine}"
    )

    overall_deadline = asyncio.get_event_loop().time() + timeout

    winner, pending_tasks = await _priority_race(
        nvidia_task=nvidia_task,
        other_tasks=other_tasks,
        nvidia_priority=cfg.get("nvidia_priority", True),
        overall_deadline=overall_deadline,
    )

    # Cancel all still-running tasks
    for task in pending_tasks:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    if winner is None:
        raise RuntimeError(f"All {total_tasks} LLM providers failed or timed out.")

    logger.info(
        f"🏆 Winner: {winner.provider} | latency={winner.latency_s}s "
        f"| tokens={winner.tokens_used}"
    )

    # ── Refinement pass ───────────────────────────────────────────────────────
    if should_refine:
        polished = await _refine_response(
            raw=winner.text,
            original_messages=messages,
            mode=mode,
            source_provider=winner.provider,
        )
        return polished, winner.provider

    return winner.text, winner.provider


# ── Drop-in replacement for call_openrouter ───────────────────────────────────


async def call_llm_racing(
    model: str,
    messages: list,
    mode: str = "synthesis",
    **kwargs,
) -> Any:
    """
    Drop-in replacement for call_openrouter().
    Races all providers, DeepSeek-V3.2 preferred, refines the winner.
    Returns a response-like object with .choices[0].message.content set.
    """
    max_tokens = kwargs.pop("max_tokens", 3500)
    temperature = kwargs.pop("temperature", 0.3)

    text, provider = await call_racing_llm(
        messages=messages,
        mode=mode,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    class _FakeMessage:
        content: str = ""

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    resp = _FakeResponse()
    resp.choices[0].message.content = text
    return resp
