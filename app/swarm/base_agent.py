"""
BaseSwarmAgent — Foundation class for all dynamically spawned swarm agents.

Every agent in the DaddysAI swarm inherits from this class. It provides:
  • Unique agent identity (ID, type, display name)
  • Lifecycle management (IDLE → RUNNING → DONE / FAILED / DISPOSED)
  • Structured message passing (input → output via SwarmMessage)
  • Access to ALL system tools (NSE, TA, OI, prediction, web, KB)
  • Heartbeat / timeout tracking
  • Parent-child agent relationships (for sub-agent spawning)
  • Rich logging with agent context

Lifecycle::

    agent = MyAgent(agent_id="ta-001", parent_id="orch-0")
    result = await agent.run(SwarmMessage(task="analyze RELIANCE", payload={...}))
    await agent.dispose()

Agent states::

    IDLE     → not yet started
    RUNNING  → actively processing
    DONE     → completed successfully, result available
    FAILED   → error occurred, error details in state
    DISPOSED → cleaned up, no longer usable
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent status
# ---------------------------------------------------------------------------


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DISPOSED = "disposed"


# ---------------------------------------------------------------------------
# Message / result containers
# ---------------------------------------------------------------------------


@dataclass
class SwarmMessage:
    """
    Input message delivered to an agent when it is spawned.

    Attributes:
        task        Human-readable task description
        payload     Arbitrary dict with task-specific data
                    (query, symbol, df_dict, option_chain, etc.)
        priority    0 = normal, 1 = high, 2 = critical
        timeout_s   Hard deadline in seconds (agent kills itself after this)
        trace_id    Propagated request trace ID for end-to-end logging
    """

    task: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    timeout_s: float = 120.0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class AgentResult:
    """
    Output produced by an agent after it finishes.

    Attributes:
        agent_id    Which agent produced this
        agent_type  Human-readable type label
        status      Final lifecycle status
        data        Main result payload (type depends on agent)
        summary     One-paragraph human-readable summary for the orchestrator
        signal      Optional directional signal: 'bullish' | 'bearish' | 'neutral'
        confidence  0.0 – 1.0 confidence in the result
        error       Error message if status == FAILED
        duration_s  Wall-clock seconds the agent ran for
        sub_results Results from any child agents spawned by this agent
        metadata    Extra metadata dict (tools used, models, etc.)
    """

    agent_id: str
    agent_type: str
    status: AgentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    signal: str = "neutral"
    confidence: float = 0.5
    error: Optional[str] = None
    duration_s: float = 0.0
    sub_results: List["AgentResult"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "data": self.data,
            "summary": self.summary,
            "signal": self.signal,
            "confidence": round(self.confidence, 3),
            "error": self.error,
            "duration_s": round(self.duration_s, 2),
            "sub_results": [r.to_dict() for r in self.sub_results],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseSwarmAgent(ABC):
    """
    Abstract base for every DaddysAI swarm agent.

    Subclasses must implement:
        async execute(message: SwarmMessage) -> AgentResult

    Everything else (lifecycle, timeouts, logging, tool access) is
    provided by this base class.
    """

    # Human-readable type label — override in every subclass
    AGENT_TYPE: str = "base_agent"

    # Default timeout in seconds — subclasses may override
    DEFAULT_TIMEOUT_S: float = 90.0

    def __init__(
        self,
        agent_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.agent_id = agent_id or f"{self.AGENT_TYPE}-{str(uuid.uuid4())[:8]}"
        self.parent_id = parent_id
        self.config = config or {}

        self.status: AgentStatus = AgentStatus.IDLE
        self.created_at: datetime = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        self._result: Optional[AgentResult] = None
        self._children: List[BaseSwarmAgent] = []
        self._log = logging.getLogger(f"swarm.{self.AGENT_TYPE}.{self.agent_id}")

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    async def run(self, message: SwarmMessage) -> AgentResult:
        """
        Main entry point.  Wraps execute() with:
          • Status transitions
          • Wall-clock timing
          • Timeout enforcement
          • Exception catching → FAILED status
          • Structured logging
        """
        if self.status == AgentStatus.DISPOSED:
            raise RuntimeError(f"Agent {self.agent_id} is already disposed")

        self._transition(AgentStatus.RUNNING)
        self.started_at = datetime.utcnow()
        t0 = time.monotonic()
        timeout = message.timeout_s or self.DEFAULT_TIMEOUT_S

        self._log.info(
            f"▶ [{self.agent_id}] Starting | task={message.task!r} "
            f"| trace={message.trace_id} | timeout={timeout}s"
        )

        try:
            result = await asyncio.wait_for(
                self.execute(message),
                timeout=timeout,
            )
            result.duration_s = time.monotonic() - t0
            self._result = result
            self._transition(AgentStatus.DONE)

            self._log.info(
                f"✅ [{self.agent_id}] Done in {result.duration_s:.1f}s "
                f"| signal={result.signal} conf={result.confidence:.2f}"
            )
            return result

        except asyncio.TimeoutError:
            duration = time.monotonic() - t0
            err_msg = f"Agent timed out after {timeout}s"
            self._log.warning(f"⏱ [{self.agent_id}] {err_msg}")
            result = self._make_failed_result(err_msg, duration)
            self._result = result
            self._transition(AgentStatus.FAILED)
            return result

        except Exception as exc:
            duration = time.monotonic() - t0
            err_msg = f"{type(exc).__name__}: {exc}"
            self._log.error(
                f"❌ [{self.agent_id}] Failed after {duration:.1f}s: {err_msg}",
                exc_info=True,
            )
            result = self._make_failed_result(err_msg, duration)
            self._result = result
            self._transition(AgentStatus.FAILED)
            return result

        finally:
            self.ended_at = datetime.utcnow()

    async def dispose(self) -> None:
        """
        Clean up agent resources and mark as DISPOSED.
        Subclasses may override to release connections, cancel tasks, etc.
        Override teardown() for custom cleanup — don't override dispose().
        """
        await self.teardown()
        for child in self._children:
            try:
                await child.dispose()
            except Exception:
                pass
        self._children.clear()
        self._transition(AgentStatus.DISPOSED)
        self._log.debug(f"🗑 [{self.agent_id}] Disposed")

    # ------------------------------------------------------------------
    # Abstract / override points
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(self, message: SwarmMessage) -> AgentResult:
        """
        Core logic — every subclass must implement this.

        Receives a SwarmMessage with the task description and payload dict.
        Must return an AgentResult.

        Guidelines:
          • Use self.tools.* for all data access (never raw HTTP directly)
          • Use self.spawn_child() to create sub-agents
          • Use self._log.info/warning/error for structured logging
          • Catch expected errors; let unexpected ones bubble up to base.run()
        """
        ...

    async def teardown(self) -> None:
        """
        Optional cleanup hook — called by dispose().
        Override to release resources (DB cursors, HTTP clients, etc.).
        """
        pass

    # ------------------------------------------------------------------
    # Tool access — lazy-initialised singletons
    # ------------------------------------------------------------------

    @property
    def tools(self) -> "AgentToolbox":
        """Access the shared AgentToolbox (lazy singleton)."""
        return AgentToolbox.instance()

    # ------------------------------------------------------------------
    # Sub-agent spawning
    # ------------------------------------------------------------------

    async def spawn_child(
        self,
        agent_class: type,
        message: SwarmMessage,
        config: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Dynamically create a child agent, run it, and return its result.
        The child is auto-disposed after completion.

        Usage::

            result = await self.spawn_child(
                TechnicalAnalysisAgent,
                SwarmMessage(task="TA for RELIANCE", payload={"symbol": "RELIANCE"}),
            )
        """
        child = agent_class(parent_id=self.agent_id, config=config or {})
        self._children.append(child)
        self._log.info(
            f"🐣 [{self.agent_id}] Spawning child: {child.agent_id} "
            f"for task={message.task!r}"
        )
        try:
            result = await child.run(message)
            return result
        finally:
            await child.dispose()
            if child in self._children:
                self._children.remove(child)

    async def spawn_parallel(
        self,
        tasks: List[tuple],  # [(agent_class, SwarmMessage), ...]
    ) -> List[AgentResult]:
        """
        Spawn multiple child agents IN PARALLEL and collect all results.

        Usage::

            results = await self.spawn_parallel([
                (TechnicalAnalysisAgent, SwarmMessage(task="TA", payload={...})),
                (OIAnalysisAgent,        SwarmMessage(task="OI", payload={...})),
                (WebResearchAgent,       SwarmMessage(task="news", payload={...})),
            ])
        """
        self._log.info(f"🌊 [{self.agent_id}] Spawning {len(tasks)} parallel children")

        async def _run_one(agent_class, msg):
            return await self.spawn_child(agent_class, msg)

        results = await asyncio.gather(
            *[_run_one(cls, msg) for cls, msg in tasks],
            return_exceptions=True,
        )

        # Convert exceptions to failed AgentResults
        safe_results: List[AgentResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                cls_name = tasks[i][0].AGENT_TYPE if i < len(tasks) else "unknown"
                safe_results.append(
                    AgentResult(
                        agent_id=f"{cls_name}-error",
                        agent_type=cls_name,
                        status=AgentStatus.FAILED,
                        error=str(r),
                        summary=f"Child agent failed: {r}",
                    )
                )
            else:
                safe_results.append(r)

        return safe_results

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def result(self) -> Optional[AgentResult]:
        return self._result

    @property
    def is_done(self) -> bool:
        return self.status in (
            AgentStatus.DONE,
            AgentStatus.FAILED,
            AgentStatus.DISPOSED,
        )

    def get_info(self) -> Dict[str, Any]:
        """Return agent identity and status as a dict (for dashboards/logging)."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.AGENT_TYPE,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "children": [c.agent_id for c in self._children],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_status: AgentStatus) -> None:
        old = self.status
        self.status = new_status
        self._log.debug(f"[{self.agent_id}] {old.value} → {new_status.value}")

    def _make_failed_result(self, error: str, duration: float) -> AgentResult:
        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            status=AgentStatus.FAILED,
            error=error,
            summary=f"Agent {self.AGENT_TYPE} failed: {error}",
            duration_s=duration,
        )

    def _ok(
        self,
        data: Dict[str, Any],
        summary: str = "",
        signal: str = "neutral",
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> AgentResult:
        """
        Convenience helper for subclasses to build a successful AgentResult.

        Usage (inside execute())::

            return self._ok(
                data={"indicators": {...}},
                summary="RELIANCE is bullish on RSI+MACD confluence.",
                signal="bullish",
                confidence=0.78,
            )
        """
        return AgentResult(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            status=AgentStatus.DONE,
            data=data,
            summary=summary,
            signal=signal,
            confidence=confidence,
            metadata=metadata or {},
        )


# ---------------------------------------------------------------------------
# AgentToolbox — shared, lazy-loaded access to all system tools
# ---------------------------------------------------------------------------


class AgentToolbox:
    """
    Centralised toolbox that every swarm agent uses to access system services.

    Lazy singletons avoid importing heavy modules at startup.
    All attributes are properties that cache on first access.
    """

    _instance: Optional["AgentToolbox"] = None

    @classmethod
    def instance(cls) -> "AgentToolbox":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Data Pipeline ─────────────────────────────────────────────────────

    @property
    def data_pool(self):
        from ..data_pipeline.data_pool import get_data_pool

        return get_data_pool()

    @property
    def data_fetcher(self):
        from ..data_pipeline.data_fetcher import MarketDataFetcher

        return MarketDataFetcher()

    # ── Analytics ─────────────────────────────────────────────────────────

    @property
    def technical_engine(self):
        from ..analytics.technical_engine import TechnicalEngine

        return TechnicalEngine()

    @property
    def pattern_detector(self):
        from ..analytics.pattern_detector import PatternDetector

        return PatternDetector()

    @property
    def oi_analyzer(self):
        from ..analytics.oi_analyzer import OIAnalyzer

        return OIAnalyzer()

    @property
    def shock_detector(self):
        from ..analytics.shock_detector import ShockDetector

        return ShockDetector()

    @property
    def buyer_seller_analyzer(self):
        from ..analytics.buyer_seller_analyzer import BuyerSellerAnalyzer

        return BuyerSellerAnalyzer()

    @property
    def breadth_analyzer(self):
        from ..analytics.market_breadth import MarketBreadthAnalyzer

        return MarketBreadthAnalyzer()

    # ── NSE / Market Data ─────────────────────────────────────────────────

    async def fetch_nse_quote(self, symbol: str) -> Dict[str, Any]:
        from ..tools.nse_scraper import fetch_nse_quote

        return await fetch_nse_quote(symbol)

    async def fetch_option_chain(self, symbol: str = "NIFTY") -> Dict[str, Any]:
        from ..tools.nse_scraper import fetch_option_chain

        return await fetch_option_chain(symbol)

    async def fetch_fii_dii(self) -> Dict[str, Any]:
        from ..tools.nse_scraper import fetch_fii_dii

        return await fetch_fii_dii()

    async def fetch_market_status(self) -> Dict[str, Any]:
        from ..tools.nse_scraper import fetch_market_status

        return await fetch_market_status()

    async def fetch_live_price(self, symbol: str) -> Dict[str, Any]:
        from ..tools.nse_scraper import fetch_google_price, fetch_nse_quote

        result = await fetch_nse_quote(symbol)
        if "error" not in result:
            return result
        return await fetch_google_price(symbol)

    # ── Web Search + Research ─────────────────────────────────────────────

    async def web_search(self, query: str) -> str:
        from ..tools.tool_executor import _search_web_groq

        return await _search_web_groq(query)

    async def search_knowledge_base(self, query: str) -> Dict[str, Any]:
        from ..rag import search_knowledge_base

        return await search_knowledge_base(query)

    # ── Groq LLM ──────────────────────────────────────────────────────────

    def get_llm_client(self):
        """Get a Groq client (rotated key). For web search + GPT-OSS only."""
        from ..config.key_rotator import get_groq_client

        return get_groq_client()

    async def call_llm(self, model: str, messages: list, **kwargs):
        """
        Call Groq LLM with automatic key rotation on 401 errors.
        Use for GPT-OSS deep reasoning and Compound web search ONLY.
        For all other LLM calls, use call_openrouter() instead.
        """
        from groq import AsyncGroq as _AsyncGroq

        from ..config.key_rotator import get_rotator as _get_rotator

        try:
            all_keys = list(_get_rotator().api_keys)
        except RuntimeError:
            from ..config.settings import settings as _s

            all_keys = [_s.groq_api_key]

        last_error = None
        for key_idx, api_key in enumerate(all_keys):
            try:
                client = _AsyncGroq(api_key=api_key)
                response = await client.chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                return response
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "401" in err_str or "invalid_api_key" in err_str:
                    logger.debug(
                        f"call_llm: key {key_idx + 1}/{len(all_keys)} failed (401) on {model}"
                    )
                    continue
                elif "429" in err_str or "rate_limit" in err_str:
                    import asyncio

                    await asyncio.sleep(1)
                    continue
                else:
                    raise  # unexpected error, don't retry

        raise last_error  # all keys exhausted

    # ── OpenRouter LLM (primary provider) ─────────────────────────────────

    def get_openrouter_client(self):
        """Get an OpenRouter client (rotated key). Primary LLM provider."""
        from ..config.openrouter_client import get_openrouter_client
        return get_openrouter_client()

    async def call_openrouter(self, model: str, messages: list, **kwargs):
        """
        Call LLM with multi-provider racing (OpenRouter + NVIDIA NIM + Groq).
        Fires all configured providers simultaneously; the fastest wins.
        Falls back to plain OpenRouter if racing module fails.
        """
        from ..config.racing_llm import call_llm_racing

        # Infer mode from max_tokens so the racing module picks right models
        max_tokens = kwargs.get("max_tokens", 3500)
        temperature = kwargs.get("temperature", 0.3)

        if max_tokens <= 800:
            mode = "fast"
        elif max_tokens >= 3000:
            mode = "deep"
        else:
            mode = "synthesis"

        try:
            return await call_llm_racing(
                model=model,
                messages=messages,
                mode=mode,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as racing_err:
            # Graceful fallback to plain OpenRouter if racing module errors
            logger.warning(f"Racing LLM failed ({racing_err}), falling back to OpenRouter")
            from ..config.openrouter_client import call_openrouter
            return await call_openrouter(model=model, messages=messages, **kwargs)

    def get_model(self, task_type: str = "fast") -> str:
        """Get the best model for a task. Returns OpenRouter model when available."""
        from ..config import ModelType, settings

        type_map = {
            "fast": ModelType.FAST,
            "analysis": ModelType.ANALYSIS,
            "reasoning": ModelType.REASONING_DEEP,
            "compound": ModelType.COMPOUND,
            "kb_rag": ModelType.KB_RAG,
        }
        mt = type_map.get(task_type, ModelType.FAST)
        
        # Use OpenRouter for non-compound, non-vision tasks
        if settings.openrouter_available and mt not in (ModelType.COMPOUND, ModelType.COMPOUND_MINI, ModelType.VISION):
            return settings.get_openrouter_model(mt)
        return settings.get_model_for_task(mt)


    # ── OHLCV with auto-fetch fallback ────────────────────────────────────

    async def get_ohlcv(
        self,
        symbol: str,
        days: int = 365,
        interval: str = "1d",
    ):
        """
        Get OHLCV DataFrame.
        1. Try data pool (MongoDB) first
        2. Fall back to yfinance live fetch — ALWAYS works even without MongoDB
        """
        import pandas as pd

        # Try MongoDB / data pool first
        df = pd.DataFrame()
        try:
            df = await self.data_pool.get_ohlcv(symbol, days=days, interval=interval)
        except Exception as exc:
            logger.debug(f"get_ohlcv: data_pool error for {symbol}: {exc} — falling back to live fetch")

        if df is not None and len(df) > 20:
            return df

        # Fallback: always fetch live from yfinance
        period_map = {30: "1mo", 90: "3mo", 180: "6mo", 365: "1y", 730: "2y"}
        period = "1y"
        for threshold, p in sorted(period_map.items()):
            if days <= threshold:
                period = p
                break

        logger.info(f"get_ohlcv: live yfinance fetch for {symbol} (period={period})")
        df = await self.data_fetcher.fetch_ohlcv(
            symbol, period=period, interval=interval
        )
        if df is not None and not df.empty:
            # Store for future use only if DB is available
            try:
                await self.data_pool.store_ohlcv(symbol, df, interval=interval)
            except Exception:
                pass  # Store failure is non-critical
        return df
