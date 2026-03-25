"""
DaddysAI Swarm Package
======================
Phase 2 · Autonomous Multi-Agent Intelligence System

Initialises the swarm subsystem:
  1. Creates the global AgentRegistry singleton
  2. Registers every available agent type
  3. Starts the auto-reaper background task (call init_swarm() at startup)
  4. Exports the MasterOrchestrator for use by the main workflow

Architecture (matches the diagram):
  User Query
      ↓
  MasterOrchestrator          ← plans, dispatches, collects, reports
      ↓  (parallel)
  ┌─────────────────────────────────────────────────────────┐
  │ TechnicalAnalysisAgent  OIAnalysisAgent  WebResearchAgent│
  │ GlobalMarketAgent  ShockDetectionAgent  PredictionAgent  │
  │ FundamentalsAgent  SentimentAgent  DataFetchAgent        │
  └─────────────────────────────────────────────────────────┘
      ↓  (results collected)
  Analysis Phase
      ↓
  ReportAgent               ← asset sub-agent (generates final report)
      ↓
  Delivered to User
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _register_all_agents() -> None:
    """
    Register every agent type in the global AgentRegistry.
    Called once at import time — safe to call multiple times (idempotent).
    """
    from .agent_registry import get_registry

    registry = get_registry()

    # ── Technical Analysis ────────────────────────────────────────────────
    try:
        from .agents.technical_analysis_agent import TechnicalAnalysisAgent

        registry.register_class(
            TechnicalAnalysisAgent,
            description=(
                "Full TA suite: RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, "
                "CCI, VWAP, Ichimoku, Supertrend, EMA/SMA crossovers, OBV, MFI, "
                "candlestick & chart patterns, support/resistance, pivot points."
            ),
            max_concurrent=5,
            default_timeout=60.0,
            tags=["analytics", "technical", "indicators", "patterns"],
        )
        logger.debug("Registered: technical_analysis")
    except Exception as e:
        logger.warning(f"Could not register TechnicalAnalysisAgent: {e}")

    # ── OI Analysis ───────────────────────────────────────────────────────
    try:
        from .agents.oi_analysis_agent import OIAnalysisAgent

        registry.register_class(
            OIAnalysisAgent,
            description=(
                "Option chain analysis: PCR, Max Pain, OI walls (support/resistance), "
                "GEX (gamma exposure), IV skew, OI change classification "
                "(buildup/unwinding), ATM strike analysis."
            ),
            max_concurrent=3,
            default_timeout=45.0,
            tags=["options", "oi", "derivatives", "pcr", "max_pain"],
        )
        logger.debug("Registered: oi_analysis")
    except Exception as e:
        logger.warning(f"Could not register OIAnalysisAgent: {e}")

    # ── Web Research ──────────────────────────────────────────────────────
    try:
        from .agents.web_research_agent import WebResearchAgent

        registry.register_class(
            WebResearchAgent,
            description=(
                "Autonomous multi-step web research with self-directed follow-up "
                "queries, source credibility scoring, deduplication, and automatic "
                "knowledge base learning (stores findings to Qdrant KB)."
            ),
            max_concurrent=4,
            default_timeout=180.0,
            tags=["research", "web", "news", "kb", "learning"],
        )
        logger.debug("Registered: web_research")
    except Exception as e:
        logger.warning(f"Could not register WebResearchAgent: {e}")

    # ── Global Market ─────────────────────────────────────────────────────
    try:
        from .agents.global_market_agent import GlobalMarketAgent

        registry.register_class(
            GlobalMarketAgent,
            description=(
                "Global financial markets monitor: US/EU/Asia indices, commodities "
                "(gold, crude, copper), forex (DXY, USD/INR), US bond yields, crypto. "
                "Computes India impact score and FII flow prediction."
            ),
            max_concurrent=3,
            default_timeout=120.0,
            tags=["global", "macro", "commodities", "forex", "bonds", "fii"],
        )
        logger.debug("Registered: global_market")
    except Exception as e:
        logger.warning(f"Could not register GlobalMarketAgent: {e}")

    # ── Shock Detection ───────────────────────────────────────────────────
    try:
        from .agents.shock_detection_agent import ShockDetectionAgent

        registry.register_class(
            ShockDetectionAgent,
            description=(
                "Market shock & crash detection: volatility spike (Z-score), "
                "drawdown analysis, circuit breaker probability, VIX fear gauge, "
                "flash crash detection, momentum exhaustion, gap analysis."
            ),
            max_concurrent=4,
            default_timeout=60.0,
            tags=["risk", "shock", "crash", "volatility", "drawdown"],
        )
        logger.debug("Registered: shock_detection")
    except Exception as e:
        logger.warning(f"Could not register ShockDetectionAgent: {e}")

    # ── Prediction ────────────────────────────────────────────────────────
    try:
        from .agents.prediction_agent import PredictionAgent

        registry.register_class(
            PredictionAgent,
            description=(
                "ML ensemble price & trend prediction: RandomForest trend classifier "
                "(1d/5d/20d horizon), LSTM price range predictor, market regime "
                "detection, confidence scoring with uncertainty quantification."
            ),
            max_concurrent=2,
            default_timeout=120.0,
            tags=["prediction", "ml", "forecast", "lstm", "trend"],
        )
        logger.debug("Registered: prediction")
    except Exception as e:
        logger.warning(f"Could not register PredictionAgent: {e}")

    # ── Fundamentals ──────────────────────────────────────────────────────
    try:
        from .agents.fundamentals_agent import FundamentalsAgent

        registry.register_class(
            FundamentalsAgent,
            description=(
                "Company fundamentals via yfinance: PE, PB, ROE, ROA, EPS, "
                "revenue, margins, debt/equity, market cap, 52-week range, "
                "institutional holdings, sector/industry classification."
            ),
            max_concurrent=5,
            default_timeout=45.0,
            tags=["fundamentals", "valuation", "financials", "pe", "roe"],
        )
        logger.debug("Registered: fundamentals")
    except Exception as e:
        logger.warning(f"Could not register FundamentalsAgent: {e}")

    # ── Sentiment ─────────────────────────────────────────────────────────
    try:
        from .agents.sentiment_agent import SentimentAgent

        registry.register_class(
            SentimentAgent,
            description=(
                "News & social sentiment analysis: searches latest news, extracts "
                "sentiment (positive/negative/neutral), scores headlines by relevance "
                "and recency, aggregates into an overall sentiment signal."
            ),
            max_concurrent=4,
            default_timeout=60.0,
            tags=["sentiment", "news", "social", "nlp"],
        )
        logger.debug("Registered: sentiment")
    except Exception as e:
        logger.warning(f"Could not register SentimentAgent: {e}")

    # ── Data Fetch ────────────────────────────────────────────────────────
    try:
        from .agents.data_fetch_agent import DataFetchAgent

        registry.register_class(
            DataFetchAgent,
            description=(
                "Fetches and stores OHLCV data for any NSE symbol into the data pool. "
                "Validates data quality, fills small gaps, and returns the clean DataFrame."
            ),
            max_concurrent=8,
            default_timeout=45.0,
            tags=["data", "ohlcv", "fetch", "storage"],
        )
        logger.debug("Registered: data_fetch")
    except Exception as e:
        logger.warning(f"Could not register DataFetchAgent: {e}")

    # ── Report (Asset Sub-Agent) ──────────────────────────────────────────
    try:
        from .agents.report_agent import ReportAgent

        registry.register_class(
            ReportAgent,
            description=(
                "Final report generator (asset sub-agent): synthesises all research "
                "agent findings into a structured markdown report with sections for "
                "executive summary, technical analysis, OI, global cues, prediction, "
                "and risk warnings. Always runs last in the orchestration plan."
            ),
            max_concurrent=3,
            default_timeout=120.0,
            tags=["report", "synthesis", "output", "asset"],
        )
        logger.debug("Registered: report")
    except Exception as e:
        logger.warning(f"Could not register ReportAgent: {e}")

    registered = list(get_registry()._types.keys())
    logger.info(
        f"✅ Swarm registry ready: {len(registered)} agent types — {registered}"
    )


# ---------------------------------------------------------------------------
# Register all agents at import time
# ---------------------------------------------------------------------------
_register_all_agents()


# ---------------------------------------------------------------------------
# Startup / shutdown hooks (called from app lifespan)
# ---------------------------------------------------------------------------


async def init_swarm() -> None:
    """
    Async startup hook — call from FastAPI lifespan after DB init.

    Starts:
      • AgentRegistry auto-reaper (disposes stale agents every 60s)
      • DataIngestionScheduler (autonomous market data collection)
    """
    from .agent_registry import get_registry

    registry = get_registry()
    await registry.start_reaper()
    logger.info("🕷️  Swarm registry reaper started")

    # Start the data ingestion scheduler
    try:
        from ..data_pipeline.ingestion_scheduler import get_scheduler

        scheduler = get_scheduler()
        await scheduler.start()
        logger.info("📅 DataIngestionScheduler started")
    except Exception as e:
        logger.warning(f"DataIngestionScheduler start failed (non-critical): {e}")


async def shutdown_swarm() -> None:
    """
    Async shutdown hook — call from FastAPI lifespan on shutdown.

    Stops:
      • AgentRegistry auto-reaper
      • DataIngestionScheduler
      • Disposes all live agents
    """
    from .agent_registry import get_registry

    registry = get_registry()

    # Dispose all running agents
    try:
        disposed = await registry.dispose_all()
        logger.info(f"Disposed {disposed} live agent(s) on shutdown")
    except Exception as e:
        logger.warning(f"Agent disposal error: {e}")

    await registry.stop_reaper()
    logger.info("Swarm registry reaper stopped")

    try:
        from ..data_pipeline.ingestion_scheduler import get_scheduler

        scheduler = get_scheduler()
        await scheduler.stop()
        logger.info("DataIngestionScheduler stopped")
    except Exception as e:
        logger.warning(f"Scheduler stop error: {e}")


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

from .agent_registry import AgentRegistry, get_registry
from .base_agent import (
    AgentResult,
    AgentStatus,
    AgentToolbox,
    BaseSwarmAgent,
    SwarmMessage,
)
from .orchestrator import MasterOrchestrator, OrchestratorResult, get_orchestrator

__all__ = [
    # Orchestrator
    "MasterOrchestrator",
    "OrchestratorResult",
    "get_orchestrator",
    # Registry
    "AgentRegistry",
    "get_registry",
    # Base agent primitives
    "BaseSwarmAgent",
    "SwarmMessage",
    "AgentResult",
    "AgentStatus",
    "AgentToolbox",
    # Lifecycle
    "init_swarm",
    "shutdown_swarm",
]
