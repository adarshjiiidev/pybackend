"""
DataIngestionScheduler — Autonomous background market data ingestion.

Runs as a background service that:
  • Fetches EOD (end-of-day) OHLCV data for all NIFTY50 + extra symbols daily
  • Fetches intraday 5-minute bars during market hours (9:15–15:30 IST)
  • Refreshes fundamentals weekly
  • Detects market hours automatically (IST timezone)
  • Uses APScheduler with asyncio bridge

Lifecycle::

    scheduler = get_scheduler()
    await scheduler.start()
    # ... app runs ...
    await scheduler.stop()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from datetime import time as dtime
from typing import List, Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IST = ZoneInfo("Asia/Kolkata")

# NSE market hours (IST)
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# Pre-open session starts at 9:00; post-close till 16:00 for MF/EOD
PRE_OPEN = dtime(9, 0)
POST_CLOSE = dtime(16, 0)

# NSE trading days: Monday (0) through Friday (4)
TRADING_DAYS = {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Market hours helpers
# ---------------------------------------------------------------------------


def is_market_open(now_ist: Optional[datetime] = None) -> bool:
    """Return True if NSE is currently in the regular trading session."""
    now = now_ist or datetime.now(IST)
    if now.weekday() not in TRADING_DAYS:
        return False
    t = now.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def is_trading_day(now_ist: Optional[datetime] = None) -> bool:
    """Return True if today is a weekday (not a weekend)."""
    now = now_ist or datetime.now(IST)
    return now.weekday() in TRADING_DAYS


def minutes_to_close(now_ist: Optional[datetime] = None) -> int:
    """Return minutes remaining until market close (negative if already closed)."""
    now = now_ist or datetime.now(IST)
    close_dt = now.replace(
        hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0
    )
    delta = close_dt - now
    return int(delta.total_seconds() / 60)


# ---------------------------------------------------------------------------
# Ingestion jobs
# ---------------------------------------------------------------------------


async def _ingest_eod_all() -> None:
    """
    End-of-day job: fetch daily OHLCV for all tracked symbols.
    Runs every trading day at 16:05 IST (after market close + settlement).
    """
    logger.info("📥 EOD ingestion job started")
    try:
        from .data_fetcher import ALL_TRACKED_SYMBOLS, MarketDataFetcher
        from .data_pool import get_data_pool

        fetcher = MarketDataFetcher()
        pool = get_data_pool()

        success_count = 0
        fail_count = 0

        # Fetch 5 trading days of fresh data to catch up on any missed days
        batch_results = await fetcher.fetch_bulk_ohlcv(
            symbols=ALL_TRACKED_SYMBOLS,
            period="5d",
            interval="1d",
            max_concurrent=6,
        )

        for symbol, df in batch_results.items():
            if df is not None and not df.empty:
                written = await pool.store_ohlcv(symbol, df, interval="1d")
                if written >= 0:
                    success_count += 1
                else:
                    fail_count += 1
            else:
                fail_count += 1

        logger.info(
            f"✅ EOD ingestion complete: {success_count} ok / {fail_count} failed "
            f"out of {len(ALL_TRACKED_SYMBOLS)} symbols"
        )
    except Exception as exc:
        logger.error(f"EOD ingestion job crashed: {exc}", exc_info=True)


async def _ingest_intraday_batch() -> None:
    """
    Intraday job: fetch 5-minute bars for NIFTY50 during market hours.
    Runs every 10 minutes between 9:15 and 15:30 IST on trading days.
    """
    now = datetime.now(IST)
    if not is_market_open(now):
        logger.debug("Intraday job skipped — market closed")
        return

    logger.info("📊 Intraday ingestion job started")
    try:
        from .data_fetcher import NIFTY50_UNIVERSE, MarketDataFetcher
        from .data_pool import get_data_pool

        fetcher = MarketDataFetcher()
        pool = get_data_pool()

        # Indices + top NIFTY50 for intraday (limit concurrent requests)
        intraday_symbols = ["NIFTY", "BANKNIFTY", "SENSEX"] + NIFTY50_UNIVERSE[:20]

        batch_results = await fetcher.fetch_bulk_ohlcv(
            symbols=intraday_symbols,
            period="2d",  # last 2 days gives us today's bars + yesterday
            interval="5m",
            max_concurrent=5,
        )

        written_total = 0
        for symbol, df in batch_results.items():
            if df is not None and not df.empty:
                written = await pool.store_ohlcv(symbol, df, interval="5m")
                written_total += written

        logger.info(
            f"✅ Intraday ingestion: {written_total} new bars for "
            f"{len(intraday_symbols)} symbols"
        )
    except Exception as exc:
        logger.error(f"Intraday ingestion job crashed: {exc}", exc_info=True)


async def _ingest_weekly_history() -> None:
    """
    Weekly job: ensure full 2-year history for all NIFTY50 symbols.
    Runs every Sunday at 04:00 IST to warm up the data pool.
    """
    logger.info("📚 Weekly history ingestion job started")
    try:
        from .data_fetcher import ALL_TRACKED_SYMBOLS, MarketDataFetcher
        from .data_pool import get_data_pool

        fetcher = MarketDataFetcher()
        pool = get_data_pool()

        # Check which symbols need a full refresh (missing or stale > 7 days)
        symbols_to_refresh: List[str] = []
        for sym in ALL_TRACKED_SYMBOLS:
            if not await pool.is_symbol_fresh(sym, max_age_hours=168):  # 7 days
                symbols_to_refresh.append(sym)

        if not symbols_to_refresh:
            logger.info("Weekly job: all symbols already fresh, skipping")
            return

        logger.info(f"Weekly job: refreshing {len(symbols_to_refresh)} stale symbols")

        batch_results = await fetcher.fetch_bulk_ohlcv(
            symbols=symbols_to_refresh,
            period="2y",
            interval="1d",
            max_concurrent=4,  # lower concurrency for big pulls
        )

        written_total = 0
        for symbol, df in batch_results.items():
            if df is not None and not df.empty:
                written = await pool.store_ohlcv(symbol, df, interval="1d")
                written_total += written

        logger.info(
            f"✅ Weekly history ingestion: {written_total} bars stored for "
            f"{len(symbols_to_refresh)} symbols"
        )
    except Exception as exc:
        logger.error(f"Weekly history job crashed: {exc}", exc_info=True)


async def _refresh_fundamentals() -> None:
    """
    Fundamentals refresh job: fetch company data for all NIFTY50 stocks.
    Runs every Tuesday and Friday at 06:00 IST.
    """
    logger.info("🏦 Fundamentals refresh job started")
    try:
        from .data_fetcher import NIFTY50_UNIVERSE, MarketDataFetcher
        from .data_pool import get_data_pool

        fetcher = MarketDataFetcher()
        pool = get_data_pool()

        success = 0
        fail = 0

        # Fetch fundamentals sequentially (yfinance info is slow per call)
        for symbol in NIFTY50_UNIVERSE:
            try:
                # Skip if fundamentals are still fresh (< 48h old)
                cached = await pool.get_fundamentals(symbol)
                if cached:
                    continue

                data = await fetcher.fetch_fundamentals(symbol)
                if "error" not in data:
                    await pool.store_fundamentals(symbol, data)
                    success += 1
                else:
                    fail += 1

                await asyncio.sleep(0.5)  # gentle pacing
            except Exception as sym_exc:
                logger.debug(f"Fundamentals error for {symbol}: {sym_exc}")
                fail += 1

        logger.info(f"✅ Fundamentals refresh: {success} ok / {fail} failed")
    except Exception as exc:
        logger.error(f"Fundamentals refresh job crashed: {exc}", exc_info=True)


async def _seed_universe() -> None:
    """
    One-time seed job: ensure the symbol universe collection is populated.
    Runs once at startup.
    """
    try:
        from .data_pool import get_data_pool

        pool = get_data_pool()
        await pool.seed_universe()
        logger.info("✅ Symbol universe seeded")
    except Exception as exc:
        logger.error(f"Universe seed failed: {exc}", exc_info=True)


async def _initial_data_warmup() -> None:
    """
    Startup warmup: fetch 1-year history for all symbols if pool is empty.
    Runs once 30 seconds after scheduler start.
    """
    logger.info("🔥 Initial data warmup started (checking pool emptiness)...")
    try:
        from .data_pool import get_data_pool

        pool = get_data_pool()
        coverage = await pool.get_data_coverage()

        if coverage.get("total_bars", 0) < 100:
            logger.info(
                "Pool appears empty — triggering 1-year history fetch for all symbols"
            )
            await _ingest_weekly_history()
        else:
            logger.info(
                f"Pool already has {coverage['total_bars']} bars — skipping warmup"
            )
    except Exception as exc:
        logger.error(f"Initial warmup failed: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Scheduler class
# ---------------------------------------------------------------------------


class DataIngestionScheduler:
    """
    APScheduler-based autonomous data ingestion scheduler.

    Jobs registered:
      eod_ingest          — daily at 16:05 IST (Mon–Fri)
      intraday_ingest     — every 10 min, 9:10–15:35 IST (Mon–Fri)
      weekly_history      — every Sunday at 04:00 IST
      fundamentals        — Tue & Fri at 06:00 IST
    """

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start all background ingestion jobs."""
        if self._running:
            logger.warning("Scheduler already running — skipping start()")
            return

        logger.info("🚀 Starting DataIngestionScheduler...")

        self._scheduler = AsyncIOScheduler(timezone=IST)

        # ── EOD daily ingestion (Mon–Fri at 16:05 IST) ────────────────────
        self._scheduler.add_job(
            _ingest_eod_all,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=16,
                minute=5,
                timezone=IST,
            ),
            id="eod_ingest",
            name="EOD Daily Ingestion",
            replace_existing=True,
            misfire_grace_time=3600,  # run up to 1h late if missed
        )

        # ── Intraday 5-min bars (Mon–Fri, every 10 min, 9:10–15:35 IST) ──
        self._scheduler.add_job(
            _ingest_intraday_batch,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="10,20,30,40,50",
                timezone=IST,
            ),
            id="intraday_ingest",
            name="Intraday 5-Min Ingestion",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # ── Weekly full-history pull (every Sunday at 04:00 IST) ──────────
        self._scheduler.add_job(
            _ingest_weekly_history,
            trigger=CronTrigger(
                day_of_week="sun",
                hour=4,
                minute=0,
                timezone=IST,
            ),
            id="weekly_history",
            name="Weekly History Ingestion",
            replace_existing=True,
            misfire_grace_time=7200,
        )

        # ── Fundamentals refresh (Tue & Fri at 06:00 IST) ─────────────────
        self._scheduler.add_job(
            _refresh_fundamentals,
            trigger=CronTrigger(
                day_of_week="tue,fri",
                hour=6,
                minute=0,
                timezone=IST,
            ),
            id="fundamentals_refresh",
            name="Fundamentals Refresh",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        self._scheduler.start()
        self._running = True
        logger.info("✅ DataIngestionScheduler started with 4 jobs")

        # Run seed + warmup as fire-and-forget background tasks
        asyncio.create_task(self._delayed_startup())

    async def _delayed_startup(self) -> None:
        """Seed universe + warmup data pool shortly after startup."""
        await asyncio.sleep(5)  # let the app fully initialize first
        await _seed_universe()
        await asyncio.sleep(10)
        await _initial_data_warmup()

    async def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if not self._running or self._scheduler is None:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception as exc:
            logger.warning(f"Scheduler shutdown warning: {exc}")
        finally:
            self._running = False
            logger.info("DataIngestionScheduler stopped")

    def get_jobs_info(self) -> List[dict]:
        """Return info about all registered jobs (for health/status endpoints)."""
        if not self._scheduler:
            return []
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": (
                        job.next_run_time.isoformat() if job.next_run_time else None
                    ),
                    "trigger": str(job.trigger),
                }
            )
        return jobs

    async def trigger_eod_now(self) -> None:
        """Manually trigger an EOD ingestion immediately (for admin use)."""
        logger.info("Manual EOD ingestion triggered")
        await _ingest_eod_all()

    async def trigger_intraday_now(self) -> None:
        """Manually trigger an intraday batch immediately."""
        logger.info("Manual intraday ingestion triggered")
        await _ingest_intraday_batch()

    async def trigger_warmup_now(self) -> None:
        """Force a full history warmup (re-fetches all symbols)."""
        logger.info("Manual warmup triggered")
        await _ingest_weekly_history()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_scheduler: Optional[DataIngestionScheduler] = None


def get_scheduler() -> DataIngestionScheduler:
    """Return the global DataIngestionScheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = DataIngestionScheduler()
    return _scheduler
