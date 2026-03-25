"""
DataPool — MongoDB-backed historical market data pool.

Stores OHLCV bars, fundamentals, and symbol metadata so agents can
query rich history without hitting external APIs on every request.

Collections:
  ohlcv_bars        — one doc per (symbol, date, interval)
  symbol_universe   — tracked symbols + metadata
  fundamentals_cache — company fundamentals (TTL refreshed daily)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from beanie import Document, Indexed
from pydantic import Field

logger = logging.getLogger(__name__)


def _is_db_ready() -> bool:
    """Return True only if MongoDB is connected and Beanie has been initialized.
    
    Database.db is set and init_beanie() is called atomically inside connect(),
    so checking db is not None is sufficient.
    """
    try:
        from ..config.database import Database
        return Database.db is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Beanie document models (registered in app/config/database.py)
# ---------------------------------------------------------------------------


class OHLCVBar(Document):
    """Single OHLCV bar stored in MongoDB."""

    symbol: Indexed(str) = Field(..., description="NSE trading symbol (uppercase)")
    interval: str = Field(default="1d", description="Bar interval: 1d, 1h, 15m …")
    timestamp: Indexed(datetime) = Field(..., description="Bar open timestamp (UTC)")
    open: float = Field(..., description="Open price (INR)")
    high: float = Field(..., description="High price (INR)")
    low: float = Field(..., description="Low price (INR)")
    close: float = Field(..., description="Close price (INR)")
    volume: float = Field(default=0.0, description="Traded volume")
    source: str = Field(default="yfinance", description="Data source identifier")
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "ohlcv_bars"
        indexes = [
            [("symbol", 1), ("interval", 1), ("timestamp", -1)],
            [("symbol", 1), ("timestamp", -1)],
        ]


class SymbolMeta(Document):
    """Symbol metadata — universe membership, last update, etc."""

    symbol: Indexed(str, unique=True) = Field(..., description="NSE symbol")
    company_name: Optional[str] = Field(default=None)
    sector: Optional[str] = Field(default=None)
    industry: Optional[str] = Field(default=None)
    market_cap_cr: Optional[float] = Field(
        default=None, description="Market cap in ₹ crores"
    )
    is_nifty50: bool = Field(default=False)
    is_tracked: bool = Field(default=True)
    first_ingested: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    data_start_date: Optional[datetime] = Field(
        default=None, description="Earliest bar date in DB"
    )
    data_end_date: Optional[datetime] = Field(
        default=None, description="Latest bar date in DB"
    )
    bar_count: int = Field(default=0, description="Total bars stored")

    class Settings:
        name = "symbol_universe"
        indexes = ["symbol", "is_nifty50", "is_tracked"]


class FundamentalsCache(Document):
    """Fundamentals data cached per symbol (refreshed daily)."""

    symbol: Indexed(str, unique=True) = Field(..., description="NSE symbol")
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Raw fundamentals dict"
    )
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(hours=24)
    )

    class Settings:
        name = "fundamentals_cache"
        indexes = [
            "symbol",
            # TTL index on expires_at is created in database.py
        ]

    def is_fresh(self) -> bool:
        return datetime.utcnow() < self.expires_at


# ---------------------------------------------------------------------------
# DataPool
# ---------------------------------------------------------------------------

_NIFTY50 = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "ITC",
    "SBIN",
    "BHARTIARTL",
    "KOTAKBANK",
    "LT",
    "HCLTECH",
    "AXISBANK",
    "ASIANPAINT",
    "MARUTI",
    "BAJFINANCE",
    "TITAN",
    "SUNPHARMA",
    "TATAMOTORS",
    "WIPRO",
    "NESTLEIND",
    "ULTRACEMCO",
    "POWERGRID",
    "NTPC",
    "TATASTEEL",
    "JSWSTEEL",
    "ONGC",
    "COALINDIA",
    "TECHM",
    "HINDALCO",
    "DIVISLAB",
    "APOLLOHOSP",
    "CIPLA",
    "EICHERMOT",
    "BAJAJFINSV",
    "BRITANNIA",
    "DRREDDY",
    "GRASIM",
    "BPCL",
    "HEROMOTOCO",
    "M&M",
    "ADANIPORTS",
    "INDUSINDBK",
    "BAJAJ-AUTO",
    "LTIM",
    "SBILIFE",
    "HDFCLIFE",
    "UPL",
    "TATACONSUM",
    "VEDL",
]

_INDICES = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"]


class DataPool:
    """
    Central data access layer for historical market data.

    All read / write operations go through this class so callers never
    interact with MongoDB directly.

    Usage::

        pool = get_data_pool()
        await pool.store_ohlcv("RELIANCE", df)
        df = await pool.get_ohlcv("RELIANCE", days=365)
        price = await pool.get_latest_price("RELIANCE")
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def store_ohlcv(
        self,
        symbol: str,
        df: pd.DataFrame,
        interval: str = "1d",
        source: str = "yfinance",
        upsert: bool = True,
    ) -> int:
        """
        Persist a OHLCV DataFrame to MongoDB.

        Args:
            symbol: NSE symbol (case-insensitive, stored uppercase)
            df: DataFrame with columns open/high/low/close/volume + DatetimeIndex
            interval: Bar interval string
            source: Source identifier
            upsert: If True, skip existing bars (idempotent re-ingestion)

        Returns:
            Number of bars newly written
        """
        sym = symbol.upper()
        if not _is_db_ready():
            logger.debug(f"store_ohlcv: DB not ready — skipping persist for {sym}")
            return 0
        if df.empty:
            logger.warning(f"store_ohlcv: empty DataFrame for {sym}, skipping")
            return 0

        # Validate required columns
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            logger.error(f"store_ohlcv: missing columns {missing} for {sym}")
            return 0

        written = 0

        for ts, row in df.iterrows():
            try:
                bar_ts = pd.Timestamp(ts).to_pydatetime()
                if bar_ts.tzinfo is not None:
                    bar_ts = bar_ts.replace(tzinfo=None)  # store as UTC naive

                if upsert:
                    existing = await OHLCVBar.find_one(
                        OHLCVBar.symbol == sym,
                        OHLCVBar.interval == interval,
                        OHLCVBar.timestamp == bar_ts,
                    )
                    if existing:
                        continue

                bar = OHLCVBar(
                    symbol=sym,
                    interval=interval,
                    timestamp=bar_ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                    source=source,
                )
                await bar.insert()
                written += 1
            except Exception as exc:
                logger.debug(f"Bar insert error ({sym} {ts}): {exc}")

        # Update symbol metadata
        await self._update_symbol_meta(sym, df)
        logger.info(
            f"store_ohlcv: {sym} → {written} new bars written (interval={interval})"
        )
        return written

    async def store_fundamentals(self, symbol: str, data: Dict[str, Any]) -> None:
        """Upsert fundamentals for a symbol."""
        sym = symbol.upper()
        if not _is_db_ready():
            logger.debug(f"store_fundamentals: DB not ready — skipping for {sym}")
            return
        try:
            existing = await FundamentalsCache.find_one(FundamentalsCache.symbol == sym)
            if existing:
                existing.data = data
                existing.fetched_at = datetime.utcnow()
                existing.expires_at = datetime.utcnow() + timedelta(hours=24)
                await existing.save()
            else:
                doc = FundamentalsCache(symbol=sym, data=data)
                await doc.insert()
        except Exception as exc:
            logger.error(f"store_fundamentals error for {sym}: {exc}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_ohlcv(
        self,
        symbol: str,
        days: int = 365,
        interval: str = "1d",
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data from the pool.

        Args:
            symbol: NSE symbol
            days: Lookback window in calendar days
            interval: Bar interval
            end_date: Latest date to include (default: now)

        Returns:
            DataFrame sorted ascending with columns: open, high, low, close, volume
        """
        sym = symbol.upper()
        if not _is_db_ready():
            logger.debug(f"get_ohlcv: DB not ready — returning empty for {sym}")
            return pd.DataFrame()
        end_dt = end_date or datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)

        try:
            bars = (
                await OHLCVBar.find(
                    OHLCVBar.symbol == sym,
                    OHLCVBar.interval == interval,
                    OHLCVBar.timestamp >= start_dt,
                    OHLCVBar.timestamp <= end_dt,
                )
                .sort(+OHLCVBar.timestamp)
                .to_list()
            )

            if not bars:
                logger.debug(f"get_ohlcv: no data found for {sym} (last {days}d)")
                return pd.DataFrame()

            records = [
                {
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ]
            df = pd.DataFrame(records).set_index("timestamp")
            df.index = pd.DatetimeIndex(df.index)
            df["symbol"] = sym
            return df
        except Exception as exc:
            logger.error(f"get_ohlcv error for {sym}: {exc}")
            return pd.DataFrame()

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return the most recent close price stored in the pool."""
        sym = symbol.upper()
        try:
            bar = (
                await OHLCVBar.find(
                    OHLCVBar.symbol == sym,
                    OHLCVBar.interval == "1d",
                )
                .sort(-OHLCVBar.timestamp)
                .first_or_none()
            )
            return bar.close if bar else None
        except Exception as exc:
            logger.error(f"get_latest_price error for {sym}: {exc}")
            return None

    async def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return cached fundamentals (None if expired or missing)."""
        sym = symbol.upper()
        if not _is_db_ready():
            return None
        try:
            doc = await FundamentalsCache.find_one(FundamentalsCache.symbol == sym)
            if doc and doc.is_fresh():
                return doc.data
            return None
        except Exception as exc:
            logger.error(f"get_fundamentals error for {sym}: {exc}")
            return None

    async def get_symbols_universe(
        self,
        nifty50_only: bool = False,
        tracked_only: bool = True,
    ) -> List[str]:
        """Return list of tracked symbols from the universe collection."""
        try:
            query = []
            if nifty50_only:
                query.append(SymbolMeta.is_nifty50 == True)
            if tracked_only:
                query.append(SymbolMeta.is_tracked == True)

            metas = await SymbolMeta.find(*query).to_list()
            if metas:
                return [m.symbol for m in metas]
            # Fallback: return hardcoded universe if DB is empty
            return _NIFTY50 if nifty50_only else _NIFTY50 + _INDICES
        except Exception as exc:
            logger.error(f"get_symbols_universe error: {exc}")
            return _NIFTY50

    async def get_symbol_meta(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return metadata dict for a symbol."""
        sym = symbol.upper()
        try:
            meta = await SymbolMeta.find_one(SymbolMeta.symbol == sym)
            if meta:
                return meta.model_dump()
            return None
        except Exception as exc:
            logger.error(f"get_symbol_meta error for {sym}: {exc}")
            return None

    async def get_data_coverage(self) -> Dict[str, Any]:
        """
        Return data coverage summary: how many symbols, bars, date range.
        Useful for health checks and debugging.
        """
        if not _is_db_ready():
            return {"status": "unavailable", "total_bars": 0, "total_symbols": 0}
        try:
            total_symbols = await SymbolMeta.count()
            total_bars = await OHLCVBar.count()
            earliest = await OHLCVBar.find().sort(+OHLCVBar.timestamp).first_or_none()
            latest = await OHLCVBar.find().sort(-OHLCVBar.timestamp).first_or_none()

            return {
                "total_symbols": total_symbols,
                "total_bars": total_bars,
                "earliest_date": earliest.timestamp.isoformat() if earliest else None,
                "latest_date": latest.timestamp.isoformat() if latest else None,
                "status": "healthy" if total_bars > 0 else "empty",
            }
        except Exception as exc:
            logger.error(f"get_data_coverage error: {exc}")
            return {"status": "error", "error": str(exc)}

    async def is_symbol_fresh(self, symbol: str, max_age_hours: int = 25) -> bool:
        """
        Return True if symbol has data ingested within max_age_hours.
        Used by the scheduler to decide whether to re-fetch.
        """
        sym = symbol.upper()
        try:
            meta = await SymbolMeta.find_one(SymbolMeta.symbol == sym)
            if not meta or not meta.last_updated:
                return False
            cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
            return meta.last_updated > cutoff
        except Exception:
            return False

    async def delete_symbol_data(self, symbol: str) -> int:
        """Delete all OHLCV bars for a symbol (for data reset)."""
        sym = symbol.upper()
        try:
            result = await OHLCVBar.find(OHLCVBar.symbol == sym).delete()
            await SymbolMeta.find(SymbolMeta.symbol == sym).delete()
            count = result.deleted_count if hasattr(result, "deleted_count") else 0
            logger.info(f"Deleted {count} bars for {sym}")
            return count
        except Exception as exc:
            logger.error(f"delete_symbol_data error for {sym}: {exc}")
            return 0

    # ------------------------------------------------------------------
    # Seed universe
    # ------------------------------------------------------------------

    async def seed_universe(self) -> None:
        """
        Ensure all NIFTY50 + index symbols exist in symbol_universe collection.
        Safe to call multiple times (idempotent).
        """
        if not _is_db_ready():
            logger.warning("seed_universe: DB not ready — skipping universe seed")
            return
        for sym in _NIFTY50:
            await self._ensure_symbol_meta(sym, is_nifty50=True)
        for sym in _INDICES:
            await self._ensure_symbol_meta(sym, is_nifty50=False)
        logger.info(
            f"Universe seeded: {len(_NIFTY50)} equity + {len(_INDICES)} index symbols"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_symbol_meta(
        self, symbol: str, is_nifty50: bool = False
    ) -> Optional[SymbolMeta]:
        """Get or create a SymbolMeta document. Returns None if DB unavailable."""
        sym = symbol.upper()
        if not _is_db_ready():
            return None
        existing = await SymbolMeta.find_one(SymbolMeta.symbol == sym)
        if existing:
            return existing
        meta = SymbolMeta(symbol=sym, is_nifty50=is_nifty50)
        await meta.insert()
        return meta

    async def _update_symbol_meta(self, symbol: str, df: pd.DataFrame) -> None:
        """Update SymbolMeta after a successful ingestion."""
        sym = symbol.upper()
        try:
            meta = await SymbolMeta.find_one(SymbolMeta.symbol == sym)
            if not meta:
                meta = SymbolMeta(symbol=sym, is_nifty50=(sym in _NIFTY50))

            # Update date range
            if not df.empty:
                df_start = df.index.min()
                df_end = df.index.max()

                if isinstance(df_start, pd.Timestamp):
                    df_start = df_start.to_pydatetime().replace(tzinfo=None)
                if isinstance(df_end, pd.Timestamp):
                    df_end = df_end.to_pydatetime().replace(tzinfo=None)

                if not meta.data_start_date or df_start < meta.data_start_date:
                    meta.data_start_date = df_start
                if not meta.data_end_date or df_end > meta.data_end_date:
                    meta.data_end_date = df_end

            meta.last_updated = datetime.utcnow()
            meta.bar_count = await OHLCVBar.find(OHLCVBar.symbol == sym).count()
            await meta.save()
        except Exception as exc:
            logger.debug(f"_update_symbol_meta failed for {sym}: {exc}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pool: Optional[DataPool] = None


def get_data_pool() -> DataPool:
    """Return the global DataPool singleton."""
    global _pool
    if _pool is None:
        _pool = DataPool()
    return _pool
