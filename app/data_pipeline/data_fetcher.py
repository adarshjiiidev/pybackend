"""
MarketDataFetcher — Multi-source historical and live market data fetcher.

Sources (in priority order):
  1. yfinance  — historical OHLCV, fundamentals (free, reliable)
  2. NSE fast scraper — live quotes, option chain, FII/DII
  3. Google Finance scraper — live index prices (fast fallback)

NSE symbol convention:
  Equity  → symbol + ".NS"   (e.g. "RELIANCE.NS")
  Indices → special mapping  (e.g. "^NSEI" for Nifty 50)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------

# Map NSE trading symbols → yfinance tickers
NSE_TO_YF: Dict[str, str] = {
    # Indices
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCAP": "^NSEMDCP50",
    # Stocks — added automatically via _nse_equity_ticker()
}

# yfinance period / interval validated combos
VALID_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}

# NIFTY 50 universe (canonical NSE trading symbols)
NIFTY50_UNIVERSE: List[str] = [
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

# Extra Indian market symbols worth tracking
EXTRA_SYMBOLS: List[str] = [
    "NIFTY",
    "BANKNIFTY",
    "SENSEX",  # Indices
    "ADANIENT",
    "ADANIGREEN",
    "ADANIPOWER",  # Adani group
    "ZOMATO",
    "NYKAA",
    "PAYTM",  # New-age tech
    "HAL",
    "BEL",
    "BEML",  # Defence PSUs
    "IRFC",
    "IRCTC",
    "IREDA",  # Railway PSUs
    "DMART",
    "ABFRL",  # Retail
    "PNB",
    "BANKBARODA",
    "CANBK",  # PSU banks
    "SAIL",
    "NMDC",  # Metal PSUs
]

ALL_TRACKED_SYMBOLS: List[str] = list(dict.fromkeys(NIFTY50_UNIVERSE + EXTRA_SYMBOLS))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nse_equity_ticker(symbol: str) -> str:
    """Convert NSE equity symbol to yfinance ticker (append '.NS')."""
    symbol = symbol.upper().strip()
    if symbol in NSE_TO_YF:
        return NSE_TO_YF[symbol]
    # Skip index-like symbols
    if symbol.startswith("^"):
        return symbol
    return f"{symbol}.NS"


def _parse_period_to_days(period: str) -> int:
    """Parse yfinance-style period string to number of days."""
    mapping = {
        "1d": 1,
        "5d": 5,
        "7d": 7,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "ytd": 365,
        "max": 3650,
    }
    return mapping.get(period.lower(), 365)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class MarketDataFetcher:
    """
    Fetches OHLCV + fundamentals from multiple sources.

    Usage::

        fetcher = MarketDataFetcher()
        df = await fetcher.fetch_ohlcv("RELIANCE", period="1y")
        quote = await fetcher.fetch_live_quote("RELIANCE")
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.5):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: NSE symbol (e.g. "RELIANCE") or index ("NIFTY")
            period: lookback period — "1mo", "3mo", "6mo", "1y", "2y", "5y"
            interval: bar interval — "1d", "1wk", "1mo", "1h", "15m", etc.

        Returns:
            DataFrame with columns: open, high, low, close, volume, symbol
            Index: DatetimeIndex (UTC-normalised)
        """
        ticker = _nse_equity_ticker(symbol)
        logger.info(
            f"Fetching OHLCV: {symbol} ({ticker}) | period={period} interval={interval}"
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                df = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._yf_download(ticker, period, interval),
                )
                if df is not None and not df.empty:
                    df = self._normalise_columns(df, symbol)
                    logger.info(f"✅ Fetched {len(df)} rows for {symbol}")
                    return df
            except Exception as exc:
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for {symbol}: {exc}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)

        logger.error(f"All {self.max_retries} attempts failed for {symbol}")
        return pd.DataFrame()

    async def fetch_ohlcv_range(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data between two explicit dates."""
        ticker = _nse_equity_ticker(symbol)
        end = end or datetime.utcnow()
        logger.info(f"Fetching range: {symbol} from {start.date()} to {end.date()}")

        for attempt in range(1, self.max_retries + 1):
            try:
                df = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._yf_download_range(ticker, start, end, interval),
                )
                if df is not None and not df.empty:
                    return self._normalise_columns(df, symbol)
            except Exception as exc:
                logger.warning(
                    f"Range fetch attempt {attempt} failed for {symbol}: {exc}"
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)

        return pd.DataFrame()

    async def fetch_live_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch live quote — tries NSE scraper first, then Google Finance.
        Returns dict with: ltp, change, changePercent, open, high, low,
                           prevClose, volume, timestamp, source
        """
        try:
            from ..tools.nse_scraper import fetch_google_price, fetch_nse_quote

            # Indices via Google (faster for indices)
            index_symbols = {"NIFTY", "NIFTY50", "BANKNIFTY", "SENSEX", "FINNIFTY"}
            if symbol.upper() in index_symbols:
                result = await fetch_google_price(symbol)
                if "error" not in result:
                    return result

            # NSE scraper for equities
            result = await fetch_nse_quote(symbol)
            if "error" not in result:
                return result

            # Fallback: Google Finance
            return await fetch_google_price(symbol)
        except Exception as exc:
            logger.error(f"Live quote error for {symbol}: {exc}")
            return {"error": str(exc), "symbol": symbol}

    async def fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data (PE, EPS, market cap, etc.) via yfinance.
        """
        ticker = _nse_equity_ticker(symbol)
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._yf_info(ticker),
            )
            return self._parse_fundamentals(info, symbol)
        except Exception as exc:
            logger.error(f"Fundamentals fetch error for {symbol}: {exc}")
            return {"error": str(exc), "symbol": symbol}

    async def fetch_bulk_ohlcv(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_concurrent: int = 8,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple symbols concurrently with rate limiting.

        Returns:
            Dict[symbol → DataFrame]
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results: Dict[str, pd.DataFrame] = {}

        async def _fetch_one(sym: str) -> None:
            async with semaphore:
                df = await self.fetch_ohlcv(sym, period=period, interval=interval)
                results[sym] = df
                # Small delay to avoid hammering yfinance
                await asyncio.sleep(0.2)

        await asyncio.gather(*[_fetch_one(s) for s in symbols], return_exceptions=True)
        return results

    def get_nifty50_universe(self) -> List[str]:
        """Return NIFTY50 constituent symbols."""
        return NIFTY50_UNIVERSE.copy()

    def get_all_tracked_symbols(self) -> List[str]:
        """Return all symbols tracked by the system."""
        return ALL_TRACKED_SYMBOLS.copy()

    # ------------------------------------------------------------------
    # yfinance internals (sync — run in executor)
    # ------------------------------------------------------------------

    def _yf_download(
        self, ticker: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Synchronous yfinance download wrapper."""
        try:
            t = yf.Ticker(ticker)
            df = t.history(
                period=period, interval=interval, auto_adjust=True, actions=False
            )
            return df if not df.empty else None
        except Exception as exc:
            logger.debug(f"yf.Ticker.history failed: {exc}")
            return None

    def _yf_download_range(
        self, ticker: str, start: datetime, end: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        """Synchronous yfinance download with explicit date range."""
        try:
            t = yf.Ticker(ticker)
            df = t.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                actions=False,
            )
            return df if not df.empty else None
        except Exception as exc:
            logger.debug(f"yf range download failed: {exc}")
            return None

    def _yf_info(self, ticker: str) -> Dict[str, Any]:
        """Synchronous yfinance info fetch."""
        return yf.Ticker(ticker).info or {}

    # ------------------------------------------------------------------
    # Data normalization
    # ------------------------------------------------------------------

    def _normalise_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalise yfinance DataFrame column names to lowercase snake_case
        and ensure required columns exist.
        """
        df = df.copy()

        # yfinance returns MultiIndex columns for bulk download; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(c).lower() for c in col).strip("_") for col in df.columns
            ]
        else:
            df.columns = [str(c).lower() for c in df.columns]

        # Rename standard yfinance columns
        rename_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "adj close": "adj_close",
            "dividends": "dividends",
            "stock splits": "stock_splits",
        }
        df.rename(columns=rename_map, inplace=True)

        # Ensure required columns
        for col in ("open", "high", "low", "close"):
            if col not in df.columns:
                logger.warning(f"Missing column '{col}' for {symbol}")
                df[col] = float("nan")

        if "volume" not in df.columns:
            df["volume"] = 0

        # Add symbol column
        df["symbol"] = symbol.upper()

        # Normalise index to UTC DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)

        # Drop rows where all OHLC are NaN
        df.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)

        # Sort chronologically
        df.sort_index(inplace=True)

        return df

    def _parse_fundamentals(self, info: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Parse yfinance info dict into clean fundamentals."""

        def _safe(key: str, default=None):
            val = info.get(key, default)
            return val if val not in (None, "N/A", "None", float("inf")) else default

        return {
            "symbol": symbol,
            "company_name": _safe("longName") or _safe("shortName") or symbol,
            "sector": _safe("sector"),
            "industry": _safe("industry"),
            "market_cap": _safe("marketCap"),
            "market_cap_cr": round(_safe("marketCap", 0) / 1e7, 2),  # INR crores
            "current_price": _safe("currentPrice") or _safe("regularMarketPrice"),
            "pe_ratio": _safe("trailingPE"),
            "forward_pe": _safe("forwardPE"),
            "pb_ratio": _safe("priceToBook"),
            "ps_ratio": _safe("priceToSalesTrailing12Months"),
            "peg_ratio": _safe("pegRatio"),
            "eps": _safe("trailingEps"),
            "forward_eps": _safe("forwardEps"),
            "book_value": _safe("bookValue"),
            "dividend_yield": _safe("dividendYield"),
            "roe": _safe("returnOnEquity"),
            "roa": _safe("returnOnAssets"),
            "revenue": _safe("totalRevenue"),
            "revenue_cr": round((_safe("totalRevenue") or 0) / 1e7, 2),
            "net_income": _safe("netIncomeToCommon"),
            "debt_to_equity": _safe("debtToEquity"),
            "current_ratio": _safe("currentRatio"),
            "quick_ratio": _safe("quickRatio"),
            "gross_margins": _safe("grossMargins"),
            "operating_margins": _safe("operatingMargins"),
            "profit_margins": _safe("profitMargins"),
            "52w_high": _safe("fiftyTwoWeekHigh"),
            "52w_low": _safe("fiftyTwoWeekLow"),
            "50d_avg": _safe("fiftyDayAverage"),
            "200d_avg": _safe("twoHundredDayAverage"),
            "beta": _safe("beta"),
            "avg_volume": _safe("averageVolume"),
            "avg_volume_10d": _safe("averageVolume10days"),
            "shares_outstanding": _safe("sharesOutstanding"),
            "float_shares": _safe("floatShares"),
            "held_by_institutions": _safe("institutionsPercentHeld"),
            "held_by_insiders": _safe("heldPercentInsiders"),
            "short_ratio": _safe("shortRatio"),
            "exchange": _safe("exchange"),
            "currency": _safe("currency", "INR"),
            "source": "yfinance",
            "timestamp": datetime.utcnow().isoformat(),
        }
