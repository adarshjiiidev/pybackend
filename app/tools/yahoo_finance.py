"""
Yahoo Finance API integration for market data.
Provides async tools for fetching stock, crypto, and index data.
"""

import yfinance as yf
from typing import Optional, Any
import asyncio
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def async_yfinance(func):
    """Decorator to run synchronous yfinance calls in executor."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    return wrapper


@async_yfinance
def _fetch_stock_info(symbol: str) -> dict[str, Any]:
    """Fetch stock information (sync)."""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return {
        "symbol": symbol,
        "name": info.get("longName", symbol),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "marketCap": info.get("marketCap"),
        "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previousClose": info.get("previousClose"),
        "open": info.get("open"),
        "dayHigh": info.get("dayHigh"),
        "dayLow": info.get("dayLow"),
        "volume": info.get("volume"),
        "avgVolume": info.get("averageVolume"),
        "peRatio": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "dividendYield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "description": info.get("longBusinessSummary", "")[:500] if info.get("longBusinessSummary") else ""
    }


@async_yfinance
def _fetch_historical_data(symbol: str, period: str = "1mo") -> dict[str, Any]:
    """Fetch historical OHLCV data (sync)."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    
    if hist.empty:
        return {"error": "No data available"}
    
    return {
        "symbol": symbol,
        "period": period,
        "data": hist.reset_index().to_dict(orient="records")
    }


@async_yfinance
def _fetch_crypto_data(symbol: str) -> dict[str, Any]:
    """Fetch cryptocurrency data (sync)."""
    # Ensure symbol has proper suffix for crypto
    if not symbol.endswith("-USD"):
        symbol = f"{symbol}-USD"
    
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="1d")
    
    return {
        "symbol": symbol,
        "name": info.get("name", symbol),
        "currentPrice": info.get("regularMarketPrice"),
        "previousClose": info.get("previousClose"),
        "dayChange": info.get("regularMarketChangePercent"),
        "volume": info.get("volume24Hr") or info.get("volume"),
        "marketCap": info.get("marketCap"),
        "circulatingSupply": info.get("circulatingSupply")
    }


@async_yfinance
def _fetch_market_indices() -> dict[str, Any]:
    """Fetch Indian market indices (sync)."""
    indices = {
        "NIFTY50": "^NSEI",
        "SENSEX": "^BSESN",
        "NIFTYBANK": "^NSEBANK"
    }
    
    results = {}
    for name, symbol in indices.items():
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            results[name] = {
                "symbol": symbol,
                "price": info.get("regularMarketPrice"),
                "previousClose": info.get("previousClose"),
                "change": info.get("regularMarketChange"),
                "changePercent": info.get("regularMarketChangePercent")
            }
        except Exception as e:
            logger.error(f"Failed to fetch {name}: {e}")
            results[name] = {"error": str(e)}
    
    return results


async def get_stock_info(symbol: str) -> dict[str, Any]:
    """
    Get comprehensive stock information.
    
    Args:
        symbol: Stock symbol (use .NS for NSE, .BO for BSE)
        
    Returns:
        Dictionary with stock info
    """
    try:
        # Ensure Indian stocks have proper suffix
        if not any(symbol.endswith(suffix) for suffix in [".NS", ".BO"]):
            # Default to NSE
            symbol = f"{symbol}.NS"
        
        return await _fetch_stock_info(symbol)
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}


async def get_historical_data(
    symbol: str,
    period: str = "1mo"
) -> dict[str, Any]:
    """
    Get historical OHLCV data.
    
    Args:
        symbol: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        
    Returns:
        Dictionary with historical data
    """
    try:
        if not any(symbol.endswith(suffix) for suffix in [".NS", ".BO", "-USD"]):
            symbol = f"{symbol}.NS"
        
        return await _fetch_historical_data(symbol, period)
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}


async def get_crypto_data(symbol: str) -> dict[str, Any]:
    """
    Get cryptocurrency data.
    
    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)
        
    Returns:
        Dictionary with crypto data
    """
    try:
        return await _fetch_crypto_data(symbol)
    except Exception as e:
        logger.error(f"Error fetching crypto data for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}


async def get_market_indices() -> dict[str, Any]:
    """
    Get Indian market indices (Nifty 50, Sensex, Bank Nifty).
    
    Returns:
        Dictionary with index data
    """
    try:
        return await _fetch_market_indices()
    except Exception as e:
        logger.error(f"Error fetching market indices: {e}")
        return {"error": str(e)}
