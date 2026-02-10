"""
Advanced Technical Analysis Module
Calculates RSI, Bollinger Bands, Fibonacci retracements, MACD, and other indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from .symbol_mapper import normalize_symbol

logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands (upper, middle, lower)."""
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band
    }


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_fibonacci_retracements(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    
    levels = {
        '0%': high,
        '23.6%': high - (0.236 * diff),
        '38.2%': high - (0.382 * diff),
        '50%': high - (0.5 * diff),
        '61.8%': high - (0.618 * diff),
        '78.6%': high - (0.786 * diff),
        '100%': low
    }
    
    return levels


def calculate_support_resistance(prices: pd.Series, window: int = 20) -> Dict[str, float]:
    """Calculate support and resistance levels."""
    recent_high = prices.tail(window).max()
    recent_low = prices.tail(window).min()
    
    # Find local peaks and troughs
    rolling_max = prices.rolling(window=window, center=True).max()
    rolling_min = prices.rolling(window=window, center=True).min()
    
    resistance_levels = prices[prices == rolling_max].unique()
    support_levels = prices[prices == rolling_min].unique()
    
    return {
        'resistance': float(recent_high) if not np.isnan(recent_high) else None,
        'support': float(recent_low) if not np.isnan(recent_low) else None,
        'resistance_levels': sorted(resistance_levels[-3:].tolist(), reverse=True) if len(resistance_levels) > 0 else [],
        'support_levels': sorted(support_levels[-3:].tolist()) if len(support_levels) > 0 else []
    }


def calculate_moving_averages(prices: pd.Series) -> Dict[str, float]:
    """Calculate various moving averages."""
    return {
        'sma_20': float(prices.tail(20).mean()) if len(prices) >= 20 else None,
        'sma_50': float(prices.tail(50).mean()) if len(prices) >= 50 else None,
        'sma_200': float(prices.tail(200).mean()) if len(prices) >= 200 else None,
        'ema_12': float(prices.ewm(span=12, adjust=False).mean().iloc[-1]) if len(prices) >= 12 else None,
        'ema_26': float(prices.ewm(span=26, adjust=False).mean().iloc[-1]) if len(prices) >= 26 else None,
    }


def calculate_volatility(prices: pd.Series, period: int = 20) -> Dict[str, float]:
    """Calculate volatility metrics."""
    returns = prices.pct_change().dropna()
    
    return {
        'std_dev': float(returns.std()),
        'variance': float(returns.var()),
        'daily_volatility': float(returns.tail(period).std()),
        'annualized_volatility': float(returns.std() * np.sqrt(252))  # 252 trading days
    }


async def get_technical_indicators(symbol: str, period: str = "6mo") -> Dict[str, Any]:
    """
    Get comprehensive technical indicators for a stock using Browser Search.
    
    Args:
        symbol: Stock symbol (NSE format, e.g., "RELIANCE", "TCS")
        period: Historical period (1mo, 3mo, 6mo, 1y, 2y, 5y)
    
    Returns:
        Dictionary with all technical indicators and analysis
    """
    try:
        from .browser_search import browser_search_historical_data
        
        # Get historical data using Browser Search
        hist_data = await browser_search_historical_data(symbol, period=period)
        
        if "error" in hist_data or not hist_data.get("historical"):
            return {"error": f"Failed to fetch historical data for {symbol}"}
        
        # Convert to DataFrame
        df = pd.DataFrame(hist_data.get("historical", []))
        
        if df.empty or 'close' not in df.columns:
            return {"error": f"Insufficient data for {symbol}"}
        
        # Ensure numeric close prices
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        
        if len(df) < 20:
            return {"error": f"Not enough data for {symbol}"}
        
        # Calculate indicators
        closes = df['close'].values
        rsi = calculate_rsi(closes)
        sma_20 = calculate_sma(closes, 20)
        sma_50 = calculate_sma(closes, 50) if len(closes) >= 50 else None
        ema_12 = calculate_ema(closes, 12)
        ema_26 = calculate_ema(closes, 26)
        macd_line, signal_line, macd_histogram = calculate_macd(closes)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
        
        return {
            "symbol": symbol,
            "rsi": rsi,
            "rsi_interpretation": interpret_rsi(rsi),
            "sma_20": sma_20,
            "sma_50": sma_50,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": macd_histogram,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "current_price": float(closes[-1]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Technical indicators error for {symbol}: {e}")
        return {"error": str(e), "symbol": symbol}



def interpret_rsi(rsi: Optional[float]) -> str:
    """Interpret RSI value."""
    if rsi is None:
        return "N/A"
    
    if rsi >= 70:
        return "Overbought (potential sell signal)"
    elif rsi <= 30:
        return "Oversold (potential buy signal)"
    elif rsi >= 60:
        return "Strong uptrend"
    elif rsi <= 40:
        return "Weak/bearish"
    else:
        return "Neutral"


def interpret_bollinger_bands(price: float, bb: Dict[str, Optional[float]]) -> str:
    """Interpret Bollinger Bands position."""
    if not all([bb.get('upper'), bb.get('middle'), bb.get('lower')]):
        return "N/A"
    
    upper = bb['upper']
    middle = bb['middle']
    lower = bb['lower']
    
    if price >= upper:
        return "Price at upper band - overbought, potential reversal"
    elif price <= lower:
        return "Price at lower band - oversold, potential bounce"
    elif price > middle:
        return "Price above middle band - bullish"
    elif price < middle:
        return "Price below middle band - bearish"
    else:
        return "Price at middle band - neutral"


def interpret_macd(macd: Dict[str, Optional[float]]) -> str:
    """Interpret MACD."""
    if not all([macd.get('macd'), macd.get('signal')]):
        return "N/A"
    
    macd_line = macd['macd']
    signal_line = macd['signal']
    
    if macd_line > signal_line and macd_line > 0:
        return "Bullish (MACD above signal and zero line)"
    elif macd_line > signal_line:
        return "Bullish crossover (MACD crossed above signal)"
    elif macd_line < signal_line and macd_line < 0:
        return "Bearish (MACD below signal and zero line)"
    elif macd_line < signal_line:
        return "Bearish crossover (MACD crossed below signal)"
    else:
        return "Neutral"


def generate_trading_signals(
    price: float,
    rsi: Optional[float],
    bb: Dict[str, Optional[float]],
    macd: Dict[str, Optional[float]],
    ma: Dict[str, Optional[float]]
) -> Dict[str, str]:
    """Generate overall trading signals based on indicators."""
    signals = {
        'trend': 'neutral',
        'momentum': 'neutral',
        'overall': 'neutral'
    }
    
    # Trend analysis (Moving Averages)
    if ma.get('sma_20') and ma.get('sma_50'):
        if price > ma['sma_20'] > ma['sma_50']:
            signals['trend'] = 'strong_uptrend'
        elif price > ma['sma_20']:
            signals['trend'] = 'uptrend'
        elif price < ma['sma_20'] < ma['sma_50']:
            signals['trend'] = 'strong_downtrend'
        elif price < ma['sma_20']:
            signals['trend'] = 'downtrend'
    
    # Momentum analysis (RSI + MACD)
    if rsi:
        if rsi > 70:
            signals['momentum'] = 'overbought'
        elif rsi < 30:
            signals['momentum'] = 'oversold'
        elif rsi > 60:
            signals['momentum'] = 'bullish'
        elif rsi < 40:
            signals['momentum'] = 'bearish'
    
    # Overall signal
    bullish_count = 0
    bearish_count = 0
    
    if rsi and rsi < 30:
        bullish_count += 1
    if rsi and rsi > 70:
        bearish_count += 1
    
    if bb.get('lower') and price <= bb['lower']:
        bullish_count += 1
    if bb.get('upper') and price >= bb['upper']:
        bearish_count += 1
    
    if macd.get('macd') and macd.get('signal'):
        if macd['macd'] > macd['signal']:
            bullish_count += 1
        else:
            bearish_count += 1
    
    if bullish_count > bearish_count:
        signals['overall'] = 'bullish'
    elif bearish_count > bullish_count:
        signals['overall'] = 'bearish'
    
    return signals
