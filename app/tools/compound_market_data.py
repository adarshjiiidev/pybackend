"""
Compound AI Market Data Fetcher
Uses Groq's Compound AI (web search) and Browser Search for unlimited market data.
- Primary: Compound AI web search (fast, text snippets)
- Fallback: Browser Search (comprehensive, interactive navigation)
No rate limits, no API keys needed for data sources.
Automatic retry with exponential backoff for resilience.
"""

import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from groq import AsyncGroq
from ..config import settings
from ..utils.retry import async_retry
from ..utils.fallback import FallbackChain

logger = logging.getLogger(__name__)


@async_retry(
    max_attempts=3,
    backoff_factor=2.0,
    initial_delay=1.0,
    exceptions=(Exception,)
)
async def get_stock_price_compound(symbol: str, search_web_func) -> Dict[str, Any]:
    """
    Get current stock price using Compound AI web search.
    Auto-retries up to 3 times on failure with exponential backoff.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS")
        search_web_func: Web search function from tool executor
    
    Returns:
        Dict with price information
    """
    try:
        # Format query for Indian stock market
        query = f"current stock price {symbol} NSE BSE India live LTP"
        
        logger.info(f"🔍 Searching stock price for {symbol} using Compound AI")
        result = await search_web_func(query)
        
        # Parse the result
        price_data = _parse_price_from_search(result, symbol)
        
        if price_data:
            logger.info(f"✅ Found price for {symbol}: {price_data.get('ltp', 'N/A')}")
            return price_data
        else:
            logger.warning(f"⚠️ Could not parse price for {symbol}")
            return {
                "symbol": symbol,
                "error": "Price not found in search results",
                "raw_result": result[:200]  # First 200 chars for debugging
            }
    
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e)
        }


@async_retry(max_attempts=3, backoff_factor=2.0)
async def get_company_info_compound(symbol: str, search_web_func) -> Dict[str, Any]:
    """
    Get company fundamentals using Compound AI web search.
    Auto-retries on failure.
    
    Args:
        symbol: Stock symbol
        search_web_func: Web search function
    
    Returns:
        Dict with company information
    """
    try:
        query = f"{symbol} stock fundamental analysis PE ratio market cap revenue profit India"
        
        logger.info(f"🔍 Searching company info for {symbol} using Compound AI")
        result = await search_web_func(query)
        
        # Parse fundamentals from search result
        info = _parse_company_info(result, symbol)
        
        if info:
            logger.info(f"✅ Found company info for {symbol}")
            return info
        else:
            return {
                "symbol": symbol,
                "error": "Company info not found",
                "raw_result": result[:200]
            }
    
    except Exception as e:
        logger.error(f"Error fetching company info for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e)
        }


@async_retry(max_attempts=3, backoff_factor=2.0)
async def get_market_indices_compound(search_web_func) -> Dict[str, Any]:
    """
    Get current market indices (Nifty, Sensex, Bank Nifty) using Compound AI.
    Auto-retries on failure.
    
    Args:
        search_web_func: Web search function
    
    Returns:
        Dict with index values
    """
    try:
        query = "Nifty 50 Sensex Bank Nifty current live value today India"
        
        logger.info("🔍 Searching market indices using Compound AI")
        result = await search_web_func(query)
        
        indices = _parse_indices(result)
        
        if indices:
            logger.info(f"✅ Found market indices")
            return indices
        else:
            return {
                "error": "Could not parse indices",
                "raw_result": result[:200]
            }
    
    except Exception as e:
        logger.error(f"Error fetching market indices: {e}")
        return {"error": str(e)}


@async_retry(max_attempts=2, backoff_factor=1.5)
async def get_stock_news_compound(symbol: str, search_web_func) -> Dict[str, Any]:
    """
    Get latest news for a stock using Compound AI.
    Auto-retries on failure (fewer attempts for news).
    
    Args:
        symbol: Stock symbol
        search_web_func: Web search function
    
    Returns:
        Dict with news information
    """
    try:
        today = datetime.now().strftime("%B %Y")
        query = f"{symbol} stock latest news today {today} India"
        
        logger.info(f"🔍 Searching news for {symbol} using Compound AI")
        result = await search_web_func(query)
        
        return {
            "symbol": symbol,
            "news": result,
            "fetched_at": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e)
        }


# Helper functions for parsing search results

def _parse_price_from_search(text: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Parse price information from web search results."""
    try:
        # Look for common price patterns
        # Pattern 1: "₹2,450.75" or "Rs. 2450.75"
        price_patterns = [
            r'₹\s*([0-9,]+\.?[0-9]*)',
            r'Rs\.?\s*([0-9,]+\.?[0-9]*)',
            r'INR\s*([0-9,]+\.?[0-9]*)',
            r'price[:\s]+([0-9,]+\.?[0-9]*)',
            r'LTP[:\s]+([0-9,]+\.?[0-9]*)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    price = float(price_str)
                    return {
                        "symbol": symbol,
                        "ltp": price,
                        "currency": "INR",
                        "fetched_at": datetime.utcnow().isoformat(),
                        "source": "compound_ai_web_search"
                    }
                except ValueError:
                    continue
        
        # If no pattern matched, return None
        return None
    
    except Exception as e:
        logger.error(f"Error parsing price: {e}")
        return None


def _parse_company_info(text: str, symbol: str) -> Optional[Dict[str, Any]]:
    """Parse company fundamental information from web search results."""
    try:
        info = {
            "symbol": symbol,
            "fetched_at": datetime.utcnow().isoformat(),
            "source": "compound_ai_web_search"
        }
        
        # Extract PE ratio
        pe_pattern = r'P/E|PE|price.to.earnings[:\s]+([0-9]+\.?[0-9]*)'
        pe_match = re.search(pe_pattern, text, re.IGNORECASE)
        if pe_match:
            try:
                info["pe_ratio"] = float(pe_match.group(1))
            except:
                pass
        
        # Extract market cap
        mcap_patterns = [
            r'market cap[:\s]+₹?\s*([0-9,]+\.?[0-9]*)\s*(cr|crore|billion|lakh)',
            r'mcap[:\s]+₹?\s*([0-9,]+\.?[0-9]*)\s*(cr|crore|billion|lakh)',
        ]
        for pattern in mcap_patterns:
            mcap_match = re.search(pattern, text, re.IGNORECASE)
            if mcap_match:
                try:
                    value = float(mcap_match.group(1).replace(',', ''))
                    unit = mcap_match.group(2).lower()
                    info["market_cap"] = f"{value} {unit}"
                    break
                except:
                    pass
        
        # Store raw snippet for context
        info["summary"] = text[:500]  # First 500 chars
        
        return info if len(info) > 3 else None  # At least symbol + 2 data points
    
    except Exception as e:
        logger.error(f"Error parsing company info: {e}")
        return None


def _parse_indices(text: str) -> Optional[Dict[str, Any]]:
    """Parse market indices from web search results."""
    try:
        indices = {
            "fetched_at": datetime.utcnow().isoformat(),
            "source": "compound_ai_web_search"
        }
        
        # Nifty 50
        nifty_pattern = r'Nifty\s*50?[:\s]+([0-9,]+\.?[0-9]*)'
        nifty_match = re.search(nifty_pattern, text, re.IGNORECASE)
        if nifty_match:
            try:
                indices["nifty_50"] = float(nifty_match.group(1).replace(',', ''))
            except:
                pass
        
        # Sensex
        sensex_pattern = r'Sensex[:\s]+([0-9,]+\.?[0-9]*)'
        sensex_match = re.search(sensex_pattern, text, re.IGNORECASE)
        if sensex_match:
            try:
                indices["sensex"] = float(sensex_match.group(1).replace(',', ''))
            except:
                pass
        
        # Bank Nifty
        banknifty_pattern = r'Bank\s*Nifty[:\s]+([0-9,]+\.?[0-9]*)'
        banknifty_match = re.search(banknifty_pattern, text, re.IGNORECASE)
        if banknifty_match:
            try:
                indices["bank_nifty"] = float(banknifty_match.group(1).replace(',', ''))
            except:
                pass
        
        return indices if len(indices) > 2 else None  # At least 1 index value
    
    except Exception as e:
        logger.error(f"Error parsing indices: {e}")
        return None
