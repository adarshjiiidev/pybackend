"""
Browser Search using Groq - Powered by Exa
Interactive web browsing for comprehensive financial data retrieval.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

from groq import AsyncGroq
from ..config import settings
from ..utils.retry import async_retry

logger = logging.getLogger(__name__)


@async_retry(max_attempts=3, initial_delay=1.0)
async def browser_search_historical_data(
    symbol: str,
    period: str = "6mo"
) -> Dict[str, Any]:
    """
    Fetch historical stock data using Groq Browser Search.
    
    Browser search navigates financial websites interactively to gather
    comprehensive historical price data.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS")
        period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)
    
    Returns:
        Dictionary with historical price data
    """
    try:
        client = AsyncGroq(api_key=settings.groq_api_key)
        
        # Use Browser Search supported model
        model = "openai/gpt-oss-20b"
        
        query = f"""
Find historical stock price data for {symbol} for the last {period}.
I need:
- Daily closing prices
- High and low prices for each day
- Trading volume
- Date for each entry

Search financial websites like NSE India, MoneyControl, or Yahoo Finance.
Return the data in a structured format with dates and prices.
"""
        
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ],
            model=model,
            temperature=0.3,
            max_completion_tokens=2048,
            tool_choice="required",
            tools=[
                {
                    "type": "browser_search"
                }
            ]
        )
        
        content = response.choices[0].message.content
        
        # Parse the response to extract structured data
        historical_data = _parse_historical_response(content, symbol)
        
        return {
            "symbol": symbol,
            "period": period,
            "historical": historical_data,
            "source": "browser_search",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Browser search failed for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "source": "browser_search"
        }


@async_retry(max_attempts=3, initial_delay=1.0)
async def browser_search_company_info(symbol: str) -> Dict[str, Any]:
    """
    Fetch comprehensive company information using Browser Search.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dictionary with company information
    """
    try:
        client = AsyncGroq(api_key=settings.groq_api_key)
        model = "openai/gpt-oss-20b"
        
        query = f"""
Find comprehensive information about {symbol} stock:
- Current price
- Market cap
- P/E ratio
- 52-week high and low
- Company description
- Industry and sector
- Recent news or developments

Search NSE India, MoneyControl, or similar reliable financial sources.
"""
        
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ],
            model=model,
            temperature=0.3,
            max_completion_tokens=1024,
            tool_choice="required",
            tools=[
                {
                    "type": "browser_search"
                }
            ]
        )
        
        content = response.choices[0].message.content
        
        return {
            "symbol": symbol,
            "info": content,
            "source": "browser_search",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Browser search company info failed for {symbol}: {e}")
        return {
            "error": str(e),
            "symbol": symbol
        }


def _parse_historical_response(content: str, symbol: str) -> List[Dict[str, Any]]:
    """
    Parse browser search response to extract historical data.
    
    This attempts to extract structured price data from the response.
    The response format may vary, so we use heuristics to parse it.
    """
    historical = []
    
    # Try to find date-price patterns in the response
    # Common formats: "2024-01-15: 2850.50" or "Jan 15, 2024 - Rs 2850.50"
    
    # Pattern 1: YYYY-MM-DD format
    pattern1 = r'(\d{4}-\d{2}-\d{2}).*?(\d+\.?\d*)'
    matches1 = re.findall(pattern1, content)
    
    # Pattern 2: Month Day, Year format
    pattern2 = r'([A-Z][a-z]+ \d{1,2}, \d{4}).*?(\d+\.?\d*)'
    matches2 = re.findall(pattern2, content)
    
    if matches1:
        for date_str, price_str in matches1[:100]:  # Limit to 100 entries
            try:
                historical.append({
                    "date": date_str,
                    "close": float(price_str),
                    "symbol": symbol
                })
            except ValueError:
                continue
    elif matches2:
        for date_str, price_str in matches2[:100]:
            try:
                historical.append({
                    "date": date_str,
                    "close": float(price_str),
                    "symbol": symbol
                })
            except ValueError:
                continue
    
    # If no structured data found, return empty with note
    if not historical:
        logger.warning(f"Could not parse historical data for {symbol} from browser search")
    
    return historical


@async_retry(max_attempts=2, initial_delay=1.0)
async def browser_search_general(query: str) -> str:
    """
    General purpose browser search for any financial query.
    
    Args:
        query: Search query
    
    Returns:
        Search results as text
    """
    try:
        client = AsyncGroq(api_key=settings.groq_api_key)
        model = "openai/gpt-oss-20b"
        
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ],
            model=model,
            temperature=0.3,
            max_completion_tokens=1500,
            tool_choice="required",
            tools=[
                {
                    "type": "browser_search"
                }
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"General browser search failed: {e}")
        return f"Search error: {str(e)}"
