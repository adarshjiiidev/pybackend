"""
Tool executor for financial analysis - NO YAHOO FINANCE
Uses NSE Scraper and Compound AI only.
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

from groq import AsyncGroq
from ..config import settings, ModelType
from ..database import MarketDataCacheManager  
from ..tools.nse_scraper import fetch_nse_quote, fetch_fii_dii, fetch_option_chain, fetch_market_status
from ..tools.technical_analysis import get_technical_indicators
from ..tools.nse_cache import get_nse_cache

logger = logging.getLogger(__name__)
cache = MarketDataCacheManager()


# Simple in-memory cache for web search results (prevents redundant searches)
_web_search_cache: Dict[str, tuple[str, datetime]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cache_key(query: str) -> str:
    """Generate cache key from query."""
    return query.lower().strip()


def _is_cache_valid(cached_time: datetime) -> bool:
    """Check if cached result is still valid."""
    return (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS


# Groq Compound AI web search
async def _search_web_groq(query: str) -> str:
    """
    Search web using Groq Compound AI with caching and truncation.
    - Caches results for 5 minutes to avoid redundant searches
    - Truncates results to prevent 413 Payload Too Large errors
    - Uses key rotation for better rate limit distribution
    """
    # Check cache first
    cache_key = _get_cache_key(query)
    if cache_key in _web_search_cache:
        cached_result, cached_time = _web_search_cache[cache_key]
        if _is_cache_valid(cached_time):
            logger.info(f"✅ Using cached web search result for: {query[:50]}...")
            return cached_result
    
    try:
        # Use key rotation for web search
        from ..config.key_rotator import get_groq_client
        client = get_groq_client()
        model = settings.get_model_for_task(ModelType.COMPOUND)
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Search and summarize concisely. Provide key facts with sources. Keep response under 800 words to avoid payload limits."
                },
                {
                    "role": "user",
                    "content": f"Search: {query}"
                }
            ],
            temperature=0.3,
            max_tokens=1024  # Limit to prevent 413 errors
        )
        
        result = response.choices[0].message.content
        
        # Truncate if still too long (safety check)
        if len(result) > 3000:
            result = result[:3000] + "... [truncated for size]"
            logger.warning(f"⚠️ Truncated web search result from {len(result)} to 3000 chars")
        
        # Cache the result
        _web_search_cache[cache_key] = (result, datetime.now())
        logger.info(f"🔍 Web search completed and cached for: {query[:50]}...")
        
        return result
    except Exception as e:
        logger.error(f"Groq web search error: {e}")
        return f"Search error: {str(e)}"


async def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a financial analysis tool.
    
    Available tools:
    - search_web: Web search via Compound AI
    - fetch_nse_quote: Get NSE stock quote
    - fetch_fii_dii: Get FII/DII data
    - fetch_option_chain: Get options data
    - fetch_market_status: Market status
    - get_technical_indicators: Technical analysis (DISABLED - requires historical data)
    """
    try:
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")
        
        if tool_name == "search_web":
            # Groq Compound AI web search
            query = arguments.get("query", "")
            result = await _search_web_groq(query)
            return {"result": result, "source": "groq_compound_ai"}
        
        elif tool_name == "fetch_nse_quote":
            symbol = arguments.get("symbol")
            if not symbol:
                return {"error": "Symbol parameter is required"}
            data = await fetch_nse_quote(symbol)
            return data
        
        elif tool_name == "fetch_fii_dii":
            data = await fetch_fii_dii()
            return data
        
        elif tool_name == "fetch_option_chain":
            symbol = arguments.get("symbol", "NIFTY")
            data = await fetch_option_chain(symbol)
            return data
        
        elif tool_name == "fetch_market_status":
            data = await fetch_market_status()
            return data
        
        elif tool_name == "get_technical_indicators":
            # DISABLED - requires historical data
            return {
                "error": "Technical indicators temporarily disabled",
                "message": "Historical data source required (Yahoo Finance removed)"
            }

        elif tool_name == "search_knowledge_base":
            from ..rag import search_knowledge_base as kb_search
            query = arguments.get("query", "")
            if not query:
                return {"error": "Query parameter is required for search_knowledge_base"}
            result = await kb_search(query)
            return result

        elif tool_name == "search_financial_news":
            # Delegate to search_web for now
            query = arguments.get("query", "")
            return await execute_tool("search_web", {"query": f"financial news {query}"})

        elif tool_name == "get_stock_fundamentals":
            symbol = arguments.get("symbol")
            if symbol:
                return await execute_tool("fetch_nse_quote", {"symbol": symbol})
            return {"error": "Symbol parameter is required"}

        elif tool_name == "get_market_sentiment":
            return await execute_tool("fetch_fii_dii", {})

        elif tool_name == "compare_stocks":
            symbols = arguments.get("symbols", [])
            if not symbols:
                return {"error": "Symbols array is required for compare_stocks"}
            # Parallel fetch — all quotes concurrently instead of sequential
            capped = symbols[:5]
            data_list = await asyncio.gather(
                *[fetch_nse_quote(sym) for sym in capped]
            )
            results = [
                {"symbol": sym, "data": data}
                for sym, data in zip(capped, data_list)
            ]
            return {"comparison": results}

        elif tool_name == "get_sector_analysis":
            return await execute_tool("search_web", {"query": arguments.get("sector", "") + " sector analysis India"})

        elif tool_name == "calculate_portfolio_optimization":
            return await execute_tool("search_web", {"query": f"portfolio optimization {arguments.get('stocks', [])} risk {arguments.get('risk_level', 'moderate')}"})
        
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}
            
    except Exception as e:
        logger.error(f"Tool execution error for {tool_name}: {e}")
        return {"error": str(e), "tool": tool_name}
