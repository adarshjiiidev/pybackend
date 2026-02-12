"""
Tool executor for financial analysis - NO YAHOO FINANCE
Uses NSE Scraper and Compound AI only.
"""

import logging
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


# Groq Compound AI web search
async def _search_web_groq(query: str) -> str:
    """Search web using Groq Compound AI."""
    try:
        client = AsyncGroq(api_key=settings.groq_api_key)
        model = settings.get_model_for_task(ModelType.COMPOUND)
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a web search assistant. Search the web and provide accurate, current information with sources. Be concise but comprehensive."
                },
                {
                    "role": "user",
                    "content": f"Search the web and provide information about: {query}"
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
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
            results = []
            for sym in symbols[:5]:  # Max 5
                data = await fetch_nse_quote(sym)
                results.append({"symbol": sym, "data": data})
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
