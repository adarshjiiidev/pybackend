"""
Tool executor for financial analysis - NO YAHOO FINANCE
Uses NSE Scraper and Compound AI only.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from groq import AsyncGroq

from ..config import ModelType, settings
from ..database import MarketDataCacheManager
from ..tools.nse_cache import get_nse_cache
from ..tools.nse_scraper import (
    fetch_fii_dii,
    fetch_google_price,
    fetch_market_status,
    fetch_nse_quote,
    fetch_option_chain,
)
from ..tools.technical_analysis import get_technical_indicators

logger = logging.getLogger(__name__)
cache = MarketDataCacheManager()


# Simple in-memory cache for web search results (prevents redundant searches)
_web_search_cache: Dict[str, tuple[str, datetime]] = {}
CACHE_TTL_SECONDS = 30  # 30-second TTL — fresh enough for market data, saves API quota


def _get_cache_key(query: str) -> str:
    """Generate cache key from query."""
    return query.lower().strip()


def _is_cache_valid(cached_time: datetime) -> bool:
    """Check if cached result is still valid."""
    return (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS
# ── Parallel Web Search: OpenRouter first → Groq Compound fallback ────────────

async def _search_web_openrouter(query: str) -> str:
    """Search web using OpenRouter model (primary — free, no RPD limits)."""
    try:
        from ..config.openrouter_client import get_openrouter_client
        client = get_openrouter_client()
        
        truncated_query = query[:300] if len(query) > 300 else query
        
        response = await client.chat.completions.create(
            model=settings.or_model_analysis,  # step-3.5-flash — strong at search synthesis
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial research assistant focused on Indian markets (NSE/BSE). "
                        "Search the web and provide factual, data-rich answers. "
                        "Include specific numbers, dates, and sources. Keep under 600 words."
                    ),
                },
                {"role": "user", "content": truncated_query},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        
        result = response.choices[0].message.content
        if result and len(result) > 3000:
            result = result[:3000] + "... [truncated]"
            logger.warning("⚠️ Truncated OpenRouter web search result to 3000 chars")
        
        if result:
            logger.info(f"🔍 OpenRouter web search completed for: {query[:50]}...")
            return result
        return ""
    except Exception as e:
        logger.warning(f"OpenRouter web search failed: {e}")
        return ""


async def _search_web_groq_compound(query: str) -> str:
    """Search web using Groq Compound AI (fallback — has built-in web search)."""
    from ..config.key_rotator import get_rotator as _get_rotator

    try:
        _rotator = _get_rotator()
        all_keys = list(_rotator.api_keys)
    except RuntimeError:
        from ..config.settings import settings as _s
        all_keys = [_s.groq_api_key]

    truncated_query = query[:150] if len(query) > 150 else query  # keep query short
    models_to_try = ["groq/compound-mini", "groq/compound"]  # mini first (smaller payload)

    for model in models_to_try:
        for key_idx, api_key in enumerate(all_keys):
            try:
                client = AsyncGroq(api_key=api_key)
                create_kwargs = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Search web. Short factual answer.",
                        },
                        {"role": "user", "content": truncated_query},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 400,  # keep response small to avoid 413
                    # NOTE: search_settings removed — caused 413 by triggering more fetches
                }

                response = await client.chat.completions.create(**create_kwargs)
                result = response.choices[0].message.content

                if result and len(result) > 3000:
                    result = result[:3000] + "... [truncated]"
                    logger.warning("⚠️ Truncated Groq web search result to 3000 chars")

                if result:
                    logger.info(
                        f"🔍 Groq web search completed for: {query[:50]}... "
                        f"(model={model}, key={key_idx + 1}/{len(all_keys)})"
                    )
                    return result

            except Exception as e:
                err_str = str(e)
                if "413" in err_str or "request_too_large" in err_str:
                    logger.warning(f"📦 Payload too large for {model}, skipping...")
                    break
                elif "429" in err_str or "rate_limit" in err_str:
                    await asyncio.sleep(1)
                    continue
                elif "401" in err_str:
                    continue
                else:
                    logger.error(f"Groq web search error on {model}: {e}")
                    break

    return ""


async def _search_web_groq(query: str) -> str:
    """
    Fast dual-track web search — MCP first, Groq compound as fallback.
    Phase 1: DDG + Google scrape via MCP (1-5s, no API keys, best quality)
    Phase 2: Groq Compound fallback (15-20s, only if MCP fails)
    OpenRouter LLM search is SKIPPED — it adds 10s latency for no gain over MCP.
    """
    # Check cache first
    cache_key = _get_cache_key(query)
    if cache_key in _web_search_cache:
        cached_result, cached_time = _web_search_cache[cache_key]
        if _is_cache_valid(cached_time):
            logger.info(f"✅ Cache hit for: {query[:50]}...")
            return cached_result

    # Phase 1: MCP scrape (fast, free, no API limits)
    try:
        from .web_search_mcp import fast_scrape_search
        scrape_result = await asyncio.wait_for(fast_scrape_search(query), timeout=10.0)  # extra time for Google parse
        if scrape_result and len(scrape_result) > 30:  # lowered threshold from 50
            logger.info(f"🔍 MCP scrape succeeded for: {query[:50]}...")
            _web_search_cache[cache_key] = (scrape_result, datetime.now())
            return scrape_result
        else:
            logger.debug(f"MCP scrape returned thin results ({len(scrape_result) if scrape_result else 0} chars) for: {query[:50]}")
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug(f"MCP scrape failed: {e}")

    # Phase 2: OpenRouter LLM with built-in web search (reliable, never 413s)
    # Groq compound is NOT used here — it consistently 413s due to internal fetch payload issues
    try:
        result = await asyncio.wait_for(_search_web_openrouter(query), timeout=25.0)
        if result:
            _web_search_cache[cache_key] = (result, datetime.now())
            return result
    except asyncio.TimeoutError:
        logger.warning("⏳ OpenRouter web search timed out after 25s")
    except Exception as e:
        logger.warning(f"OpenRouter web search error: {e}")

    logger.error("All web search providers failed")
    return "Search temporarily unavailable. Please try again."


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

    Security: symbol arguments are sanitized via sanitize_symbol() before use.
    Errors are logged internally; generic messages are returned to callers.
    """
    try:
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        # ── Sanitize symbol arguments before they reach any downstream call ──
        # Prevents path traversal / injection if a compromised model generates
        # a malicious tool argument (e.g. symbol="../../etc/passwd").
        if "symbol" in arguments and arguments["symbol"]:
            try:
                from ..utils.sanitizer import sanitize_symbol

                arguments["symbol"] = sanitize_symbol(str(arguments["symbol"]))
            except Exception as san_err:
                logger.warning(
                    f"Symbol sanitization rejected '{arguments.get('symbol')}': {san_err}"
                )
                return {"error": "Invalid symbol format", "tool": tool_name}

        if "symbols" in arguments and isinstance(arguments["symbols"], list):
            try:
                from ..utils.sanitizer import sanitize_symbol

                arguments["symbols"] = [
                    sanitize_symbol(str(s)) for s in arguments["symbols"] if s
                ]
            except Exception as san_err:
                logger.warning(f"Symbols sanitization rejected list: {san_err}")
                return {"error": "Invalid symbol in list", "tool": tool_name}

        if tool_name == "search_web":
            query = arguments.get("query", "").strip()
            # MCP search: DDG + Google scrape (fast) → OpenRouter → Groq Compound
            result = await _search_web_groq(query)
            return {"result": result, "source": "mcp_search"}

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
            symbol = arguments.get("symbol", "NIFTY")
            try:
                result = await get_technical_indicators(symbol)
                return result
            except Exception as ti_err:
                logger.warning(f"Technical indicators failed for {symbol}: {ti_err}, falling back to web search")
                return await execute_tool(
                    "search_web",
                    {"query": f"{symbol} technical analysis RSI MACD moving average India NSE"},
                )

        elif tool_name == "search_knowledge_base":
            from ..rag import search_knowledge_base as kb_search

            query = arguments.get("query", "")
            if not query:
                return {
                    "error": "Query parameter is required for search_knowledge_base"
                }
            result = await kb_search(query)
            return result

        elif tool_name == "search_financial_news":
            query = arguments.get("query", "")
            # Route through news-specific sites for better relevance
            news_query = (
                f"{query} latest news site:moneycontrol.com OR site:economictimes.indiatimes.com "
                f"OR site:livemint.com OR site:business-standard.com"
            )
            return await execute_tool("search_web", {"query": news_query})

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
            data_list = await asyncio.gather(*[fetch_nse_quote(sym) for sym in capped])
            results = [
                {"symbol": sym, "data": data} for sym, data in zip(capped, data_list)
            ]
            return {"comparison": results}

        elif tool_name == "get_sector_analysis":
            return await execute_tool(
                "search_web",
                {"query": arguments.get("sector", "") + " sector analysis India"},
            )

        elif tool_name == "calculate_portfolio_optimization":
            return await execute_tool(
                "search_web",
                {
                    "query": f"portfolio optimization {arguments.get('stocks', [])} risk {arguments.get('risk_level', 'moderate')}"
                },
            )

        elif tool_name == "get_index_history":
            symbols = arguments.get("symbols", [])
            days = int(arguments.get("days", 90))
            if not symbols:
                return {"error": "symbols array is required"}

            # Map friendly names → yfinance tickers
            TICKER_MAP = {
                "NIFTY50": "^NSEI",
                "NIFTY": "^NSEI",
                "SENSEX": "^BSESN",
                "BSE": "^BSESN",
                "BANKNIFTY": "^NSEBANK",
                "NIFTYIT": "^CNXIT",
                "NIFTYBANK": "^NSEBANK",
            }

            try:
                import yfinance as yf
                from datetime import timedelta, date as _date
                end_dt = _date.today()
                start_dt = end_dt - timedelta(days=days)

                result = {}
                for sym in symbols[:3]:  # cap at 3 to avoid heavy payloads
                    ticker = TICKER_MAP.get(sym.upper(), f"^{sym.upper()}")
                    try:
                        df = yf.download(
                            ticker,
                            start=start_dt.strftime("%Y-%m-%d"),
                            end=end_dt.strftime("%Y-%m-%d"),
                            interval="1d",
                            progress=False,
                            auto_adjust=True,
                        )
                        if df is None or df.empty:
                            result[sym] = {"error": f"No data for {sym}"}
                            continue

                        # Flatten MultiIndex columns if present
                        if hasattr(df.columns, "levels"):
                            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                        df.columns = [str(c).lower() for c in df.columns]

                        bars = []
                        for idx, row in df.iterrows():
                            time_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                            try:
                                bar = {
                                    "time": time_str,
                                    "open": round(float(row.get("open", row["close"])), 2),
                                    "high": round(float(row.get("high", row["close"])), 2),
                                    "low": round(float(row.get("low", row["close"])), 2),
                                    "close": round(float(row["close"]), 2),
                                    "value": round(float(row["close"]), 2),
                                }
                                bars.append(bar)
                            except Exception:
                                continue

                        result[sym] = {
                            "bars": bars,
                            "count": len(bars),
                            "latest_close": bars[-1]["close"] if bars else None,
                            "ticker": ticker,
                        }
                        logger.info(f"✅ get_index_history: {sym} ({ticker}) — {len(bars)} bars")
                    except Exception as sym_err:
                        logger.warning(f"Index history failed for {sym}: {sym_err}")
                        result[sym] = {"error": str(sym_err)}

                return result

            except ImportError:
                return {"error": "yfinance not installed. Run: pip install yfinance"}
            except Exception as e:
                logger.error(f"get_index_history error: {e}")
                return {"error": str(e)}

        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        # Log full details internally; return a safe generic message to the caller
        logger.error(f"Tool execution error for {tool_name}: {e}", exc_info=True)
        return {"error": "Tool execution failed. Please try again.", "tool": tool_name}
