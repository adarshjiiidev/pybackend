"""
Tool implementations for advanced financial analysis.
Implements all tools defined in tool_definitions.py
"""

import asyncio
from typing import Any, Optional
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..tools.yahoo_finance import get_stock_info, get_historical_data, get_crypto_data, get_market_indices
from ..database import MarketDataCacheManager
from ..tools.nse_scraper import fetch_nse_quote, fetch_fii_dii, fetch_option_chain, fetch_market_status
from ..tools.technical_analysis import get_technical_indicators
from ..tools.nse_cache import get_nse_cache

logger = logging.getLogger(__name__)
cache = MarketDataCacheManager()


class FinancialTools:
    """Implementations of all financial analysis tools."""
    
    @staticmethod
    async def get_stock_fundamentals(symbol: str, include_financials: bool = True) -> dict[str, Any]:
        """Get comprehensive fundamental data."""
        try:
            # Ensure NSE suffix
            if not symbol.endswith(('.NS', '.BO')):
                symbol = f"{symbol}.NS"
            
            data = await get_stock_info(symbol)
            
            if include_financials and not data.get("error"):
                # Add financial statements (simplified for MVP)
                ticker = yf.Ticker(symbol)
                try:
                    financials = ticker.financials
                    data["has_financials"] = not financials.empty if financials is not None else False
                except:
                    data["has_financials"] = False
            
            return data
        except Exception as e:
            logger.error(f"Error in get_stock_fundamentals: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def get_technical_indicators(
        symbol: str,
        indicators: Optional[list[str]] = None,
        period: str = "3mo"
    ) -> dict[str, Any]:
        """Calculate technical indicators."""
        try:
            if not symbol.endswith(('.NS', '.BO', '-USD')):
                symbol = f"{symbol}.NS"
            
            hist_data = await get_historical_data(symbol, period=period)
            
            if hist_data.get("error"):
                return hist_data
            
            df = pd.DataFrame(hist_data["data"])
            if df.empty:
                return {"error": "No data available"}
            
            results = {"symbol": symbol, "indicators": {}}
            
            # Calculate requested indicators
            if not indicators:
                indicators = ["RSI", "MACD", "SMA"]
            
            if "RSI" in indicators:
                # Simple RSI calculation
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                results["indicators"]["RSI"] = float(rsi.iloc[-1]) if not rsi.empty else None
            
            if "SMA" in indicators:
                sma_20 = df['Close'].rolling(window=20).mean()
                sma_50 = df['Close'].rolling(window=50).mean()
                results["indicators"]["SMA_20"] = float(sma_20.iloc[-1]) if not sma_20.empty else None
                results["indicators"]["SMA_50"] = float(sma_50.iloc[-1]) if not sma_50.empty else None
            
            if "EMA" in indicators:
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                results["indicators"]["EMA_12"] = float(ema_12.iloc[-1]) if not ema_12.empty else None
                results["indicators"]["EMA_26"] = float(ema_26.iloc[-1]) if not ema_26.empty else None
            
            return results
        except Exception as e:
            logger.error(f"Error in get_technical_indicators: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def search_financial_news(query: str, limit: int = 5, days: int = 7) -> dict[str, Any]:
        """Search for financial news (simulated for MVP)."""
        # For MVP, return placeholder. In production, integrate NewsAPI or similar
        return {
            "query": query,
            "articles": [
                {
                    "title": f"Latest updates on {query}",
                    "source": "Financial Express",
                    "published_at": datetime.utcnow().isoformat(),
                    "summary": f"Recent developments regarding {query} in Indian markets."
                }
            ],
            "note": "News integration requires API key. This is simulated data for MVP."
        }
    
    @staticmethod
    async def get_market_sentiment(market: str = "INDIA") -> dict[str, Any]:
        """Analyze overall market sentiment."""
        try:
            if market == "INDIA":
                indices = await get_market_indices()
                
                sentiment = {
                    "market": market,
                    "indices": indices,
                    "overall_sentiment": "NEUTRAL"
                }
                
                # Simple sentiment logic based on index changes
                if indices.get("NIFTY50", {}).get("changePercent", 0) > 1:
                    sentiment["overall_sentiment"] = "BULLISH"
                elif indices.get("NIFTY50", {}).get("changePercent", 0) < -1:
                    sentiment["overall_sentiment"] = "BEARISH"
                
                return sentiment
            
            return {"error": f"Market {market} not yet implemented"}
        except Exception as e:
            logger.error(f"Error in get_market_sentiment: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def compare_stocks(symbols: list[str], metrics: Optional[list[str]] = None) -> dict[str, Any]:
        """Compare multiple stocks."""
        try:
            comparison = {"stocks": {}}
            
            for symbol in symbols[:5]:  # Limit to 5
                stock_data = await FinancialTools.get_stock_fundamentals(symbol, include_financials=False)
                if not stock_data.get("error"):
                    comparison["stocks"][symbol] = {
                        "pe": stock_data.get("peRatio"),
                        "price": stock_data.get("currentPrice"),
                        "market_cap": stock_data.get("marketCap"),
                        "dividend_yield": stock_data.get("dividendYield")
                    }
            
            return comparison
        except Exception as e:
            logger.error(f"Error in compare_stocks: {e}")
            return {"error": str(e)}
    
    @staticmethod
    async def get_sector_analysis(sector: str, period: str = "1m") -> dict[str, Any]:
        """Get sector analysis (simplified for MVP)."""
        return {
            "sector": sector,
            "period": period,
            "note": "Sector analysis requires additional data sources. Placeholder for MVP."
        }
    
    @staticmethod
    async def search_web(query: str, num_results: int = 5) -> dict[str, Any]:
        """Web search (simulated for MVP)."""
        return {
            "query": query,
            "results": [
                {"title": f"Search result for {query}", "url": "#", "snippet": "Placeholder"}
            ],
            "note": "Web search requires SerpAPI key. This is simulated data for MVP."
        }
    
    @staticmethod
    async def calculate_portfolio_optimization(
        stocks: list[str],
        risk_level: str,
        amount: Optional[float] = None
    ) -> dict[str, Any]:
        """Portfolio optimization (simplified for MVP)."""
        allocations = {}
        
        if risk_level == "conservative":
            weights = [0.4, 0.3, 0.2, 0.1]  # Example
        elif risk_level == "aggressive":
            weights = [0.3, 0.3, 0.2, 0.2]
        else:  # moderate
            weights = [0.35, 0.30, 0.20, 0.15]
        
        for i, stock in enumerate(stocks[:len(weights)]):
            allocations[stock] = weights[i]
        
        return {
            "risk_level": risk_level,
            "allocations": allocations,
            "total_amount": amount,
            "note": "Simplified allocation. Full optimization requires historical correlation analysis."
        }


# Tool dispatcher
async def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name with given arguments."""
    tools_map = {
        "get_stock_fundamentals": FinancialTools.get_stock_fundamentals,
        "get_technical_indicators": FinancialTools.get_technical_indicators,
        "search_financial_news": FinancialTools.search_financial_news,
        "get_market_sentiment": FinancialTools.get_market_sentiment,
        "compare_stocks": FinancialTools.compare_stocks,
        "get_sector_analysis": FinancialTools.get_sector_analysis,
        "search_web": FinancialTools.search_web,
        "calculate_portfolio_optimization": FinancialTools.calculate_portfolio_optimization,
        "search_knowledge_base": search_knowledge_base_tool,
        "get_technical_indicators": get_technical_indicators,
        "fetch_nse_quote": fetch_nse_quote_cached,
        "fetch_fii_dii": fetch_fii_dii_cached,
        "fetch_option_chain": fetch_option_chain_cached,
        "fetch_market_status": fetch_market_status_cached,
        "scrape_with_puppeteer": scrape_with_puppeteer_tool
    }
    
    tool_func = tools_map.get(tool_name)
    if not tool_func:
        return {"error": f"Unknown tool: {tool_name}"}
    
    try:
        result = await tool_func(**arguments)
        return result
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return {"error": str(e)}


async def search_knowledge_base_tool(query: str) -> dict[str, Any]:
    """Search knowledge base tool wrapper."""
    from ..rag import search_knowledge_base
    return await search_knowledge_base(query)


async def scrape_with_puppeteer_tool(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Puppeteer scraping tool wrapper."""
    from .puppeteer_client import scrape_with_puppeteer
    return await scrape_with_puppeteer(endpoint, params)


# Fast NSE tools with caching
async def fetch_nse_quote_cached(symbol: str) -> dict[str, Any]:
    """Fetch NSE quote with caching - FAST (0.5-1.5s or instant if cached)."""
    cache = get_nse_cache()
    cached_data = cache.get("quote", symbol)
    
    if cached_data:
        return cached_data
    
    data = await fetch_nse_quote(symbol)
    if not data.get("error"):
        cache.set("quote", data, symbol)
    return data


async def fetch_fii_dii_cached() -> dict[str, Any]:
    """Fetch FII/DII with caching - FAST (1-2s or instant if cached)."""
    cache = get_nse_cache()
    cached_data = cache.get("fii_dii")
    
    if cached_data:
        return cached_data
    
    data = await fetch_fii_dii()
    if not data.get("error"):
        cache.set("fii_dii", data)
    return data


async def fetch_option_chain_cached(symbol: str = "NIFTY") -> dict[str, Any]:
    """Fetch option chain with caching - FAST (1-2s or instant if cached)."""
    cache = get_nse_cache()
    cached_data = cache.get("option_chain", symbol)
    
    if cached_data:
        return cached_data
    
    data = await fetch_option_chain(symbol)
    if not data.get("error"):
        cache.set("option_chain", data, symbol)
    return data


async def fetch_market_status_cached() -> dict[str, Any]:
    """Fetch market status with caching - FAST."""
    cache =  get_nse_cache()
    cached_data = cache.get("market_status")
    
    if cached_data:
        return cached_data
    
    data = await fetch_market_status()
    if not data.get("error"):
        cache.set("market_status", data)
    return data
