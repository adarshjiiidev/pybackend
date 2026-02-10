"""
Real-time Market Analysis Agent
Focuses on current market data, live prices, and intraday movements.
Uses NSE data and Compound AI for market information.
"""

from groq import AsyncGroq
import logging
from typing import Dict, Any, List
from datetime import datetime

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..tools.formatting import format_for_llm
from ..tools.nse_scraper import fetch_nse_quote
from ..tools.symbol_mapper import normalize_symbol, get_display_name
from ..tools.formatting import clean_response
from ..database import MarketDataCacheManager

logger = logging.getLogger(__name__)


class RealtimeAnalysisAgent:
    """Provides real-time market analysis and short-term insights using analysis model."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        # Use analysis model for technical analysis
        self.model = settings.get_model_for_task(ModelType.ANALYSIS)
        self.temperature = 0.6  # Slightly lower for technical precision
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.ANALYSIS)
        self.cache = MarketDataCacheManager()
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Perform real-time market analysis.
        Focuses on intraday movements, technical indicators, and short-term signals.
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        symbols = entities.get("symbols", [])
        
        # Always fetch market indices for context
        # indices_data = await self._fetch_market_indices()
        
        tool_results = {}
        
        # Fetch stock data for symbols
        if symbols:
            for symbol in symbols[:3]:  # Limit to 3 symbols for performance
                stock_data = await self._fetch_realtime_data(symbol)
                tool_results[symbol] = stock_data
        
        # DISABLED: Market indices removed with Yahoo Finance
        # indices = await self._fetch_market_indices()
        # tool_results["market_indices"] = indices
        
        # Generate AI analysis
        analysis = await self._generate_analysis(
            query=query,
            tool_results=tool_results,
            conversation_history=state.get("conversation_history", [])
        )
        state["final_response"] = analysis
        state["execution_metadata"] = {
            "agent": "realtime_analysis",
            "symbols_analyzed": list(tool_results.keys())
        }
        
        return state
    
    # DISABLED: Market indices fetch removed with Yahoo Finance
    # async def _fetch_market_indices(self) -> dict:
    #     """Fetch Indian market indices with caching."""
    #     async def fetch_fresh():
    #         return await get_market_indices()
    #     
    #     return await self.cache.get_or_fetch(
    #         symbol="INDICES",
    #         data_type="indices",
    #         fetch_func=fetch_fresh,
    #         ttl_seconds=30  # 30 second cache for indices
    #     )
    
    async def _fetch_realtime_data(self, symbol: str) -> dict:
        """
        Fetch real-time stock data using NSE scraper only.
        No Yahoo Finance dependency.
        """
        async def fetch_fresh():
            # Normalize symbol
            display_name = get_display_name(symbol)
            
            logger.info(f"Fetching data for {symbol} -> {display_name}")
            
            # Get current price from NSE
            nse_data = await fetch_nse_quote(symbol)
            
            return {
                "info": nse_data,
                "symbol": symbol,
                "display_name": display_name
            }
        
        # Return fresh data directly (cache can be added later if needed)
        return await fetch_fresh()
    
    async def _generate_analysis(
        self,
        query: str,
        tool_results: dict,
        conversation_history: list
    ) -> str:
        """Generate real-time market analysis."""
        
        system_prompt = """You are Daddys AI Real-Time Analysis Engine - FULLY AUTONOMOUS.

**Your Role:** Provide instant, actionable technical analysis and intraday insights.

**🤖 AUTONOMOUS MODE ENABLED:**
You proactively fetch ALL data you need. NEVER say "I don't have information."

**Auto-Data Gathering Protocol:**
1. Stock query? → AUTO-FETCH:
   - fetch_nse_quote (instant NSE price)
   - search_web("[symbol] technical analysis India") for additional context
   - get_technical_indicators for RSI, MACD, momentum

2. Index query? → AUTO-FETCH market indices

3. Need historical data? → AUTO-FETCH without asking

4. Unsure about a stock? → search_web("[symbol] latest news") for context

**Critical Rules:**
- ✅ Always fetch FRESH data (no cache assumptions)
- ✅ Use fetch_nse_quote as primary source for Indian stocks
- ✅ Use search_web as fallback for non-NSE or when fetch fails
- ✅ Compare current price with support/resistance levels
- ❌ NEVER respond without fetching current data first

**Output Style:**
- Precise price levels (entry, SL, target)
- Clear technical signals (RSI overbought/oversold, MACD crossover)
- Risk assessment
- Use ₹ for prices
- Be decisive yet professional

**Remember:** You're AUTONOMOUS - fetch data first, analyze second, respond third."""
        
        # Build context with actual data
        context = "Real-time Market Data:\n\n"
        
        # Add indices data
        if "indices" in tool_results:
            context += format_for_llm(tool_results["indices"], "indices") + "\n\n"
        
        # Add stock-specific data
        for symbol, data in tool_results.items():
            if symbol != "indices" and isinstance(data, dict):
                if "info" in data:
                    context += f"Stock: {symbol}\n"
                    context += format_for_llm(data["info"], "stock") + "\n"

        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in conversation_history[-5:]:
            messages.append(msg)
        
        user_message = f"{context}\nUser Query: {query}\n\nProvide focused analysis."
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            analysis = response.choices[0].message.content
            logger.info("Real-time analysis generated successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Real-time analysis agent error: {e}")
            return f"I encountered an error while analyzing real-time market data. Please try again. Error: {str(e)}"
