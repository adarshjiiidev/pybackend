"""
Real-time Analysis Agent - Intraday price movements and technical signals.
Provides short-term trading insights and volatility analysis.
Uses ANALYSIS model for technical analysis tasks.
"""

from groq import AsyncGroq
import logging

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..tools import get_stock_info, get_historical_data, get_market_indices, format_for_llm
from ..tools.symbol_mapper import normalize_symbol, is_index, get_display_name
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
        indices_data = await self._fetch_market_indices()
        
        tool_results = {"indices": indices_data}
        
        # Fetch stock data for symbols (with proper Yahoo Finance symbols)
        if symbols:
            for symbol in symbols[:3]:
                stock_data = await self._fetch_realtime_data(symbol)
                tool_results[symbol] = stock_data
        
        state["tool_results"] = tool_results
        
        # Generate real-time analysis
        analysis = await self._generate_analysis(query, tool_results, state.get("conversation_history", []))
        
        state["final_response"] = analysis
        state["execution_metadata"] = {
            "agent": "realtime_analysis",
            "symbols_analyzed": list(tool_results.keys())
        }
        
        return state
    
    async def _fetch_market_indices(self) -> dict:
        """Fetch Indian market indices with caching."""
        async def fetch_fresh():
            return await get_market_indices()
        
        return await self.cache.get_or_fetch(
            symbol="INDICES",
            data_type="indices",
            fetch_func=fetch_fresh,
            ttl_seconds=30  # 30 second cache for indices
        )
    
    async def _fetch_realtime_data(self, symbol: str) -> dict:
        """Fetch real-time stock data using Yahoo Finance with correct symbol mapping."""
        async def fetch_fresh():
            # Normalize symbol to correct Yahoo Finance format
            # HUL -> HINDUNILVR.NS, NIFTY -> ^NSEI, BANKNIFTY -> ^NSEBANK
            yf_symbol = normalize_symbol(symbol)
            display_name = get_display_name(symbol)
            
            logger.info(f"Fetching data for {symbol} -> {yf_symbol}")
            
            # Get current price and info
            info = await get_stock_info(yf_symbol)
            # Get recent historical for trend context
            hist = await get_historical_data(yf_symbol, period="5d")
            
            return {
                "info": info,
                "historical": hist,
                "symbol": symbol,
                "yf_symbol": yf_symbol,
                "display_name": display_name
            }
        
        return await self.cache.get_or_fetch(
            symbol=symbol,
            data_type="realtime",
            fetch_func=fetch_fresh,
            ttl_seconds=10  # 10 second cache for real-time stock prices
        )
    
    async def _generate_analysis(
        self,
        query: str,
        tool_results: dict,
        conversation_history: list
    ) -> str:
        """Generate real-time market analysis."""
        
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
        
        system_prompt = """You are the real-time analysis module of Daddys AI.

**Your Role:**
- **ANSWER THE SPECIFIC QUESTION ASKED** - If they ask for a price, give the price directly
- Provide ACTUAL data from tool results (cite real numbers, not generic statements)
- Focus on TODAY'S market action and intraday movements
- Use Indian currency (₹) and context (NSE/BSE, Nifty, Sensex)

**Response Format:**

**For Price Queries** ("what's the price", "price of", "current price"):
1. Direct answer first: "[SYMBOL] is trading at ₹[PRICE]"
2. Day's change: "+[CHANGE] (+[PERCENT]%)"
3. Day range: "High ₹[HIGH], Low ₹[LOW]"
4. Brief context if relevant

**For Index Queries** ("Nifty", "Sensex", "market status"):
1. Index levels: "NIFTY50 at [VALUE]"
2. Day's movement: "+[CHANGE] (+[PERCENT]%)"
3. Market sentiment: Brief overview

**For Analysis Queries**:
1. Current Status (with actual prices from data)
2. Technical Signals (volume, momentum)
3. Short-term Outlook
4. Key Levels to Watch

**CRITICAL Rules:**
- Always cite ACTUAL numbers from the provided data
- Be concise and direct (2-4 sentences for simple price queries)
- If asked about indices, focus on indices
- If asked about a stock, focus on that stock
- Don't make predictions - just analyze current data
- Add disclaimer only when giving trading signals"""

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
