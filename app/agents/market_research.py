"""
Market Research Agent - Uses Compound AI for company analysis
"""

import logging
from typing import Dict, Any
from datetime import datetime

from groq import AsyncGroq
from ..config import settings, ModelType
from ..models.agent_state import AgentState, AgentMode
from ..tools.formatting import format_for_llm
from ..tools.nse_scraper import fetch_nse_quote
from ..tools.symbol_mapper import normalize_symbol
from ..database import MarketDataCacheManager

logger = logging.getLogger(__name__)


class MarketResearchAgent:
    """Provides in-depth market research and fundamental analysis using reasoning model."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        # Use deep reasoning model for fundamental analysis
        self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = settings.get_temperature_for_task(ModelType.REASONING_DEEP)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.REASONING_DEEP)
        self.cache = MarketDataCacheManager()
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Perform deep market research analysis.
        Fetches stock data and provides comprehensive insights.
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        symbols = entities.get("symbols", [])
        
        # Fetch market data for identified symbols
        tool_results = {}
        
        if symbols:
            for symbol in symbols[:3]:  # Limit to 3 symbols
                stock_data = await self._fetch_stock_fundamentals(symbol)
                tool_results[symbol] = stock_data
        else:
            # Try to infer from query if no symbols extracted
            # For MVP, we'll use a default or ask LLM to identify
            pass
        
        state["tool_results"] = tool_results
        
        # Generate analysis using LLM
        analysis = await self._generate_analysis(query, tool_results, state.get("conversation_history", []))
        
        state["final_response"] = analysis
        state["execution_metadata"] = {
            "agent": "market_research",
            "symbols_analyzed": list(tool_results.keys())
        }
        
        return state
    
    async def _fetch_stock_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch stock fundamentals using NSE data."""
        try:
            logger.info(f"Fetching fundamentals for {symbol}")
            return await fetch_nse_quote(symbol)
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {"error": str(e)}
    
    async def _generate_analysis(
        self,
        query: str,
        tool_results: dict,
        conversation_history: list
    ) -> str:
        """Generate comprehensive market research analysis."""
        
        # Build context from tool results
        context = "Market Data:\n\n"
        for symbol, data in tool_results.items():
            context += format_stock_info(data) + "\n\n"
        
        system_prompt = """You are a senior equity research analyst specializing in Indian stock markets (NSE/BSE). 

Your role:
- Provide data-grounded, comprehensive fundamental analysis
- Focus on long-term investment potential
- Assess risks comprehensively (business, financial, market, regulatory)
- Compare with sector peers when relevant
- Use Indian retail investor-friendly language
- Include relevant financial metrics (PE, PB, ROE, debt ratios, etc.)
- Provide structured insights with clear sections
- Always include disclaimers about not being financial advice

Format your response with clear sections:
1. Overview
2. Financial Health
3. Growth Prospects
4. Risks & Concerns
5. Valuation Analysis
6. Verdict

IMPORTANT: 
- DO NOT include any [REASONING] or [INTERNAL] markers
- Be direct and user-facing
- Ground all statements in the provided data
- If data is insufficient, state what's missing"""

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 5 messages)
        for msg in conversation_history[-5:]:
            messages.append(msg)
        
        # Add current query with context
        user_message = f"{context}\nUser Query: {query}\n\nProvide comprehensive market research analysis."
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            analysis = response.choices[0].message.content
            logger.info("Market research analysis generated successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Market research agent error: {e}")
            return f"I encountered an error while analyzing the market data. Please try again. Error: {str(e)}"
