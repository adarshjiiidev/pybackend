"""
Real-time Market Analysis Agent - Autonomous tool calling for live market data.
AI proactively fetches current prices, technical indicators, and market news.
"""

from groq import AsyncGroq
import logging
import json
from typing import Dict, Any, List
from datetime import datetime

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..tools.formatting import format_for_llm
from ..database import MarketDataCacheManager

logger = logging.getLogger(__name__)


class RealtimeAnalysisAgent:
    """Provides real-time market analysis with autonomous data fetching."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.get_model_for_task(ModelType.ANALYSIS)
        self.temperature = 0.6
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.ANALYSIS)
        self.cache = MarketDataCacheManager()
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Perform real-time market analysis with autonomous tool usage.
        AI decides which real-time data sources to use.
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        symbols = entities.get("symbols", [])
        
        system_prompt = """You are Daddys AI Real-Time Analysis Engine - FULLY AUTONOMOUS.

**Your Role:** Provide instant, actionable technical analysis and intraday insights.

**🤖 AUTONOMOUS TOOL USAGE:**

You have FULL AUTHORITY to fetch ALL data you need. NEVER say "I don't have information."

**Auto-Fetch Protocol:**

1. **Stock price query?**
   → USE `fetch_nse_quote` immediately for NSE stocks (RELIANCE, TCS, INFY, etc.)
   → USE `get_technical_indicators` for RSI, MACD, moving averages
   → USE `search_financial_news` for latest news affecting the stock

2. **Market sentiment query?**
   → USE `get_market_sentiment` for overall market mood
   → USE `search_web` for breaking market news

3. **Technical analysis needed?**
   → USE `get_technical_indicators` with appropriate indicators
   → Calculate support/resistance from price action

4. **Option chain analysis?**
   → USE `fetch_option_chain` for NIFTY/BANKNIFTY/stocks

5. **FII/DII activity?**
   → USE `fetch_fii_dii` for institutional flows

**Critical Rules:**
- ✅ ALWAYS fetch FRESH data - no assumptions about current prices
- ✅ Use `fetch_nse_quote` as primary source for Indian stocks
- ✅ Use multiple tools if comprehensive view needed
- ✅ Cross-reference technical indicators with price action
- ❌ NEVER respond without fetching current data first

**Analysis Focus:**
- Precise price levels (entry, stop-loss, targets)
- Clear technical signals (RSI overbought/oversold, MACD crossover, moving average trends)
- Volume analysis and momentum
- Support/resistance levels
- Risk-reward ratios
- Intraday bias (bullish/bearish/neutral)

**Output Style:**
- Direct and actionable
- Use ₹ for Indian stock prices
- Specific levels, not ranges
- Include disclaimer: "Not financial advice. For educational purposes."

**Remember:** You're AUTONOMOUS - proactively fetch real-time data, analyze, then respond."""

        # Import tool definitions
        from ..tools.tool_definitions import FINANCIAL_TOOLS
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        conversation_history = state.get("conversation_history", [])
        for msg in conversation_history[-5:]:
            messages.append(msg)
        
        # Add current query with symbol hints
        context_hint = ""
        if symbols:
            context_hint = f"\n\nExtracted symbols: {', '.join(symbols)}"
        
        messages.append({
            "role": "user",
            "content": f"User Query: {query}{context_hint}\n\nProvide focused real-time analysis. USE TOOLS autonomously to fetch current data."
        })
        
        try:
            # First call with tool availability
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=FINANCIAL_TOOLS,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            tool_calls = message.tool_calls
            
            # Execute autonomous tool calls
            if tool_calls:
                logger.info(f"🛠️ AI autonomously using {len(tool_calls)} tools: {[tc.function.name for tc in tool_calls]}")
                
                from ..tools.tool_executor import execute_tool
                tool_results = []
                
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Executing: {tool_name}({tool_args})")
                    
                    result = await execute_tool(tool_name, tool_args)
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result)
                    })
                
                # Add assistant message and tool results
                messages.append(message)
                messages.extend(tool_results)
                
                # Get final analysis
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                analysis = final_response.choices[0].message.content
            else:
                analysis = message.content
            
            state["final_response"] = analysis
            state["execution_metadata"] = {
                "agent": "realtime_analysis",
                "model": self.model,
                "symbols_analyzed": symbols,
                "autonomous_tool_calls": len(tool_calls) if tool_calls else 0,
                "tools_used": [tc.function.name for tc in tool_calls] if tool_calls else []
            }
            
            logger.info(f"✅ Real-time analysis: Tools={len(tool_calls) if tool_calls else 0}, Symbols={len(symbols)}")
            return state
            
        except Exception as e:
            logger.error(f"Real-time analysis agent error: {e}")
            state["error"] = str(e)
            state["final_response"] = f"I encountered an error while analyzing real-time market data. Please try again. Error: {str(e)}"
            return state
