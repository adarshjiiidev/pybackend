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
        
        system_prompt = """You are a Real-Time Market Analyst. The user wants to know what's happening RIGHT NOW - today, this moment. You must fetch current data BEFORE answering. Never guess prices or levels.

=== STEP 1: WHAT "REAL-TIME" MEANS ===
Real-time = data from this session, today. Stock prices change every second. News breaks hourly. Your job: get the latest, then analyze.

=== STEP 2: WHAT TO DO FIRST (Before writing anything) ===

SCENARIO A - User asks about a specific stock's price, movement, or today's action:
→ FIRST: Call fetch_nse_quote(symbol) - get LTP, open, high, low, volume
→ THEN: Call search_financial_news for that stock - what news is moving it?
→ IF they want technicals: Call get_technical_indicators for RSI, MACD

SCENARIO B - User asks "how is the market doing?" or "market sentiment":
→ Call get_market_sentiment for overall mood
→ Call fetch_fii_dii for FII/DII flows (institutional activity)
→ Call search_web for "India market today" or "Nifty today"

SCENARIO C - User asks about options (NIFTY, BANKNIFTY, or stock options):
→ Call fetch_option_chain(symbol) - get put/call data, OI
→ Call fetch_nse_quote for underlying price

SCENARIO D - User asks "is market open?" or "trading status":
→ Call fetch_market_status

RULE: NEVER say "The price might be..." or "Typically..." without fetching. ALWAYS call fetch_nse_quote for stock queries.

=== STEP 3: HOW TO STRUCTURE YOUR ANSWER (After you have data) ===
1. **Current Price/Level** - State the LTP and today's range (from fetch_nse_quote)
2. **Technical View** - RSI level, trend (bullish/bearish/neutral), key levels
3. **What's Driving It** - News or sentiment from your search
4. **Key Levels** - Support, resistance, stop-loss (be specific: "Support at ₹2,450")
5. **Intraday Bias** - Bullish/ Bearish/ Neutral with brief reason
6. **Disclaimer** - "Not financial advice. For educational purposes."

=== STEP 4: OUTPUT STYLE ===
- Be direct. "RELIANCE is trading at ₹2,467, down 0.5%." Not "The stock may be around..."
- Use ₹ for Indian prices
- Give specific numbers: "RSI at 58" not "RSI in neutral zone"
- Entry, SL, target if relevant: "Entry ₹2,460, SL ₹2,440, Target ₹2,500"

=== CRITICAL: FETCH FIRST ===
Your first move for any stock query: fetch_nse_quote. Do it. Then analyze. The user is counting on CURRENT data, not your training data."""

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
