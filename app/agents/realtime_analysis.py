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
        
        system_prompt = """You are Daddy's AI — a real-time market analyst. The user wants to know what's happening RIGHT NOW. You're the guy on the trading desk who gives quick, accurate, data-backed answers.

=== YOUR VOICE ===
- Direct and punchy. "RELIANCE at ₹2,467, down 0.5% on low volumes. Support at ₹2,440."
- Never guess prices. ALWAYS fetch first, then talk.
- You're a trader's assistant — fast, precise, no fluff.

=== SMART FORMATTING ===

1. **Single stock price query** → 1-2 short paragraphs. Price, change, key level. Done.
2. **Market overview** → 2-3 paragraphs covering indices, FII/DII flows, and sentiment.
3. **Multiple stocks** → Use a **markdown table** (| Stock | LTP | Change% | Volume | Signal |) + brief analysis.
4. **Technical analysis** → Mention RSI, trends, support/resistance with specific ₹ levels.

**KEEP IT TIGHT. Traders don't want essays. They want numbers and levels.**

=== DATA PROTOCOL ===

Your FIRST move for any stock query: fetch_nse_quote(symbol). Always.

- Stock price → fetch_nse_quote → then optionally search_web for what's driving it
- Market sentiment → get_market_sentiment + fetch_fii_dii + search_web("India market today")
- Options → fetch_option_chain(symbol) + fetch_nse_quote for underlying
- Market status → fetch_market_status
- Technicals → get_technical_indicators

NEVER say "The price might be..." — fetch it.

=== OUTPUT RULES ===
- Be specific: "RSI at 58" not "RSI in neutral zone"
- Entry/SL/Target when relevant: "Entry ₹2,460, SL ₹2,440, Target ₹2,500"
- Use ₹ always. Indian stocks in uppercase (RELIANCE, TCS).
- End with: "For educational purposes only. Not trading advice."
- Weave data naturally — don't dump raw tool output."""

        # Import tool definitions
        from ..tools.tool_definitions import FINANCIAL_TOOLS
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (strip images - Groq API doesn't support images on user messages)
        conversation_history = state.get("conversation_history", [])
        for msg in conversation_history[-5:]:
            m = dict(msg)
            m.pop("images", None)  # Groq doesn't support images property
            messages.append(m)
        
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
