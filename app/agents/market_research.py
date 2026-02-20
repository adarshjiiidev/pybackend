"""
Market Research Agent - Deep fundamental analysis with autonomous tool calling.
AI autonomously decides when and which tools to use for comprehensive analysis.
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime

from groq import AsyncGroq
from ..config import settings, ModelType
from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState, AgentMode
from ..tools.formatting import format_for_llm
from ..database import MarketDataCacheManager

logger = logging.getLogger(__name__)


class MarketResearchAgent:
    """Provides in-depth market research with autonomous tool usage."""
    
    def __init__(self):
        self.client = get_groq_client()
        self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = settings.get_temperature_for_task(ModelType.REASONING_DEEP)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.REASONING_DEEP)
        self.cache = MarketDataCacheManager()
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Perform deep market research analysis with autonomous tool calling.
        AI decides which tools to use based on the query.
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        symbols = entities.get("symbols", [])
        
        # Build comprehensive system prompt with autonomous tool instructions
        system_prompt = """You are Daddy's AI — a senior equity research analyst covering Indian markets. You write like a sharp analyst at a top brokerage: data-driven but readable, structured yet conversational.

=== YOUR STYLE ===
- Lead with INSIGHT, not data dumps. "RELIANCE looks attractively valued at 22x PE 📊 vs its 5-year average of 28x" > "PE ratio is 22"
- Be opinionated (with caveats). Analysts have views. Share yours, with data backing.
- Use ₹, lakhs, crores. You're writing for Indian investors.

=== SMART FORMATTING (Adaptive Response Style) ===

**Match your format to the content:**

1. **Single stock analysis** → Use 2-3 ## sections with emojis (📌 Overview, 💼 Financials, 🎯 Verdict). Short paragraphs. End with bull/bear case.
2. **Stock comparison** → Use a **markdown table** (| Stock | CMP (₹) | PE | ROE | Verdict |) with emojis in verdict column (💎 quality, 📈 buy, ⚠️ caution). Follow with 2-3 paragraphs.
3. **Sector overview** → Lead paragraph + table of top performers + brief outlook.
4. **Quick queries** → 1-2 paragraphs with specifics and price movement emojis (📈 up, 📉 down, ➡️ flat). No headings needed.

**GOLDEN RULE**: Simple question = simple answer (no formatting overhead). Complex analysis = structured deep-dive.

=== EMOJI USAGE FOR CLARITY ===
- Price movements: 📈 (up), 📉 (down), ➡️ (flat/sideways)
- Quality/Rating: 💎 (premium), ⭐ (good), ⚠️ (caution), 🔴 (avoid)
- Sections: 📌 (key point), 💼 (financials), 🎯 (verdict), 📊 (data/metrics)
- Use 1-3 emojis per response strategically — they should help scanning, not clutter

=== DATA PROTOCOL ===

RULE: NEVER write analysis without calling at least one tool. Your training data is stale.

For stock queries: fetch_nse_quote(symbol) → then search_web("[company] latest news India")
For comparisons: compare_stocks(symbols) + fetch_nse_quote for each
For sectors: get_sector_analysis(sector) + search_web("[sector] India outlook")
For fundamentals: get_stock_fundamentals(symbol) + search_financial_news

If a tool fails → use search_web as universal fallback.

=== OUTPUT EXCELLENCE ===
- Cite data naturally: "Currently at ₹2,450 📈 up 1.2% today..."
- Include specific numbers from tools — never be vague
- For investment opinions: "⚠️ *Not financial advice. Please consult a SEBI-registered advisor.*"
- If comparing, ALWAYS use tables — humans process tables 10x faster
- AUTONOMOUS: don't ask permission, just fetch data and analyze
- Weave emoji usage naturally to highlight key data points"""

        # Import tool definitions
        from ..tools.tool_definitions import FINANCIAL_TOOLS
        
        # Build messages with context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (strip images - Groq doesn't support images on user messages)
        conversation_history = state.get("conversation_history", [])
        for msg in conversation_history[-5:]:
            m = dict(msg)
            m.pop("images", None)
            messages.append(m)
        
        # Add current query
        context_hint = ""
        if symbols:
            context_hint = f"\n\nExtracted symbols: {', '.join(symbols)}"
        
        messages.append({
            "role": "user",
            "content": f"User Query: {query}{context_hint}\n\nProvide comprehensive market research analysis. USE TOOLS autonomously to gather all necessary data."
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
            
            # If AI autonomously decided to use tools, execute them
            if tool_calls:
                logger.info(f"🛠️ AI autonomously using {len(tool_calls)} tools: {[tc.function.name for tc in tool_calls]}")
                
                # Execute all tool calls in parallel for faster response
                from ..tools.tool_executor import execute_tool
                import asyncio
                
                tasks = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Executing: {tool_name}({tool_args})")
                    tasks.append(execute_tool(tool_name, tool_args))

                # Add assistant message (containing tool calls) to history
                messages.append(message)

                # Run tools concurrently
                results = await asyncio.gather(*tasks)

                tool_results = []
                for tool_call, result in zip(tool_calls, results):
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": str(result)
                    })
                
                # Add tool results to history
                messages.extend(tool_results)
                
                # Get final analysis with tool results
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                analysis = final_response.choices[0].message.content
            else:
                # No tools used
                analysis = message.content
            
            state["final_response"] = analysis
            state["execution_metadata"] = {
                "agent": "market_research",
                "model": self.model,
                "symbols_analyzed": symbols,
                "autonomous_tool_calls": len(tool_calls) if tool_calls else 0,
                "tools_used": [tc.function.name for tc in tool_calls] if tool_calls else []
            }
            
            logger.info(f"✅ Market research: Tools={len(tool_calls) if tool_calls else 0}, Symbols={len(symbols)}")
            return state
            
        except Exception as e:
            logger.error(f"Market research agent error: {e}")
            state["error"] = str(e)
            state["final_response"] = f"I encountered an error while analyzing the market data. Please try again. Error: {str(e)}"
            return state
