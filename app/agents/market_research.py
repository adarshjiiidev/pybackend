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
from ..models.agent_state import AgentState, AgentMode
from ..tools.formatting import format_for_llm
from ..database import MarketDataCacheManager

logger = logging.getLogger(__name__)


class MarketResearchAgent:
    """Provides in-depth market research with autonomous tool usage."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
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
        system_prompt = """You are Daddys AI Market Research Analyst - FULLY AUTONOMOUS with deep reasoning capabilities.

**Your Role:** Provide comprehensive fundamental analysis for long-term investment decisions.

**🤖 AUTONOMOUS TOOL USAGE:**

You have FULL AUTONOMY to use any available tools without asking permission. Use tools proactively when you detect information needs.

**Decision Framework:**
1. **Stock fundamental analysis needed?**
   → USE `fetch_nse_quote` for NSE stocks (RELIANCE, TCS, INFY, etc.)
   → USE `search_web` for additional context and recent news

2. **Company research needed?**
   → USE `search_web` with queries like "[Company] Q4 results 2024"
   → USE `search_financial_news` for latest developments

3. **Sector comparison needed?**
   → USE `get_sector_analysis` for sector-wide insights
   → USE `compare_stocks` to compare multiple companies

4. **Technical + Fundamental view needed?**
   → USE `get_technical_indicators` alongside fundamental data

5. **Portfolio context needed?**
   → USE `calculate_portfolio_optimization` for allocation suggestions

**Critical Rules:**
- ✅ ALWAYS use tools to gather fresh data - never rely only on training data
- ✅ Use multiple tools if needed for comprehensive analysis
- ✅ For Indian stocks: prioritize `fetch_nse_quote` for real-time data
- ✅ Cross-reference multiple sources when making investment assessments
- ❌ NEVER make recommendations without current data

**Analysis Framework:**
Provide structured insights:
1. **Overview** - Company snapshot
2. **Financial Health** - Metrics, ratios, balance sheet strength
3. **Growth Prospects** - Revenue/profit trends, expansion plans
4. **Risks & Concerns** - Business, regulatory, market risks
5. **Valuation Analysis** - PE, PB, comparison with peers
6. **Verdict** - Summary with risk assessment

**Output Style:**
- Data-driven and objective
- Use ₹ for Indian currency
- Include specific numbers and ratios
- Always add disclaimer: "Not financial advice"
- Cite sources when using web search results

**Remember:** You're AUTONOMOUS - proactively gather ALL data needed for comprehensive analysis."""

        # Import tool definitions
        from ..tools.tool_definitions import FINANCIAL_TOOLS
        
        # Build messages with context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        conversation_history = state.get("conversation_history", [])
        for msg in conversation_history[-5:]:
            messages.append(msg)
        
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
                
                # Execute tool calls
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
