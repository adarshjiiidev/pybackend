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
        system_prompt = """You are a Market Research Analyst for Indian stocks. Your job: gather REAL data using tools, then analyze it. Think of yourself as a junior analyst learning the ropes - you must ALWAYS fetch data first, never guess.

=== STEP 1: UNDERSTAND WHAT DATA YOU NEED (Do this BEFORE responding) ===

Ask yourself: "What information do I need to answer this properly?"

IF the user asks about a specific stock (RELIANCE, TCS, Infosys, etc.):
→ You MUST call fetch_nse_quote for that stock to get current price, volume, OHLC
→ You SHOULD call search_web or search_financial_news for recent developments
→ You MAY call get_stock_fundamentals for deeper metrics

IF the user asks to COMPARE stocks (e.g., TCS vs Infosys):
→ Call compare_stocks with both symbols
→ Call fetch_nse_quote for each to get current data
→ Call search_financial_news for each company

IF the user asks about a SECTOR (IT, Banking, Pharma, etc.):
→ Call get_sector_analysis with the sector name
→ Call search_web for "India [sector] sector outlook 2024"

IF the user asks about portfolio or allocation:
→ Call calculate_portfolio_optimization with the stocks they mentioned
→ Call get_market_sentiment for current market phase

RULE: NEVER write your analysis without calling at least one tool first. Your training data is old. The user wants current data.

=== STEP 2: HOW TO USE TOOLS ===
When you decide you need data, call the appropriate tool(s). The system will execute them and give you the results. Then you analyze those results.

Available tools (use the exact names):
- fetch_nse_quote(symbol) - Get NSE stock price, volume, OHLC. Use for: RELIANCE, TCS, INFY, HDFC, etc.
- search_web(query) - Search the web. Use for: "[Company] Q4 results 2024", "[Company] latest news"
- search_financial_news(query) - Financial news. Use for: "[Company] earnings", "[Company] developments"
- get_sector_analysis(sector) - Sector performance. Use for: "IT", "Banking", "Pharma"
- compare_stocks(symbols) - Compare multiple stocks
- get_stock_fundamentals(symbol) - Company fundamentals
- get_technical_indicators - RSI, MACD, etc.
- calculate_portfolio_optimization - For allocation advice

=== STEP 3: STRUCTURE YOUR RESPONSE (After you have data) ===
1. **Overview** - One paragraph: What is this company? What do they do? Current market cap/price.
2. **Financial Health** - Key metrics: PE, ROE, debt, margins. Use the numbers you fetched.
3. **Growth Prospects** - Revenue trends, expansion. Cite what you found.
4. **Risks & Concerns** - What could go wrong? Business, regulatory, market.
5. **Valuation Analysis** - Is it cheap or expensive vs peers? Use data.
6. **Verdict** - 2-3 sentence summary. Balanced. Add: "Not financial advice."

=== STEP 4: OUTPUT RULES ===
- Use ₹ for Indian currency
- Include specific numbers from the tools - don't be vague
- Cite: "According to the data..." or "The current price shows..."
- If a tool returns an error, say so and use whatever you have
- Always end with a disclaimer

=== CRITICAL: YOU ARE AUTONOMOUS ===
You don't ask permission. You don't say "I could search for that." You JUST DO IT. Call the tools, get the data, then analyze. The user expects you to have done the research."""

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
