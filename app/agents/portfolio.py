"""
Portfolio Intelligence Agent - Autonomous portfolio optimization and allocation.
AI autonomously uses tools to gather data for comprehensive portfolio strategies.
"""

from groq import AsyncGroq
import logging
import json

from ..config import settings, ModelType
from ..models.agent_state import AgentState

logger = logging.getLogger(__name__)


class PortfolioAgent:
    """Provides portfolio allocation with autonomous tool usage."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = settings.get_temperature_for_task(ModelType.REASONING_DEEP)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.REASONING_DEEP)
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Generate portfolio allocation strategies with autonomous tool usage.
        AI decides which tools to use for comprehensive portfolio construction.
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        amount = entities.get("amount")
        
        # Build amount context
        amount_context = ""
        if amount:
            amount_str = f"₹{amount:,.0f}"
            if amount >= 10000000:
                amount_str = f"₹{amount/10000000:.2f} Cr"
            elif amount >= 100000:
                amount_str = f"₹{amount/100000:.2f} L"
            amount_context = f"\nInvestment Amount: {amount_str}"
        
        system_prompt = """You are Daddys AI Portfolio Strategist - FULLY AUTONOMOUS with deep reasoning.

**Your Role:** Design comprehensive asset allocation and portfolio strategies for Indian investors.

**🤖 AUTONOMOUS TOOL USAGE:**

You have FULL AUTHORITY to use tools to build data-driven portfolio recommendations.

**Auto-Fetch Protocol:**

1. **Stock universe selection needed?**
   → USE `get_sector_analysis` to identify strong sectors
   → USE `compare_stocks` to pick best stocks in each sector
   → USE `fetch_nse_quote` for current valuations

2. **Portfolio optimization needed?**
   → USE `calculate_portfolio_optimization` with selected stocks
   → Specify risk level based on user's risk appetite

3. **Market sentiment context needed?**
   → USE `get_market_sentiment` to understand current market phase
   → Adjust allocation based on market conditions

4. **Specific stocks mentioned?**
   → USE `get_stock_fundamentals` to evaluate quality
   → USE `get_technical_indicators` for entry timing

5. **Sector rotation strategy?**
   → USE `get_sector_analysis` to identify rotating sectors
   → USE `search_web` for macro trends affecting sectors

**Critical Rules:**
- ✅ ALWAYS use tools to build data-driven allocations
- ✅ Cross-validate stock selections with fundamentals + technicals
- ✅ Consider India-specific tax implications (LTCG, STCG, 80C, PPF, ELSS)
- ✅ Balance growth potential with downside protection
- ❌ NEVER recommend portfolio without analyzing current market data

**Portfolio Framework:**
Provide structured strategy:
1. **Risk Profile Assessment** - Conservative/Moderate/Aggressive
2. **Asset Allocation** - Equity %, Debt %, Gold %, Cash %
3. **Equity Breakdown** - Sector-wise allocation with specific stocks
4. **Tax-efficient Instruments** - ELSS, PPF, NPS considerations
5. **Rebalancing Strategy** - When and how to rebalance
6. **Risk Management** - Stop-loss, position sizing, diversification

**Output Style:**
- Specific percentage allocations
- Named stocks with rationale
- Entry strategies and price levels
- Clear tax implications
- Use ₹ for Indian currency
- Include disclaimer: "Educational purposes only, not personalized financial advice"

**Remember:** You're AUTONOMOUS - gather ALL data needed for comprehensive portfolio construction."""

        # Import tool definitions
        from ..tools.tool_definitions import FINANCIAL_TOOLS
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        conversation_history = state.get("conversation_history", [])
        for msg in conversation_history[-5:]:
            messages.append(msg)
        
        # Add current query
        messages.append({
            "role": "user",
            "content": f"{amount_context}\nUser Query: {query}\n\nProvide comprehensive portfolio strategy. USE TOOLS autonomously to gather data."
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
                
                # Get final portfolio advice
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                advice = final_response.choices[0].message.content
            else:
                advice = message.content
            
            state["final_response"] = advice
            state["execution_metadata"] = {
                "agent": "portfolio",
                "model": self.model,
                "amount_analyzed": amount,
                "autonomous_tool_calls": len(tool_calls) if tool_calls else 0,
                "tools_used": [tc.function.name for tc in tool_calls] if tool_calls else []
            }
            
            logger.info(f"✅ Portfolio strategy: Tools={len(tool_calls) if tool_calls else 0}, Amount={amount}")
            return state
            
        except Exception as e:
            logger.error(f"Portfolio agent error: {e}")
            state["error"] = str(e)
            state["final_response"] = f"I encountered an error while generating portfolio advice. Please try again. Error: {str(e)}"
            return state
