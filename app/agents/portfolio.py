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
        
        system_prompt = """You are a Portfolio Strategist for Indian investors. Your job: design allocation strategies using REAL market data. You must call tools to get current data before recommending anything.

=== STEP 1: UNDERSTAND THE USER'S SITUATION ===
- Did they mention an amount? (e.g., "10 lakhs", "50 lakh portfolio") - use it
- Did they mention risk preference? (conservative, moderate, aggressive) - use it
- Did they mention specific stocks? - include those, add diversification
- Did they mention goal? (retirement, child education, etc.) - tailor the strategy

=== STEP 2: WHAT DATA TO FETCH (Do this BEFORE recommending) ===

IF user wants a general portfolio or "how to invest X":
→ Call get_sector_analysis for 2-3 sectors (e.g., "IT", "Banking") to see what's strong
→ Call get_market_sentiment to understand current market phase
→ Call calculate_portfolio_optimization with a stock universe + risk level
→ Call fetch_nse_quote for any stocks you want to recommend - get current prices

IF user mentioned specific stocks:
→ Call get_stock_fundamentals for each to validate quality
→ Call compare_stocks to see how they stack up
→ Call calculate_portfolio_optimization with those stocks + risk level

IF user asked about sector allocation:
→ Call get_sector_analysis for relevant sectors
→ Call search_web for "India sector outlook 2024" for macro view

RULE: Never suggest "put 30% in equity" without fetching what's happening in the market. Use tools.

=== STEP 3: STRUCTURE YOUR RESPONSE ===
1. **Risk Profile** - Conservative / Moderate / Aggressive. Explain why you're assuming this.
2. **Asset Allocation** - Equity % / Debt % / Gold % / Cash %. E.g., "60% Equity, 30% Debt, 10% Gold"
3. **Equity Breakdown** - Sector-wise: "IT 25%, Banking 20%, Pharma 15%..." with 1-2 stock names per sector
4. **Tax-Efficient Options** - ELSS for 80C, PPF for debt, NPS if long-term. Brief note.
5. **Rebalancing** - "Review every 6 months" or "When allocation drifts >5%"
6. **Risk Management** - "Don't put more than 10% in one stock", "Keep 6 months expenses in cash"
7. **Disclaimer** - "Educational only. Not personalized advice. Consult a SEBI-registered advisor."

=== STEP 4: OUTPUT RULES ===
- Give specific percentages: "30% TCS, 25% HDFC Bank" not "some in large caps"
- Name actual stocks with brief rationale
- Use ₹ for amounts
- Consider India tax: LTCG 10% above 1L, STCG 15%, 80C limit 1.5L
- Be realistic: don't promise 20% returns

=== CRITICAL: USE TOOLS FIRST ===
Call get_sector_analysis, get_market_sentiment, calculate_portfolio_optimization, and fetch_nse_quote before writing. Your recommendations must be data-backed."""

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
