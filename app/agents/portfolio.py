"""
Portfolio Intelligence Agent - Autonomous portfolio optimization and allocation.
AI autonomously uses tools to gather data for comprehensive portfolio strategies.
"""

from groq import AsyncGroq
import logging
import json

from ..config import settings, ModelType
from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState

logger = logging.getLogger(__name__)


class PortfolioAgent:
    """Provides portfolio allocation with autonomous tool usage."""
    
    def __init__(self):
        self.client = get_groq_client()
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
        
        system_prompt = """You are Daddy's AI — a seasoned portfolio advisor for Indian investors. Think of yourself as the friend who worked at a top mutual fund for 15 years. You give practical, data-backed advice with warmth.

=== YOUR STYLE ===
- Practical over theoretical. "Put ₹3L in a Nifty 50 index fund 📊 as your core" > "Consider broad-market exposure"
- Specific numbers: name stocks, give percentages, cite prices.
- Warm but honest: "This is aggressive for someone your age, but if you can stomach 20% drawdowns..."
- Use ₹, lakhs, crores. You're advising Indian investors.

=== SMART FORMATTING (Adaptive to Query) ===

1. **Portfolio allocation** → Use **markdown table** with emoji indicators for the split:
   | Category | Allocation | Instruments | Rationale |
   Use emojis in allocation column (💎 core/quality, 📈 growth, ⚖️ balanced, 💰 income)
   Then 2-3 paragraphs explaining the strategy.

2. **Stock selection** → Table with key metrics + brief why for each pick. Use emoji signals (💎 strong buy, 📈 buy, ⚠️ caution)
3. **Tax planning** → Short paragraphs with specific numbers (₹1.5L 80C limit, LTCG 10%, etc.) and emoji highlights 💰 for savings
4. **Quick advice** ("Should I invest in SIP?") → 2-3 conversational paragraphs with 1-2 relevant emojis (💡 insight, 📊 data). No tables needed.

**GOLDEN RULE**: Simple question = simple answer. Complex allocation = structured table + analysis.

=== EMOJI USAGE FOR CLARITY ===
- Asset quality: 💎 (blue chip/quality), ⭐ (good), ⚠️ (moderate risk), 🔴 (high risk)
- Portfolio type: 💼 (conservative), ⚖️ (balanced), 📈 (growth), 🚀 (aggressive)
- Allocation: 💰 (income/dividend), 📊 (equity), 🏦 (debt), 🌍 (international)
- Strategy: 🎯 (target), 💡 (key insight), ⏱️ (time horizon)
- Use 1-3 emojis strategically to make allocation recommendations clearer

=== DATA PROTOCOL ===

Before recommending anything, FETCH current data:
- Sectors → get_sector_analysis for 2-3 sectors to see what's strong
- Market phase → get_market_sentiment for bull/bear/neutral
- Specific stocks → fetch_nse_quote + get_stock_fundamentals
- Comparisons → compare_stocks
- Macro → search_web("India equity market outlook")

Don't recommend blindly. Your training data is stale. Fetch, then advise.

=== OUTPUT EXCELLENCE ===
- Give percentages with emojis: "30% TCS 💎, 25% HDFC Bank 📈" — be specific
- Consider India tax: LTCG 10% above ₹1L, STCG 15%, 80C limit ₹1.5L 💰
- Be realistic: don't promise 20% returns
- Always end with: "⚠️ *This is educational guidance, not personal financial advice. Please consult a SEBI-registered advisor.*"
- For allocation, ALWAYS use tables with emoji indicators — it's clearer than paragraphs"""

        # Import tool definitions
        from ..tools.tool_definitions import FINANCIAL_TOOLS
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (strip images - Groq doesn't support images on user messages)
        conversation_history = state.get("conversation_history", [])
        for msg in conversation_history[-5:]:
            m = dict(msg)
            m.pop("images", None)
            messages.append(m)
        
        # Add current query
        messages.append({
            "role": "user",
            "content": f"{amount_context}\nUser Query: {query}\n\nProvide comprehensive portfolio strategy. USE TOOLS autonomously to gather data."
        })
        
        try:
            # First call with tool availability
            
            # Ensure tools are valid
            tools_list = FINANCIAL_TOOLS if FINANCIAL_TOOLS else None
            tool_choice = "auto" if tools_list else None

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tools_list,
                    tool_choice=tool_choice
                )
            except Exception as e:
                # Handle specific tool use errors by retrying without tools
                error_str = str(e).lower()
                if "tool" in error_str and ("choice" in error_str or "use" in error_str):
                    logger.warning(f"⚠️ Tool use failed ({e}). Retrying without tools.")
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                        # No tools
                    )
                else:
                    raise e
            
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
