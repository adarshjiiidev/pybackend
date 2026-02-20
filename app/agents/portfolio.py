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

=== STRICT FORMATTING RULES (NO BULLET POINTS) ===

**🚫 BULLET POINTS ARE COMPLETELY BANNED - Use tables or carousels**

Your ONLY formatting options:
1. **Markdown tables** for allocations, comparisons, recommendations
2. **Carousels** for step-by-step strategies or multiple recommendations
3. **Flowing paragraphs** for explanations

**1. Portfolio Allocation → MUST use table:**
| Category | Allocation | Instruments | Rationale |
|----------|------------|-------------|-----------|
| Equity 📊 | 60% | Nifty 50 index + 5 stocks | Core growth |
| Debt 🏦 | 30% | Liquid funds + bonds | Stability |
| Gold 💰 | 10% | Gold ETF | Hedge |

Then 2-3 paragraphs explaining the strategy and risk profile.

**2. Stock Recommendations → MUST use table:**
| Stock | CMP (₹) | Target | Allocation | Verdict |
|-------|---------|---------|------------|---------|
| TCS | 3,850 | 4,200 | 15% | 💎 Quality |
| HDFC Bank | 1,650 | 1,850 | 12% | 📈 Growth |

**3. Multi-step Strategy → MUST use carousel:**
````carousel
## Step 1: Emergency Fund 💰
Park ₹2L in a liquid fund for 6 months expenses before investing in equity.
<!-- slide -->
## Step 2: Core Portfolio 📊
Invest ₹5L in Nifty 50 index fund via SIP over 6 months for market exposure.
<!-- slide -->
## Step 3: Satellite Holdings 🎯
Add ₹3L in 3-5 quality stocks for alpha generation potential.
````

**4. Quick Advice → Flowing paragraphs:**
"For SIP investing 💡, start with ₹10K monthly in a Nifty index fund. This gives you disciplined accumulation and rupee cost averaging benefits."

**ABSOLUTE RULES:**
- ❌ NEVER use bullet points (-, *, •) or numbered lists
- ✅ Tables for allocations, stock picks, comparisons
- ✅ Carousels for strategies, steps, multiple recommendations
- ✅ Paragraphs for narrative explanations
- ✅ Emojis (1-3 per response) for visual clarity

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
            
            # For GPT-OSS models, use explicit tool_choice to avoid conflicts
            if "gpt-oss" in self.model.lower():
                # Phase 1: Get reasoning without tool calling
                tool_choice = "none"  # Explicitly prevent tool calling during reasoning
                logger.debug("🧠 GPT-OSS: Phase 1 reasoning without tools")
            else:
                # Regular models can use auto tool calling
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
                if "tool" in error_str and ("choice" in error_str or "use" in error_str or "400" in str(e)):
                    logger.warning(f"⚠️ Tool use failed ({e}). Retrying without tools.")
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                        # No tools at all
                    )
                else:
                    raise e
            
            message = response.choices[0].message
            tool_calls = message.tool_calls
            
            # Execute autonomous tool calls
            if tool_calls:
                logger.info(f"🛠️ AI autonomously using {len(tool_calls)} tools: {[tc.function.name for tc in tool_calls]}")
                
                from ..tools.tool_executor import execute_tool
                import asyncio
                
                # Execute all tool calls in parallel for better performance
                tasks = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    logger.info(f"Executing: {tool_name}({tool_args})")
                    tasks.append(execute_tool(tool_name, tool_args))

                # Add assistant message (containing tool calls) to history
                messages.append(message)

                # Wait for all tools concurrently
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
