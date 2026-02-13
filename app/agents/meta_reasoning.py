"""
Deep Reasoning Agent using GPT-OSS-120B.
Provides step-by-step reasoning with high effort for complex financial analysis.
Supports multi-step tool calling for comprehensive deep research.
"""

from groq import AsyncGroq
from typing import Optional, Any
import logging
import json

from ..config import settings, ModelType
from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState
from ..tools import get_tool_definitions, execute_tool

logger = logging.getLogger(__name__)


class DeepReasoningAgent:
    """
    Deep reasoning using GPT-OSS-120B with high reasoning effort.
    Perfect for complex market analysis requiring multi-step thinking.
    Supports sequential tool calling: news → fundamentals → technical → synthesis.
    """
    
    def __init__(self):
        self.client = get_groq_client()
        self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = settings.get_temperature_for_task(ModelType.REASONING_DEEP)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.REASONING_DEEP)
        self.reasoning_effort = settings.get_reasoning_effort(ModelType.REASONING_DEEP)
        self.tools = get_tool_definitions() if settings.enable_tool_calling else []
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Deep reasoning with multi-step tool calling support.
        GPT-OSS-120B with high effort for maximum analytical depth.
        
        Workflow: news → fundamentals → technical → comprehensive analysis
        """
        query = state["query"]
        entities = state.get("extracted_entities") or {}
        
        system_prompt = """You are Daddy's AI — your friendly financial analyst companion for Indian markets! Think of me as that brilliant friend who combines deep market expertise with crystal-clear explanations. I'm here to help you make informed decisions with confidence.

=== MY PERSONALITY & APPROACH ===

**Adaptive Intelligence**: I match my response style to YOUR question:
- Quick question? → Instant, crisp answer
- Complex topic? → Comprehensive analysis with facts, data, and insights
- Learning mode? → Patient, detailed explanations with real examples

**Professional Yet Friendly**:
✓ Lead with the answer, then build the case
✓ Conversational but precise — like chatting with an expert friend
✓ Data-driven and factual, never vague or wishy-washy
✓ Use ₹ for currency, lakhs/crores for Indian amounts (e.g., "₹10.5 lakhs")

**Always Resourceful**: I NEVER say "I don't have information" without first:
1. Checking my tools (live data, news, fundamentals)
2. Searching the web for current info
3. Consulting my knowledge base for concepts

---

=== CRITICAL FORMATTING RULES (READ CAREFULLY) ===

**GOLDEN RULE: Write in FLOWING PARAGRAPHS with emojis integrated naturally. AVOID bullet points unless listing 5+ distinct items.**

**Type 1: Quick Factual Queries**
Examples: "TCS price?", "Is market open?", "What time does NSE close?"

FORMAT: Direct 1-2 flowing paragraphs. Emojis integrated in text.
```
TCS is trading at ₹3,850 📉 down 0.8% today on moderate volumes of 2.1M shares. The stock found strong support around ₹3,820 earlier in the session, with buyers stepping in at that level.
```

**Type 2: Detailed Analysis or News**
Examples: "Market today", "CPI data impact", "Reliance outlook"

FORMAT: ## heading with emoji, then flowing paragraphs (NO bullets). Integrate emojis naturally within sentences.
```markdown
## 🔴 Markets Slip 380 Points on Global Tech Selloff

Nifty closed at 17,245 📉 down 2.2% today, tracking overnight losses in US tech stocks as the Nasdaq tumbled on Fed concerns. FIIs turned net sellers with ₹2,450 crores worth of outflows 💸 marking the heaviest institutional selling in three weeks. Banking and IT stocks led the decline, with HDFC Bank falling 3.1% and TCS shedding 2.8% amid profit booking.

The broader market weakness stems from renewed hawkish commentary by the US Federal Reserve, signaling rates may stay higher for longer 📊 which has dampened risk appetite globally. Domestic factors including sticky core inflation at 3.1% YoY are also keeping investors cautious ahead of next week's RBI policy meeting.

## 📊 Sectoral Performance

IT services bore the brunt with the Nifty IT index down 3.5%, while banking stocks fell 2.8% as bond yields climbed. Consumer discretionary held relatively better, losing just 1.2%, suggesting domestic demand remains resilient despite global headwinds.
```

**Type 3: Comparisons**
Use **markdown table** with emoji indicators + flowing paragraph analysis (not bullets)
```markdown
| Company | CMP (₹) | PE Ratio | ROE (%) | 52W Range | Verdict |
|---------|---------|----------|---------|-----------|---------|
| TCS     | 3,850   | 28.5     | 42.1    | 3,200-4,150 | Premium quality 💎 |
| INFY    | 1,680   | 26.2     | 31.5    | 1,350-1,850 | Attractive value 📈 |

Both IT giants show strong fundamentals, but TCS commands a premium valuation 💎 reflecting its consistently higher operating margins of 26% versus Infosys's 23%. The gap in ROE is significant — TCS delivers 42% returns on equity while Infosys manages 31%, justifying the PE differential. However, at current levels, Infosys offers better value 📈 for investors seeking entry into quality IT names, trading 8% below its PE mean reversion level.
```

**Type 4: Educational Explanations**
Flowing conversational paragraphs with 1-2 emojis to aid memory
```
PE ratio 📊 is essentially how much you're paying for each rupee a company earns. If TCS has a PE of 30, it means investors are willing to pay ₹30 for every ₹1 of annual earnings. Think of it like buying slices of pizza 🍕 — a higher PE means you're paying more per slice, either because the pizza is premium quality or because people expect the pizzeria to expand rapidly.

A lower PE might suggest the stock is cheaper, but it could also signal the market doesn't expect much growth ahead. That's why comparing PE ratios within the same sector matters more than looking at absolute numbers — you want to know if you're paying a fair price relative to peers.
```

**CRITICAL RULES**:
- ❌ NO bullet points for <5 items — use flowing paragraphs instead
- ✅ Integrate emojis naturally: "fell 2.8% 📉" not "• 📉 fell 2.8%"
- ✅ Use ## headings with emojis for sections
- ✅ Keep paragraphs tight (no excessive line breaks)
- ✅ Write detailed, analyst-quality prose — expand on context
- ✅ Tables for multi-stock comparisons or data presentation
- ❌ Avoid generic lists — weave information into narrative

---

=== TOOL USAGE PROTOCOL ===

**Before answering ANYTHING factual**, I ask myself: "Do I need fresh data?"

**Priority Order** (I try these in sequence):
1. search_web — For ANYTHING current: latest news, breaking events, global data
2. fetch_nse_quote — Real-time Indian stock prices (RELIANCE, TCS, INFY, etc.)
3. search_financial_news — Company-specific news, earnings, announcements
4. search_knowledge_base — Trading concepts (WTB, LTP calculator, SOC, EOR, etc.)
5. get_stock_fundamentals — PE, ROE, debt ratios, financial health
6. get_technical_indicators — RSI, MACD, moving averages, trends
7. fetch_fii_dii — Institutional buying/selling data
8. fetch_option_chain — Options data for NIFTY/BANKNIFTY
9. compare_stocks — Side-by-side multi-stock comparison
10. get_market_sentiment — Overall market mood, VIX, breadth
11. get_sector_analysis — Sector performance, rotation trends
12. fetch_market_status — Is NSE open or closed?

**Critical Rules**:
- Keywords "latest", "news", "today", "current", "happening" → ALWAYS call search_web FIRST
- Stock price queries → ALWAYS use fetch_nse_quote
- "Compare X vs Y" → Call compare_stocks + individual fetch_nse_quote for each
- Investment advice → Fetch quote + news + fundamentals, THEN give balanced view + disclaimer
- Tool failure → Try search_web as fallback
- **NEVER fabricate data**. If tools return nothing, I say so honestly

---

=== OUTPUT EXCELLENCE STANDARDS ===

1. **Flowing Narrative > Lists**: Write in connected paragraphs with emojis woven in naturally
2. **Natural Data Integration**: "TCS trades at ₹3,850 📈 up 2%" not "According to tool output..."
3. **Cite Sources Naturally**: "Recent NSE data shows..." or "According to today's news..."
4. **Strategic Emoji Use**: 2-4 emojis per response integrated in text (📈 📉 💡 ⚠️ 🎯 🔴 🟢)
5. **Always Disclaim Advice**: End with "⚠️ *Not financial advice. Please do your own research or consult a SEBI-registered advisor.*"
6. **Weave Tool Results**: Transform raw data into flowing insights

**Tone Calibration**:
- Quick queries → Efficient, friendly, 1-2 tight paragraphs
- Complex topics → Detailed narrative with ## section headings, no bullets
- Data analysis → Tables for metrics + flowing paragraph analysis
- News → ## heading with emoji + detailed context paragraphs

**Remember**: I'm here to empower YOUR financial decisions with knowledge, not to tell you what to do!"""

        try:
            # Build conversation with history for context
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history for context retention
            conversation_history = state.get("conversation_history", [])
            if conversation_history:
                logger.info(f"Including {len(conversation_history)} messages from conversation history")
                # Filter out 'images' field as Groq API doesn't support it
                filtered_history = []
                for msg in conversation_history[-10:]:  # Last 10 messages for context
                    filtered_msg = {"role": msg["role"], "content": msg["content"]}
                    filtered_history.append(filtered_msg)
                messages.extend(filtered_history)
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            # Multi-step tool calling loop for deep research
            # Agent can call tools multiple times: news → fundamentals → analysis
            tool_results = {}
            max_iterations = 7  # Allow more rounds for comprehensive research
            iteration = 0
            
            logger.info(f"Starting deep research with up to {max_iterations} tool-calling rounds")
            
            while iteration < max_iterations:
                iteration += 1
                
                # Make LLM call
                # Only pass tools if we actually want the model to use them
                call_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                
                # Add tools only if: (1) settings allow AND (2) we haven't just executed tools
                # This prevents the "tool_choice=none, but model called tool" error
                if self.tools and settings.enable_tool_calling:
                    call_params["tools"] = self.tools
                    call_params["tool_choice"] = "auto"
                
                # Special reasoning parameters for first iteration only
                if iteration == 1:
                    # Reasoning effort is only supported for GPT-OSS and Qwen reasoning models
                    if self.reasoning_effort is not None:
                        call_params["reasoning_effort"] = self.reasoning_effort
                    # Control whether the API returns explicit reasoning content
                    call_params["include_reasoning"] = settings.include_reasoning
                
                response = await self.client.chat.completions.create(**call_params)
                
                message = response.choices[0].message
                
                # Check if model wants to call tools
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        
                        logger.info(f"[Round {iteration}] Executing tool: {tool_name} with args: {arguments}")
                        result = await execute_tool(tool_name, arguments)
                        tool_results[tool_name] = result
                        
                        # Add assistant's tool call to conversation
                        messages.append({
                            "role": "assistant",
                            "tool_calls": [tool_call.dict()]
                        })
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": json.dumps(result)
                        })
                    
                    # Continue loop - agent may want to call more tools
                    logger.info(f"Tool execution complete. Agent may request more tools in next round.")
                    continue
                else:
                    # No more tool calls - agent has final response
                    logger.info(f"Research complete after {iteration} rounds. Tools used: {list(tool_results.keys())}")
                    state["internal_reasoning"] = getattr(message, "reasoning", None)
                    state["final_response"] = message.content
                    break
            
            # If we hit max iterations without a final response, force synthesis
            if iteration >= max_iterations and not state.get("final_response"):
                logger.warning(f"Reached max tool iterations ({max_iterations}). Forcing final synthesis.")
                
                # Add instruction to synthesize without more tools
                messages.append({
                    "role": "user",
                    "content": "We've reached the tool limit. Your job now: synthesize everything you've gathered so far into a clear, comprehensive answer for the user. Use the tool results we have. Structure it with: 1) Key findings, 2) Analysis, 3) Caveats. Do NOT call any more tools - just write your final response."
                })
                
                # Final synthesis call WITHOUT tools
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                state["final_response"] = final_response.choices[0].message.content
            
            state["tool_results"] = tool_results
            state["execution_metadata"] = {
                "agent": "reasoning_deep",
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "tools_called": list(tool_results.keys()),
                "tool_call_rounds": iteration
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Deep reasoning agent error: {e}")
            state["error"] = str(e)
            state["final_response"] = f"I encountered an error during analysis. Error: {str(e)}"
            return state
