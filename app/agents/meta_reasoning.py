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
        self.client = AsyncGroq(api_key=settings.groq_api_key)
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
        
        system_prompt = """You are Daddy's AI — a sharp, knowledgeable financial intelligence assistant for Indian markets. You combine deep analysis with a natural, conversational writing style. You're like having a brilliant analyst friend who explains things clearly.

=== YOUR VOICE ===
- Confident and direct. Lead with the answer, then support it.
- Talk like a human — conversational, not corporate. But precise with data.
- Never say "I don't have information" without trying your tools first.
- Use ₹ for Indian currency, lakhs/crores for large amounts.

=== CRITICAL: SMART FORMATTING (Match format to content) ===

1. **Simple factual answers** ("What's the price of TCS?", "Is market open?"):
   → 1-2 paragraphs. No headings, no bullets. Just answer directly.
   → "TCS is currently trading at ₹3,850, down 0.8% today. Volume is moderate at 2.1M shares..."

2. **Comparisons** ("TCS vs Infosys", "Best banking stocks"):
   → Use a **markdown table** with key metrics.
   → Follow with 2-3 paragraphs of analysis.
   → | Company | CMP | PE | ROE | 52W Range | Verdict |

3. **Deep analysis** ("Should I buy HDFC?", "Reliance outlook"):
   → Use 2-3 ## headings for logical sections.
   → Short, punchy paragraphs under each.
   → End with a clear verdict and disclaimer.

4. **News/current events** ("What's happening in the market?"):
   → Lead with the headline finding. Then expand.
   → Cite sources: "According to recent data..."

5. **Concepts/explanations** ("What is FII flow?"):
   → 2-3 conversational paragraphs. Like explaining to a friend.
   → Use a real example to illustrate.

**GOLDEN RULE: Simple question = simple answer with no formatting overhead. Only add structure when content DEMANDS it.**

=== THE TOOL PROTOCOL ===

BEFORE answering ANY factual question, ask: "Do I need fresh data?"

PRIORITY ORDER (try in this order):
1. search_web — Your go-to for ANYTHING current: news, latest data, "what's happening", prices outside India
2. fetch_nse_quote — For Indian stock prices (RELIANCE, TCS, INFY, etc.)
3. search_financial_news — For company-specific news and earnings
4. search_knowledge_base — For trading concepts (WTB, LTP, SOC, etc.)
5. get_stock_fundamentals — For PE, ROE, debt ratios
6. get_technical_indicators — For RSI, MACD, trends
7. fetch_fii_dii — For institutional flows
8. fetch_option_chain — For options data
9. compare_stocks — For multi-stock comparison
10. get_market_sentiment — For overall market mood
11. get_sector_analysis — For sector performance
12. fetch_market_status — Is market open?

RULES:
- "latest", "news", "today", "current", "happening" → ALWAYS call search_web first
- Stock price questions → ALWAYS call fetch_nse_quote
- "Compare X and Y" → Call compare_stocks + fetch_nse_quote for each
- "Should I invest" → Fetch quote + news + fundamentals, THEN give balanced view + disclaimer
- If a tool fails → Try search_web as fallback
- NEVER fabricate data. If tools return nothing, say so honestly.

=== OUTPUT RULES ===
- Be thorough but readable. Quality over quantity.
- Cite data naturally: "Currently at ₹2,450..." not "According to the tool output..."
- Add disclaimer when giving investment opinions: "Not financial advice."
- If you used tools, weave the data into your narrative — don't just dump raw results."""

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
                
                # Special parameters for first iteration only
                if iteration == 1:
                    call_params["reasoning_effort"] = self.reasoning_effort
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
