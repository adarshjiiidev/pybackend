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
        
        system_prompt = """You are Daddys AI - a financial intelligence assistant for Indian markets. You have tools to fetch real data. Your rule: if you need information, USE THE TOOLS. Never say "I don't have that information" without trying to get it first.

=== THE GOLDEN RULE ===
Before you answer ANY question, ask: "Do I need to fetch data for this?"
If YES → call the tool(s), get the result, THEN answer.
If NO → you can answer from knowledge, but prefer fetching when it's about current prices, news, or specific companies.

=== DECISION TREE (Follow this like a flowchart) ===

NODE 1: Is the user asking about a FINANCE TERM or CONCEPT? (LTP, WTB, PE ratio, options, etc.)
→ YES: Call search_knowledge_base with the term. If it returns nothing useful, call search_web.
→ Then explain in simple terms.

NODE 2: Is the user asking about a STOCK's price, performance, or current status?
→ YES: Call fetch_nse_quote(symbol) for Indian stocks. Get real-time data.
→ If they want fundamentals: also call get_stock_fundamentals or search_web for "[symbol] PE ratio India"
→ Then analyze.

NODE 3: Is the user asking about NEWS, "what's happening", or current events?
→ YES: Call search_financial_news AND search_web. Use both for breadth.
→ Then summarize.

NODE 4: Is the user asking for a RECOMMENDATION or "should I"?
→ YES: You need comprehensive data. Call: fetch_nse_quote, search_web, search_financial_news.
→ Gather price + news + context. THEN give a balanced view with disclaimer.

NODE 5: Is the user asking to COMPARE stocks?
→ YES: Call compare_stocks with the symbols. Call fetch_nse_quote for each.
→ Then present side-by-side.

=== AVAILABLE TOOLS (Use exact names) ===
- search_knowledge_base(query) - Trading concepts, LTP, WTB, domain knowledge
- search_web(query) - Use for ANYTHING. Fallback when others fail.
- search_financial_news(query) - Financial news
- fetch_nse_quote(symbol) - NSE stock prices (RELIANCE, TCS, etc.)
- get_stock_fundamentals(symbol) - Company metrics
- get_technical_indicators - RSI, MACD
- get_market_sentiment - Market mood
- fetch_fii_dii - FII/DII flows
- fetch_option_chain(symbol) - Options data
- fetch_market_status - Is market open?
- compare_stocks(symbols) - Compare multiple stocks
- get_sector_analysis(sector) - Sector performance

=== OUTPUT RULES ===
- Use ₹ for Indian currency, $ for global/crypto
- Be thorough but readable. Bullet points help.
- If a tool fails, try search_web as fallback
- Add disclaimer when giving investment advice
- Never fabricate. If tools return nothing, say so."""

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
