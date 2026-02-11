"""
Explainer Agent - Provides clear, educational explanations.
Uses autonomous tool calling - AI decides when to use tools based on knowledge gaps.
Includes conversation history for context retention.
"""

from groq import AsyncGroq
import logging
import json

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..rag import get_kb_rag

logger = logging.getLogger(__name__)


class ExplainerAgent:
    """
    Provides clear, educational explanations using autonomous tool calling.
    AI autonomously decides when to search KB or web based on knowledge gaps.
    Remembers conversation context for follow-up questions.
    """
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.get_model_for_task(ModelType.CREATIVE)
        self.temperature = 0.2  # VERY LOW temperature to prevent hallucination
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.CREATIVE)
        self.kb_rag = get_kb_rag()
        
        # Domain-specific keywords that MUST use knowledge base
        self.domain_keywords = [
            'wtb', 'wtt', 'ltp', 'shifting', 'pressure', 'coa', 
            'support', 'resistance', 'imaginary line', 'soc',
            'state of confusion', 'weekly range', 'scenario',
            'game of percentage', 'natural weakness', '75% rule',
            'blast', 'swing', 'arbitrage', 'itm', 'otm'
        ]
    
    def _check_domain_query(self, query: str) -> bool:
        """Check if query contains domain-specific keywords or financial terms."""
        from ..tools.financial_terms import is_financial_term
        
        # Check if it's a known financial term
        if is_financial_term(query):
            return True
        
        # Check domain-specific keywords
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.domain_keywords)
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Provide educational explanations with autonomous tool usage.
        AI autonomously decides when to use KB, web search, or internal knowledge.
        """
        query = state["query"]
        from ..tools.financial_terms import is_financial_term
        
        # Always check knowledge base first for domain terms
        kb_context = ""
        kb_sources = []
        is_domain_query = self._check_domain_query(query) or is_financial_term(query)
        
        if is_domain_query:
            # Search knowledge base
            results = self.kb_rag.search(query, top_k=2)
            if results:
                logger.info(f"📚 Found {len(results)} KB files for query")
                kb_sources = [f"{r['title']} ({r['filename']})" for r in results]
                
                # Build context with source files
                kb_parts = []
                for result in results:
                    kb_parts.append(f"\n## Source: {result['title']} ({result['filename']})\n{result['content'][:1500]}")
                kb_context = "\n---\n**Knowledge Base Context:**\n" + "\n".join(kb_parts)
        
        # Build system prompt with autonomous tool usage instructions
        system_prompt = f"""You are the educational module of Daddys AI, a financial intelligence system built by Adarsh, a 14-year-old student at Daddys International School.

**Your Mission:** Make complex finance simple and accessible for Indian retail investors.

**AUTONOMOUS DECISION FRAMEWORK:**

1. **First:** Check if you have sufficient internal knowledge to answer accurately
2. **Second:** If internal knowledge is insufficient, check the Knowledge Base Context provided below
3. **Third:** If BOTH internal knowledge AND Knowledge Base are insufficient, YOU MUST use available tools autonomously:
   - `search_knowledge_base` - For domain-specific financial terms (WTB, WTT, LTP, trading concepts)
   - `search_web` - For current events, breaking news, "what's happening", latest updates, recent news
   - `fetch_nse_quote` - For stock prices and real-time market data
   - Use tools AUTONOMOUSLY when you detect knowledge gaps - don't ask, just use them

**CRITICAL WHEN TO USE TOOLS:**
- Query contains "latest", "today", "happening now", "breaking", "current" → USE `search_web` autonomously
- Financial term not in Knowledge Base Context → USE `search_knowledge_base` first
- Stock price/market data needed → USE `fetch_nse_quote`
- Unknown concept AND not in KB → USE `search_web` as fallback

**CRITICAL RULES:**
1. **ALWAYS cite the source file** when answering from Knowledge Base
2. For domain-specific terms (WTB, WTT, LTP, etc.), prioritize Knowledge Base Context
3. **Format:** Start with "📖 Source: [filename]" when using KB
4. NEVER make up definitions - if unsure, USE TOOLS
5. For follow-up questions like "summarize it", refer to conversation history

**Teaching Style:**
- Use simple language and real-world analogies
- Provide examples with Indian stocks (Reliance, TCS, HDFC, etc.)
- Break down concepts step-by-step
- Use ₹ for Indian currency
- Be conversational yet professional

**Knowledge Base Sources Available:**
{", ".join(kb_sources) if kb_sources else "None - You may need to use tools autonomously"}"""

        # Import tool definitions
        from ..tools.tool_definitions import FINANCIAL_TOOLS
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            logger.info(f"Explainer: Including {len(conversation_history)} messages for context")
            for msg in conversation_history[-5:]:  # Last 5 messages
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current query with KB context if available
        user_message = query
        if kb_context:
            user_message = f"{kb_context}\n\n---\nUser Query: {query}\n\nProvide a clear explanation. CITE SOURCE if using KB. USE TOOLS if knowledge gap detected."
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            # First call with tool availability - AI decides autonomously
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=FINANCIAL_TOOLS,
                tool_choice="auto"  # Let AI autonomously decide
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
                
                # Get final response with tool results
                final_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                explanation = final_response.choices[0].message.content
            else:
                # No tools used, direct response
                explanation = message.content
            
            # Validate response
            if explanation and len(explanation) > 50:
                state["final_response"] = explanation
            else:
                state["final_response"] = "I apologize, I couldn't generate a proper explanation. Could you rephrase your question?"
            
            state["execution_metadata"] = {
                "agent": "explainer",
                "model": self.model,
                "temperature": self.temperature,
                "used_knowledge_base": bool(kb_context),
                "kb_context_length": len(kb_context) if kb_context else 0,
                "conversation_history_length": len(conversation_history),
                "autonomous_tool_calls": len(tool_calls) if tool_calls else 0,
                "tools_used": [tc.function.name for tc in tool_calls] if tool_calls else []
            }
            
            logger.info(f"✅ Explanation: KB={bool(kb_context)}, Tools={len(tool_calls) if tool_calls else 0}, History={len(conversation_history)}")
            return state
            
        except Exception as e:
            logger.error(f"Explainer agent error: {e}")
            state["error"] = str(e)
            state["final_response"] = f"I encountered an error while generating the explanation. Please try again."
            return state
