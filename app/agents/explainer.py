"""
Explainer Agent - Provides clear, educational explanations.
FULL AUTONOMY: ReAct-style tool loop - model outputs structured tool calls,
we parse, execute, and loop. Works with any model, no Groq native tools needed.
"""

from groq import AsyncGroq
import logging
import re
import json

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..rag import get_kb_rag
from ..tools.tool_executor import execute_tool

logger = logging.getLogger(__name__)

# Tools Explainer can use autonomously (curated for educational queries)
EXPLAINER_TOOLS = [
    "search_knowledge_base",
    "search_web",
    "fetch_nse_quote",
    "search_financial_news",
    "get_stock_fundamentals",
]


class ExplainerAgent:
    """
    Provides clear, educational explanations with FULL AUTONOMOUS tool usage.
    Uses ReAct-style loop: model outputs <tool_call>...</tool_call>, we execute and continue.
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
        
        # Detect casual greetings and simple conversations
        greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'hola', 'namaste']
        casual_queries = ['how are you', 'whats up', "what's up", 'how do you do', 'how r u', 'how r you']
        query_lower = query.lower().strip()
        
        is_greeting = any(query_lower == pattern or query_lower.startswith(pattern + ' ') or query_lower.startswith(pattern + '!') for pattern in greeting_patterns)
        is_casual = any(pattern in query_lower for pattern in casual_queries)
        
        # Handle greetings naturally
        if is_greeting and len(query.split()) <= 3:
            state["final_response"] = "Hey there! 👋 I'm Daddy's AI, your financial companion. I can help you with:\n\n• **Stock analysis** - Get insights on Indian stocks like Reliance, TCS, HDFC\n• **Market data** - Real-time prices, fundamentals, technical indicators\n• **Learning** - Understand trading concepts, strategies, and financial terms\n• **Portfolio guidance** - Investment strategies and allocation advice\n\nWhat would you like to explore today?"
            state["execution_metadata"] = {"agent": "explainer", "mode": "greeting"}
            return state
        
        # Handle casual queries naturally
        if is_casual:
            state["final_response"] = "I'm doing great, thanks for asking! 😊\n\nI'm here and ready to help you with anything related to finance and investing. Whether you want to:\n\n• Analyze stocks\n• Track market trends\n• Learn about trading concepts\n• Get portfolio suggestions\n\n...I'm all yours! What's on your mind?"
            state["execution_metadata"] = {"agent": "explainer", "mode": "casual_conversation"}
            return state
        
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
        
        # Build system prompt with AUTONOMOUS tool-calling instructions (ReAct format)
        system_prompt = f"""You are a patient teacher explaining finance to Indian retail investors. Imagine you are teaching a curious child who asks "why?" and "how?" - be clear, use examples, and never assume they know jargon.

=== PART 1: YOUR TEACHING STYLE ===
- Use simple language. Replace jargon with plain words when possible.
- Use real examples: Reliance, TCS, HDFC - Indian stocks they know.
- Use ₹ for Indian currency.
- Be warm and encouraging. Say "Great question!" or "Let me break that down."
- For complex topics: start simple, then add detail. Like building blocks.

=== PART 2: WHEN TO USE TOOLS (Read this like a recipe) ===

BEFORE you answer, ask yourself: "Do I have enough information to answer accurately?"

SITUATION A - User asks about something with "latest", "today", "current", "happening", "breaking":
→ You NEED fresh data. Use search_web with a query like "latest [topic] India"
→ Example: "latest Reliance news" → search_web with query "Reliance Industries latest news today"

SITUATION B - User asks about a financial term (WTB, LTP, RSI, PE ratio, support/resistance) and it's NOT in the context below:
→ You NEED the knowledge base. Use search_knowledge_base with the term
→ Example: "What is WTB?" → search_knowledge_base with query "WTB weak towards bottom rules"

SITUATION C - User asks "what is the price of X" or "current value of [stock]":
→ You NEED real-time data. Use fetch_nse_quote with symbol (e.g. RELIANCE, TCS)
→ Or get_stock_fundamentals for more detail

SITUATION D - User asks about company earnings, quarterly results, or financial news:
→ Use search_financial_news with the company name

SITUATION E - You have enough in the context below to answer:
→ Do NOT use tools. Just explain using what you have. Cite the source.

=== PART 3: HOW TO CALL A TOOL (Exact format - copy this) ===
When you need a tool, output EXACTLY this (one tool per response, then wait for results):

<tool_call>
{{"tool": "TOOL_NAME", "arguments": {{"param": "value"}}}}
</tool_call>

Tool names and their arguments:
- search_knowledge_base: arguments = {{"query": "your search phrase"}}
- search_web: arguments = {{"query": "your search phrase"}}
- fetch_nse_quote: arguments = {{"symbol": "RELIANCE"}}  (use uppercase)
- search_financial_news: arguments = {{"query": "company or topic"}}
- get_stock_fundamentals: arguments = {{"symbol": "TCS"}}

=== PART 4: AFTER YOU GET TOOL RESULTS ===
- Read the result carefully
- Synthesize it into your explanation
- Cite the source: "According to the data..." or "Based on..."
- If the tool returned nothing useful, say "I couldn't find specific data on that, but here's what I know..."
- NEVER make up numbers or facts

=== PART 5: WHAT NEVER TO DO ===
- Never fabricate data. If unsure, say so.
- Never output more than one <tool_call> at a time - wait for results first
- Never ignore tool results and give a generic answer

=== PRE-LOADED CONTEXT (use this first before calling tools) ===
{", ".join(kb_sources) if kb_sources else "None - you will need to use search_knowledge_base or search_web to fetch information"}"""
        
        messages = [{"role": "system", "content": system_prompt}]
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            for msg in conversation_history[-5:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        user_message = f"{kb_context}\n\n---\nUser Query: {query}\n\nProvide a clear explanation. USE TOOLS if you need more info. CITE SOURCE when using KB or tool results." if kb_context else f"User Query: {query}\n\nProvide a clear explanation. USE TOOLS autonomously if you need more info."
        messages.append({"role": "user", "content": user_message})
        
        tools_used = []
        max_tool_rounds = 3
        
        try:
            for round_num in range(max_tool_rounds + 1):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content or ""
                
                # Parse for <tool_call>...</tool_call>
                tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, re.DOTALL)
                
                if not tool_call_match:
                    # No tool call - this is the final answer (strip any partial tool blocks)
                    explanation = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                    if explanation and len(explanation) > 30:
                        state["final_response"] = explanation
                    else:
                        state["final_response"] = content.strip() or "I apologize, I couldn't generate a proper explanation. Could you rephrase?"
                    break
                
                try:
                    tc = json.loads(tool_call_match.group(1))
                    tool_name = tc.get("tool", "")
                    tool_args = tc.get("arguments", {})
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool call JSON, treating as final response")
                    state["final_response"] = content.strip()
                    break
                
                if tool_name not in EXPLAINER_TOOLS:
                    logger.warning(f"Explainer requested unknown tool: {tool_name}, ignoring")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Tool '{tool_name}' is not available. Use one of: {', '.join(EXPLAINER_TOOLS)}. Or provide your answer without that tool."})
                    continue
                
                logger.info(f"🛠️ Explainer autonomous tool: {tool_name}({tool_args})")
                tools_used.append(tool_name)
                result = await execute_tool(tool_name, tool_args)
                result_str = json.dumps(result, default=str)[:2000]
                
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"[Tool Result for {tool_name}]:\n{result_str}\n\nNow synthesize and provide your complete answer to the user. Cite this data when relevant."})
            
            else:
                # Max rounds reached - strip any tool call from last response
                final_text = re.sub(r'<tool_call>.*?</tool_call>', '', content or '', flags=re.DOTALL).strip()
                state["final_response"] = final_text or "I gathered some data but couldn't fully synthesize. Please try a more specific question."
            
            state["execution_metadata"] = {
                "agent": "explainer",
                "model": self.model,
                "used_knowledge_base": bool(kb_context),
                "autonomous_tools_used": tools_used,
                "tool_rounds": len(tools_used)
            }
            logger.info(f"✅ Explainer: KB={bool(kb_context)}, Tools={tools_used}")
            return state
            
        except Exception as e:
            logger.error(f"Explainer agent error: {e}")
            state["error"] = str(e)
            state["final_response"] = "I encountered an error while generating the explanation. Please try again."
            return state
