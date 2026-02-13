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
from ..config.key_rotator import get_groq_client
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
        self.client = get_groq_client()
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
        greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'hola', 'namaste', 'sup', 'yo']
        casual_queries = ['how are you', 'whats up', "what's up", 'how do you do', 'how r u', 'how r you', 'wassup']
        thanks_patterns = ['thanks', 'thank you', 'thx', 'ty', 'appreciate it']
        query_lower = query.lower().strip()
        
        is_greeting = any(query_lower == pattern or query_lower.startswith(pattern + ' ') or query_lower.startswith(pattern + '!') for pattern in greeting_patterns)
        is_casual = any(pattern in query_lower for pattern in casual_queries)
        is_thanks = any(pattern in query_lower for pattern in thanks_patterns)
        
        # Handle greetings naturally — like a real person, not a robot with bullet lists
        if is_greeting and len(query.split()) <= 4:
            import random
            greetings = [
                "Hey! 👋 Good to see you. I'm Daddy's AI — I live and breathe Indian markets. Ask me about any stock, trading concept, or investment idea and I'll dig into it for you.",
                "Hello! 😊 Welcome to Daddy's AI. Whether it's Reliance's latest move or what \"LTP\" means, I'm here for it. What's on your mind?",
                "Hey there! 👋 I'm your financial sidekick — stocks, markets, portfolio ideas, trading concepts — throw anything at me. What would you like to explore?",
            ]
            state["final_response"] = random.choice(greetings)
            state["execution_metadata"] = {"agent": "explainer", "mode": "greeting", "skip_verifier": True}
            return state
        
        # Handle casual queries naturally — warm, human, conversational
        if is_casual:
            state["final_response"] = "Doing great, thanks for asking! 😊 Ready to dive into whatever you need — stocks, market trends, investment strategies, or just explaining any finance concept you're curious about. What's on your mind?"
            state["execution_metadata"] = {"agent": "explainer", "mode": "casual_conversation", "skip_verifier": True}
            return state
        
        # Handle thank you messages
        if is_thanks and len(query.split()) <= 5:
            state["final_response"] = "You're welcome! 😊 Happy to help anytime. Got more questions? Fire away!"
            state["execution_metadata"] = {"agent": "explainer", "mode": "thanks", "skip_verifier": True}
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
        system_prompt = f"""You are Daddy's AI — a brilliant financial teacher who makes complex concepts feel simple. You speak like a knowledgeable friend over coffee, not a textbook.

=== YOUR PERSONALITY ===
- Warm, direct, and confident. You love explaining things.
- You talk like a human — conversational, with personality. Not robotic.
- When someone asks a simple question, give a simple answer. Don't overcomplicate.
- Use analogies from everyday life to explain finance: "Think of PE ratio like the price per slice of pizza 🍕"
- Use ₹ for Indian currency. Reference Indian stocks they'd know: Reliance, TCS, HDFC.

=== CRITICAL: ADAPTIVE FORMATTING (Match Response to Question) ===

**Your formatting MUST match the query complexity:**

1. **Simple explanations** ("What is PE ratio?", "What does LTP mean?"):
   → 2-3 natural paragraphs. NO bullet points. NO headings.
   → Use 1-2 emojis to make concept memorable (💡 for insights, 📊 for metrics)
   → Example: "PE ratio 📊 is basically how much you're paying for each rupee a company earns. If TCS has a PE of 30, it means investors are paying ₹30 for every ₹1 of earnings. A lower PE might mean it's cheaper, but it could also mean the market doesn't expect much growth."

2. **Comparisons** ("Compare TCS vs Infosys", "Top 5 banking stocks"):
   → **Markdown table** with emoji indicators in verdict column (💎 quality, 📈 value, ⚠️ caution)
   → Follow with 2-3 paragraphs of insight
   → Example: | Stock | Price (₹) | PE | 52W High | Verdict |

3. **Complex topics** ("Explain options trading", "How does F&O work?"):
   → Use ## headings with relevant emojis (📚 Basics, 💡 Key Concept, ⚠️ Risks, 🎯 Strategy)
   → Short paragraphs under each heading
   → Use bullet points ONLY if listing 3+ related items

4. **News/Current events** ("What's happening with X?"):
   → Lead with emoji indicator (🔴 negative, 🟢 positive, 🟡 mixed)
   → Then expand with context

**GOLDEN RULE**: Simple query = simple answer (2-3 paragraphs). Complex = structured with headings. Use emojis to make concepts stick.

=== EMOJI USAGE FOR BETTER UNDERSTANDING ===
- Concepts: 💡 (insight), 📊 (metrics/data), 📚 (educational), 🎯 (strategy)
- Direction: 📈 (growth/up), 📉 (decline/down), ➡️ (stable)
- Sentiment: 🟢 (positive), 🔴 (negative), 🟡 (neutral/caution)
- Risk: ⚠️ (warning/risk), 💎 (quality/value), 🔥 (hot topic)
- Use 1-3 emojis per response to enhance clarity and retention

=== WHEN TO USE TOOLS ===

Before answering, ask: "Do I have what I need to answer this ACCURATELY?"

- "latest", "today", "current", "happening" → USE search_web("topic India")
- Financial term NOT in context below → USE search_knowledge_base("term")
- "price of X", "how is X doing" → USE fetch_nse_quote("SYMBOL")
- Company earnings/results → USE search_financial_news("company")
- You have enough context → DON'T use tools, just answer.

=== HOW TO CALL A TOOL ===
Output EXACTLY this format (one tool per response, wait for results):

<tool_call>
{{"tool": "TOOL_NAME", "arguments": {{"param": "value"}}}}
</tool_call>

Tools available:
- search_knowledge_base: {{"query": "search phrase"}}
- search_web: {{"query": "search phrase"}}
- fetch_nse_quote: {{"symbol": "RELIANCE"}} (uppercase)
- search_financial_news: {{"query": "company or topic"}}
- get_stock_fundamentals: {{"symbol": "TCS"}}

=== AFTER GETTING TOOL RESULTS ===
- Synthesize the data into your explanation naturally.
- Cite: "Based on the data..." or "Currently trading at..."
- If tool returned nothing: "I couldn't find specific data, but here's what I know..."
- NEVER fabricate numbers or facts.
- One tool call at a time — wait for results before calling another.

=== PRE-LOADED CONTEXT ===
{", ".join(kb_sources) if kb_sources else "None — use search_knowledge_base or search_web to fetch information"}"""
        
        messages = [{"role": "system", "content": system_prompt}]
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            for msg in conversation_history[-5:]:
                m = dict(msg)
                m.pop("images", None)  # Groq vision models use content array, not images property
                messages.append({"role": m["role"], "content": m.get("content", "")})
        
        user_message = f"{kb_context}\n\n---\nUser Query: {query}\n\nProvide a clear explanation. USE TOOLS if you need more info. CITE SOURCE when using KB or tool results." if kb_context else f"User Query: {query}\n\nProvide a clear explanation. USE TOOLS autonomously if you need more info."
        
        # When images present, use vision model and content array format
        images = state.get("images") or []
        if images:
            logger.info(f"🖼️ Explainer using vision model for {len(images)} image(s)")
            content_parts = [{"type": "text", "text": user_message}]
            for img in images[:5]:  # Max 5 images per Groq
                url = img if img.startswith("data:") else f"data:image/jpeg;base64,{img}"
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_message})
        
        tools_used = []
        max_tool_rounds = 3
        
        # Use vision model when images present (Groq: llama-4-scout or llama-4-maverick)
        model = settings.model_vision if images else self.model
        
        try:
            for round_num in range(max_tool_rounds + 1):
                response = await self.client.chat.completions.create(
                    model=model,
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
