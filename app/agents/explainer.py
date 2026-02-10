"""
Explainer Agent - Provides clear, educational explanations.
Forces knowledge base usage for domain-specific terms to prevent hallucination.
Includes conversation history for context retention.
"""

from groq import AsyncGroq
import logging

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..rag import get_kb_rag

logger = logging.getLogger(__name__)


class ExplainerAgent:
    """
    Provides clear, educational explanations using knowledge base.
    ALWAYS searches knowledge base for domain-specific terms to prevent hallucination.
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
        Provide educational explanations with conversation memory and knowledge base usage.
        Uses AI-driven decision from router to determine if web search is needed.
        """
        query = state["query"]
        from ..tools.financial_terms import is_financial_term, get_term_definition
        
        # Check if router AI decided web search is needed
        needs_web_search = state.get("needs_web_search", False)
        
        # If AI determined web search is needed, use browser search
        if needs_web_search:
            logger.info(f"🌐 AI detected web search needed for: {query}")
            try:
                from ..tools.browser_search import browser_search_general
                search_result = await browser_search_general(query)
                
                system_prompt = """You are Daddys AI's news explainer. Present current information clearly and concisely.
                
Format your response:
1. Start with a brief summary
2. Key points in bullet form  
3. Provide context if needed
4. Keep it accurate and factual

Use the web search results provided to give up-to-date information."""
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Web Search Results:\n{search_result}\n\nUser Query: {query}\n\nProvide a clear, informative response based on these search results."}
                ]
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=self.max_tokens
                )
                
                state["final_response"] = response.choices[0].message.content
                state["execution_metadata"] = {
                    "agent": "explainer",
                    "used_browser_search": True,
                    "is_current_events": True
                }
                logger.info(f"✅ Current events response generated using browser search")
                return state
                
            except Exception as e:
                logger.error(f"Browser search failed: {e}, falling back to standard explanation")
                # Fall through to standard explanation
        
        # Check if this is a known financial term or domain-specific query  
        is_known_financial_term = is_financial_term(query) and not needs_web_search
        is_domain_query = self._check_domain_query(query) and not needs_web_search
        
        # ALWAYS search knowledge base for financial terms and domain queries
        kb_context = ""
        kb_sources = []
        
        if is_known_financial_term or is_domain_query:
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
            else:
                # No KB results - automatically use web search instead of saying "I need to search"
                logger.info(f"📚 No KB results for '{query}' - automatically using web search")
                try:
                    from ..tools.browser_search import browser_search_general
                    search_result = await browser_search_general(query)
                    
                    system_prompt = """You are Daddys AI's explainer. Provide clear, educational explanations based on web search results.
                    
Format your response:
1. Start with a brief definition/explanation
2. Break down key concepts
3. Provide examples if relevant
4. Keep it simple and educational

Use the web search results provided to give accurate information."""
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Web Search Results:\n{search_result}\n\nUser Query: {query}\n\nProvide a clear, educational explanation based on these search results."}
                    ]
                    
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=self.max_tokens
                    )
                    
                    state["final_response"] = response.choices[0].message.content
                    state["execution_metadata"] = {
                        "agent": "explainer",
                        "used_browser_search": True,
                        "fallback_from_kb": True
                    }
                    logger.info(f"✅ Auto web search successful for unknown term")
                    return state
                    
                except Exception as e:
                    logger.error(f"Auto web search failed: {e}")
                    # Will fall through to normal response without KB context
        
        system_prompt = f"""You are the educational module of Daddys AI, a financial intelligence system built by Adarsh, a 14-year-old student at Daddys International School.

**Your Mission:** Make complex finance simple and accessible for Indian retail investors.

**CRITICAL RULES:**
1. **ALWAYS cite the source file** when answering from Knowledge Base
2. For domain-specific terms (WTB, WTT, LTP, etc.), ONLY use the Knowledge Base Context provided
3. **Format:** Start your response with "📖 Source: [filename]" when using KB
4. NEVER make up definitions for technical terms
5. If you don't have verified information, say "I need to search the knowledge base for this"
6. For follow-up questions like "summarize it", refer to the conversation history

**Teaching Style:**
- Use simple language and real-world analogies
- Provide examples with Indian stocks (Reliance, TCS, HDFC, etc.)
- Break down concepts step-by-step
- Use ₹ for Indian currency
- Be conversational yet professional
- Remember context from previous messages in this conversation

**Knowledge Base Sources Available:**
{", ".join(kb_sources) if kb_sources else "None"}

If Knowledge Base Context is missing for a domain term, acknowledge it instead of guessing."""

        # Build messages with conversation history for context retention
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
            user_message = f"{kb_context}\n\n---\nUser Query: {query}\n\nProvide a clear explanation and CITE THE SOURCE FILE."
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            explanation = response.choices[0].message.content
            
            # Ensure we never return the system prompt itself
            if explanation and len(explanation) > 50:  # Valid response
                state["final_response"] = explanation
            else:
                state["final_response"] = "I apologize, I couldn't generate a proper explanation. Could you rephrase your question?"
            
            state["execution_metadata"] = {
                "agent": "explainer",
                "model": self.model,
                "temperature": self.temperature,
                "used_knowledge_base": bool(kb_context),
                "kb_context_length": len(kb_context) if kb_context else 0,
                "conversation_history_length": len(conversation_history)
            }
            
            logger.info(f"Explanation generated with KB usage: {bool(kb_context)}, history: {len(conversation_history)} msgs")
            return state
            
        except Exception as e:
            logger.error(f"Explainer agent error: {e}")
            state["error"] = str(e)
            state["final_response"] = f"I encountered an error while generating the explanation. Please try again."
            return state
