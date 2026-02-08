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
        """
        query = state["query"]
        from ..tools.financial_terms import is_financial_term, get_term_definition
        
        # Check if this is a known financial term or domain-specific query
        is_known_financial_term = is_financial_term(query)
        is_domain_query = self._check_domain_query(query)
        
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
                logger.warning(f"⚠️ No KB results for financial term: {query}")
        
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
