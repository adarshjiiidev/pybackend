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
        self.temperature = 0.3  # LOW temperature to prevent hallucination
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
        """Check if query contains domain-specific keywords."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.domain_keywords)
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Provide educational explanations with conversation memory and knowledge base usage.
        """
        query = state["query"]
        
        # Check if this is a domain-specific query
        is_domain_query = self._check_domain_query(query)
        
        # ALWAYS get knowledge base context for domain queries
        kb_context = ""
        if is_domain_query:
            kb_context = self.kb_rag.get_relevant_context(query, max_chars=2500)
            logger.info(f"Retrieved knowledge base context for domain query: {len(kb_context)} chars")
        
        system_prompt = """You are the educational module of Daddys AI, a financial intelligence system built by Adarsh, a 14-year-old student at Daddys International School.

**Your Mission:** Make complex finance simple and accessible for Indian retail investors.

**CRITICAL RULES:**
1. For domain-specific terms (WTB, WTT, LTP, shifting, etc.), ONLY use the Knowledge Base Context provided below
2. NEVER make up definitions for technical terms
3. If Knowledge Base Context is provided, cite it directly and accurately
4. If you don't have verified information, say "I need to search the knowledge base for this"
5. For follow-up questions like "summarize it", refer to the conversation history

**Teaching Style:**
- Use simple language and real-world analogies
- Provide examples with Indian stocks (Reliance, TCS, HDFC, etc.)
- Break down concepts step-by-step
- Use ₹ for Indian currency
- Be conversational yet professional
- Remember context from previous messages in this conversation
- Add disclaimers only when giving specific trading recommendations

**Example - WTB (CORRECT):**
- WTB = Weak Towards Bottom
- It's when second highest OI is >=75% at BOTTOM side
- Related to Option Chain analysis and market pressure

If Knowledge Base Context is missing for a domain term, acknowledge it instead of guessing."""

        # Build messages with conversation history for context retention
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            logger.info(f"Explainer: Including {len(conversation_history)} messages for context")
            messages.extend(conversation_history[-10:])  # Last 10 for context
        
        # Add current query with KB context if available
        if kb_context:
            user_message = f"{kb_context}\n\nBased on the above knowledge base information, {query}"
        else:
            user_message = query
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens
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
