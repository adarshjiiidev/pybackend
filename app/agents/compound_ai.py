"""
Compound AI Agent - Uses Groq's built-in tools for real-time web search.
Perfect for news, real-time data, and calculations.
"""

from groq import AsyncGroq
import logging

from ..config import settings, ModelType
from ..models.agent_state import AgentState

logger = logging.getLogger(__name__)


class CompoundAgent:
    """
    Groq Compound AI with built-in web search, browser, and code execution.
    Single API call gets you a complete agentic response with real-time data.
    """
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.get_model_for_task(ModelType.COMPOUND)
        self.temperature = settings.get_temperature_for_task(ModelType.COMPOUND)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.COMPOUND)
    
    async def analyze(self, state: AgentState) -> AgentState:
        """
        Use Compound AI for real-time queries with web search.
        Automatically uses web search when needed - perfect for beating Perplexity.
        """
        query = state["query"]
        
        system_prompt = """You are a real-time financial intelligence agent with web access.

Your capabilities:
- Real-time web search for latest news and market data
- Access to current events and breaking news
- Ability to provide citations and sources
- Code execution for calculations

When answering:
1. Use web search for latest information (news, prices, events)
2. Provide citations with sources
3. Focus on accuracy and timeliness
4. Be comprehensive but concise

You're optimized for Indian markets - use NSE/BSE data and ₹ currency."""

        try:
            # AGGRESSIVE truncation for Compound AI to prevent 413 errors
            # Strategy: Only keep 1 most recent exchange + strip images (but keep full message content)
            conversation_history = state.get("conversation_history", [])
            
            # Build minimal history - last 2 messages only (1 exchange)
            filtered_history = []
            for msg in conversation_history[-2:]:  # Only last 2 messages (1 exchange)
                # Strip images completely but keep full message content
                filtered_msg = {
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", ""))  # Full content, no truncation
                }
                filtered_history.append(filtered_msg)
            
            # Build minimal payload with full query
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(filtered_history)
            messages.append({"role": "user", "content": query})  # Full query, no truncation
            
            logger.info(f"🔥 Compound AI minimal: {len(filtered_history)} history msgs, no images, full content")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            message = response.choices[0].message
            content = message.content if message.content else ""
            
            # Clean up any raw function calls that might appear
            # Compound AI should execute tools automatically, but sometimes shows the call
            if "<function>" in content or "</function>" in content:
                # Extract only text after function calls
                import re
                logger.warning("Raw function calls detected in response, cleaning up...")
                
                # Remove function call blocks
                content = re.sub(r'<function>.*?</function>', '', content, flags=re.DOTALL)
                # Clean up extra whitespace
                content = re.sub(r'\n{3,}', '\n\n', content.strip())
                
                if not content or len(content) < 10:
                    # If nothing left after cleaning, provide fallback
                    content = "I've searched for the latest information on your query. Could you please rephrase or provide more details so I can give you a better response?"
                    logger.warning("Compound AI response was mostly function calls, using fallback")
            
            state["final_response"] = content
            state["execution_metadata"] = {
                "agent": "compound_ai",
                "model": self.model,
                "executed_tools": getattr(message, "executed_tools", []),
                "truncation_level": "ultra_minimal",  # Only 1 exchange, no images
                "original_history_size": len(conversation_history)
            }
            
            logger.info(f"✅ Compound AI completed successfully with {len(content)} chars")
            return state
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"❌ Compound AI error: {error_message}")
            
            # Provide user-friendly error messages
            if "413" in error_message or "too large" in error_message.lower():
                state["error"] = "Payload too large"
                state["final_response"] = "The conversation history is too long. Please start a new conversation for the best results."
            else:
                state["error"] = error_message
                state["final_response"] = f"I encountered an error fetching real-time data. Please try again. Error: {error_message}"
            
            return state
