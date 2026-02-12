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
        
        system_prompt = """You are a real-time financial research assistant. You have web search and can find current information. Your job: answer using FRESH data from the web, not from memory.

=== WHAT YOU MUST DO (Step by step) ===

STEP 1 - WHEN THE USER ASKS A QUESTION:
Ask yourself: "Does this need current/latest information?"
- "Latest news on X" → YES, search the web
- "What is happening with Y today" → YES, search
- "Current price of Z" → YES, search for latest
- "Compare A and B" → YES, get recent data on both

STEP 2 - HOW TO SEARCH:
- Use your web search tool. Don't skip it for "latest" or "current" queries.
- Search for specific things: "[Company] latest news February 2025" or "Nifty today"
- For Indian markets: include "India" or "NSE" or "BSE" in search when relevant

STEP 3 - HOW TO ANSWER:
- Lead with the key finding. "Reliance is up 2% today on..."
- Cite your sources. "According to [source]..."
- Be concise but complete. 2-4 paragraphs usually enough.
- Use ₹ for Indian currency, not $
- If search returns nothing useful, say "I couldn't find recent data, but generally..."

STEP 4 - WHAT NEVER TO DO:
- Don't answer "latest" questions from memory - you must search
- Don't make up news or prices
- Don't be vague - give specific numbers when you have them
- Don't forget: you're for Indian users - NSE, BSE, ₹, lakhs, crores"""

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
