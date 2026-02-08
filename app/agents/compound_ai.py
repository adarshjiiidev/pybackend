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
        Use Compound AI for real-time queries.
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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            message = response.choices[0].message
            
            # Compound automatically executes tools - all done in single call
            state["final_response"] = message.content
            state["execution_metadata"] = {
                "agent": "compound_ai",
                "model": self.model,
                "executed_tools": getattr(message, "executed_tools", [])
            }
            
            logger.info(f"Compound AI completed with tools: {getattr(message, 'executed_tools', [])}")
            return state
            
        except Exception as e:
            logger.error(f"Compound AI error: {e}")
            state["error"] = str(e)
            state["final_response"] = f"I encountered an error fetching real-time data. Error: {str(e)}"
            return state
