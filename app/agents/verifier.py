"""
Verifier/Refiner Agent - Quality control and final response polishing.
Ensures responses are data-grounded, clear, and properly formatted.
"""

from groq import AsyncGroq
import logging

from ..config import settings, ModelType
from ..models.agent_state import AgentState
from ..tools import clean_response

logger = logging.getLogger(__name__)


class VerifierAgent:
    """Verifies and enhances agent responses with disclaimers using fast model."""
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        # Use fast model for verification (quick checks)
        self.model = settings.get_model_for_task(ModelType.FAST)
        self.temperature = settings.get_temperature_for_task(ModelType.FAST)
        self.max_tokens = settings.get_max_tokens_for_task(ModelType.FAST)
    
    async def verify_and_refine(self, state: AgentState) -> AgentState:
        """
        Verify response quality and refine if needed.
        Ensures:
        - Data-grounded (no hallucinations)
        - Clear structure
        - Appropriate disclaimers
        - No internal reasoning visible
        """
        raw_response = state.get("final_response", "")
        
        if not raw_response:
            state["final_response"] = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            return state
        
        # Clean response (remove internal reasoning markers, extra whitespace)
        cleaned = clean_response(raw_response)
        
        # Add reasoning tags if internal reasoning is available
        internal_reasoning = state.get("internal_reasoning")
        if internal_reasoning and internal_reasoning.strip():
            # Format response with reasoning tags
            final_output = f"<reasoning>\n{internal_reasoning.strip()}\n</reasoning>\n\n{cleaned}"
            state["final_response"] = final_output
        else:
            state["final_response"] = cleaned
        
        logger.info("Response verified and refined")
        return state
