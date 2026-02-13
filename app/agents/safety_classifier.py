"""
Safety Classifier Agent - Pre-check risk detection.
Classifies queries by risk level and determines allowed response types.
"""

from groq import AsyncGroq
import logging
import json
from typing import Literal

from ..config import settings, ModelType
from ..config.key_rotator import get_groq_client
from ..models.agent_state import AgentState

logger = logging.getLogger(__name__)


class SafetyClassifierAgent:
    """
    Classifies query risk level BEFORE research begins.
    
    Risk Levels:
    - LOW: General education, market overview
    - MEDIUM: Specific analysis with uncertainty
    - HIGH: Personal advice, price predictions, buy/sell instructions
    
    Determines allowed response type and modifies agent behavior.
    """
    
    def __init__(self):
        self.client = get_groq_client()
        self.model = settings.get_model_for_task(ModelType.FAST)
        self.temperature = 0.2  # Low for consistent safety classification
        self.max_tokens = 200
    
    async def classify_risk(self, state: AgentState) -> AgentState:
        """
        Classify query risk level and set response constraints.
        """
        query = state["query"]
        
        system_prompt = """You are a safety classifier for financial queries. Your job: figure out how risky the user's question is, so we can respond appropriately. Think step by step.

=== STEP 1: READ THE QUERY ===
What is the user asking? Write it in your own words.

=== STEP 2: ASK THESE QUESTIONS ===

QUESTION A - Is the user asking for PERSONAL ADVICE they might act on?
- "Should I buy X?" - YES
- "What stock should I invest in?" - YES
- "When should I enter/exit?" - YES
- "Will X reach ₹Y?" - YES (price prediction)
- If YES to any → risk_level = HIGH, allowed_response_type = "education_only"

QUESTION B - Is the user asking for SPECIFIC ANALYSIS that could influence a decision?
- "Analyze TCS fundamentals" - YES
- "Compare Reliance vs TCS" - YES
- "Is Infosys overvalued?" - YES
- If YES but NOT personal advice → risk_level = MEDIUM, allowed_response_type = "analysis_with_disclaimer"

QUESTION C - Is the user asking for GENERAL LEARNING?
- "What is PE ratio?" - YES
- "How does options trading work?" - YES
- "Explain support and resistance" - YES
- If YES → risk_level = LOW, allowed_response_type = "general_info"

=== STEP 3: PICK risk_level ===
- HIGH: Personal advice, buy/sell suggestions, price predictions, "should I"
- MEDIUM: Stock analysis, comparisons, valuations (informational)
- LOW: Definitions, concepts, how things work (educational)

=== STEP 4: OUTPUT JSON ===
{
  "risk_level": "LOW" or "MEDIUM" or "HIGH",
  "risk_factors": ["list", "why", "it's", "this", "level"],
  "allowed_response_type": "education_only" or "analysis_with_disclaimer" or "general_info",
  "rationale": "One sentence: why you chose this level."
}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify: {query}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            safety_data = json.loads(response.choices[0].message.content)
            
            risk_level = safety_data.get("risk_level", "MEDIUM")
            
            # Update state with safety metadata
            state["safety_classification"] = {
                "risk_level": risk_level,
                "risk_factors": safety_data.get("risk_factors", []),
                "allowed_response_type": safety_data.get("allowed_response_type", "analysis_with_disclaimer"),
                "rationale": safety_data.get("rationale", "")
            }
            
            # Set response constraints
            if risk_level == "HIGH":
                state["response_mode"] = "education_only"
                state["enable_buy_sell_language"] = False
            elif risk_level == "MEDIUM":
                state["response_mode"] = "analysis_with_disclaimer"
                state["enable_buy_sell_language"] = False
            else:
                state["response_mode"] = "general_info"
                state["enable_buy_sell_language"] = False
            
            logger.info(f"Safety classification: {risk_level}, Mode: {state['response_mode']}")
            return state
            
        except Exception as e:
            logger.error(f"Safety classifier error: {e}")
            # Default to MEDIUM risk on error (conservative)
            state["safety_classification"] = {
                "risk_level": "MEDIUM",
                "risk_factors": ["classification_error"],
                "allowed_response_type": "analysis_with_disclaimer",
                "rationale": "Error in classification, defaulting to safe mode"
            }
            state["response_mode"] = "analysis_with_disclaimer"
            state["enable_buy_sell_language"] = False
            return state
