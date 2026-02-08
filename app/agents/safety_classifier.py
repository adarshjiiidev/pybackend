"""
Safety Classifier Agent - Pre-check risk detection.
Classifies queries by risk level and determines allowed response types.
"""

from groq import AsyncGroq
import logging
import json
from typing import Literal

from ..config import settings, ModelType
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
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.get_model_for_task(ModelType.FAST)
        self.temperature = 0.2  # Low for consistent safety classification
        self.max_tokens = 200
    
    async def classify_risk(self, state: AgentState) -> AgentState:
        """
        Classify query risk level and set response constraints.
        """
        query = state["query"]
        
        system_prompt = """You are a financial safety classifier.

Classify queries into risk levels:

**HIGH RISK** (personal advice, predictions, buy/sell):
- "Should I buy/sell X?"
- "What stock should I invest in?"
- "Will X reach Y price?"
- "When should I enter/exit?"

**MEDIUM RISK** (specific analysis with uncertainty):
- "Analyze X stock fundamentals"
- "Compare X vs Y"
- "Is X overvalued?"

**LOW RISK** (education, general info):
- "What is P/E ratio?"
- "How does market work?"
- "Explain technical analysis"

Output JSON:
{
  "risk_level": "LOW|MEDIUM|HIGH",
  "risk_factors": ["factor1", "factor2"],
  "allowed_response_type": "education_only|analysis_with_disclaimer|general_info",
  "rationale": "brief explanation"
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
