"""
Validation Agent - Post-check fact validation.
Scans draft responses for hallucinations, unsourced claims, and overconfidence.
"""

from groq import AsyncGroq
import logging
import re
from typing import List, Dict, Any

from ..config import settings, ModelType
from ..models.agent_state import AgentState

logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Validates final responses for:
    - Fabricated numbers
    - Unsourced claims
    - Overconfident language
    - Missing timestamps
    
    Auto-modifies responses to be safer.
    """
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.get_model_for_task(ModelType.FAST)
        self.temperature = 0.1
        self.max_tokens = 1000
        
        # Patterns for validation
        self.risky_patterns = [
            (r"\bwill\s+(reach|hit|go\s+to|be)\s+[\d,]+", "price_prediction"),
            (r"\b(definitely|certainly|guaranteed|100%)\b", "overconfidence"),
            (r"\b(buy|sell|invest\s+in)\s+\w+\s+now\b", "direct_advice"),
            (r"\btarget\s+price[:\s]+[\d,]+", "price_target"),
        ]
    
    async def validate(self, state: AgentState) -> AgentState:
        """
        Validate and sanitize final response.
        """
        draft_response = state.get("final_response", "")
        risk_level = state.get("safety_classification", {}).get("risk_level", "MEDIUM")
        response_mode = state.get("response_mode", "analysis_with_disclaimer")
        
        # Run validation checks
        violations = self._detect_violations(draft_response)
        
        if violations:
            logger.warning(f"Validation violations detected: {violations}")
            
            # Auto-sanitize response
            sanitized_response = self._sanitize_response(
                draft_response, 
                violations, 
                risk_level,
                response_mode
            )
            
            state["final_response"] = sanitized_response
            state["validation_violations"] = violations
            state["response_sanitized"] = True
        else:
            state["validation_violations"] = []
            state["response_sanitized"] = False
         
        # Add appropriate disclaimer based on risk level
        state["final_response"] = self._add_disclaimer(
            state["final_response"],
            risk_level,
            response_mode
        )
        
        logger.info(f"Validation complete. Violations: {len(violations)}, Sanitized: {state['response_sanitized']}")
        return state
    
    def _detect_violations(self, text: str) -> List[Dict[str, Any]]:
        """Detect validation violations."""
        violations = []
        
        for pattern, violation_type in self.risky_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                violations.append({
                    "type": violation_type,
                    "text": match.group(0),
                    "position": match.span()
                })
        
        # Check for numeric claims without context
        number_pattern = r"\b(P/E|PE|EPS|price|value)\s*[:\s]+[\d,.]+"
        numbers = re.findall(number_pattern, text, re.IGNORECASE)
        if numbers:
            # Check if "as of" or "source" appears nearby
            if not re.search(r"\b(as of|source|according to|dated)\b", text, re.IGNORECASE):
                violations.append({
                    "type": "unsourced_numbers",
                    "text": str(numbers),
                    "position": None
                })
        
        return violations
    
    def _sanitize_response(
        self, 
        text: str, 
        violations: List[Dict], 
        risk_level: str,
        response_mode: str
    ) -> str:
        """Auto-sanitize response by replacing risky language."""
        
        sanitized = text
        
        # Replace overconfident language
        sanitized = re.sub(
            r"\b(definitely|certainly|guaranteed|100%)\b",
            "potentially",
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Soften "will" predictions
        sanitized = re.sub(
            r"\bwill\s+(reach|hit|go\s+to)",
            r"may \1",
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Remove direct buy/sell language if risk is HIGH
        if risk_level == "HIGH" or response_mode == "education_only":
            sanitized = re.sub(
                r"\b(you should|I recommend you)\s+(buy|sell|invest)",
                r"one could consider",
                sanitized,
                flags=re.IGNORECASE
            )
        
        # Add uncertainty markers
        if "price target" in str(violations).lower():
            sanitized = sanitized.replace("Target price:", "Potential price range:")
        
        return sanitized
    
    def _add_disclaimer(self, text: str, risk_level: str, response_mode: str) -> str:
        """Add appropriate disclaimer based on risk level."""
        
        # Check if disclaimer already exists
        if "not financial advice" in text.lower() or "disclaimer:" in text.lower():
            return text
        
        if response_mode == "education_only" or risk_level == "HIGH":
            disclaimer = "\n\n**IMPORTANT NOTICE:** This is educational information only, not personalized financial advice. Investment decisions should be made after consulting with a SEBI-registered financial advisor. Past performance does not guarantee future results."
        elif risk_level == "MEDIUM":
            disclaimer = "\n\n**Disclaimer:** This analysis is for informational purposes only and not financial advice. Please conduct your own research and consult professionals before making investment decisions."
        else:
            disclaimer = "\n\n**Note:** This is general information for educational purposes."
        
        return text + disclaimer
