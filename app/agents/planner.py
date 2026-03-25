"""
Planner Agent - Breaks queries into research steps.
Creates structured research plans for autonomous loops.
"""

from groq import AsyncGroq
import logging
import json
from datetime import datetime
from typing import List

from ..config import settings, ModelType
from ..config.research_config import MAX_RESEARCH_ITERATIONS
try:
    from ..config.openrouter_client import get_openrouter_client
    _HAS_OPENROUTER = True
except ImportError:
    _HAS_OPENROUTER = False

from ..config.key_rotator import get_groq_client
from ..models.research_state import ResearchState, ResearchStep

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Plans multi-step research for complex queries.
    Determines what data sources to use and in what order.
    """
    
    def __init__(self):
        # OpenRouter first priority, Groq fallback
        if _HAS_OPENROUTER and settings.openrouter_available:
            from ..config.openrouter_client import get_openrouter_client as _get_or
            self.client = _get_or()
            self._provider = "openrouter"
        else:
            self.client = get_groq_client()
            self._provider = "groq"
        if hasattr(self, '_provider') and self._provider == "openrouter":
            self.model = settings.get_openrouter_model(ModelType.REASONING_DEEP)
        else:
            self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = 0.4  # Low for structured planning
        self.max_tokens = 2048
    
    async def create_plan(self, state: ResearchState) -> ResearchState:
        """
        Create a structured research plan from the user query.
        
        Returns updated state with research_plan populated.
        """
        query = state["query"]
        
        system_prompt = """You are a research planner. Your job: take a complex question and break it into 2-4 smaller steps that we can research one by one. Think like a child breaking a big project into homework steps.

=== STEP 1: UNDERSTAND THE MAIN QUESTION ===
What does the user really want to know? Write it in one sentence.

=== STEP 2: WHAT INFORMATION DO WE NEED? ===
To answer fully, what do we need to find out? List 2-4 sub-questions.
Example: "Compare TCS vs Infosys" needs:
- Sub-question 1: What are TCS's key metrics? (market_data)
- Sub-question 2: What are Infosys's key metrics? (market_data)
- Sub-question 3: What's the latest news on both? (web_search)

=== STEP 3: PICK THE RIGHT DATA SOURCE FOR EACH STEP ===
- market_data: Stock prices, PE, technicals, fundamentals. Use when: "What is X's price?", "Compare financials"
- knowledge_base: Trading concepts, WTB, LTP, strategies. Use when: "Explain WTB", "What is LTP calculator"
- web_search: News, current events, "what's happening". Use when: "Latest on X", "Recent developments"
- crypto_narrative: Crypto-specific. Use when: "Bitcoin", "ETH", "crypto market"

=== STEP 4: ORDER THE STEPS ===
- Usually: get the data first (market_data, web_search), then analyze
- Or: understand the concept first (knowledge_base), then apply

=== STEP 5: OUTPUT JSON ===
{
  "steps": [
    {
      "step_number": 1,
      "question": "Exact sub-question we will research",
      "data_source": "market_data" or "knowledge_base" or "web_search" or "crypto_narrative",
      "rationale": "Why we need this step"
    }
  ],
  "estimated_confidence": 0.0 to 1.0
}

Keep it 2-4 steps. Quality over quantity. Each step should be one clear question."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create research plan for: {query}"}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            plan_json = json.loads(response.choices[0].message.content)
            
            # Convert to ResearchStep format
            research_steps: List[ResearchStep] = []
            for step in plan_json.get("steps", [])[:MAX_RESEARCH_ITERATIONS]:
                research_steps.append({
                    "step_number": step["step_number"],
                    "question": step["question"],
                    "data_source": step["data_source"],
                    "status": "pending",
                    "result": None,
                    "confidence": 0.0
                })
            
            state["research_plan"] = research_steps
            state["current_step"] = 0
            state["iteration_count"] = 0
            state["should_continue"] = True
            state["started_at"] = datetime.utcnow()
            
            logger.info(f"Created research plan with {len(research_steps)} steps")
            return state
            
        except Exception as e:
            logger.error(f"Planner error: {e}")
            state["should_continue"] = False
            state["stop_reason"] = "planning_error"
            return state
