"""
Planner Agent - Breaks queries into research steps.
Creates structured research plans for autonomous loops.
"""

from groq import AsyncGroq
import logging
import json
from datetime import datetime

from ..config import settings, ModelType
from ..config.research_config import MAX_RESEARCH_ITERATIONS
from ..models.research_state import ResearchState, ResearchStep

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Plans multi-step research for complex queries.
    Determines what data sources to use and in what order.
    """
    
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.get_model_for_task(ModelType.REASONING_DEEP)
        self.temperature = 0.4  # Low for structured planning
        self.max_tokens = 2048
    
    async def create_plan(self, state: ResearchState) -> ResearchState:
        """
        Create a structured research plan from the user query.
        
        Returns updated state with research_plan populated.
        """
        query = state["query"]
        
        system_prompt = """You are a research planner for financial market analysis.

Break down complex queries into 2-4 focused research steps.

Available data sources:
- market_data: Stock prices, technical indicators, fundamentals
- knowledge_base: WTB/WTT rules, LTP calculator, trading concepts
- web_search: Latest news, real-time events
- crypto_narrative: Crypto-specific analysis

Output JSON:
{
  "steps": [
    {
      "step_number": 1,
      "question": "specific sub-question",
      "data_source": "market_data|knowledge_base|web_search|crypto_narrative",
      "rationale": "why this step"
    }
  ],
  "estimated_confidence": 0.8
}

Keep it focused - quality over quantity."""

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
