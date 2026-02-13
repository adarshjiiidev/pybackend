"""
Researcher Agent - Executes research steps iteratively.
Gathers data from various sources based on the plan.
"""

from groq import AsyncGroq
import logging
from typing import Dict, Any
from datetime import datetime

from ..config import settings, ModelType
from ..config.key_rotator import get_groq_client
from ..models.research_state import ResearchState
from ..tools import execute_tool
from ..rag import get_kb_rag

logger = logging.getLogger(__name__)


class ResearcherAgent:
    """
    Executes ONE research step per iteration.
    Calls appropriate tools based on data_source in the plan.
    """
    
    def __init__(self):
        self.client = get_groq_client()
        self.model = settings.get_model_for_task(ModelType.ANALYSIS)
        self.temperature = 0.5
        self.max_tokens = 3000
        self.kb_rag = get_kb_rag()
    
    async def execute_step(self, state: ResearchState) -> ResearchState:
        """
        Execute the current research step and append results to gathered_data.
        """
        current_idx = state["current_step"]
        research_plan = state["research_plan"]
        
        if current_idx >= len(research_plan):
            state["should_continue"] = False
            state["stop_reason"] = "all_steps_completed"
            return state
        
        step = research_plan[current_idx]
        question = step["question"]
        data_source = step["data_source"]
        
        logger.info(f"Executing step {current_idx + 1}: {question} from {data_source}")
        
        try:
            result = await self._fetch_data(question, data_source)
            
            # Update step status
            step["status"] = "completed"
            step["result"] = result
            step["confidence"] = result.get("confidence", 0.5)
            
            # Append to gathered data
            state["gathered_data"].append({
                "step_number": step["step_number"],
                "question": question,
                "source": data_source,
                "data": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Move to next step
            state["current_step"] += 1
            state["iteration_count"] += 1
            
            logger.info(f"Step {current_idx + 1} completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Research step {current_idx + 1} failed: {e}")
            step["status"] = "failed"
            step["confidence"] = 0.0
            state["current_step"] += 1
            state["iteration_count"] += 1
            return state
    
    async def _fetch_data(self, question: str, source: str) -> Dict[str, Any]:
        """Fetch data from the specified source."""
        
        if source == "knowledge_base":
            # Search knowledge base
            results = self.kb_rag.search(question, top_k=2)
            return {
                "source": "knowledge_base",
                "results": [
                    {
                        "title": r["title"],
                        "content": r["content"][:500]
                    }
                    for r in results
                ],
                "confidence": 0.9 if results else 0.3
            }
        
        elif source == "market_data":
            # Extract symbols and call market data tools
            # Simplified - you'd parse the question for symbols
            return {
                "source": "market_data",
                "data": "market data placeholder",
                "confidence": 0.7
            }
        
        elif source == "web_search":
            # Use Compound AI or web search tool
            return {
                "source": "web_search",
                "data": "web search placeholder",
                "confidence": 0.6
            }
        
        else:
            return {
                "source": source,
                "data": "generic data",
                "confidence": 0.5
            }
