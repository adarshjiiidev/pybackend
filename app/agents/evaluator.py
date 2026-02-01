"""
Evaluator Agent - Scores data quality and confidence.
Determines if research should continue or stop.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from ..config.research_config import (
    DATA_QUALITY_WEIGHT,
    DATA_FRESHNESS_WEIGHT,
    AGREEMENT_WEIGHT,
    COMPLETENESS_WEIGHT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MAX_RESEARCH_ITERATIONS
)
from ..models.research_state import ResearchState

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """
    Evaluates research progress and decides whether to continue.
    Scores: quality, freshness, agreement, completeness.
    """
    
    async def evaluate(self, state: ResearchState) -> ResearchState:
        """
        Evaluate gathered data and update confidence scores.
        Decide if loop should continue.
        """
        gathered_data = state["gathered_data"]
        research_plan = state["research_plan"]
        iteration_count = state["iteration_count"]
        
        # Score each dimension
        quality_score = self._score_quality(gathered_data)
        freshness_score = self._score_freshness(gathered_data)
        agreement_score = self._score_agreement(gathered_data)
        completeness_score = self._score_completeness(gathered_data, research_plan)
        
        # Weighted overall confidence
        confidence_score = (
            quality_score * DATA_QUALITY_WEIGHT +
            freshness_score * DATA_FRESHNESS_WEIGHT +
            agreement_score * AGREEMENT_WEIGHT +
            completeness_score * COMPLETENESS_WEIGHT
        )
        
        # Update state
        state["data_quality_score"] = quality_score
        state["data_freshness_score"] = freshness_score
        state["agreement_score"] = agreement_score
        state["completeness_score"] = completeness_score
        state["confidence_score"] = confidence_score
        
        # Decision logic
        should_continue = True
        stop_reason = ""
        
        if confidence_score >= DEFAULT_CONFIDENCE_THRESHOLD:
            should_continue = False
            stop_reason = "confidence_met"
            logger.info(f"Confidence threshold met: {confidence_score:.2f}")
        
        elif iteration_count >= MAX_RESEARCH_ITERATIONS:
            should_continue = False
            stop_reason = "max_iterations"
            logger.info(f"Max iterations reached: {iteration_count}")
        
        elif state["current_step"] >= len(research_plan):
            should_continue = False
            stop_reason = "all_steps_completed"
        
        state["should_continue"] = should_continue
        state["stop_reason"] = stop_reason
        
        logger.info(f"Evaluation: confidence={confidence_score:.2f}, continue={should_continue}")
        return state
    
    def _score_quality(self, data: List[Dict[str, Any]]) -> float:
        """Score data quality (0-1)."""
        if not data:
            return 0.0
        
        # Average confidence from data sources
        confidences = [d.get("data", {}).get("confidence", 0.5) for d in data]
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _score_freshness(self, data: List[Dict[str, Any]]) -> float:
        """Score data freshness (0-1)."""
        if not data:
            return 0.0
        
        # Check timestamps - newer is better
        now = datetime.utcnow()
        scores = []
        
        for item in data:
            try:
                timestamp = datetime.fromisoformat(item.get("timestamp", now.isoformat()))
                age_hours = (now - timestamp).total_seconds() / 3600
                
                # Decay function: 1.0 for <1hr, 0.5 for 24hr, 0.0 for >7 days
                if age_hours < 1:
                    scores.append(1.0)
                elif age_hours < 24:
                    scores.append(0.8)
                elif age_hours < 168:  # 7 days
                    scores.append(0.5)
                else:
                    scores.append(0.2)
            except:
                scores.append(0.5)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _score_agreement(self, data: List[Dict[str, Any]]) -> float:
        """Score agreement between sources (0-1)."""
        if len(data) < 2:
            return 0.7  # Single source - moderate confidence
        
        # Simplified: check if multiple sources exist
        sources = set(d.get("source") for d in data)
        
        # More diverse sources = higher agreement score
        if len(sources) >= 3:
            return 0.9
        elif len(sources) == 2:
            return 0.75
        else:
            return 0.6
    
    def _score_completeness(self, data: List[Dict[str, Any]], plan: List) -> float:
        """Score completeness of research (0-1)."""
        if not plan:
            return 0.5
        
        completed = sum(1 for step in plan if step["status"] == "completed")
        total = len(plan)
        
        return completed / total if total > 0 else 0.0
