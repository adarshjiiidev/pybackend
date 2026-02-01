"""
Research State Model
Extended state for autonomous research loops with confidence tracking.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class ResearchStep(TypedDict):
    """Single research step in the plan."""
    step_number: int
    question: str
    data_source: str  # 'market_data', 'knowledge_base', 'web_search', 'crypto_narrative'
    status: str  # 'pending', 'completed', 'failed'
    result: Optional[Dict[str, Any]]
    confidence: float


class ResearchState(TypedDict):
    """State for autonomous research loops."""
    # Original query
    query: str
    session_id: str
    
    # Research planning
    research_plan: List[ResearchStep]
    current_step: int
    
    # Gathered data
    gathered_data: List[Dict[str, Any]]
    
    # Confidence tracking
    confidence_score: float
    data_quality_score: float
    data_freshness_score: float
    agreement_score: float
    completeness_score: float
    
    # Loop control
    iteration_count: int
    should_continue: bool
    stop_reason: str  # 'confidence_met', 'max_iterations', 'timeout', 'error'
    
    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Final output
    final_answer: str
    key_findings: List[str]
    risks_uncertainties: List[str]
    data_freshness_indicator: str
    
    # Metadata
    execution_metadata: Dict[str, Any]
