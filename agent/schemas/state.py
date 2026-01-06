from typing import List, Dict, Any, Optional, TypedDict
from langchain_core.messages import BaseMessage
from agent.schemas.models import FinancialQuery, MarketMetrics

class AgentState(TypedDict):
    """
    State dictionary for the financial agent graph.
    """
    messages: List[BaseMessage]
    parsed_query: Optional[FinancialQuery]
    fetched_data: Optional[Dict[str, Any]]  # Raw data from tools
    normalized_metrics: Optional[List[MarketMetrics]]
    retrieved_docs: Optional[List[str]] # Context from Pinecone
    analysis_result: Optional[Dict[str, Any]] # Intermediate reasoning results
    final_response: Optional[str] # Final formatted string
    error: Optional[str]
