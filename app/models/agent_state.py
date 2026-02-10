"""
LangGraph state models for agent orchestration.
Defines the shared state that flows through all agent nodes.
"""

from typing import TypedDict, Literal, Optional, Any
from enum import Enum


class AgentMode(str, Enum):
    """Available agent operation modes."""
    AUTO = "auto"
    MARKET_RESEARCH = "market_research"
    REALTIME_ANALYSIS = "realtime_analysis"
    PORTFOLIO = "portfolio"
    EXPLAINER = "explainer"
    CRYPTO = "crypto"


class AgentState(TypedDict):
    """
    Shared state for LangGraph workflow.
    This state is passed through all agent nodes and accumulates context.
    """
    
    # User input
    query: str
    images: Optional[list[str]]  # Base64 encoded images
    mode: AgentMode
    session_id: str
    
    # Routing decisions
    selected_mode: Optional[str]
    extracted_entities: Optional[dict[str, Any]]  # symbols, timeframes, etc.
    needs_web_search: Optional[bool]  # AI-driven decision for web search
    enable_research_loop: Optional[bool]  # Deep research flag
    has_vision_content: Optional[bool]  # Has images flag
    
    # Conversation context
    conversation_history: list[dict[str, str]]  # [{role, content}, ...]
    
    # Tool results
    tool_results: Optional[dict[str, Any]]  # Market data, calculations, etc.
    
    # Agent reasoning (internal, not shown to user)
    internal_reasoning: Optional[str]
    
    # Final output
    final_response: Optional[str]
    
    # Metadata
    execution_metadata: Optional[dict[str, Any]]  # Timings, model used, etc.
    error: Optional[str]
