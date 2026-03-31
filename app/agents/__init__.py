"""Agent implementations for Daaddys AI."""

from .meta_reasoning import DeepReasoningAgent
from .compound_ai import CompoundAgent
from .router import RouterAgent
from .market_research import MarketResearchAgent
from .realtime_analysis import RealtimeAnalysisAgent
from .portfolio import PortfolioAgent
from .explainer import ExplainerAgent
from .verifier import VerifierAgent
from .quick_reply import QuickReplyAgent

# Research loop agents
from .planner import PlannerAgent
from .researcher import ResearcherAgent
from .evaluator import EvaluatorAgent
from .refiner import RefinerAgent

__all__ = [
    "DeepReasoningAgent",
    "CompoundAgent",
    "RouterAgent",
    "MarketResearchAgent",
    "RealtimeAnalysisAgent",
    "PortfolioAgent",
    "ExplainerAgent",
    "VerifierAgent",
    "QuickReplyAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "EvaluatorAgent",
    "RefinerAgent"
]

