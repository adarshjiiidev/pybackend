"""
LangGraph workflow definition.
Orchestrates multi-agent system with conditional routing including GPT-OSS-120B and Compound AI.
"""

from langgraph.graph import StateGraph, END
from typing import Literal
import logging

from ..models.agent_state import AgentState, AgentMode
from ..agents import (
    DeepReasoningAgent,
    CompoundAgent,
    RouterAgent,
    MarketResearchAgent,
    RealtimeAnalysisAgent,
    PortfolioAgent,
    ExplainerAgent,
    CryptoAgent,
    VerifierAgent
)
from ..config import settings

logger = logging.getLogger(__name__)


# Initialize agents
router_agent = RouterAgent()
deep_reasoning_agent = DeepReasoningAgent()
compound_agent = CompoundAgent()
market_research_agent = MarketResearchAgent()
realtime_agent = RealtimeAnalysisAgent()
portfolio_agent = PortfolioAgent()
explainer_agent = ExplainerAgent()
crypto_agent = CryptoAgent()
verifier_agent = VerifierAgent()


# Node functions
async def router_node(state: AgentState) -> AgentState:
    """Router node - classifies intent and selects mode."""
    logger.info("Executing router node")
    return await router_agent.classify_intent(state)


async def deep_reasoning_node(state: AgentState) -> AgentState:
    """Deep reasoning node - GPT-OSS-120B with high effort."""
    logger.info("Executing deep reasoning node (GPT-OSS-120B)")
    return await deep_reasoning_agent.analyze(state)


async def compound_ai_node(state: AgentState) -> AgentState:
    """Compound AI node - real-time web search and tools."""
    logger.info("Executing Compound AI node")
    return await compound_agent.analyze(state)


async def market_research_node(state: AgentState) -> AgentState:
    """Market research node - deep fundamental analysis."""
    logger.info("Executing market research node")
    return await market_research_agent.analyze(state)


async def realtime_node(state: AgentState) -> AgentState:
    """Real-time analysis node - intraday insights."""
    logger.info("Executing real-time analysis node")
    return await realtime_agent.analyze(state)


async def portfolio_node(state: AgentState) -> AgentState:
    """Portfolio intelligence node - allocation strategies."""
    logger.info("Executing portfolio node")
    return await portfolio_agent.analyze(state)


async def explainer_node(state: AgentState) -> AgentState:
    """Explainer node - educational content."""
    logger.info("Executing explainer node")
    return await explainer_agent.analyze(state)


async def crypto_node(state: AgentState) -> AgentState:
    """Crypto intelligence node - cryptocurrency analysis."""
    logger.info("Executing crypto node")
    return await crypto_agent.analyze(state)


async def verifier_node(state: AgentState) -> AgentState:
    """Verifier node - quality control and refinement."""
    logger.info("Executing verifier node")
    return await verifier_agent.verify_and_refine(state)


# Routing function
def route_to_agent(state: AgentState) -> Literal[
    "deep_reasoning",
    "compound_ai",
    "market_research",
    "realtime_analysis",
    "portfolio",
    "explainer",
    "crypto"
]:
    """
    Conditional routing based on selected mode and query content.
    Routes to Compound AI for real-time needs, otherwise to specialist agents.
    """
    selected_mode = state.get("selected_mode", AgentMode.MARKET_RESEARCH.value)
    query_lower = state["query"].lower()
    
    # Route to Compound AI for real-time queries (beats Perplexity with live data)
    if settings.prefer_compound_for_realtime and settings.enable_compound_ai:
        realtime_keywords = ["latest", "news", "today", "current", "now", "breaking", "recent", "update"]
        if any(kw in query_lower for kw in realtime_keywords):
            logger.info("Routing to Compound AI for real-time query")
            return "compound_ai"
    
    # Route explainer queries to explainer
    if selected_mode == AgentMode.EXPLAINER.value:
        return "explainer"
    
    # Route portfolio queries to portfolio
    elif selected_mode == AgentMode.PORTFOLIO.value:
        return "portfolio"
    
    # Route crypto queries to crypto
    elif selected_mode == AgentMode.CRYPTO.value:
        return "crypto"
    
    # Route real-time analysis to realtime agent
    elif selected_mode == AgentMode.REALTIME_ANALYSIS.value:
        return "realtime_analysis"
    
    # Route market research to deep reasoning (GPT-OSS-120B)
    elif selected_mode == AgentMode.MARKET_RESEARCH.value:
        return "deep_reasoning"
    
    # Default to deep reasoning
    else:
        return "deep_reasoning"


def create_agent_graph() -> StateGraph:
    """
    Create the LangGraph workflow with GPT-OSS-120B and Compound AI.
    
    Graph structure:
    START -> router -> [conditional routing] -> 
      - compound_ai (for real-time queries)
      - deep_reasoning (for complex analysis)
      - specialist agents (market_research, realtime, portfolio, explainer, crypto)
    -> verifier -> END
    """
    
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("deep_reasoning", deep_reasoning_node)
    workflow.add_node("compound_ai", compound_ai_node)
    workflow.add_node("market_research", market_research_node)
    workflow.add_node("realtime_analysis", realtime_node)
    workflow.add_node("portfolio", portfolio_node)
    workflow.add_node("explainer", explainer_node)
    workflow.add_node("crypto", crypto_node)
    workflow.add_node("verifier", verifier_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional routing from router to agents
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "deep_reasoning": "deep_reasoning",
            "compound_ai": "compound_ai",
            "market_research": "market_research",
            "realtime_analysis": "realtime_analysis",
            "portfolio": "portfolio",
            "explainer": "explainer",
            "crypto": "crypto"
        }
    )
    
    # All agent nodes go to verifier
    workflow.add_edge("deep_reasoning", "verifier")
    workflow.add_edge("compound_ai", "verifier")
    workflow.add_edge("market_research", "verifier")
    workflow.add_edge("realtime_analysis", "verifier")
    workflow.add_edge("portfolio", "verifier")
    workflow.add_edge("explainer", "verifier")
    workflow.add_edge("crypto", "verifier")
    
    # Verifier goes to END
    workflow.add_edge("verifier", END)
    
    # Compile graph
    app = workflow.compile()
    
    logger.info("LangGraph workflow created with GPT-OSS-120B and Compound AI")
    return app


# Global graph instance
agent_graph = create_agent_graph()


async def run_agent_workflow(
    query: str,
    mode: str = "auto",
    session_id: str = None,
    conversation_history: list = None
) -> AgentState:
    """
    Execute the agent workflow.
    
    Args:
        query: User query
        mode: Agent mode (auto, market_research, etc.)
        session_id: Session identifier
        conversation_history: Previous conversation messages
    
    Returns:
        Final agent state with response
    """
    # Initialize state
    initial_state: AgentState = {
        "query": query,
        "mode": AgentMode(mode) if mode != "auto" else AgentMode.AUTO,
        "session_id": session_id or "default",
        "selected_mode": None,
        "extracted_entities": None,
        "conversation_history": conversation_history or [],
        "tool_results": None,
        "internal_reasoning": None,
        "final_response": None,
        "execution_metadata": None,
        "error": None
    }
    
    try:
        # Run workflow
        logger.info(f"Starting workflow for query: {query[:100]}...")
        final_state = await agent_graph.ainvoke(initial_state)
        logger.info("Workflow completed successfully")
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow execution error: {e}")
        initial_state["error"] = str(e)
        initial_state["final_response"] = "I encountered an error processing your request. Please try again."
        return initial_state
