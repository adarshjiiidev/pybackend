"""
Autonomous Research Loop using LangGraph.
Implements bounded iterative research with confidence-based stopping.
"""

from langgraph.graph import StateGraph, END
from typing import Literal
import logging
from datetime import datetime

from ..models.research_state import ResearchState
from ..agents.planner import PlannerAgent
from ..agents.researcher import ResearcherAgent
from ..agents.evaluator import EvaluatorAgent
from ..agents.refiner import RefinerAgent
from ..config.research_config import MAX_RESEARCH_ITERATIONS

logger = logging.getLogger(__name__)


# Initialize research agents
planner = PlannerAgent()
researcher = ResearcherAgent()
evaluator = EvaluatorAgent()
refiner = RefinerAgent()


# Node functions
async def plan_research(state: ResearchState) -> ResearchState:
    """Plan research steps."""
    logger.info("PLANNER: Creating research plan")
    return await planner.create_plan(state)


async def execute_research(state: ResearchState) -> ResearchState:
    """Execute ONE research step."""
    logger.info(f"RESEARCHER: Executing step {state['current_step'] + 1}")
    return await researcher.execute_step(state)


async def evaluate_progress(state: ResearchState) -> ResearchState:
    """Evaluate confidence and decide if loop continues."""
    logger.info("EVALUATOR: Scoring research quality")
    return await evaluator.evaluate(state)


async def synthesize_findings(state: ResearchState) -> ResearchState:
    """Final synthesis of all research."""
    logger.info("REFINER: Synthesizing final answer")
    return await refiner.refine(state)


# Conditional routing function - THE KEY TO THE LOOP
def should_continue_research(state: ResearchState) -> Literal["continue", "finish"]:
    """
    Decision point: continue research loop or proceed to synthesis.
    
    This is the bounded loop condition that prevents infinite loops.
    """
    should_continue = state.get("should_continue", False)
    iteration_count = state.get("iteration_count", 0)
    stop_reason = state.get("stop_reason", "")
    
    # Hard safety bounds
    if iteration_count >= MAX_RESEARCH_ITERATIONS:
        logger.info(f"DECISION: STOP - Max iterations ({MAX_RESEARCH_ITERATIONS}) reached")
        return "finish"
    
    if not should_continue:
        logger.info(f"DECISION: STOP - {stop_reason}")
        return "finish"
    
    logger.info(f"DECISION: CONTINUE - Iteration {iteration_count + 1}/{MAX_RESEARCH_ITERATIONS}")
    return "continue"


def create_research_loop() -> StateGraph:
    """
    Create the autonomous research loop graph.
    
    Flow:
    START → PLAN → RESEARCH → EVALUATE → [DECISION]
                                             ↓
                                    continue → RESEARCH (loop back)
                                             ↓
                                      finish → REFINE → END
    """
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("plan", plan_research)
    workflow.add_node("research", execute_research)
    workflow.add_node("evaluate", evaluate_progress)
    workflow.add_node("refine", synthesize_findings)
    
    # Set entry point
    workflow.set_entry_point("plan")
    
    # Linear flow through first iteration
    workflow.add_edge("plan", "research")
    workflow.add_edge("research", "evaluate")
    
    # CONDITIONAL LOOP EDGE - The heart of autonomous research
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_research,
        {
            "continue": "research",  # Loop back for another iteration
            "finish": "refine"       # Exit loop and synthesize
        }
    )
    
    # Final edge
    workflow.add_edge("refine", END)
    
    # Compile
    app = workflow.compile()
    
    logger.info("Research loop graph created with bounded iterations")
    return app


# Create singleton instance
research_loop_graph = create_research_loop()


async def run_research_loop(query: str, session_id: str) -> dict:
    """
    Execute the research loop for a complex query.
    
    Returns final synthesized results with confidence scores.
    """
    # Initialize state
    initial_state: ResearchState = {
        "query": query,
        "session_id": session_id,
        "research_plan": [],
        "current_step": 0,
        "gathered_data": [],
        "confidence_score": 0.0,
        "data_quality_score": 0.0,
        "data_freshness_score": 0.0,
        "agreement_score": 0.0,
        "completeness_score": 0.0,
        "iteration_count": 0,
        "should_continue": True,
        "stop_reason": "",
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "final_answer": "",
        "key_findings": [],
        "risks_uncertainties": [],
        "data_freshness_indicator": "Unknown",
        "execution_metadata": {}
    }
    
    # Run the loop
    final_state = await research_loop_graph.ainvoke(initial_state)
    
    # Return structured results
    return {
        "final_answer": final_state.get("final_answer", ""),
        "key_findings": final_state.get("key_findings", []),
        "risks_uncertainties": final_state.get("risks_uncertainties", []),
        "confidence_score": final_state.get("confidence_score", 0.0),
        "data_freshness": final_state.get("data_freshness_indicator", "Unknown"),
        "iterations_completed": final_state.get("iteration_count", 0),
        "stop_reason": final_state.get("stop_reason", ""),
        "execution_time": (
            (final_state.get("completed_at") - final_state.get("started_at")).total_seconds()
            if final_state.get("completed_at") else 0
        )
    }
