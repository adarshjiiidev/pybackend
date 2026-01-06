from typing import Literal
from langgraph.graph import StateGraph, END
from agent.schemas.state import AgentState
from agent.nodes.input_parsing import parse_input
from agent.nodes.query_planner import plan_query
from agent.nodes.data_fetch import fetch_data
from agent.nodes.data_normalization import normalize_data
from agent.nodes.embedding import embed_knowledge
from agent.nodes.retrieval import retrieve_context
from agent.nodes.reasoning import analyze_market
from agent.nodes.response_generation import generate_response

def route_query(state: AgentState) -> Literal["fetch", "reason"]:
    """
    Conditional logic to determine the path after planning.
    - general_chat and options_trading: Skip data fetch (use knowledge)
    - market_data and comparative_analysis: Fetch yfinance data
    """
    query = state.get('parsed_query')
    if not query:
        return "fetch"
        
    # Skip data fetch for general chat and options trading
    if query.intent in ["general_chat", "options_trading"]:
        return "reason"
    
    # market_data and comparative_analysis need yfinance
    return "fetch"

def build_graph():
    """
    Constructs the LangGraph.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("input_parsing", parse_input)
    workflow.add_node("query_planner", plan_query)
    workflow.add_node("data_fetch", fetch_data)
    workflow.add_node("data_normalization", normalize_data)
    workflow.add_node("embedding", embed_knowledge)
    workflow.add_node("retrieval", retrieve_context)
    workflow.add_node("reasoning", analyze_market)
    workflow.add_node("response_generation", generate_response)
    
    # Define edges
    workflow.set_entry_point("input_parsing")
    
    def check_parsing_error(state: AgentState):
        if state.get("error"):
            return "end"
        return "continue"

    workflow.add_conditional_edges(
        "input_parsing",
        check_parsing_error,
        {
            "end": END,
            "continue": "query_planner"
        }
    )
    
    workflow.add_conditional_edges(
        "query_planner",
        route_query,
        {
            "fetch": "data_fetch",
            "reason": "reasoning"
        }
    )
    
    # Sequential flow for data fetching path
    workflow.add_edge("data_fetch", "data_normalization")
    workflow.add_edge("data_normalization", "embedding")
    workflow.add_edge("embedding", "retrieval") 
    workflow.add_edge("retrieval", "reasoning")
    # Note: Embedding creates new memory, Retrieval gets *existing* memory. 
    # Ideally Retrieval matches the query. 
    
    workflow.add_edge("reasoning", "response_generation")
    workflow.add_edge("response_generation", END)
    
    return workflow.compile()
