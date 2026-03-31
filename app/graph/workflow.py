"""
LangGraph workflow definition.
Orchestrates multi-agent system with conditional routing including GPT-OSS-120B and Compound AI.
"""

import logging
from typing import Awaitable, Callable, Literal, Optional

from langgraph.graph import END, StateGraph

from ..models.agent_state import AgentState, AgentMode
from ..agents import (
    DeepReasoningAgent,
    CompoundAgent,
    RouterAgent,
    MarketResearchAgent,
    RealtimeAnalysisAgent,
    PortfolioAgent,
    ExplainerAgent,
    VerifierAgent,
    QuickReplyAgent,
)
from ..agents.vision_reader import VisionReaderAgent
from ..config import settings

logger = logging.getLogger(__name__)

StatusEmitter = Callable[[dict], Awaitable[None]]



import hashlib as _hashlib
from collections import OrderedDict as _OD
from datetime import datetime as _dt

# ── Simple TTL-LRU response cache (avoids full pipeline for repeated queries) ─
_RESPONSE_CACHE: "_OD[str, tuple[dict, _dt]]" = _OD()
_CACHE_MAX = 80          # max entries
_CACHE_TTL_SECONDS = 300 # 5-minute TTL

def _cache_key(query: str, mode: str) -> str:
    raw = f"{mode}::{query.strip().lower()}"
    return _hashlib.md5(raw.encode()).hexdigest()

def _cache_get(key: str) -> "dict | None":
    if key not in _RESPONSE_CACHE:
        return None
    state, ts = _RESPONSE_CACHE[key]
    if (_dt.now() - ts).total_seconds() > _CACHE_TTL_SECONDS:
        del _RESPONSE_CACHE[key]
        return None
    _RESPONSE_CACHE.move_to_end(key)  # LRU update
    return state

def _cache_set(key: str, state: dict) -> None:
    if key in _RESPONSE_CACHE:
        _RESPONSE_CACHE.move_to_end(key)
    _RESPONSE_CACHE[key] = (state, _dt.now())
    if len(_RESPONSE_CACHE) > _CACHE_MAX:
        _RESPONSE_CACHE.popitem(last=False)

# Initialize agents
vision_reader_agent = VisionReaderAgent()
quick_reply_agent = QuickReplyAgent()
router_agent = RouterAgent()
deep_reasoning_agent = DeepReasoningAgent()
compound_agent = CompoundAgent()
market_research_agent = MarketResearchAgent()
realtime_agent = RealtimeAnalysisAgent()
portfolio_agent = PortfolioAgent()
explainer_agent = ExplainerAgent()
verifier_agent = VerifierAgent()


async def _prefetch_kb(query: str) -> str:
    """Async KB search â€” runs in parallel with routing/status emissions."""
    try:
        from ..rag.qdrant_kb import get_qdrant_rag
        qdrant_rag = get_qdrant_rag()
        results = qdrant_rag.search(query, top_k=3)
        if not results:
            return ""
        good = [r for r in results if r.get("score", 1.0) >= 0.25]
        if not good:
            return ""
        return "\n\n".join(
            f"[{r.get('source', 'KB')}]\n{r.get('content', '').strip()}"
            for r in good[:3]
        )
    except Exception as e:
        logger.debug(f"KB prefetch skipped: {e}")
        return ""


async def _prefetch_web(query: str) -> str:
    """
    Async web search using MCP (DDG + Google scrape).
    Fires immediately after routing, parallel to KB search.
    Result injected into state["web_context"] before executor starts.
    """
    try:
        from ..tools.web_search_mcp import fast_scrape_search
        import asyncio as _a
        result = await _a.wait_for(fast_scrape_search(query), timeout=10.0)
        return (result or "").strip()
    except Exception as e:
        logger.debug(f"Web prefetch skipped: {e}")
        return ""


# Node functions
async def quick_reply_node(state: AgentState) -> AgentState:
    """
    Pre-router node. Handles pure greetings/small-talk instantly.
    If conversational: sets final_response and returns â€” no router call needed.
    If not conversational: returns state unchanged.
    """
    return await quick_reply_agent.handle(state)


async def vision_reader_node(state: AgentState) -> AgentState:
    """
    Pre-router vision node.
    Reads images â†’ extracts NLP scenario â†’ enriches query.
    Always runs first; is a no-op when no images are present.
    """
    return await vision_reader_agent.read_and_enrich(state)


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


# REMOVED: crypto_node (CryptoAgent not available)


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
    "explainer"
]:
    """
    Conditional routing based on selected mode and query content.
    
    Priority: GPT-OSS (deep_reasoning) for most queries â€” it has search_web tool
    and is MUCH faster than Compound. Compound only for deep multi-source research.
    """
    selected_mode = state.get("selected_mode", AgentMode.MARKET_RESEARCH.value)
    query_lower = state["query"].lower()
    
    # Route explainer queries first (greetings, terms, concepts)
    if selected_mode == AgentMode.EXPLAINER.value:
        return "explainer"
    
    # Route portfolio queries
    if selected_mode == AgentMode.PORTFOLIO.value:
        return "portfolio"
    
    # Route realtime analysis (prices, intraday)
    if selected_mode == AgentMode.REALTIME_ANALYSIS.value:
        return "realtime_analysis"
    
    # Compound AI: ONLY for deep multi-source research tasks that EXPLICITLY need
    # exhaustive web crawling across many sources (it's slow but thorough).
    # Examples: "compare quarterly results of top 10 IT stocks", "comprehensive sector report"
    if settings.enable_compound_ai:
        deep_research_signals = [
            "comprehensive report",
            "detailed analysis across",
            "compare quarterly results",
            "in-depth research",
            "full sector report",
        ]
        if any(signal in query_lower for signal in deep_research_signals):
            logger.info("Routing to Compound AI for deep multi-source research")
            return "compound_ai"
    
    # Everything else â†’ GPT-OSS deep_reasoning (has search_web tool, fast)
    # This includes: "latest news", "what's happening", "current", all market_research
    return "deep_reasoning"


def create_agent_graph() -> StateGraph:
    """
    Create the LangGraph workflow.

    Graph structure:
    START â†’ vision_reader (no-op if no images) â†’ router â†’ [conditional routing] â†’
      - deep_reasoning / compound_ai (research)
      - realtime_analysis (prices, intraday)
      - portfolio
      - explainer (concepts, education)
    â†’ verifier â†’ END

    Image queries: vision_reader converts image to NLP text, enriches query,
    then router classifies normally â€” no special-case image paths.
    """

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("vision_reader", vision_reader_node)  # â† new pre-router
    workflow.add_node("router", router_node)
    workflow.add_node("deep_reasoning", deep_reasoning_node)
    workflow.add_node("compound_ai", compound_ai_node)
    workflow.add_node("market_research", market_research_node)
    workflow.add_node("realtime_analysis", realtime_node)
    workflow.add_node("portfolio", portfolio_node)
    workflow.add_node("explainer", explainer_node)
    workflow.add_node("verifier", verifier_node)

    # Entry point: vision_reader first (no-op for text queries)
    workflow.set_entry_point("vision_reader")

    # vision_reader always flows to router
    workflow.add_edge("vision_reader", "router")

    # Router conditional routing
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "deep_reasoning": "deep_reasoning",
            "compound_ai": "compound_ai",
            "market_research": "market_research",
            "realtime_analysis": "realtime_analysis",
            "portfolio": "portfolio",
            "explainer": "explainer"
        }
    )

    # All agent nodes go to verifier
    workflow.add_edge("deep_reasoning", "verifier")
    workflow.add_edge("compound_ai", "verifier")
    workflow.add_edge("market_research", "verifier")
    workflow.add_edge("realtime_analysis", "verifier")
    workflow.add_edge("portfolio", "verifier")
    workflow.add_edge("explainer", "verifier")

    # Verifier goes to END
    workflow.add_edge("verifier", END)

    app = workflow.compile()
    logger.info("LangGraph workflow created with vision_reader pre-router")
    return app


# Global graph instance
agent_graph = create_agent_graph()

_NODE_EXECUTORS = {
    "vision_reader": vision_reader_node,
    "deep_reasoning": deep_reasoning_node,
    "compound_ai": compound_ai_node,
    "market_research": market_research_node,
    "realtime_analysis": realtime_node,
    "portfolio": portfolio_node,
    "explainer": explainer_node,
}


async def run_agent_workflow_with_events(
    initial_state: AgentState,
    emit_status: Optional[StatusEmitter] = None,
) -> AgentState:
    """
    Execute the workflow sequentially so the caller can stream meaningful progress.

    This avoids recompiling the graph per request and lets the API surface
    phases such as routing, KB lookup, web search, analysis, and verification.
    """

    async def _emit(event: dict) -> None:
        if emit_status:
            await emit_status(event)

    state = initial_state

    # ── LRU response cache check ──────────────────────────────────────────
    if not initial_state.get('images'):  # never cache vision queries
        _ck = _cache_key(initial_state.get('query', ''), initial_state.get('mode', 'auto'))
        _cached = _cache_get(_ck)
        if _cached:
            logger.info(f"⚡ Cache HIT: {initial_state.get('query', '')[:50]!r}")
            if emit_status:
                await emit_status({'phase': 'finalizing', 'stage': 'finalizing',
                                   'message': 'Answer ready.', 'progress_pct': 100})
            return dict(initial_state) | _cached
    else:
        _ck = None


    # â”€â”€ STEP 0: QuickReply â€” instant return for greetings/small-talk â”€â”€â”€â”€â”€â”€
    state = await quick_reply_node(state)
    if state.get("final_response"):
        # Greeting handled instantly â€” skip the entire pipeline
        await _emit({"phase": "finalizing", "stage": "finalizing",
                     "message": "Done!", "progress_pct": 100})
        return state

    # â”€â”€ STEP 1: Vision reader (no-op for text queries) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if state.get("images"):
        await _emit(
            {
                "phase": "vision",
                "stage": "vision",
                "message": "Reading attachments and extracting chart context...",
                "progress_pct": 8,
            }
        )
    state = await vision_reader_node(state)

    await _emit(
        {
            "phase": "routing",
            "stage": "routing",
            "message": "Understanding the question and choosing the best specialist...",
            "progress_pct": 18,
        }
    )
    state = await router_node(state)

    selected_mode = state.get("selected_mode", AgentMode.MARKET_RESEARCH.value)
    use_kb = bool(state.get("use_kb"))
    needs_search = bool(state.get("enable_research_loop"))
    entities = state.get("extracted_entities") or {}
    symbols = entities.get("symbols") or []

    # â”€â”€ Fire async KB prefetch immediately â€” runs while we emit status â”€â”€â”€â”€â”€
    # By the time executor starts, KB search is almost certainly done.
    import asyncio as _asyncio
    # Fire BOTH searches immediately after routing — run in parallel during status emissions
    kb_task  = _asyncio.create_task(_prefetch_kb(state["query"])) if use_kb else None
    web_task = _asyncio.create_task(_prefetch_web(state["query"]))

    await _emit(
        {
            "phase": "routing_complete",
            "stage": "routing_complete",
            "message": f"Routed to {selected_mode.replace('_', ' ')}.",
            "progress_pct": 24,
            "selected_mode": selected_mode,
            "use_kb": use_kb,
            "needs_search": needs_search,
            "symbols": symbols,
        }
    )

    if use_kb:
        await _emit(
            {
                "phase": "knowledge",
                "stage": "knowledge",
                "message": "Searching the knowledge base for domain context...",
                "progress_pct": 30,
                "selected_mode": selected_mode,
            }
        )

    if needs_search:
        await _emit(
            {
                "phase": "search",
                "stage": "search",
                "message": "Searching the web and gathering fresh market data...",
                "progress_pct": 38,
                "selected_mode": selected_mode,
                "symbols": symbols,
            }
        )

    route = route_to_agent(state)
    executor = _NODE_EXECUTORS[route]

    # ─── Await BOTH prefetch results before executor fires ─────────────────
    if kb_task is not None:
        try:
            kb_ctx = await _asyncio.wait_for(kb_task, timeout=6.0)
            if kb_ctx:
                state["kb_context"] = kb_ctx
                logger.info(f"📚 KB prefetch ready: {len(kb_ctx)} chars")
            await _emit({
                "phase": "kb_ready",
                "stage": "kb_ready",
                "message": "Knowledge base searched",
                "progress_pct": 33,
                "selected_mode": selected_mode,
            })
        except Exception as kb_err:
            logger.debug(f"KB prefetch await failed: {kb_err}")

    try:
        web_ctx = await _asyncio.wait_for(web_task, timeout=12.0)
        if web_ctx:
            state["web_context"] = web_ctx
            logger.info(f"🌐 Web prefetch ready: {len(web_ctx)} chars")
        await _emit({
            "phase": "web_ready",
            "stage": "web_ready",
            "message": "Web search complete",
            "progress_pct": 45,
            "selected_mode": selected_mode,
        })
    except Exception as web_err:
        logger.debug(f"Web prefetch await failed: {web_err}")

    await _emit(
        {
            "phase": "analysis",
            "stage": "analysis",
            "message": f"{route.replace('_', ' ').title()} is analyzing the evidence...",
            "progress_pct": 58,
            "selected_mode": selected_mode,
            "agent": route,
        }
    )
    state = await executor(state)

    metadata = state.get("execution_metadata") or {}
    tool_names = metadata.get("tools_used") or metadata.get("tools_called") or []
    if tool_names:
        await _emit(
            {
                "phase": "analysis_complete",
                "stage": "analysis_complete",
                "message": f"Finished analysis using {len(tool_names)} tool call(s).",
                "progress_pct": 76,
                "tools_used": tool_names,
                "selected_mode": selected_mode,
            }
        )

    if not metadata.get("skip_verifier"):
        await _emit(
            {
                "phase": "verification",
                "stage": "verification",
                "message": "Polishing the final answer for clarity and structure...",
                "progress_pct": 88,
                "selected_mode": selected_mode,
            }
        )
        state = await verifier_node(state)

    await _emit(
        {
            "phase": "finalizing",
            "stage": "finalizing",
            "message": "Final response ready. Sending it to the chat...",
            "progress_pct": 96,
            "selected_mode": selected_mode,
        }
    )
    # ── Cache successful responses ────────────────────────────────────────
    if (_ck and state.get('final_response') and not state.get('error')
            and not state.get('is_conversational')):
        _cache_set(_ck, {
            'final_response': state.get('final_response'),
            'execution_metadata': state.get('execution_metadata'),
        })
    return state


async def run_agent_workflow(
    query: str,
    mode: str = "auto",
    session_id: str = None,
    conversation_history: list = None,
    images: list = None
) -> AgentState:
    """
    Execute the agent workflow.
    
    Args:
        query: User query
        mode: Agent mode (auto, market_research, etc.)
        session_id: Session identifier
        conversation_history: Previous conversation messages
        images: List of base64 encoded images
    
    Returns:
        Final agent state with response
    """
    # Initialize state
    initial_state: AgentState = {
        "query": query,
        "images": images,
        "mode": AgentMode(mode) if mode != "auto" else AgentMode.AUTO,
        "session_id": session_id or "default",
        "selected_mode": None,
        "extracted_entities": None,
        "conversation_history": conversation_history or [],
        "tool_results": None,
        "kb_context": None,       # populated async after routing
        "web_context": None,       # populated async after routing
        "internal_reasoning": None,
        "final_response": None,
        "execution_metadata": None,
        "error": None
    }
    
    try:
        # Run workflow
        logger.info(f"Starting workflow for query: {query[:100]}...")
        final_state = await run_agent_workflow_with_events(initial_state)
        logger.info("Workflow completed successfully")
        return final_state
        
    except Exception as e:
        logger.error(f"Workflow execution error: {e}")
        initial_state["error"] = str(e)
        initial_state["final_response"] = "I encountered an error processing your request. Please try again."
        return initial_state

