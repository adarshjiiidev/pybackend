"""
MasterOrchestrator — Ultra-Smart Swarm Orchestration Engine
============================================================
Phase 2 · DaddysAI Core Intelligence

Implements the exact flow from the architecture diagram:

  01  User sends message
      ↓
  02  Orchestrator plans approach
      • Loads relevant skills
      • Creates task plan with dependency graph
      • Determines parallel vs sequential research needs
      ↓
  03  Entity file created
      • Writes extracted entities (symbols, topics, companies) to Workspace
      ↓
  04  Parallel research dispatch
      • Spawns N sub-agents simultaneously (each researches one domain)
      ↓
  05  Results collected
      • Each sub-agent saves findings to Workspace
      • Orchestrator reads all findings into unified result set
      ↓
  06  Analysis phase
      • Runs analysis scripts on collected data
      • Generates charts / statistics / signals
      ↓
  07  Asset sub-agent spawned
      • Spawns ReportAgent to create formatted report
      ↓
  08  Report generated
      • Full markdown / structured report with tables, charts, narrative
      ↓
  09  Delivered to user
      • Final response with total agents used and metadata

Key features:
  - LLM-driven task planning (decides which agents to spawn, in what order)
  - Dynamic agent creation and disposal
  - Real-time status tracking per agent
  - Shared Workspace for inter-agent communication
  - Web surfing manager for autonomous research
  - Confidence-weighted result aggregation
  - Self-healing: retry failed agents with different strategies
  - Streaming status events (SSE-ready via workspace event bus)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task Plan structures
# ---------------------------------------------------------------------------


@dataclass
class AgentTask:
    """
    One unit of work in the orchestrator's task plan.

    Attributes:
        task_id       Unique identifier for this task
        agent_type    Registry key of the agent to spawn
        task_name     Human-readable name shown in status updates
        payload       Dict passed as SwarmMessage.payload to the agent
        depends_on    List of task_ids that must complete before this runs
        priority      0=normal, 1=high (determines spawn order within a batch)
        optional      If True, failure doesn't block dependent tasks
        timeout_s     Per-task timeout override
        retry_count   How many times to retry on failure
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_type: str = ""
    task_name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    priority: int = 0
    optional: bool = False
    timeout_s: float = 120.0
    retry_count: int = 1


@dataclass
class OrchestratorPlan:
    """
    The full execution plan for a user query.

    Attributes:
        query           Original user query
        intent          Detected intent: 'stock_analysis' | 'market_overview' |
                        'research' | 'prediction' | 'report' | 'global' | 'general'
        entities        Extracted entities: symbols, topics, timeframes
        tasks           Ordered list of AgentTask objects
        plan_reasoning  LLM's reasoning for why it chose these agents
        estimated_s     Estimated total time in seconds
        complexity      'quick' | 'standard' | 'deep'
    """

    query: str = ""
    intent: str = "general"
    entities: Dict[str, Any] = field(default_factory=dict)
    tasks: List[AgentTask] = field(default_factory=list)
    plan_reasoning: str = ""
    estimated_s: float = 30.0
    complexity: str = "standard"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "entities": self.entities,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "agent_type": t.agent_type,
                    "task_name": t.task_name,
                    "depends_on": t.depends_on,
                    "priority": t.priority,
                    "optional": t.optional,
                }
                for t in self.tasks
            ],
            "plan_reasoning": self.plan_reasoning,
            "estimated_s": self.estimated_s,
            "complexity": self.complexity,
        }


# ---------------------------------------------------------------------------
# Orchestration result
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    """
    Final result returned to the user after full orchestration.
    """

    query: str
    final_response: str
    report_md: str = ""
    signal: str = "neutral"
    confidence: float = 0.5
    agents_used: int = 0
    agents_succeeded: int = 0
    agents_failed: int = 0
    total_duration_s: float = 0.0
    plan: Optional[OrchestratorPlan] = None
    agent_summaries: List[Dict[str, Any]] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "final_response": self.final_response,
            "report_md": self.report_md,
            "signal": self.signal,
            "confidence": round(self.confidence, 3),
            "agents_used": self.agents_used,
            "agents_succeeded": self.agents_succeeded,
            "agents_failed": self.agents_failed,
            "total_duration_s": round(self.total_duration_s, 2),
            "plan": self.plan.to_dict() if self.plan else None,
            "agent_summaries": self.agent_summaries,
            "key_findings": self.key_findings,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Master Orchestrator
# ---------------------------------------------------------------------------


class MasterOrchestrator:
    """
    The central intelligence that coordinates the entire DaddysAI swarm.

    Usage::

        orchestrator = MasterOrchestrator()
        result = await orchestrator.run(
            query="Deep analysis of RELIANCE with prediction and report",
            session_id="sess-123",
            on_status=lambda event: print(event),  # optional SSE callback
        )
        # result.final_response  → full answer for user
        # result.report_md       → structured report
        # result.agent_summaries → what each agent found

    Status callback receives dicts::

        {
            "phase": "planning" | "dispatching" | "collecting" | "analyzing" | "reporting",
            "message": "Spawning TechnicalAnalysisAgent for RELIANCE...",
            "agent_id": "...",
            "progress_pct": 45,
            "timestamp": "...",
        }
    """

    # Maximum agents to spawn in one swarm run
    MAX_AGENTS_PER_RUN = 20

    # Phases (match the image diagram steps)
    PHASE_PLANNING = "planning"
    PHASE_ENTITIES = "entities"
    PHASE_DISPATCHING = "dispatching"
    PHASE_COLLECTING = "collecting"
    PHASE_ANALYZING = "analyzing"
    PHASE_REPORTING = "reporting"
    PHASE_DONE = "done"

    def __init__(self) -> None:
        self._log = logging.getLogger("swarm.orchestrator")

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    async def run(
        self,
        query: str,
        session_id: str = "",
        mode: str = "auto",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        images: Optional[List[str]] = None,
    ) -> OrchestratorResult:
        """
        Execute the full orchestration pipeline for a user query.

        Args:
            query               User's natural language query
            session_id          Session identifier for workspace
            mode                Agent mode hint from router
            conversation_history Previous messages for context
            on_status           Optional callback for real-time status events
            images              Base64-encoded images (for vision analysis)

        Returns:
            OrchestratorResult with final_response + full structured data
        """
        run_id = str(uuid.uuid4())[:8]
        t_start = time.monotonic()
        self._log.info(f"🎯 Orchestrator run [{run_id}]: {query[:100]!r}")

        # ── Create workspace for this run ──────────────────────────────────
        workspace = self._create_workspace(run_id, query, session_id)

        async def _status(phase: str, message: str, progress: int = 0, **kwargs):
            event = {
                "run_id": run_id,
                "phase": phase,
                "message": message,
                "progress_pct": progress,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs,
            }
            workspace.emit(event)
            if on_status:
                try:
                    on_status(event)
                except Exception:
                    pass
            self._log.info(f"[{run_id}] {phase.upper()}: {message}")

        try:
            # ────────────────────────────────────────────────────────────────
            # STEP 02: Planning
            # ────────────────────────────────────────────────────────────────
            await _status(
                self.PHASE_PLANNING,
                "Orchestrator analysing query and planning approach...",
                5,
            )
            plan = await self._plan(query, mode, conversation_history or [], images)
            workspace.set_plan(plan)
            await _status(
                self.PHASE_PLANNING,
                f"Plan ready: {len(plan.tasks)} agents | intent={plan.intent} | complexity={plan.complexity}",
                10,
                plan=plan.to_dict(),
            )

            # ────────────────────────────────────────────────────────────────
            # STEP 03: Entity file created
            # ────────────────────────────────────────────────────────────────
            await _status(self.PHASE_ENTITIES, "Writing entities to workspace...", 15)
            workspace.write_entities(plan.entities or {})
            entity_summary = self._describe_entities(plan.entities)
            await _status(
                self.PHASE_ENTITIES,
                f"Entities: {entity_summary}",
                18,
                entities=plan.entities,
            )

            # ────────────────────────────────────────────────────────────────
            # STEP 04+05: Parallel dispatch + collect
            # ────────────────────────────────────────────────────────────────
            await _status(
                self.PHASE_DISPATCHING,
                f"Dispatching {len(plan.tasks)} agents in parallel batches...",
                20,
            )

            agent_results = await self._execute_plan(plan, workspace, _status)

            await _status(
                self.PHASE_COLLECTING,
                f"Collected results from {len(agent_results)} agents.",
                70,
                agent_count=len(agent_results),
            )

            # ────────────────────────────────────────────────────────────────
            # STEP 06: Analysis phase
            # ────────────────────────────────────────────────────────────────
            await _status(
                self.PHASE_ANALYZING, "Running analysis on collected data...", 75
            )
            analysis = await self._analyze_results(
                query, plan, agent_results, workspace
            )
            workspace.write_analysis(analysis)
            await _status(
                self.PHASE_ANALYZING,
                f"Analysis complete. Signal: {analysis.get('signal', 'neutral')} | Confidence: {analysis.get('confidence', 0.5):.0%}",
                82,
                signal=analysis.get("signal"),
                confidence=analysis.get("confidence"),
            )

            # ────────────────────────────────────────────────────────────────
            # STEP 07+08: Asset agent (Report generation)
            # ────────────────────────────────────────────────────────────────
            await _status(
                self.PHASE_REPORTING,
                "Spawning ReportAgent to generate final report...",
                85,
            )
            final_response, report_md = await self._generate_report(
                query, plan, agent_results, analysis, workspace
            )
            await _status(self.PHASE_REPORTING, "Report generated.", 95)

            # ────────────────────────────────────────────────────────────────
            # STEP 09: Deliver to user
            # ────────────────────────────────────────────────────────────────
            duration = time.monotonic() - t_start
            succeeded = sum(1 for r in agent_results if r.get("status") == "done")
            failed = sum(1 for r in agent_results if r.get("status") == "failed")

            await _status(
                self.PHASE_DONE,
                f"Done in {duration:.1f}s. Total sub-agents: {len(plan.tasks)} "
                f"({succeeded} research + 1 report). Delivered to user.",
                100,
                duration_s=round(duration, 2),
            )

            result = OrchestratorResult(
                query=query,
                final_response=final_response,
                report_md=report_md,
                signal=analysis.get("signal", "neutral"),
                confidence=float(analysis.get("confidence", 0.5)),
                agents_used=len(plan.tasks) + 1,  # +1 for report agent
                agents_succeeded=succeeded,
                agents_failed=failed,
                total_duration_s=duration,
                plan=plan,
                agent_summaries=[r for r in agent_results if r.get("summary")],
                key_findings=analysis.get("key_findings", []),
                artifacts=workspace.get_artifacts(),
                metadata={
                    "run_id": run_id,
                    "session_id": session_id,
                    "intent": plan.intent,
                    "complexity": plan.complexity,
                    "plan_reasoning": plan.plan_reasoning,
                    "entities": plan.entities,
                },
            )

            self._log.info(
                f"✅ Orchestrator [{run_id}] done in {duration:.1f}s | "
                f"agents={len(plan.tasks) + 1} | signal={result.signal}"
            )
            return result

        except Exception as exc:
            duration = time.monotonic() - t_start
            self._log.error(
                f"❌ Orchestrator [{run_id}] crashed after {duration:.1f}s: {exc}",
                exc_info=True,
            )
            await _status("error", f"Orchestrator error: {exc}", 0)
            # Graceful degradation — return a minimal result
            return OrchestratorResult(
                query=query,
                final_response=await self._emergency_fallback(query, str(exc)),
                signal="neutral",
                confidence=0.2,
                total_duration_s=time.monotonic() - t_start,
                metadata={"run_id": run_id, "error": str(exc)},
            )

    # -----------------------------------------------------------------------
    # STEP 02: Planning — LLM decides which agents to spawn
    # -----------------------------------------------------------------------

    async def _plan(
        self,
        query: str,
        mode: str,
        history: List[Dict[str, str]],
        images: Optional[List[str]],
    ) -> OrchestratorPlan:
        """
        Use LLM to analyse the query and produce a structured execution plan.

        The planner:
          1. Detects intent (stock analysis, market overview, research, etc.)
          2. Extracts entities (symbols, sectors, timeframes, topics)
          3. Decides which agent types are needed
          4. Sets dependencies (e.g. prediction needs TA first)
          5. Estimates complexity and time
        """
        try:
            plan = await self._llm_plan(query, mode, history, images)
            return plan
        except Exception as exc:
            self._log.warning(f"LLM planning failed ({exc}), using rule-based fallback")
            return self._rule_based_plan(query, mode)

    async def _llm_plan(
        self,
        query: str,
        mode: str,
        history: List[Dict[str, str]],
        images: Optional[List[str]],
    ) -> OrchestratorPlan:
        """LLM-powered task planner."""
        from .base_agent import AgentToolbox

        toolbox = AgentToolbox.instance()
        client = toolbox.get_llm_client()
        model = toolbox.get_model("fast")

        history_text = ""
        if history:
            recent = history[-4:]
            history_text = "\n".join(
                f"{m['role'].upper()}: {m.get('content', '')[:200]}" for m in recent
            )

        has_images = bool(images)

        system_prompt = """You are the master planning brain of DaddysAI — an ultra-smart financial AI orchestrator.

Your job: analyse the user's query and produce a precise JSON execution plan.

Available agent types and their capabilities:
  - "technical_analysis"   → RSI, MACD, BB, patterns, S/R levels, Supertrend, Ichimoku. Needs: symbol
  - "oi_analysis"          → Option chain, PCR, Max Pain, GEX, IV Skew. Needs: symbol (NIFTY/BANKNIFTY/equity)
  - "web_research"         → Multi-step autonomous web research, news, reports. Needs: query, domain_focus
  - "global_market"        → Global indices, DXY, crude, gold, FII prediction. Needs: focus
  - "shock_detection"      → Crash/volatility anomaly detection. Needs: symbol
  - "prediction"           → ML price/trend prediction. Needs: symbol
  - "fundamentals"         → PE, ROE, revenue, sector, financials. Needs: symbol
  - "sentiment"            → News + social sentiment for a stock/topic. Needs: query
  - "data_fetch"           → Fetch + store fresh OHLCV for any symbol. Needs: symbol
  - "report"               → Generate final structured report (ALWAYS last). Needs: results from other agents

Rules:
  1. Spawn the minimum number of agents needed. Don't over-spawn.
  2. Always spawn "web_research" for any query needing news/current events.
  3. Always spawn "report" as the LAST task with depends_on=[all other task_ids].
  4. For simple price/greeting queries → complexity=quick, 1-2 agents only.
  5. For deep analysis → complexity=deep, 5-8 agents.
  6. Set depends_on carefully — prediction depends on technical_analysis.
  7. For images → add "vision" payload key.
  8. Symbols must be uppercase NSE symbols (RELIANCE, TCS, NIFTY, etc.)"""

        user_prompt = f"""User query: "{query}"
Mode hint: {mode}
Has images: {has_images}

Recent conversation:
{history_text or "(none)"}

Produce a JSON execution plan. Output ONLY valid JSON, no markdown, no extra text:

{{
  "intent": "stock_analysis | market_overview | research | prediction | report | global | options | general | greeting",
  "entities": {{
    "symbols": ["RELIANCE", "NIFTY"],
    "sector": "Banking",
    "timeframe": "1y",
    "topics": ["earnings", "Q3 results"],
    "companies": ["Reliance Industries"],
    "is_index": false,
    "needs_live_price": true,
    "needs_options": false,
    "needs_global": false,
    "needs_prediction": false,
    "needs_report": true
  }},
  "tasks": [
    {{
      "task_id": "t1",
      "agent_type": "technical_analysis",
      "task_name": "Technical Analysis — RELIANCE",
      "payload": {{"symbol": "RELIANCE", "days": 365}},
      "depends_on": [],
      "priority": 1,
      "optional": false,
      "timeout_s": 60
    }},
    {{
      "task_id": "t_report",
      "agent_type": "report",
      "task_name": "Generate Final Report",
      "payload": {{"include_sections": ["analysis", "prediction", "news", "levels"]}},
      "depends_on": ["t1"],
      "priority": 0,
      "optional": false,
      "timeout_s": 90
    }}
  ],
  "plan_reasoning": "One sentence explaining why these agents were chosen.",
  "estimated_s": 45,
  "complexity": "standard"
}}"""

        try:
            from ..config.openrouter_client import call_openrouter
            from ..config.settings import Settings as _Settings
            _s = _Settings()
            _model = toolbox.get_model("fast")
            response = await call_openrouter(
                _model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2500,  # increased from 1500 — full plan JSON can be large
            )
        except Exception:
            # Fallback to direct client if call_openrouter not available
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2500,
            )

        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown code blocks
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

        data = json.loads(raw)

        # Build AgentTask objects
        tasks: List[AgentTask] = []
        for t in data.get("tasks", []):
            tasks.append(
                AgentTask(
                    task_id=str(t.get("task_id", str(uuid.uuid4())[:8])),
                    agent_type=str(t.get("agent_type", "")),
                    task_name=str(t.get("task_name", t.get("agent_type", ""))),
                    payload=dict(t.get("payload", {})),
                    depends_on=list(t.get("depends_on", [])),
                    priority=int(t.get("priority", 0)),
                    optional=bool(t.get("optional", False)),
                    timeout_s=float(t.get("timeout_s", 90.0)),
                    retry_count=int(t.get("retry_count", 1)),
                )
            )

        # Cap to MAX_AGENTS
        tasks = tasks[: self.MAX_AGENTS_PER_RUN]

        # Ensure a report task always exists (add one if missing)
        if tasks and not any(t.agent_type == "report" for t in tasks):
            non_report_ids = [t.task_id for t in tasks if t.agent_type != "report"]
            tasks.append(
                AgentTask(
                    task_id="t_report_auto",
                    agent_type="report",
                    task_name="Generate Final Report",
                    payload={},
                    depends_on=non_report_ids,
                    priority=0,
                    timeout_s=120.0,
                )
            )

        # ── Guardrails: always inject web_research + TA for stock queries ──
        # The LLM planner sometimes forgets these; ensure they're always present
        has_web = any(t.agent_type == "web_research" for t in tasks)
        has_ta = any(t.agent_type == "technical_analysis" for t in tasks)
        entities = data.get("entities", {})
        symbols = entities.get("symbols", [])
        primary_sym = symbols[0] if symbols else ""
        intent = str(data.get("intent", "general"))
        is_stock_query = bool(primary_sym) or intent in (
            "stock_analysis", "prediction", "options", "research"
        )
        report_task = next((t for t in tasks if t.agent_type == "report"), None)
        non_report_ids_all = [t.task_id for t in tasks if t.agent_type != "report"]

        if is_stock_query and not has_web:
            web_task = AgentTask(
                task_id="t_web_guard",
                agent_type="web_research",
                task_name=f"Web Research: {query[:50]}",
                payload={"query": query, "depth": 3, "domain_focus": "indian_markets"},
                depends_on=[],
                priority=1,
                timeout_s=120.0,
            )
            # Insert before report task
            insert_idx = next(
                (i for i, t in enumerate(tasks) if t.agent_type == "report"),
                len(tasks),
            )
            tasks.insert(insert_idx, web_task)
            non_report_ids_all.append("t_web_guard")

        if is_stock_query and not has_ta and primary_sym:
            ta_task = AgentTask(
                task_id="t_ta_guard",
                agent_type="technical_analysis",
                task_name=f"Technical Analysis: {primary_sym}",
                payload={"symbol": primary_sym, "days": 365},
                depends_on=[],
                priority=1,
                timeout_s=90.0,
            )
            insert_idx = next(
                (i for i, t in enumerate(tasks) if t.agent_type == "report"),
                len(tasks),
            )
            tasks.insert(insert_idx, ta_task)
            non_report_ids_all.append("t_ta_guard")

        # Update report task's depends_on to include new guard tasks
        if report_task and (
            (is_stock_query and not has_web) or (is_stock_query and not has_ta and primary_sym)
        ):
            report_task.depends_on = list(set(report_task.depends_on) | set(non_report_ids_all))


        return OrchestratorPlan(
            query=query,
            intent=str(data.get("intent", "general")),
            entities=dict(data.get("entities", {})),
            tasks=tasks,
            plan_reasoning=str(data.get("plan_reasoning", "")),
            estimated_s=float(data.get("estimated_s", 30.0)),
            complexity=str(data.get("complexity", "standard")),
        )

    def _rule_based_plan(self, query: str, mode: str) -> OrchestratorPlan:
        """
        Fallback rule-based planner when LLM planning fails.
        Uses keyword matching to determine agents.
        """
        q = query.lower()
        symbols = self._extract_symbols_simple(query)
        tasks: List[AgentTask] = []
        intent = "general"

        # Detect intent
        if any(
            k in q for k in ("price", "ltp", "quote", "where is", "how much", "level")
        ):
            intent = "realtime"
        elif any(
            k in q
            for k in (
                "technical",
                "rsi",
                "macd",
                "chart",
                "pattern",
                "support",
                "resistance",
            )
        ):
            intent = "stock_analysis"
        elif any(
            k in q for k in ("option", "oi", "pcr", "max pain", "call", "put", "strike")
        ):
            intent = "options"
        elif any(k in q for k in ("predict", "forecast", "target", "buy or sell")):
            intent = "prediction"
        elif any(
            k in q
            for k in ("global", "us market", "nikkei", "nasdaq", "dow", "crude", "gold")
        ):
            intent = "global"
        elif any(
            k in q for k in ("news", "latest", "today", "recent", "report", "research")
        ):
            intent = "research"
        elif any(
            k in q
            for k in ("fundamental", "pe", "roe", "earnings", "revenue", "profit")
        ):
            intent = "research"

        # Primary symbol
        primary_sym = symbols[0] if symbols else "NIFTY"

        # Always add web research for most queries
        t_web = AgentTask(
            task_id="t_web",
            agent_type="web_research",
            task_name=f"Research: {query[:50]}",
            payload={"query": query, "depth": 2, "domain_focus": "indian_markets"},
            depends_on=[],
            timeout_s=90.0,
        )
        tasks.append(t_web)

        if intent in ("stock_analysis", "prediction", "realtime") and primary_sym:
            tasks.append(
                AgentTask(
                    task_id="t_ta",
                    agent_type="technical_analysis",
                    task_name=f"TA: {primary_sym}",
                    payload={"symbol": primary_sym, "days": 365},
                    depends_on=[],
                    timeout_s=60.0,
                )
            )

        if intent == "options" or "nifty" in q or "banknifty" in q:
            sym = "BANKNIFTY" if "banknifty" in q else "NIFTY"
            tasks.append(
                AgentTask(
                    task_id="t_oi",
                    agent_type="oi_analysis",
                    task_name=f"OI Analysis: {sym}",
                    payload={"symbol": sym},
                    depends_on=[],
                    timeout_s=45.0,
                )
            )

        if intent == "global" or any(
            k in q for k in ("global", "us market", "crude", "gold", "fii")
        ):
            tasks.append(
                AgentTask(
                    task_id="t_global",
                    agent_type="global_market",
                    task_name="Global Market Monitor",
                    payload={"focus": "india_impact"},
                    depends_on=[],
                    timeout_s=90.0,
                )
            )

        if intent == "prediction" and primary_sym:
            tasks.append(
                AgentTask(
                    task_id="t_pred",
                    agent_type="prediction",
                    task_name=f"Prediction: {primary_sym}",
                    payload={"symbol": primary_sym},
                    depends_on=["t_ta"]
                    if any(t.task_id == "t_ta" for t in tasks)
                    else [],
                    timeout_s=90.0,
                )
            )

        # Always end with report
        all_ids = [t.task_id for t in tasks]
        tasks.append(
            AgentTask(
                task_id="t_report",
                agent_type="report",
                task_name="Generate Final Report",
                payload={},
                depends_on=all_ids,
                timeout_s=90.0,
            )
        )

        return OrchestratorPlan(
            query=query,
            intent=intent,
            entities={"symbols": symbols, "primary_symbol": primary_sym},
            tasks=tasks,
            plan_reasoning=f"Rule-based plan for {intent} intent with {len(symbols)} symbols.",
            estimated_s=len(tasks) * 20.0,
            complexity="standard",
        )

    # -----------------------------------------------------------------------
    # STEP 04+05: Execute plan — parallel batch dispatch + collect
    # -----------------------------------------------------------------------

    async def _execute_plan(
        self,
        plan: OrchestratorPlan,
        workspace: "SwarmWorkspace",
        status_cb: Callable,
    ) -> List[Dict[str, Any]]:
        """
        Execute all tasks in the plan respecting dependency order.

        Algorithm:
          • Find tasks with no unmet dependencies → run as parallel batch
          • Wait for batch to complete
          • Mark completed tasks as done
          • Repeat until all tasks scheduled (excluding 'report' type)
          • report task runs last, separately

        Returns list of agent result dicts.
        """
        from .agent_registry import get_registry

        registry = get_registry()
        all_results: Dict[str, Any] = {}  # task_id → AgentResult dict
        completed: set = set()  # task_ids that are done
        failed: set = set()  # task_ids that failed

        # Separate report task (runs last)
        report_tasks = [t for t in plan.tasks if t.agent_type == "report"]
        research_tasks = [t for t in plan.tasks if t.agent_type != "report"]

        progress_base = 20
        progress_max = 68
        n_tasks = max(len(research_tasks), 1)

        batch_num = 0

        # Keep running until all research tasks are scheduled
        while len(completed) + len(failed) < len(research_tasks):
            # Find tasks ready to run (all deps met or failed-but-optional)
            ready: List[AgentTask] = []
            for task in research_tasks:
                if task.task_id in completed or task.task_id in failed:
                    continue
                deps_met = all(
                    dep in completed
                    or dep in failed  # failed optional deps don't block
                    for dep in task.depends_on
                )
                if deps_met:
                    ready.append(task)

            if not ready:
                # Circular dependency or all tasks blocked by failures
                # Mark remaining as skipped
                remaining = [
                    t
                    for t in research_tasks
                    if t.task_id not in completed and t.task_id not in failed
                ]
                self._log.warning(
                    f"No ready tasks found, {len(remaining)} tasks skipped (blocked)"
                )
                for t in remaining:
                    failed.add(t.task_id)
                break

            # Sort by priority (high first)
            ready.sort(key=lambda t: t.priority, reverse=True)
            batch_num += 1

            await status_cb(
                self.PHASE_DISPATCHING,
                f"Batch {batch_num}: spawning {len(ready)} agents in parallel — "
                + ", ".join(t.task_name for t in ready),
                progress_base
                + int((len(completed) / n_tasks) * (progress_max - progress_base)),
                batch=batch_num,
                agents=[t.task_name for t in ready],
            )

            # Register agents in workspace
            for task in ready:
                workspace.register_agent_sync(
                    task.task_id, task.agent_type, task.task_name
                )

            # Spawn all ready agents in parallel
            spawn_results = await asyncio.gather(
                *[self._spawn_task(task, workspace, registry) for task in ready],
                return_exceptions=True,
            )

            # Process results
            for task, result in zip(ready, spawn_results):
                if isinstance(result, Exception):
                    self._log.error(
                        f"Task {task.task_id} ({task.task_name}) raised: {result}"
                    )
                    workspace.update_agent(task.task_id, "failed", error=str(result))
                    if not task.optional:
                        failed.add(task.task_id)
                    else:
                        completed.add(task.task_id)
                    all_results[task.task_id] = {
                        "task_id": task.task_id,
                        "agent_type": task.agent_type,
                        "task_name": task.task_name,
                        "status": "failed",
                        "error": str(result),
                        "summary": f"{task.task_name} failed: {result}",
                        "data": {},
                        "signal": "neutral",
                        "confidence": 0.0,
                    }
                else:
                    # result is a plain dict here (not BaseException)
                    r: dict = result  # type: ignore[assignment]
                    workspace.update_agent(
                        task.task_id,
                        r.get("status", "done"),
                        summary=r.get("summary", ""),
                        signal=r.get("signal", "neutral"),
                    )
                    all_results[task.task_id] = r
                    if r.get("status") == "failed" and not task.optional:
                        failed.add(task.task_id)
                    else:
                        completed.add(task.task_id)

                    # Write findings to workspace
                    if r.get("data"):
                        workspace.write_finding(
                            agent_id=task.task_id,
                            agent_type=task.agent_type,
                            finding_type="result",
                            content=r.get("summary", "") or "",
                            signal=r.get("signal", "neutral"),
                            confidence=float(r.get("confidence", 0.5) or 0.5),
                        )

            await status_cb(
                self.PHASE_COLLECTING,
                f"Batch {batch_num} done: {len([r for r in ready if r.task_id in completed])} succeeded, "
                f"{len([r for r in ready if r.task_id in failed])} failed",
                progress_base
                + int((len(completed) / n_tasks) * (progress_max - progress_base)),
            )

        return list(all_results.values())

    async def _spawn_task(
        self,
        task: AgentTask,
        workspace: "SwarmWorkspace",
        registry: Any,
    ) -> Dict[str, Any]:
        """
        Spawn a single agent task and return a normalised result dict.
        Handles retry logic.
        """
        from .base_agent import SwarmMessage

        last_error = None
        for attempt in range(max(1, task.retry_count)):
            try:
                msg = SwarmMessage(
                    task=task.task_name,
                    payload=task.payload,
                    timeout_s=task.timeout_s,
                )

                if registry.type_exists(task.agent_type):
                    result = await registry.spawn(
                        agent_type=task.agent_type,
                        message=msg,
                        parent_id="orchestrator",
                    )
                    return {
                        "task_id": task.task_id,
                        "agent_type": task.agent_type,
                        "task_name": task.task_name,
                        "status": result.status.value,
                        "data": result.data,
                        "summary": result.summary,
                        "signal": result.signal,
                        "confidence": result.confidence,
                        "duration_s": result.duration_s,
                        "error": result.error,
                    }
                else:
                    # Agent type not registered — run inline fallback
                    self._log.warning(
                        f"Agent type {task.agent_type!r} not in registry — using inline fallback"
                    )
                    return await self._inline_fallback(task, workspace)

            except Exception as exc:
                last_error = exc
                self._log.warning(
                    f"Task {task.task_id} attempt {attempt + 1}/{task.retry_count} failed: {exc}"
                )
                if attempt < task.retry_count - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))

        return {
            "task_id": task.task_id,
            "agent_type": task.agent_type,
            "task_name": task.task_name,
            "status": "failed",
            "data": {},
            "summary": f"{task.task_name} failed after {task.retry_count} attempt(s): {last_error}",
            "signal": "neutral",
            "confidence": 0.0,
            "error": str(last_error),
        }

    async def _inline_fallback(
        self, task: AgentTask, workspace: "SwarmWorkspace"
    ) -> Dict[str, Any]:
        """
        Inline execution for agent types that haven't been registered yet.
        Uses web search as a universal fallback.
        """
        from .base_agent import AgentToolbox

        toolbox = AgentToolbox.instance()
        query = task.payload.get("query", task.task_name)
        symbol = task.payload.get("symbol", "")

        search_q = (
            f"{symbol} {query} India market analysis"
            if symbol
            else f"{query} India finance"
        )
        try:
            result_text = await toolbox.web_search(search_q)
            return {
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "task_name": task.task_name,
                "status": "done",
                "data": {"web_result": result_text[:2000]},
                "summary": result_text[:500],
                "signal": "neutral",
                "confidence": 0.4,
                "error": None,
            }
        except Exception as exc:
            return {
                "task_id": task.task_id,
                "agent_type": task.agent_type,
                "task_name": task.task_name,
                "status": "failed",
                "data": {},
                "summary": f"Inline fallback failed: {exc}",
                "signal": "neutral",
                "confidence": 0.0,
                "error": str(exc),
            }

    # -----------------------------------------------------------------------
    # STEP 06: Analysis — synthesise all agent results
    # -----------------------------------------------------------------------

    async def _analyze_results(
        self,
        query: str,
        plan: OrchestratorPlan,
        agent_results: List[Dict[str, Any]],
        workspace: "SwarmWorkspace",
    ) -> Dict[str, Any]:
        """
        Run the analysis phase:
          • Aggregate signals from all agents (weighted voting)
          • Extract key findings
          • Compute overall confidence
          • Use LLM to cross-validate and extract insights
        """
        if not agent_results:
            return {
                "signal": "neutral",
                "confidence": 0.3,
                "key_findings": ["No agent results available."],
                "signal_breakdown": {},
                "warnings": ["All agents failed or returned empty results."],
            }

        succeeded = [r for r in agent_results if r.get("status") == "done"]

        # ── Signal aggregation (weighted voting) ──────────────────────────
        signal_weights = {
            "technical_analysis": 2.0,
            "prediction": 2.5,
            "oi_analysis": 1.8,
            "global_market": 1.5,
            "shock_detection": 2.0,  # shock bearish signal carries high weight
            "fundamentals": 1.5,
            "sentiment": 1.2,
            "web_research": 1.0,
            "data_fetch": 0.5,
        }

        bull_score = 0.0
        bear_score = 0.0
        total_weight = 0.0
        signal_breakdown: Dict[str, Any] = {}

        for r in succeeded:
            atype = r.get("agent_type", "")
            sig = r.get("signal", "neutral")
            conf = float(r.get("confidence", 0.5))
            w = signal_weights.get(atype, 1.0) * conf

            if sig == "bullish":
                bull_score += w
            elif sig == "bearish":
                bear_score += w
            total_weight += w

            signal_breakdown[r.get("task_name", atype)] = {
                "signal": sig,
                "confidence": round(conf, 3),
                "weight": w,
                "summary": (r.get("summary", ""))[:200],
            }

        # Final signal
        if total_weight > 0:
            net = (bull_score - bear_score) / total_weight
        else:
            net = 0.0

        if net > 0.3:
            overall_signal = "bullish"
        elif net < -0.3:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Confidence = weighted average of agent confidences (not bull/bear ratio)
        # This prevents confidence=0 when all agents return neutral signal
        if total_weight > 0:
            overall_confidence = min(
                1.0,
                sum(
                    float(r.get("confidence", 0.5)) * signal_weights.get(r.get("agent_type", ""), 1.0)
                    for r in succeeded
                ) / total_weight
            )
        else:
            overall_confidence = 0.3

        # ── Key findings extraction ───────────────────────────────────────
        key_findings = []
        for r in sorted(succeeded, key=lambda x: x.get("confidence", 0), reverse=True):
            summary = r.get("summary", "").strip()
            if summary and len(summary) > 30:
                # Take first meaningful line — strip raw markdown headers and symbols
                for line in summary.split("\n"):
                    line = line.strip()
                    # Skip raw markdown headers, empty lines, and very short lines
                    if not line or len(line) < 20:
                        continue
                    if line.startswith("#") or line.startswith("```") or line.startswith(":::"):
                        continue
                    if line.startswith("*Generated:") or line.startswith("**Query"):
                        continue
                    # Strip leading bullets/dashes
                    line = line.lstrip("•-*> ").strip()
                    if line and line not in key_findings:
                        key_findings.append(line[:250])
                        break
            if len(key_findings) >= 8:
                break

        # ── Warnings ─────────────────────────────────────────────────────
        warnings = []
        failed_count = len(agent_results) - len(succeeded)
        if failed_count > 0:
            warnings.append(
                f"{failed_count} agent(s) failed — results may be incomplete."
            )

        shock_result = next(
            (r for r in succeeded if r.get("agent_type") == "shock_detection"), None
        )
        if shock_result and shock_result.get("signal") == "bearish":
            warnings.append(
                "⚠️ Shock/anomaly detected — exercise extra caution. "
                + (shock_result.get("summary", "")[:150])
            )

        return {
            "signal": overall_signal,
            "confidence": round(overall_confidence, 3),
            "net_score": round(net, 3),
            "bull_score": round(bull_score, 3),
            "bear_score": round(bear_score, 3),
            "key_findings": key_findings,
            "signal_breakdown": signal_breakdown,
            "agents_succeeded": len(succeeded),
            "agents_failed": failed_count,
            "warnings": warnings,
        }

    # -----------------------------------------------------------------------
    # STEP 07+08: Report generation — asset sub-agent
    # -----------------------------------------------------------------------

    async def _generate_report(
        self,
        query: str,
        plan: OrchestratorPlan,
        agent_results: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        workspace: "SwarmWorkspace",
    ) -> Tuple[str, str]:
        """
        Spawn the ReportAgent (asset sub-agent) to produce the final report.
        Falls back to LLM synthesis if ReportAgent is unavailable.

        Returns:
            (final_response: str, report_md: str)
        """
        from .agent_registry import get_registry
        from .base_agent import SwarmMessage

        registry = get_registry()

        # Build a rich context for the report agent
        report_payload = {
            "query": query,
            "intent": plan.intent,
            "entities": plan.entities,
            "analysis": analysis,
            "agent_results": [
                {
                    "agent_type": r.get("agent_type"),
                    "task_name": r.get("task_name"),
                    "status": r.get("status"),
                    "summary": r.get("summary", "")[:3000],  # full context for LLM synthesis
                    "signal": r.get("signal"),
                    "confidence": r.get("confidence"),
                    "data": r.get("data", {}),
                }
                for r in agent_results
                if r.get("status") == "done"
            ],
            "key_findings": analysis.get("key_findings", []),
            "warnings": analysis.get("warnings", []),
            "plan_reasoning": plan.plan_reasoning,
        }

        try:
            if registry.type_exists("report"):
                msg = SwarmMessage(
                    task="Generate Final Report",
                    payload=report_payload,
                    timeout_s=120.0,
                )
                result = await registry.spawn(
                    agent_type="report",
                    message=msg,
                    parent_id="orchestrator",
                )
                if result.status.value == "done":
                    report_md = result.data.get("report_md", "")
                    final_resp = result.data.get("final_response", result.summary)
                    if report_md and final_resp:
                        return final_resp, report_md
        except Exception as exc:
            self._log.warning(
                f"ReportAgent failed: {exc} — falling back to LLM synthesis"
            )

        # Fallback: direct LLM synthesis
        return await self._llm_synthesise_report(query, plan, agent_results, analysis)

    async def _llm_synthesise_report(
        self,
        query: str,
        plan: OrchestratorPlan,
        agent_results: List[Dict[str, Any]],
        analysis: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Fallback LLM synthesis when ReportAgent is unavailable.
        Produces a comprehensive markdown report directly.
        """
        from .base_agent import AgentToolbox

        toolbox = AgentToolbox.instance()
        client = toolbox.get_llm_client()
        model = toolbox.get_model("reasoning")

        succeeded = [r for r in agent_results if r.get("status") == "done"]

        # Build context
        findings_text = "\n\n".join(
            f"### {r.get('task_name', r.get('agent_type', 'Agent'))}\n"
            f"Signal: {r.get('signal', 'neutral').upper()} | "
            f"Confidence: {r.get('confidence', 0.5):.0%}\n"
            f"{r.get('summary', 'No summary.')[:600]}"
            for r in succeeded
        )

        key_findings_text = "\n".join(
            f"• {f}" for f in analysis.get("key_findings", [])
        )

        warnings_text = (
            "\n".join(f"⚠️ {w}" for w in analysis.get("warnings", []))
            if analysis.get("warnings")
            else ""
        )

        system = """You are Daddy's AI — India's most advanced financial intelligence system.
You synthesise multi-agent research into a precise, actionable report for Indian investors.
Write with the authority of a senior analyst. Use ₹ for prices. Be specific with numbers.
Format beautifully with markdown — headers, bullets, tables where appropriate.
Always end with: ⚠️ *Not financial advice. Consult a SEBI-registered advisor.*"""

        user_prompt = f"""User asked: "{query}"
Intent: {plan.intent} | Signal: {analysis.get("signal", "neutral").upper()} | Confidence: {analysis.get("confidence", 0.5):.0%}

KEY FINDINGS:
{key_findings_text or "(No findings)"}

{warnings_text}

DETAILED AGENT FINDINGS:
{findings_text or "(No detailed findings)"}

Write a comprehensive, structured response that:
1. Directly answers the user's question
2. Includes all relevant data from the agent findings
3. Provides actionable insights with specific numbers
4. Uses proper markdown formatting (## headers, bullet points, tables)
5. Includes a clear verdict/recommendation
6. Mentions key levels (support/resistance) if available
7. Notes any warnings or risks

Total agents used: {len(succeeded)} | Total sub-agents: {plan.entities.get("total_agents", len(succeeded))}"""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=3000,
            )
            report_md = (response.choices[0].message.content or "").strip()
            # Final response is the report itself
            return report_md, report_md
        except Exception as exc:
            self._log.error(f"LLM synthesis failed: {exc}")
            # Last resort: join summaries
            fallback = f"## Analysis for: {query}\n\n"
            for r in succeeded[:5]:
                fallback += f"**{r.get('task_name', 'Agent')}**: {r.get('summary', '')[:300]}\n\n"
            fallback += "\n⚠️ *Not financial advice. Consult a SEBI-registered advisor.*"
            return fallback, fallback

    # -----------------------------------------------------------------------
    # Emergency fallback
    # -----------------------------------------------------------------------

    async def _emergency_fallback(self, query: str, error: str) -> str:
        """
        Last-resort response when the entire orchestration pipeline fails.
        Uses the existing simple web search as fallback.
        """
        try:
            from .base_agent import AgentToolbox

            toolbox = AgentToolbox.instance()
            client = toolbox.get_llm_client()
            model = toolbox.get_model("fast")

            web_result = await toolbox.web_search(f"{query} India stock market")

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial assistant. Answer based on the search results provided.",
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nSearch results:\n{str(web_result)[:2000]}",
                    },
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return (
                f"I encountered a technical issue processing your request. "
                f"Please try again or rephrase your query.\n\n"
                f"Query: {query}\n\n"
                f"⚠️ *Not financial advice.*"
            )

    # -----------------------------------------------------------------------
    # Workspace factory
    # -----------------------------------------------------------------------

    def _create_workspace(
        self, run_id: str, query: str, session_id: str
    ) -> "SwarmWorkspace":
        """Create a fresh workspace for this orchestration run."""
        return SwarmWorkspace(run_id=run_id, query=query, session_id=session_id)

    # -----------------------------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------------------------

    def _describe_entities(self, entities: Dict[str, Any]) -> str:
        parts = []
        syms = entities.get("symbols", [])
        if syms:
            parts.append(f"symbols={syms}")
        sector = entities.get("sector")
        if sector:
            parts.append(f"sector={sector}")
        topics = entities.get("topics", [])
        if topics:
            parts.append(f"topics={topics}")
        return ", ".join(parts) if parts else "general query"

    def _extract_symbols_simple(self, query: str) -> List[str]:
        """Quick regex-based symbol extraction (used in fallback planner)."""
        import re

        # Common NSE symbols pattern
        nse_known = [
            "NIFTY",
            "BANKNIFTY",
            "SENSEX",
            "FINNIFTY",
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "INFY",
            "ICICIBANK",
            "SBIN",
            "BHARTIARTL",
            "ITC",
            "KOTAKBANK",
            "LT",
            "HCLTECH",
            "AXISBANK",
            "MARUTI",
            "BAJFINANCE",
            "TITAN",
            "SUNPHARMA",
            "TATAMOTORS",
            "WIPRO",
            "TATASTEEL",
            "ADANIENT",
            "ADANIPORTS",
            "ONGC",
            "NTPC",
            "POWERGRID",
            "COALINDIA",
            "JSWSTEEL",
            "HINDALCO",
            "GRASIM",
            "APOLLOHOSP",
            "DIVISLAB",
            "DRREDDY",
            "CIPLA",
            "TECHM",
            "ULTRACEMCO",
            "ASIANPAINT",
            "NESTLEIND",
        ]
        q_upper = query.upper()
        found = [s for s in nse_known if s in q_upper]

        # Also catch unknown ALL-CAPS sequences (2–12 chars)
        extras = re.findall(r"\b[A-Z]{2,12}\b", query)
        # Filter to likely stock symbols (no common English words)
        stop_words = {
            "IN",
            "IS",
            "IT",
            "AT",
            "BE",
            "AN",
            "OR",
            "IF",
            "UP",
            "DO",
            "BY",
            "GO",
            "HI",
            "NO",
            "SO",
            "TO",
            "VS",
            "WAS",
            "THE",
            "FOR",
            "AND",
            "BUT",
            "NOT",
            "ARE",
            "HAS",
            "HAD",
            "THIS",
            "THAT",
            "WITH",
            "WHAT",
            "FROM",
            "HAVE",
            "WILL",
            "INDIA",
            "MARKET",
            "STOCK",
            "PRICE",
            "TODAY",
            "NIFTY",
        }
        for e in extras:
            if e not in stop_words and e not in found and len(e) >= 3:
                found.append(e)

        return found[:6]  # max 6 symbols


# ---------------------------------------------------------------------------
# Minimal SwarmWorkspace (inline stub used by orchestrator)
# ---------------------------------------------------------------------------


class SwarmWorkspace:
    """
    Lightweight in-memory workspace shared between orchestrator and agents
    during a single run. Stores findings, agent statuses, plan, and events.
    """

    def __init__(self, run_id: str, query: str, session_id: str = ""):
        self.run_id = run_id
        self.query = query
        self.session_id = session_id
        self._plan: Optional[OrchestratorPlan] = None
        self._entities: Dict[str, Any] = {}
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._findings: List[Dict[str, Any]] = []
        self._analysis: Dict[str, Any] = {}
        self._artifacts: Dict[str, Any] = {}
        self._events: List[Dict[str, Any]] = []
        self._event_callbacks: List[Callable] = []

    def set_plan(self, plan: OrchestratorPlan) -> None:
        self._plan = plan

    def get_plan(self) -> Optional[OrchestratorPlan]:
        return self._plan

    def write_entities(self, entities: Dict[str, Any]) -> None:
        self._entities.update(entities)

    def get_entities(self) -> Dict[str, Any]:
        return dict(self._entities)

    def register_agent(self, agent_id: str, agent_type: str, task_name: str) -> None:
        self._agents[agent_id] = {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "task_name": task_name,
            "status": "running",
            "summary": "",
            "signal": "neutral",
            "started_at": datetime.utcnow().isoformat(),
        }

    def register_agent_sync(self, agent_id: str, agent_type: str, task_name: str) -> None:
        """Alias for register_agent (sync version expected by _execute_plan)."""
        self.register_agent(agent_id, agent_type, task_name)

    def update_agent(
        self,
        agent_id: str,
        status: str,
        summary: str = "",
        signal: str = "neutral",
        error: str = "",
    ) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].update(
                {
                    "status": status,
                    "summary": summary,
                    "signal": signal,
                    "error": error,
                    "finished_at": datetime.utcnow().isoformat(),
                }
            )

    def get_agent_statuses(self) -> List[Dict[str, Any]]:
        return list(self._agents.values())

    def write_finding(
        self,
        agent_id: str,
        agent_type: str,
        finding_type: str,
        content: str,
        data: Dict[str, Any] = None,
        signal: str = "neutral",
        confidence: float = 0.5,
    ) -> None:
        self._findings.append(
            {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "finding_type": finding_type,
                "content": content,
                "data": data or {},
                "signal": signal,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def get_all_findings(self) -> List[Dict[str, Any]]:
        return list(self._findings)

    def write_analysis(self, analysis: Dict[str, Any]) -> None:
        self._analysis.update(analysis)

    def get_analysis(self) -> Dict[str, Any]:
        return dict(self._analysis)

    def get_artifacts(self) -> Dict[str, Any]:
        return dict(self._artifacts)

    def write_artifact(self, key: str, value: Any) -> None:
        self._artifacts[key] = value

    def emit(self, event: Dict[str, Any]) -> None:
        self._events.append(event)
        for cb in self._event_callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def subscribe(self, callback: Callable) -> None:
        self._event_callbacks.append(callback)

    def get_events(self) -> List[Dict[str, Any]]:
        return list(self._events)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_orchestrator: Optional[MasterOrchestrator] = None


def get_orchestrator() -> MasterOrchestrator:
    """Return the global MasterOrchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MasterOrchestrator()
    return _orchestrator
