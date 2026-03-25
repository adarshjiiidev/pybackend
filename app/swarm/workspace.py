"""
SwarmWorkspace — Shared Agent Memory & Results Board
=====================================================
Phase 2 · Swarm Core

The Workspace is the central "whiteboard" shared by all agents in a swarm run.
Every agent writes its findings here. The Orchestrator reads everything.

Analogous to:
  • The "results.csv" in the image diagram
  • A shared filesystem in a multi-process system
  • A bulletin board in a trading war-room

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                    SwarmWorkspace                       │
  │                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
  │  │  agent_logs  │  │   findings   │  │   artifacts  │  │
  │  │  (timeline)  │  │  (key facts) │  │  (reports,   │  │
  │  │              │  │              │  │   charts)    │  │
  │  └──────────────┘  └──────────────┘  └──────────────┘  │
  │                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
  │  │  agent_stats │  │  data_store  │  │  task_plan   │  │
  │  │  (status,    │  │  (OHLCV,     │  │  (what was   │  │
  │  │   timing)    │  │   quotes)    │  │   planned)   │  │
  │  └──────────────┘  └──────────────┘  └──────────────┘  │
  └─────────────────────────────────────────────────────────┘

Thread / asyncio safety: All writes use asyncio.Lock().
Observers: Register callbacks to receive real-time event notifications.

Usage::

    ws = SwarmWorkspace(run_id="run-abc123")

    # Agent writes a finding
    await ws.write_finding("web_research-001", "RELIANCE Q3 profit ₹19,000cr", source="ET")

    # Agent updates its status
    await ws.update_agent_status("technical_analysis-001", "running", progress=0.4)

    # Agent stores raw data
    await ws.store_data("RELIANCE_ohlcv", df.to_dict())

    # Agent saves an artifact (report text, chart data, etc.)
    await ws.save_artifact("final_report", content=report_md, artifact_type="markdown")

    # Orchestrator reads everything
    findings = ws.get_all_findings()
    agents   = ws.get_all_agent_statuses()
    plan     = ws.get_task_plan()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data containers
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Workspace event types broadcast to observers."""

    AGENT_SPAWNED = "agent_spawned"
    AGENT_STATUS = "agent_status"
    AGENT_DONE = "agent_done"
    AGENT_FAILED = "agent_failed"
    FINDING_ADDED = "finding_added"
    ARTIFACT_SAVED = "artifact_saved"
    DATA_STORED = "data_stored"
    PLAN_SET = "plan_set"
    PHASE_CHANGED = "phase_changed"
    RUN_COMPLETED = "run_completed"
    ERROR = "error"


@dataclass
class WorkspaceEvent:
    """A single workspace event (appended to the timeline)."""

    event_type: EventType
    agent_id: Optional[str]
    timestamp: datetime
    payload: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "run_id": self.run_id,
        }


@dataclass
class Finding:
    """
    A single piece of intelligence written by an agent.

    Attributes:
        finding_id   Auto-generated UUID
        agent_id     Which agent wrote this
        agent_type   Type label of the writing agent
        content      The actual finding text or structured data
        source       Where the finding came from (URL, tool name, etc.)
        confidence   0.0–1.0 reliability score
        signal       'bullish' | 'bearish' | 'neutral' | None
        tags         Free-form labels (e.g. ['price', 'RELIANCE', 'fundamental'])
        timestamp    When it was written
        finding_type 'fact' | 'signal' | 'data' | 'prediction' | 'alert'
    """

    agent_id: str
    agent_type: str
    content: Any  # str or dict
    source: str = ""
    confidence: float = 0.5
    signal: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    finding_type: str = "fact"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "content": self.content,
            "source": self.source,
            "confidence": round(self.confidence, 3),
            "signal": self.signal,
            "tags": self.tags,
            "finding_type": self.finding_type,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AgentStatus:
    """Live status of a single agent in the swarm."""

    agent_id: str
    agent_type: str
    status: str = "idle"  # idle | running | done | failed | disposed
    progress: float = 0.0  # 0.0–1.0
    task: str = ""  # current task description
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error: Optional[str] = None
    result_signal: Optional[str] = None
    result_confidence: float = 0.0
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status,
            "progress": round(self.progress, 3),
            "task": self.task,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "error": self.error,
            "result_signal": self.result_signal,
            "result_confidence": round(self.result_confidence, 3),
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class Artifact:
    """
    A generated artifact: report, chart, CSV, etc.

    Agents save artifacts here; the Report Agent reads them to build
    the final output delivered to the user.
    """

    artifact_id: str
    name: str
    content: Any
    artifact_type: str = "text"  # text | markdown | json | csv | html | pdf_data
    created_by: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        # Don't include full content in to_dict to keep summaries small
        content_preview = (
            str(self.content)[:200] + "…"
            if len(str(self.content)) > 200
            else str(self.content)
        )
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "artifact_type": self.artifact_type,
            "created_by": self.created_by,
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "content_preview": content_preview,
        }


@dataclass
class TaskPlan:
    """
    The orchestrator's task plan for this run.

    Contains:
      - The user's original query
      - The orchestrator's interpretation
      - List of planned agent tasks (in order or parallel groups)
      - Current execution phase
    """

    query: str
    intent: str = ""
    complexity: str = "moderate"  # simple | moderate | complex | deep
    agent_tasks: List[Dict[str, Any]] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    current_phase: str = "planning"  # planning | research | analysis | reporting | done
    estimated_agents: int = 0
    estimated_time_s: float = 30.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "complexity": self.complexity,
            "agent_tasks": self.agent_tasks,
            "parallel_groups": self.parallel_groups,
            "current_phase": self.current_phase,
            "estimated_agents": self.estimated_agents,
            "estimated_time_s": self.estimated_time_s,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# SwarmWorkspace
# ---------------------------------------------------------------------------


class SwarmWorkspace:
    """
    Central shared memory for a single swarm run.

    One workspace is created per user query by the MasterOrchestrator.
    All agents in that run share the same workspace instance.

    Lifecycle:
        ws = SwarmWorkspace(run_id="run-xyz")
        ws.set_task_plan(plan)
        # ... agents spawn and write to workspace ...
        summary = ws.get_summary()
        await ws.close()
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        query: str = "",
        session_id: str = "",
    ) -> None:
        self.run_id: str = run_id or f"run-{str(uuid.uuid4())[:8]}"
        self.query: str = query
        self.session_id: str = session_id
        self.created_at = datetime.utcnow()
        self.closed_at: Optional[datetime] = None

        # ── Core stores ───────────────────────────────────────────────────
        self._findings: List[Finding] = []
        self._agent_stats: Dict[str, AgentStatus] = {}
        self._artifacts: Dict[str, Artifact] = {}
        self._data_store: Dict[str, Any] = {}
        self._timeline: List[WorkspaceEvent] = []
        self._task_plan: Optional[TaskPlan] = None
        self._phase: str = "planning"
        self._phase_log: List[Dict[str, Any]] = []

        # ── Observers (callbacks) ─────────────────────────────────────────
        self._observers: List[Callable[[WorkspaceEvent], None]] = []

        # ── Concurrency ───────────────────────────────────────────────────
        self._lock = asyncio.Lock()

        self._log = logging.getLogger(f"workspace.{self.run_id}")
        self._log.info(f"🗂 Workspace {self.run_id} created for query: {query[:80]!r}")

    # ------------------------------------------------------------------
    # Task Plan
    # ------------------------------------------------------------------

    async def set_task_plan(self, plan: TaskPlan) -> None:
        """Set the orchestrator's task plan for this run."""
        async with self._lock:
            self._task_plan = plan
        try:
            await self._emit(
                EventType.PLAN_SET,
                None,
                {
                    "complexity": plan.complexity,
                    "estimated_agents": plan.estimated_agents,
                    "agent_tasks": len(plan.agent_tasks),
                },
            )
        except Exception:
            pass
        self._log.info(
            f"📋 Task plan set | complexity={plan.complexity} | "
            f"agents={plan.estimated_agents} | tasks={len(plan.agent_tasks)}"
        )

    def get_task_plan(self) -> Optional[TaskPlan]:
        return self._task_plan

    async def set_phase(self, phase: str) -> None:
        """
        Update the current execution phase and log the transition.
        Phases: planning → research → analysis → reporting → done
        """
        old_phase = self._phase
        async with self._lock:
            self._phase = phase
            self._phase_log.append(
                {
                    "from": old_phase,
                    "to": phase,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        try:
            await self._emit(
                EventType.PHASE_CHANGED,
                None,
                {
                    "from": old_phase,
                    "to": phase,
                },
            )
        except Exception:
            pass
        self._log.info(f"🔄 Phase: {old_phase} → {phase}")

    @property
    def current_phase(self) -> str:
        return self._phase

    # ------------------------------------------------------------------
    # Agent status tracking
    # ------------------------------------------------------------------

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        task: str = "",
    ) -> None:
        """Register a newly spawned agent in the workspace."""
        status = AgentStatus(
            agent_id=agent_id,
            agent_type=agent_type,
            status="idle",
            task=task,
        )
        async with self._lock:
            self._agent_stats[agent_id] = status

        try:
            await self._emit(
                EventType.AGENT_SPAWNED,
                agent_id,
                {
                    "agent_type": agent_type,
                    "task": task,
                },
            )
        except Exception:
            pass
        self._log.debug(f"🐣 Registered agent: {agent_id} ({agent_type})")

    async def update_agent_status(
        self,
        agent_id: str,
        status: str,
        progress: Optional[float] = None,
        task: Optional[str] = None,
        error: Optional[str] = None,
        signal: Optional[str] = None,
        confidence: Optional[float] = None,
        duration_s: Optional[float] = None,
    ) -> None:
        """Update an agent's live status (progress, phase, error, etc.)."""
        async with self._lock:
            rec = self._agent_stats.get(agent_id)
            if rec is None:
                rec = AgentStatus(agent_id=agent_id, agent_type="unknown")
                self._agent_stats[agent_id] = rec

            rec.status = status
            if progress is not None:
                rec.progress = min(1.0, max(0.0, float(progress)))
            if task is not None:
                rec.task = str(task)
            if error is not None:
                rec.error = str(error)
            if signal is not None:
                rec.result_signal = str(signal)
            if confidence is not None:
                rec.result_confidence = float(confidence)
            if duration_s is not None:
                rec.duration_s = float(duration_s)

            if status == "running" and rec.started_at is None:
                rec.started_at = datetime.utcnow()
            elif status in ("done", "failed", "disposed"):
                rec.ended_at = datetime.utcnow()
                if rec.started_at and not duration_s:
                    rec.duration_s = (rec.ended_at - rec.started_at).total_seconds()

        # Emit appropriate event
        event_type = EventType.AGENT_STATUS
        if status == "done":
            event_type = EventType.AGENT_DONE
        elif status == "failed":
            event_type = EventType.AGENT_FAILED

        try:
            await self._emit(
                event_type,
                agent_id,
                {
                    "status": status,
                    "progress": progress,
                    "task": task,
                    "error": error,
                },
            )
        except Exception:
            pass

    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        return self._agent_stats.get(agent_id)

    def get_all_agent_statuses(self) -> List[AgentStatus]:
        return list(self._agent_stats.values())

    def get_running_agents(self) -> List[AgentStatus]:
        return [s for s in self._agent_stats.values() if s.status == "running"]

    def get_done_agents(self) -> List[AgentStatus]:
        return [s for s in self._agent_stats.values() if s.status == "done"]

    def get_failed_agents(self) -> List[AgentStatus]:
        return [s for s in self._agent_stats.values() if s.status == "failed"]

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    async def write_finding(
        self,
        agent_id: str,
        content: Any,
        agent_type: str = "",
        source: str = "",
        confidence: float = 0.5,
        signal: Optional[str] = None,
        tags: Optional[List[str]] = None,
        finding_type: str = "fact",
    ) -> Finding:
        """
        Write a new finding to the workspace.

        Any agent can call this; the orchestrator reads all findings
        in the analysis phase.

        Args:
            agent_id:     ID of the writing agent
            content:      The finding — text string or structured dict
            agent_type:   Agent type label
            source:       Where the finding came from
            confidence:   0.0–1.0 reliability score
            signal:       Optional directional signal
            tags:         Labels for filtering / grouping
            finding_type: 'fact' | 'signal' | 'data' | 'prediction' | 'alert'

        Returns:
            The Finding object (for reference if needed)
        """
        # Resolve agent_type from registry if not supplied
        if not agent_type:
            rec = self._agent_stats.get(agent_id)
            agent_type = rec.agent_type if rec else "unknown"

        finding = Finding(
            agent_id=agent_id,
            agent_type=agent_type,
            content=content,
            source=source,
            confidence=confidence,
            signal=signal,
            tags=tags or [],
            finding_type=finding_type,
        )

        async with self._lock:
            self._findings.append(finding)

        try:
            await self._emit(
                EventType.FINDING_ADDED,
                agent_id,
                {
                    "finding_id": finding.finding_id,
                    "finding_type": finding_type,
                    "signal": signal,
                    "confidence": confidence,
                    "content_preview": str(content)[:120],
                },
            )
        except Exception:
            pass
        self._log.debug(
            f"✍ Finding from {agent_id} [{finding_type}] confidence={confidence:.2f}"
        )
        return finding

    def get_all_findings(
        self,
        agent_type: Optional[str] = None,
        finding_type: Optional[str] = None,
        signal_filter: Optional[str] = None,
        min_confidence: float = 0.0,
        tags_any: Optional[List[str]] = None,
    ) -> List[Finding]:
        """
        Retrieve findings with optional filters.

        Args:
            agent_type:     Only findings from agents of this type
            finding_type:   Only findings of this type (fact/signal/etc.)
            signal_filter:  Only findings with this signal
            min_confidence: Minimum confidence threshold
            tags_any:       At least one of these tags must be present

        Returns:
            Sorted list of Finding objects (newest first)
        """
        results = list(self._findings)

        if agent_type:
            results = [f for f in results if f.agent_type == agent_type]
        if finding_type:
            results = [f for f in results if f.finding_type == finding_type]
        if signal_filter:
            results = [f for f in results if f.signal == signal_filter]
        if min_confidence > 0:
            results = [f for f in results if f.confidence >= min_confidence]
        if tags_any:
            tag_set = set(tags_any)
            results = [f for f in results if tag_set & set(f.tags)]

        return sorted(results, key=lambda f: f.timestamp, reverse=True)

    def get_findings_as_text(
        self,
        max_findings: int = 50,
        min_confidence: float = 0.3,
    ) -> str:
        """
        Return all findings formatted as a readable text block.
        Used by the orchestrator when prompting the LLM for synthesis.
        """
        findings = self.get_all_findings(min_confidence=min_confidence)[:max_findings]
        if not findings:
            return "(No findings available)"

        lines = []
        for f in findings:
            signal_tag = f" [{f.signal.upper()}]" if f.signal else ""
            src_tag = f" | src: {f.source}" if f.source else ""
            content_str = str(f.content)[:200] if f.content else ""
            lines.append(
                f"[{f.agent_type}{signal_tag}] conf={f.confidence:.2f}{src_tag}: {content_str}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    async def write_artifact(
        self,
        key: str,
        artifact_type: str,
        content: Any,
        agent_id: str = "",
        description: str = "",
        format: str = "json",
    ) -> Artifact:
        """
        Store a named artifact (chart, table, report section, CSV data).

        Args:
            key:           Unique name for this artifact
            artifact_type: 'chart' | 'table' | 'report' | 'data' | 'csv'
            content:       The artifact content
            agent_id:      Which agent produced it
            description:   Human-readable description
            format:        Data format hint

        Returns:
            The Artifact object
        """
        import uuid as _uuid

        artifact = Artifact(
            artifact_id=str(_uuid.uuid4())[:12],
            name=key,
            content=content,
            artifact_type=artifact_type,
            created_by=agent_id,
        )
        async with self._lock:
            self._artifacts[key] = artifact

        try:
            await self._emit(
                EventType.ARTIFACT_SAVED,
                agent_id,
                {
                    "key": key,
                    "artifact_type": artifact_type,
                    "description": description,
                },
            )
        except Exception:
            pass
        self._log.debug(f"📎 Artifact written: {key} ({artifact_type})")
        return artifact

    def get_artifact(self, key: str) -> Optional[Artifact]:
        return self._artifacts.get(key)

    def get_all_artifacts(self) -> List[Artifact]:
        return list(self._artifacts.values())

    def get_artifacts(self) -> Dict[str, Any]:
        """Return artifacts as a plain dict (for serialization)."""
        return {
            k: {
                "key": a.name,
                "type": a.artifact_type,
                "created_by": a.created_by,
                "content": a.content
                if not isinstance(a.content, bytes)
                else "<binary>",
                "created_at": a.timestamp.isoformat(),
            }
            for k, a in self._artifacts.items()
        }

    # ------------------------------------------------------------------
    # Data store (generic key-value)
    # ------------------------------------------------------------------

    async def store(self, key: str, value: Any) -> None:
        """Store arbitrary data accessible to all agents."""
        async with self._lock:
            self._data_store[key] = value

    def load(self, key: str, default: Any = None) -> Any:
        """Retrieve data from the shared store."""
        return self._data_store.get(key, default)

    def set_plan(self, plan: Any) -> None:
        """Set plan (accepts OrchestratorPlan or TaskPlan)."""
        self._task_plan = plan

    def write_entities(self, entities: Dict[str, Any]) -> None:
        """Write extracted entities to the data store."""
        self._data_store["entities"] = entities

    def write_analysis(self, analysis: Dict[str, Any]) -> None:
        """Write analysis results to the data store."""
        self._data_store["analysis"] = analysis

    def register_agent_sync(
        self, agent_id: str, agent_type: str, task_name: str
    ) -> None:
        """Sync register (for orchestrator compatibility)."""
        from datetime import datetime as _dt

        self._agent_stats[agent_id] = AgentStatus(
            agent_id=agent_id,
            agent_type=agent_type,
            status="running",
            task=task_name,
            started_at=_dt.utcnow(),
        )

    def update_agent(
        self,
        agent_id: str,
        status: str,
        summary: str = "",
        signal: str = "neutral",
        error: str = "",
    ) -> None:
        """Sync update agent status (for orchestrator compatibility)."""
        from datetime import datetime

        rec = self._agent_stats.get(agent_id)
        if rec:
            rec.status = status
            rec.result_signal = signal
            rec.error = error
            if status in ("done", "failed"):
                rec.ended_at = datetime.utcnow()
                if rec.started_at:
                    rec.duration_s = (rec.ended_at - rec.started_at).total_seconds()

    def get_agent_statuses(self) -> List[Dict[str, Any]]:
        """Return agent statuses as list of dicts."""
        return [s.to_dict() for s in self._agent_stats.values()]

    # ------------------------------------------------------------------
    # Event bus
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to workspace events."""
        self._observers.append(callback)

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit a plain event dict (used by orchestrator)."""
        ws_event = WorkspaceEvent(
            event_type=EventType.AGENT_STATUS,
            agent_id=None,
            payload=event,
            timestamp=datetime.utcnow(),
        )
        self._timeline.append(ws_event)
        for obs in self._observers:
            try:
                obs(event)
            except Exception:
                pass

    async def _emit(
        self,
        event_type: "EventType",
        agent_id: Optional[str],
        payload: Dict[str, Any],
    ) -> None:
        """Internal async event emitter."""
        event = WorkspaceEvent(
            event_type=event_type,
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            payload=payload,
        )
        async with self._lock:
            self._timeline.append(event)

        for obs in self._observers:
            try:
                obs(event.to_dict())
            except Exception:
                pass

    def get_events(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._timeline]

    # ------------------------------------------------------------------
    # Snapshot / serialization
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a complete snapshot of the workspace state.
        Used for debugging, logging, and final report assembly.
        """
        return {
            "run_id": self.run_id,
            "query": self.query,
            "session_id": self.session_id,
            "phase": self._phase,
            "phase_log": self._phase_log,
            "agents": {aid: s.to_dict() for aid, s in self._agent_stats.items()},
            "findings_count": len(self._findings),
            "findings": [
                {
                    "agent_id": f.agent_id,
                    "agent_type": f.agent_type,
                    "finding_type": f.finding_type,
                    "signal": f.signal,
                    "confidence": f.confidence,
                    "content": str(f.content)[:300],
                    "timestamp": f.timestamp.isoformat(),
                }
                for f in self._findings[:50]
            ],
            "artifacts": list(self._artifacts.keys()),
            "data_store_keys": list(self._data_store.keys()),
            "timeline_events": len(self._timeline),
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"SwarmWorkspace(run_id={self.run_id!r}, phase={self._phase!r}, "
            f"agents={len(self._agent_stats)}, findings={len(self._findings)})"
        )
