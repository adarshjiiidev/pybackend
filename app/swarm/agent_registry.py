"""
AgentRegistry — Central registry for all DaddysAI swarm agents.

Responsibilities:
  • Catalog every available agent *type* (registered at import time)
  • Track every *live* agent instance by ID and status
  • Factory: instantiate any registered agent type by name
  • Lifecycle hooks: register spawn → mark done/failed → dispose
  • Telemetry: per-type stats (runs, avg duration, fail rate)
  • Concurrency guard: optional per-type concurrency limit
  • Auto-reap: background task that disposes DONE/FAILED agents after TTL

Usage::

    registry = get_registry()

    # Register a custom agent type (done once at module level)
    @registry.register("my_custom_agent")
    class MyAgent(BaseSwarmAgent): ...

    # Spawn an instance
    agent = registry.create("technical_analysis")
    result = await agent.run(SwarmMessage(task="..."))

    # Inspect live agents
    info = registry.get_all_info()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type

from .base_agent import AgentResult, AgentStatus, BaseSwarmAgent, SwarmMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent type descriptor
# ---------------------------------------------------------------------------


@dataclass
class AgentTypeDescriptor:
    """
    Metadata about a registered agent type.

    Attributes:
        name            Unique string key used to look up the type
        agent_class     The concrete subclass of BaseSwarmAgent
        description     Human-readable description of what the agent does
        max_concurrent  Max simultaneous live instances (0 = unlimited)
        default_timeout Default timeout in seconds for new instances
        tags            Free-form labels for grouping (e.g. ['analytics', 'technical'])
    """

    name: str
    agent_class: Type[BaseSwarmAgent]
    description: str = ""
    max_concurrent: int = 0  # 0 = unlimited
    default_timeout: float = 90.0
    tags: List[str] = field(default_factory=list)

    # Runtime stats (mutated by registry)
    total_runs: int = 0
    total_failures: int = 0
    total_duration_s: float = 0.0

    @property
    def avg_duration_s(self) -> float:
        runs = self.total_runs - self.total_failures
        return (self.total_duration_s / runs) if runs > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        return (self.total_failures / self.total_runs) if self.total_runs > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_concurrent": self.max_concurrent,
            "default_timeout": self.default_timeout,
            "tags": self.tags,
            "stats": {
                "total_runs": self.total_runs,
                "total_failures": self.total_failures,
                "avg_duration_s": round(self.avg_duration_s, 2),
                "failure_rate": round(self.failure_rate, 3),
            },
        }


# ---------------------------------------------------------------------------
# Live agent record
# ---------------------------------------------------------------------------


@dataclass
class LiveAgentRecord:
    """
    Tracks a single live (or recently completed) agent instance.
    """

    agent: BaseSwarmAgent
    spawned_at: datetime = field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None
    trace_id: str = ""
    task_description: str = ""

    @property
    def agent_id(self) -> str:
        return self.agent.agent_id

    @property
    def agent_type(self) -> str:
        return self.agent.AGENT_TYPE

    @property
    def status(self) -> AgentStatus:
        return self.agent.status

    @property
    def age_s(self) -> float:
        return (datetime.utcnow() - self.spawned_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "task": self.task_description,
            "spawned_at": self.spawned_at.isoformat(),
            "age_s": round(self.age_s, 1),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """
    Central registry for DaddysAI swarm agents.

    Thread-safe for asyncio single-loop usage.
    All mutating operations acquire an asyncio.Lock.
    """

    # How long a DONE/FAILED agent record is kept before auto-reap (seconds)
    REAP_TTL_S: float = 300.0  # 5 minutes

    # Interval between auto-reap passes (seconds)
    REAP_INTERVAL_S: float = 60.0

    def __init__(self) -> None:
        # type name → descriptor
        self._types: Dict[str, AgentTypeDescriptor] = {}

        # agent_id → live record
        self._live: Dict[str, LiveAgentRecord] = {}

        # concurrency semaphores keyed by type name
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

        self._lock = asyncio.Lock()
        self._reaper_task: Optional[asyncio.Task] = None
        self._started = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str = "",
        max_concurrent: int = 0,
        default_timeout: float = 90.0,
        tags: Optional[List[str]] = None,
    ) -> Callable[[Type[BaseSwarmAgent]], Type[BaseSwarmAgent]]:
        """
        Class decorator that registers an agent type.

        Usage::

            registry = get_registry()

            @registry.register(
                "technical_analysis",
                description="Runs full TA suite on OHLCV data",
                tags=["analytics", "technical"],
            )
            class TechnicalAnalysisAgent(BaseSwarmAgent):
                AGENT_TYPE = "technical_analysis"
                ...
        """

        def decorator(cls: Type[BaseSwarmAgent]) -> Type[BaseSwarmAgent]:
            descriptor = AgentTypeDescriptor(
                name=name,
                agent_class=cls,
                description=description,
                max_concurrent=max_concurrent,
                default_timeout=default_timeout,
                tags=tags or [],
            )
            self._types[name] = descriptor

            if max_concurrent > 0:
                self._semaphores[name] = asyncio.Semaphore(max_concurrent)

            logger.debug(f"Registered agent type: {name!r} → {cls.__name__}")
            return cls

        return decorator

    def register_class(
        self,
        agent_class: Type[BaseSwarmAgent],
        description: str = "",
        max_concurrent: int = 0,
        default_timeout: float = 90.0,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Directly register an agent class (no decorator syntax).
        Uses AGENT_TYPE attribute as the registry key.
        """
        name = agent_class.AGENT_TYPE
        descriptor = AgentTypeDescriptor(
            name=name,
            agent_class=agent_class,
            description=description,
            max_concurrent=max_concurrent,
            default_timeout=default_timeout,
            tags=tags or [],
        )
        self._types[name] = descriptor
        if max_concurrent > 0:
            self._semaphores[name] = asyncio.Semaphore(max_concurrent)
        logger.debug(f"Registered agent type (direct): {name!r}")

    # ------------------------------------------------------------------
    # Factory: create + track
    # ------------------------------------------------------------------

    def create(
        self,
        agent_type: str,
        agent_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseSwarmAgent:
        """
        Instantiate a registered agent type by name.

        Args:
            agent_type  Registered name (e.g. "technical_analysis")
            agent_id    Optional explicit ID; auto-generated if None
            parent_id   Parent agent ID (for child agents in a swarm)
            config      Optional config dict forwarded to the agent

        Returns:
            Unstarted agent instance (call await agent.run(message) to start)

        Raises:
            KeyError  if agent_type is not registered
        """
        descriptor = self._types.get(agent_type)
        if descriptor is None:
            available = sorted(self._types.keys())
            raise KeyError(
                f"Unknown agent type: {agent_type!r}. Available types: {available}"
            )

        effective_id = agent_id or f"{agent_type}-{str(uuid.uuid4())[:8]}"
        agent = descriptor.agent_class(
            agent_id=effective_id,
            parent_id=parent_id,
            config=config or {},
        )
        # Override timeout from descriptor if not set in config
        if hasattr(agent, "DEFAULT_TIMEOUT_S"):
            agent.DEFAULT_TIMEOUT_S = descriptor.default_timeout

        return agent

    async def spawn(
        self,
        agent_type: str,
        message: SwarmMessage,
        parent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Full spawn lifecycle:
          1. create() the agent
          2. Register it as live
          3. Acquire concurrency semaphore (if limit set)
          4. run() it
          5. Record result stats
          6. Return result (agent stays in live dict until reaped)

        This is the primary way the orchestrator launches agents.
        """
        agent = self.create(agent_type, parent_id=parent_id, config=config)
        record = LiveAgentRecord(
            agent=agent,
            parent_id=parent_id,
            trace_id=message.trace_id,
            task_description=message.task,
        )

        async with self._lock:
            self._live[agent.agent_id] = record

        semaphore = self._semaphores.get(agent_type)

        try:
            if semaphore:
                async with semaphore:
                    result = await agent.run(message)
            else:
                result = await agent.run(message)

            # Update type stats
            descriptor = self._types[agent_type]
            descriptor.total_runs += 1
            if result.status == AgentStatus.FAILED:
                descriptor.total_failures += 1
            descriptor.total_duration_s += result.duration_s

            return result

        except Exception as exc:
            # Should not reach here (base.run() catches everything),
            # but guard anyway.
            logger.error(
                f"Registry.spawn: unhandled exception for {agent.agent_id}: {exc}"
            )
            async with self._lock:
                self._live.pop(agent.agent_id, None)
            raise

    async def spawn_parallel(
        self,
        tasks: List[Dict[str, Any]],
        parent_id: Optional[str] = None,
    ) -> List[AgentResult]:
        """
        Spawn multiple agents in parallel and return all results.

        Args:
            tasks: list of dicts, each with keys:
                     agent_type (str)
                     message    (SwarmMessage)
                     config     (dict, optional)
            parent_id: common parent for all spawned agents

        Returns:
            List of AgentResult in the same order as tasks.
        """
        coroutines = [
            self.spawn(
                agent_type=t["agent_type"],
                message=t["message"],
                parent_id=parent_id,
                config=t.get("config"),
            )
            for t in tasks
        ]

        raw_results = await asyncio.gather(*coroutines, return_exceptions=True)

        results: List[AgentResult] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                atype = tasks[i]["agent_type"] if i < len(tasks) else "unknown"
                results.append(
                    AgentResult(
                        agent_id=f"{atype}-error",
                        agent_type=atype,
                        status=AgentStatus.FAILED,
                        error=str(r),
                        summary=f"Spawn failed: {r}",
                    )
                )
            else:
                results.append(r)

        return results

    # ------------------------------------------------------------------
    # Disposal
    # ------------------------------------------------------------------

    async def dispose_agent(self, agent_id: str) -> bool:
        """
        Dispose a specific live agent by ID.

        Returns True if found and disposed, False if not found.
        """
        async with self._lock:
            record = self._live.get(agent_id)
            if record is None:
                return False

        try:
            await record.agent.dispose()
        except Exception as exc:
            logger.warning(f"Error disposing agent {agent_id}: {exc}")

        async with self._lock:
            self._live.pop(agent_id, None)

        logger.info(f"🗑 Disposed agent {agent_id}")
        return True

    async def dispose_all(
        self,
        agent_type: Optional[str] = None,
        status_filter: Optional[List[AgentStatus]] = None,
    ) -> int:
        """
        Dispose all agents (optionally filtered by type and/or status).

        Returns:
            Number of agents disposed.
        """
        async with self._lock:
            candidates = {
                aid: rec
                for aid, rec in list(self._live.items())
                if (agent_type is None or rec.agent_type == agent_type)
                and (status_filter is None or rec.agent.status in status_filter)
            }
            for aid in candidates:
                self._live.pop(aid, None)

        count = 0
        for record in candidates.values():
            try:
                await record.agent.dispose()
                count += 1
            except Exception as exc:
                logger.warning(f"Error disposing {record.agent_id}: {exc}")

        logger.info(
            f"Disposed {count} agent(s) (type={agent_type}, filter={status_filter})"
        )
        return count

    # ------------------------------------------------------------------
    # Inspection / telemetry
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[BaseSwarmAgent]:
        """Return a live agent instance by ID, or None."""
        record = self._live.get(agent_id)
        return record.agent if record else None

    def get_live_agents(
        self,
        agent_type: Optional[str] = None,
        status_filter: Optional[List[AgentStatus]] = None,
    ) -> List[LiveAgentRecord]:
        """Return live agent records, optionally filtered."""
        records = list(self._live.values())
        if agent_type:
            records = [r for r in records if r.agent_type == agent_type]
        if status_filter:
            records = [r for r in records if r.agent.status in status_filter]
        return records

    def get_all_info(self) -> Dict[str, Any]:
        """
        Return a full snapshot of the registry for health checks and dashboards.
        """
        live_by_type: Dict[str, int] = defaultdict(int)
        for rec in self._live.values():
            live_by_type[rec.agent_type] += 1

        return {
            "registered_types": {
                name: desc.to_dict() for name, desc in self._types.items()
            },
            "live_agents": [r.to_dict() for r in self._live.values()],
            "live_count": len(self._live),
            "live_by_type": dict(live_by_type),
            "running_count": sum(
                1 for r in self._live.values() if r.agent.status == AgentStatus.RUNNING
            ),
        }

    def list_types(self) -> List[Dict[str, Any]]:
        """Return all registered agent type descriptors as dicts."""
        return [desc.to_dict() for desc in self._types.values()]

    def type_exists(self, agent_type: str) -> bool:
        return agent_type in self._types

    # ------------------------------------------------------------------
    # Auto-reaper background task
    # ------------------------------------------------------------------

    async def start_reaper(self) -> None:
        """
        Start the background auto-reaper task.
        Call once at application startup (via lifespan).
        """
        if self._started:
            return
        self._started = True
        self._reaper_task = asyncio.create_task(
            self._reaper_loop(), name="agent-registry-reaper"
        )
        logger.info("🧹 AgentRegistry auto-reaper started")

    async def stop_reaper(self) -> None:
        """
        Stop the background auto-reaper. Call at shutdown.
        """
        if self._reaper_task and not self._reaper_task.done():
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
        self._started = False
        logger.info("AgentRegistry auto-reaper stopped")

    async def _reaper_loop(self) -> None:
        """
        Periodically scan for DONE / FAILED agents older than REAP_TTL_S
        and dispose them to free memory.
        """
        while True:
            try:
                await asyncio.sleep(self.REAP_INTERVAL_S)
                await self._reap_stale()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"Reaper loop error: {exc}")

    async def _reap_stale(self) -> None:
        """Dispose agents in terminal states that have been sitting idle too long."""
        terminal_statuses = {AgentStatus.DONE, AgentStatus.FAILED}
        now = datetime.utcnow()
        to_reap: List[str] = []

        async with self._lock:
            for aid, rec in self._live.items():
                if rec.agent.status in terminal_statuses:
                    age = (now - rec.spawned_at).total_seconds()
                    if age > self.REAP_TTL_S:
                        to_reap.append(aid)

        for aid in to_reap:
            await self.dispose_agent(aid)

        if to_reap:
            logger.debug(f"🧹 Reaped {len(to_reap)} stale agent(s)")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    """Return the global AgentRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
