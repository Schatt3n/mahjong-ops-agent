"""Persistence contract for resumable Agent loop executions."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..agent_state import AgentRunState
from ..models import StateTransition


class AgentRunStore(Protocol):
    """Store checkpoints using an owner lease and conversation-version CAS."""

    def create_agent_run(self, state: AgentRunState) -> AgentRunState: ...

    def agent_run(self, run_id: str) -> AgentRunState | None: ...

    def save_agent_run(
        self,
        state: AgentRunState,
        *,
        expected_lease_owner: str | None = None,
    ) -> bool: ...

    def recoverable_agent_runs(
        self,
        *,
        at: datetime,
        limit: int = 100,
    ) -> list[AgentRunState]: ...

    def claim_agent_run(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_until: datetime,
        at: datetime,
    ) -> AgentRunState | None: ...

    def supersede_agent_runs(
        self,
        conversation_id: str,
        *,
        current_version: int,
        trace_id: str,
    ) -> list[StateTransition]: ...


__all__ = ["AgentRunStore"]
