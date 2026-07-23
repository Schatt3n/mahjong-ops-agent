"""In-memory resumable Agent-run persistence."""

from __future__ import annotations

from datetime import datetime

from ...agent_state import AgentRunState, AgentRunStatus
from ...models import StateTransition, now


class InMemoryAgentRunStoreMixin:
    """Use the same lease/CAS semantics as SQLite for deterministic tests."""

    __slots__ = ()

    def create_agent_run(self, state: AgentRunState) -> AgentRunState:
        with self._lock:
            if state.run_id in self.agent_runs:
                raise ValueError(f"agent run already exists: {state.run_id}")
            stored = _copy_state(state)
            self.agent_runs[stored.run_id] = stored
            return _copy_state(stored)

    def agent_run(self, run_id: str) -> AgentRunState | None:
        with self._lock:
            state = self.agent_runs.get(run_id)
            return _copy_state(state) if state is not None else None

    def save_agent_run(
        self,
        state: AgentRunState,
        *,
        expected_lease_owner: str | None = None,
    ) -> bool:
        with self._lock:
            existing = self.agent_runs.get(state.run_id)
            if existing is None:
                return False
            if expected_lease_owner is not None and existing.lease_owner != expected_lease_owner:
                return False
            state.updated_at = now()
            self.agent_runs[state.run_id] = _copy_state(state)
            return True

    def recoverable_agent_runs(
        self,
        *,
        at: datetime,
        limit: int = 100,
    ) -> list[AgentRunState]:
        with self._lock:
            candidates = [
                state
                for state in self.agent_runs.values()
                if state.status == AgentRunStatus.RECOVERABLE
                or (
                    state.status == AgentRunStatus.RUNNING
                    and state.lease_until is not None
                    and state.lease_until <= at
                )
            ]
            candidates.sort(key=lambda item: (item.updated_at, item.run_id))
            return [_copy_state(item) for item in candidates[: max(1, int(limit))]]

    def claim_agent_run(
        self,
        run_id: str,
        *,
        lease_owner: str,
        lease_until: datetime,
        at: datetime,
    ) -> AgentRunState | None:
        with self._lock:
            state = self.agent_runs.get(run_id)
            if state is None:
                return None
            recoverable = state.status == AgentRunStatus.RECOVERABLE
            expired = (
                state.status == AgentRunStatus.RUNNING
                and state.lease_until is not None
                and state.lease_until <= at
            )
            if not recoverable and not expired:
                return None
            state.status = AgentRunStatus.RUNNING
            state.lease_owner = str(lease_owner)
            state.lease_until = lease_until
            state.attempts += 1
            state.updated_at = at
            return _copy_state(state)

    def supersede_agent_runs(
        self,
        conversation_id: str,
        *,
        current_version: int,
        trace_id: str,
    ) -> list[StateTransition]:
        transitions: list[StateTransition] = []
        with self._lock:
            for state in self.agent_runs.values():
                if (
                    state.conversation_id != conversation_id
                    or state.run_version >= int(current_version)
                    or state.status not in {AgentRunStatus.RUNNING, AgentRunStatus.RECOVERABLE}
                ):
                    continue
                old = state.status.value
                state.status = AgentRunStatus.SUPERSEDED
                state.runtime_status = AgentRunStatus.SUPERSEDED.value
                state.last_error = "newer conversation message superseded this execution"
                state.lease_owner = ""
                state.lease_until = None
                state.completed_at = now()
                state.updated_at = state.completed_at
                transition = StateTransition(
                    entity_type="agent_run",
                    entity_id=state.run_id,
                    from_status=old,
                    to_status=state.status.value,
                    reason="newer_conversation_version",
                    trace_id=trace_id,
                )
                self.transitions.append(transition)
                transitions.append(transition)
        return transitions


def _copy_state(state: AgentRunState) -> AgentRunState:
    return AgentRunState.from_dict(state.to_dict())


__all__ = ["InMemoryAgentRunStoreMixin"]
