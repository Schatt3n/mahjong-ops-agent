"""实现内存存储层的智能体runs数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from datetime import datetime

from ...agent_state import AgentRunState, AgentRunStatus
from ...models import StateTransition, now


class InMemoryAgentRunStoreMixin:
    """使用与 SQLite 相同的租约/CAS 语义，以保证测试的确定性。"""

    __slots__ = ()

    def has_unfinished_user_goal(self, conversation_id: str, sender_id: str) -> bool:
        """只认当前版本的未结输入与运行，旧版遗留凭证不能永久触发主模型重评。"""
        with self._lock:
            current_version = self.conversation_version(conversation_id)
            if any(state.conversation_id == conversation_id and state.message.get("sender_id") == sender_id
                   and state.run_version == current_version
                   and state.status in {AgentRunStatus.RUNNING, AgentRunStatus.RECOVERABLE} for state in self.agent_runs.values()):
                return True
            return any(turn.role.value == "user" and turn.sender_id == sender_id and turn.metadata.get("input_receipt_key")
                       and turn.metadata.get("conversation_version") == current_version
                       and self.idempotent_message_result(str(turn.metadata["input_receipt_key"])) is None
                       for turn in self.turns.get(conversation_id, []))

    def create_agent_run(self, state: AgentRunState) -> AgentRunState:
        """创建智能体run，并返回持久化后的权威结果。"""
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
        """保存智能体run，确保后续读取能够得到一致快照。"""
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
        """原子领取智能体run，防止并发消费者重复处理。"""
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
