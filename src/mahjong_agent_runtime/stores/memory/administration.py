"""InMemory administration store operations."""

from __future__ import annotations

from typing import Any
from ...models import new_id

class InMemoryAdministrationStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def clear_runtime_state(
        self,
        *,
        include_customers: bool = False,
        include_badcases: bool = False,
    ) -> dict[str, int]:
        with self._lock:
            deleted = {
                "games": len(self.games),
                "invite_drafts": len(self.invite_drafts),
                "outbound_message_drafts": len(self.outbound_message_drafts),
                "room_reservations": len(self.room_reservations),
                "state_transitions": len(self.transitions),
                "conversation_turns": sum(len(items) for items in self.turns.values()),
                "conversation_checkpoints": len(self.conversation_checkpoints),
                "task_context_checkpoints": len(self.task_context_checkpoints),
                "task_contexts": len(self.task_contexts),
                "conversation_versions": len(self.conversation_versions),
                "idempotency_ledger": len(self.idempotency_ledger),
                "message_results": len(self.message_results),
                "message_references": len(self.message_references),
                "task_memories": len(self.task_memories),
                "pending_memory_candidates": len(self.pending_memory_candidates),
                "pending_input_batches": len(self.pending_input_batches),
                "scheduled_tasks": len(self.scheduled_tasks),
                "waiting_demands": len(self.waiting_demands),
                "agent_runs": len(self.agent_runs),
                "customers": len(self.customers) if include_customers else 0,
                "customer_relationships": len(self.customer_relationships) if include_customers else 0,
                "badcases": len(self.badcases) if include_badcases else 0,
            }
            self.games.clear()
            self.invite_drafts.clear()
            self.outbound_message_drafts.clear()
            self.room_reservations.clear()
            self.transitions.clear()
            self.turns.clear()
            self.conversation_checkpoints.clear()
            self.task_context_checkpoints.clear()
            self.task_contexts.clear()
            self.conversation_versions.clear()
            self.idempotency_ledger.clear()
            self.idempotency_claimed_at.clear()
            self.message_results.clear()
            self.message_references.clear()
            self.task_memories.clear()
            self.pending_memory_candidates.clear()
            self.pending_input_batches.clear()
            self.scheduled_tasks.clear()
            self.waiting_demands.clear()
            self.agent_runs.clear()
            if include_customers:
                self.customers.clear()
                self.customer_relationships.clear()
            if include_badcases:
                self.badcases.clear()
            return deleted

    def record_badcase(self, payload: dict[str, Any], *, trace_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            record = {"badcase_id": new_id("badcase"), "trace_id": trace_id, "conversation_id": conversation_id, **dict(payload)}
            self.badcases.append(record)
            return record
