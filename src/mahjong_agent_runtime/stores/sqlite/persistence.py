"""SQLite persistence store operations."""

from __future__ import annotations

from ...models import (
    ConversationTaskContext,
    InviteDraft,
    MessageReference,
    OutboundMessageDraft,
    PendingInputBatch,
    PendingMemoryCandidate,
    ScheduledAgentTask,
    StateTransition,
    TaskMemory,
)
from .serialization import _dumps

class SQLitePersistenceStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def _save_scheduled_agent_task(self, task: ScheduledAgentTask) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_scheduled_agent_tasks(
                task_id, task_type, aggregate_id, conversation_id, status,
                due_at, lease_until, idempotency_key, payload, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                task_type=excluded.task_type,
                aggregate_id=excluded.aggregate_id,
                conversation_id=excluded.conversation_id,
                status=excluded.status,
                due_at=excluded.due_at,
                lease_until=excluded.lease_until,
                idempotency_key=excluded.idempotency_key,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                task.task_id,
                task.task_type,
                task.aggregate_id,
                task.conversation_id,
                task.status.value,
                task.due_at.isoformat(),
                task.lease_until.isoformat() if task.lease_until else None,
                task.idempotency_key,
                _dumps(task.to_dict()),
                task.updated_at.isoformat(),
            ),
        )

    def _save_invite(self, draft: InviteDraft) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_invite_drafts(draft_id, game_id, customer_id, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(draft_id) DO UPDATE SET
                game_id=excluded.game_id,
                customer_id=excluded.customer_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (draft.draft_id, draft.game_id, draft.customer_id, draft.status.value, _dumps(draft.to_dict()), draft.updated_at.isoformat()),
        )

    def _save_outbound_message_draft(self, draft: OutboundMessageDraft) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_outbound_message_drafts(draft_id, conversation_id, recipient_id, channel, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(draft_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                recipient_id=excluded.recipient_id,
                channel=excluded.channel,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                draft.draft_id,
                draft.conversation_id,
                draft.recipient_id,
                draft.channel,
                draft.status.value,
                _dumps(draft.to_dict()),
                draft.updated_at.isoformat(),
            ),
        )

    def _save_message_reference(self, reference: MessageReference) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_message_references(
                message_id,
                conversation_id,
                business_ref_type,
                business_ref_id,
                payload,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id, message_id) DO UPDATE SET
                business_ref_type=excluded.business_ref_type,
                business_ref_id=excluded.business_ref_id,
                payload=excluded.payload
            """,
            (
                reference.message_id,
                reference.conversation_id,
                reference.business_ref_type,
                reference.business_ref_id,
                _dumps(reference.to_dict()),
                reference.created_at.isoformat(),
            ),
        )

    def _save_task_context(self, context: ConversationTaskContext) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_task_contexts(
                task_context_id,
                conversation_id,
                customer_id,
                status,
                payload,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_context_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                customer_id=excluded.customer_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                context.task_context_id,
                context.conversation_id,
                context.customer_id,
                context.status,
                _dumps(context.to_dict()),
                context.updated_at.isoformat(),
            ),
        )

    def _save_task_memory(self, memory: TaskMemory) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_task_memories(memory_id, conversation_id, customer_id, target_customer_id, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                customer_id=excluded.customer_id,
                target_customer_id=excluded.target_customer_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                memory.memory_id,
                memory.conversation_id,
                memory.customer_id,
                memory.target_customer_id or "",
                memory.status,
                _dumps(memory.to_dict()),
                memory.updated_at.isoformat(),
            ),
        )

    def _save_pending_memory_candidate(self, candidate: PendingMemoryCandidate) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_pending_memory_candidates(
                candidate_id,
                conversation_id,
                customer_id,
                target_customer_id,
                status,
                risk_level,
                payload,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                customer_id=excluded.customer_id,
                target_customer_id=excluded.target_customer_id,
                status=excluded.status,
                risk_level=excluded.risk_level,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                candidate.candidate_id,
                candidate.conversation_id,
                candidate.customer_id,
                candidate.target_customer_id or "",
                candidate.status,
                candidate.risk_level,
                _dumps(candidate.to_dict()),
                candidate.updated_at.isoformat(),
            ),
        )

    def _save_pending_input_batch(self, batch: PendingInputBatch) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_pending_input_batches(
                batch_key,
                batch_id,
                conversation_id,
                sender_id,
                version,
                status,
                quiet_deadline,
                payload,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(batch_key) DO UPDATE SET
                batch_id=excluded.batch_id,
                conversation_id=excluded.conversation_id,
                sender_id=excluded.sender_id,
                version=excluded.version,
                status=excluded.status,
                quiet_deadline=excluded.quiet_deadline,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                batch.batch_key,
                batch.batch_id,
                batch.conversation_id,
                batch.sender_id,
                batch.version,
                batch.status.value,
                batch.quiet_deadline.isoformat(),
                _dumps(batch.to_dict()),
                batch.updated_at.isoformat(),
            ),
        )

    def _append_transition(self, transition: StateTransition) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_state_transitions(trace_id, entity_type, entity_id, occurred_at, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transition.trace_id,
                transition.entity_type,
                transition.entity_id,
                transition.occurred_at.isoformat(),
                _dumps(transition.to_dict()),
            ),
        )
