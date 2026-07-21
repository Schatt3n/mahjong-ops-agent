"""SQLite conversation store operations."""

from __future__ import annotations

from typing import Any
from datetime import datetime
from ...models import (
    ConversationCheckpoint,
    ConversationRole,
    ConversationTaskContext,
    ConversationTurn,
    DEFAULT_TZ,
    InviteStatus,
    OutboundDraftStatus,
    StateTransition,
    new_id,
)
from .serialization import (
    _checkpoint_from_payload,
    _dumps,
    _loads,
    _now_iso,
    _task_context_from_payload,
    _task_memory_from_payload,
    _turn_from_payload,
)

class SQLiteConversationStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def append_user_turn(self, message, trace_id: str) -> None:
        task_context = self.current_task_context(message.conversation_id, message.sender_id)
        metadata = dict(getattr(message, "metadata", {}) or {})
        if task_context is not None:
            metadata["task_context_id"] = task_context.task_context_id
        self.append_turn(
            message.conversation_id,
            ConversationTurn(
                role=ConversationRole.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                metadata=metadata,
                occurred_at=message.sent_at,
            ),
        )

    def append_assistant_turn(
        self,
        conversation_id: str,
        text: str,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        task_context = self.latest_task_context(conversation_id)
        turn_metadata = dict(metadata or {})
        if task_context is not None:
            turn_metadata.setdefault("task_context_id", task_context.task_context_id)
        self.append_turn(
            conversation_id,
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                content=text,
                trace_id=trace_id,
                metadata=turn_metadata,
            ),
        )

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        task_context = self.latest_task_context(conversation_id)
        metadata = {"task_context_id": task_context.task_context_id} if task_context else {}
        self.append_turn(
            conversation_id,
            ConversationTurn(role=ConversationRole.TOOL, content=text, trace_id=trace_id, metadata=metadata),
        )

    def append_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_conversation_turns(conversation_id, trace_id, role, occurred_at, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, turn.trace_id, turn.role.value, turn.occurred_at.isoformat(), _dumps(turn.to_dict())),
            )

    def recent_turns(self, conversation_id: str, limit: int = 30) -> list[ConversationTurn]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload
                FROM runtime_conversation_turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, int(limit)),
            ).fetchall()
            turns = [_turn_from_payload(_loads(row["payload"])) for row in rows]
            return list(reversed(turns))

    def get_conversation_checkpoint(self, conversation_id: str) -> ConversationCheckpoint | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_conversation_checkpoints WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            return _checkpoint_from_payload(_loads(row["payload"]))

    def current_task_context(self, conversation_id: str, customer_id: str) -> ConversationTaskContext | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload
                FROM runtime_task_contexts
                WHERE conversation_id = ? AND customer_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (conversation_id, customer_id),
            ).fetchone()
            return _task_context_from_payload(_loads(row["payload"])) if row else None

    def latest_task_context(self, conversation_id: str) -> ConversationTaskContext | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload
                FROM runtime_task_contexts
                WHERE conversation_id = ? AND status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
            return _task_context_from_payload(_loads(row["payload"])) if row else None

    def activate_task_context(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        trace_id: str,
        activity_at: datetime,
        started_at: datetime,
        reason: str,
        force_new: bool,
        archive_previous: bool,
    ) -> tuple[ConversationTaskContext, list[StateTransition]]:
        with self._write_transaction():
            from ...models import new_id

            transitions: list[StateTransition] = []
            previous = self.current_task_context(conversation_id, customer_id)
            if previous is not None and not force_new:
                previous.updated_at = activity_at
                previous.source_trace_id = trace_id
                self._save_task_context(previous)
                return previous, transitions

            if previous is not None:
                previous.status = "closed"
                previous.closed_at = activity_at
                previous.updated_at = activity_at
                self._save_task_context(previous)
                transitions.append(
                    StateTransition(
                        "task_context",
                        previous.task_context_id,
                        "active",
                        "closed",
                        reason,
                        trace_id,
                    )
                )

            if archive_previous:
                rows = self._connection.execute(
                    """
                    SELECT payload
                    FROM runtime_task_memories
                    WHERE conversation_id = ? AND status = 'active'
                      AND (customer_id = ? OR target_customer_id = ?)
                    """,
                    (conversation_id, customer_id, customer_id),
                ).fetchall()
                previous_context_id = previous.task_context_id if previous else None
                for row in rows:
                    memory = _task_memory_from_payload(_loads(row["payload"]))
                    memory_context_id = str(memory.metadata.get("task_context_id") or "")
                    belongs_to_previous = bool(previous_context_id and memory_context_id == previous_context_id)
                    predates_new_context = memory.updated_at < started_at
                    if not belongs_to_previous and not predates_new_context:
                        continue
                    memory.status = "archived"
                    memory.updated_at = activity_at
                    self._save_task_memory(memory)
                    transitions.append(
                        StateTransition(
                            "task_memory",
                            memory.memory_id,
                            "active",
                            "archived",
                            "task_context_reset",
                            trace_id,
                        )
                    )

            context = ConversationTaskContext(
                task_context_id=new_id("task_context"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                reset_reason=reason,
                previous_task_context_id=previous.task_context_id if previous else None,
                source_trace_id=trace_id,
                started_at=started_at,
                updated_at=activity_at,
            )
            self._save_task_context(context)
            transitions.append(
                StateTransition("task_context", context.task_context_id, None, "active", reason, trace_id)
            )
            for transition in transitions:
                self._append_transition(transition)
            return context, transitions

    def conversation_version(self, conversation_id: str) -> int:
        key = conversation_id or "default"
        with self._lock:
            row = self._connection.execute(
                "SELECT version FROM runtime_conversation_versions WHERE conversation_id = ?",
                (key,),
            ).fetchone()
            return int(row["version"]) if row else 0

    def advance_conversation_version(
        self,
        conversation_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> tuple[int, StateTransition]:
        key = conversation_id or "default"
        with self._lock, self._connection:
            row = self._connection.execute(
                """
                INSERT INTO runtime_conversation_versions(conversation_id, version, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    version=runtime_conversation_versions.version + 1,
                    updated_at=excluded.updated_at
                RETURNING version
                """,
                (key, _now_iso()),
            ).fetchone()
            if row is None:
                raise RuntimeError("conversation version update returned no row")
            new = int(row["version"])
            old = new - 1
            transition = StateTransition(
                "conversation_version",
                key,
                str(old),
                str(new),
                reason or "user_message_received",
                trace_id,
            )
            self._append_transition(transition)
            return new, transition

    def supersede_pending_outputs(
        self,
        conversation_id: str,
        *,
        sender_id: str | None = None,
        trace_id: str,
        reason: str,
    ) -> tuple[dict[str, int], list[StateTransition]]:
        key = conversation_id or "default"
        with self._lock, self._connection:
            transitions: list[StateTransition] = []
            counts = {
                "invite_drafts": 0,
                "outbound_message_drafts": 0,
                "assistant_replies": 0,
            }
            game_ids = {game.game_id for game in self.games.values() if game.conversation_id == key}
            sender_is_pending_candidate = bool(
                sender_id
                and any(
                    draft.game_id in game_ids
                    and draft.customer_id == sender_id
                    and draft.status == InviteStatus.PENDING_APPROVAL
                    for draft in self.invite_drafts.values()
                )
            )
            for draft in self.invite_drafts.values():
                if sender_is_pending_candidate:
                    continue
                if draft.game_id not in game_ids or draft.status != InviteStatus.PENDING_APPROVAL:
                    continue
                old = draft.status.value
                draft.status = InviteStatus.SUPERSEDED
                draft.updated_at = datetime.now(DEFAULT_TZ)
                draft.metadata = {
                    **dict(draft.metadata),
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["invite_drafts"] += 1
                transitions.append(StateTransition("invite_draft", draft.draft_id, old, draft.status.value, reason, trace_id))
                self._save_invite(draft)
            for draft in self.outbound_message_drafts.values():
                if sender_is_pending_candidate:
                    continue
                if draft.conversation_id != key or draft.status != OutboundDraftStatus.PENDING_APPROVAL:
                    continue
                old = draft.status.value
                draft.status = OutboundDraftStatus.SUPERSEDED
                draft.updated_at = datetime.now(DEFAULT_TZ)
                draft.metadata = {
                    **dict(draft.metadata),
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["outbound_message_drafts"] += 1
                transitions.append(
                    StateTransition("outbound_message_draft", draft.draft_id, old, draft.status.value, reason, trace_id)
                )
                self._save_outbound_message_draft(draft)
            rows = self._connection.execute(
                """
                SELECT id, payload
                FROM runtime_conversation_turns
                WHERE conversation_id = ? AND role = ?
                """,
                (key, ConversationRole.ASSISTANT.value),
            ).fetchall()
            for row in rows:
                payload = _loads(row["payload"])
                turn = _turn_from_payload(payload)
                if turn.metadata.get("delivery_status") != "pending_operator_send":
                    continue
                old = str(turn.metadata.get("delivery_status") or "")
                turn.metadata = {
                    **dict(turn.metadata),
                    "delivery_status": "superseded",
                    "superseded_by_trace_id": trace_id,
                    "superseded_reason": reason,
                }
                counts["assistant_replies"] += 1
                transitions.append(StateTransition("assistant_reply", turn.trace_id, old, "superseded", reason, trace_id))
                self._connection.execute(
                    "UPDATE runtime_conversation_turns SET payload = ? WHERE id = ?",
                    (_dumps(turn.to_dict()), row["id"]),
                )
            for transition in transitions:
                self._append_transition(transition)
            return counts, transitions

    def upsert_conversation_checkpoint(
        self,
        *,
        conversation_id: str,
        summary: str,
        facts: dict[str, Any],
        open_questions: list[str],
        trace_id: str,
    ) -> tuple[ConversationCheckpoint, StateTransition]:
        with self._lock, self._connection:
            previous = self.get_conversation_checkpoint(conversation_id)
            task_context = self.latest_task_context(conversation_id)
            checkpoint = ConversationCheckpoint(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
                task_context_id=task_context.task_context_id if task_context else None,
                source_trace_id=trace_id,
            )
            transition = StateTransition(
                "conversation_checkpoint",
                conversation_id,
                "exists" if previous else None,
                "updated",
                "update_context_checkpoint",
                trace_id,
            )
            self._connection.execute(
                """
                INSERT INTO runtime_conversation_checkpoints(conversation_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (conversation_id, _dumps(checkpoint.to_dict()), checkpoint.updated_at.isoformat()),
            )
            self._append_transition(transition)
            return checkpoint, transition
