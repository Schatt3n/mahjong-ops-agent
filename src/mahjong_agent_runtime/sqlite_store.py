from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import (
    AgentAction,
    AgentRuntimeResult,
    ConversationCheckpoint,
    ConversationRole,
    ConversationTurn,
    CustomerProfile,
    CustomerRelationship,
    DEFAULT_TZ,
    GameParticipant,
    GameStatus,
    Game,
    InviteDraft,
    InviteStatus,
    MessageReference,
    OPEN_INVITE_STATUSES,
    OutboundDraftStatus,
    OutboundMessageDraft,
    Party,
    PendingInputBatch,
    PendingInputBatchStatus,
    PendingMemoryCandidate,
    RoomReservation,
    StateTransition,
    TaskMemory,
    ToolCall,
    ToolResult,
    new_id,
    now,
)
from .store import (
    ALLOWED_GAME_TRANSITIONS,
    CONFIRMED_CANDIDATE_STATUSES,
    UNCONFIRMED_CANDIDATE_STATUSES,
    invite_status_from_candidate_status,
    normalize_game_participants,
    normalize_game_parties,
    refresh_requirement_seat_snapshot,
    normalize_requirement_with_party,
    normalize_requirement,
    score_customer,
    game_contains_customer,
    is_avoid_playing_memory,
    relationship_anchor_ids,
    relationship_context_for_sender,
    relationship_pair_key,
    pending_input_batch_key,
    parse_datetime_value,
    PENDING_INPUT_PROCESSING_LEASE_SECONDS,
    IDEMPOTENCY_CLAIM_LEASE_SECONDS,
    tool_result_is_in_progress,
    requested_seat_count_from_search_requirement,
    score_requirement,
    score_customer_relationships,
    task_memory_anchor_ids,
    join_projection,
    game_for_model_context,
    seat_count_from_payload,
    apply_game_lifecycle,
    expire_game_if_stale,
)


@dataclass(slots=True)
class SQLiteAgentStore:
    path: str | Path
    _connection: sqlite3.Connection = field(init=False, repr=False)
    _lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._migrate()

    @contextmanager
    def _write_transaction(self):
        """Acquire SQLite's write reservation before reading mutable invariants."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    @property
    def customers(self) -> dict[str, CustomerProfile]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_customers").fetchall()
            return {item.customer_id: item for item in (_customer_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def customer_relationships(self) -> dict[str, CustomerRelationship]:
        with self._lock:
            rows = self._connection.execute("SELECT pair_key, payload FROM runtime_customer_relationships").fetchall()
            return {str(row["pair_key"]): _relationship_from_payload(_loads(row["payload"])) for row in rows}

    @property
    def games(self) -> dict[str, Game]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_games").fetchall()
            return {item.game_id: item for item in (_game_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def invite_drafts(self) -> dict[str, InviteDraft]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_invite_drafts").fetchall()
            return {item.draft_id: item for item in (_invite_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def outbound_message_drafts(self) -> dict[str, OutboundMessageDraft]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_outbound_message_drafts").fetchall()
            return {item.draft_id: item for item in (_outbound_message_draft_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def room_ids(self) -> list[str]:
        with self._lock:
            rows = self._connection.execute("SELECT room_id FROM runtime_rooms ORDER BY room_id").fetchall()
            return [str(row["room_id"]) for row in rows]

    @property
    def room_reservations(self) -> dict[str, RoomReservation]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_room_reservations").fetchall()
            items = [_room_reservation_from_payload(_loads(row["payload"])) for row in rows]
            return {item.reservation_id: item for item in items}

    @property
    def conversation_checkpoints(self) -> dict[str, ConversationCheckpoint]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_conversation_checkpoints").fetchall()
            return {
                item.conversation_id: item
                for item in (_checkpoint_from_payload(_loads(row["payload"])) for row in rows)
            }

    @property
    def conversation_versions(self) -> dict[str, int]:
        with self._lock:
            rows = self._connection.execute("SELECT conversation_id, version FROM runtime_conversation_versions").fetchall()
            return {str(row["conversation_id"]): int(row["version"]) for row in rows}

    @property
    def transitions(self) -> list[StateTransition]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_state_transitions ORDER BY id").fetchall()
            return [_transition_from_payload(_loads(row["payload"])) for row in rows]

    @property
    def badcases(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_badcases ORDER BY id").fetchall()
            return [_loads(row["payload"]) for row in rows]

    @property
    def task_memories(self) -> dict[str, TaskMemory]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_task_memories").fetchall()
            return {item.memory_id: item for item in (_task_memory_from_payload(_loads(row["payload"])) for row in rows)}

    @property
    def pending_memory_candidates(self) -> dict[str, PendingMemoryCandidate]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_pending_memory_candidates").fetchall()
            return {
                item.candidate_id: item
                for item in (_pending_memory_candidate_from_payload(_loads(row["payload"])) for row in rows)
            }

    @property
    def pending_input_batches(self) -> dict[str, PendingInputBatch]:
        with self._lock:
            rows = self._connection.execute("SELECT batch_key, payload FROM runtime_pending_input_batches").fetchall()
            return {
                str(row["batch_key"]): _pending_input_batch_from_payload(_loads(row["payload"]))
                for row in rows
            }

    def upsert_customer(self, profile: CustomerProfile) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_customers(customer_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (profile.customer_id, _dumps(profile.to_dict()), _now_iso()),
            )

    def configure_rooms(self, room_ids: list[str]) -> None:
        normalized = list(dict.fromkeys(str(item).strip() for item in room_ids if str(item).strip()))
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM runtime_rooms")
            self._connection.executemany(
                "INSERT INTO runtime_rooms(room_id, updated_at) VALUES (?, ?)",
                [(room_id, _now_iso()) for room_id in normalized],
            )

    def search_room_availability(self, *, start_at: Any, end_at: Any) -> dict[str, Any]:
        start = parse_datetime_value(start_at)
        end = parse_datetime_value(end_at)
        if start is None or end is None or end <= start:
            raise ValueError("start_at and end_at must be valid datetimes with end_at after start_at")
        with self._lock:
            room_ids = [
                str(row["room_id"])
                for row in self._connection.execute("SELECT room_id FROM runtime_rooms ORDER BY room_id").fetchall()
            ]
            occupied = {
                str(row["room_id"])
                for row in self._connection.execute(
                    """
                    SELECT DISTINCT room_id
                    FROM runtime_room_reservations
                    WHERE status IN ('held', 'confirmed') AND start_at < ? AND end_at > ?
                    """,
                    (end.isoformat(), start.isoformat()),
                ).fetchall()
            }
            available = [room_id for room_id in room_ids if room_id not in occupied]
            return {
                "configured": bool(room_ids),
                "start_at": start.isoformat(),
                "end_at": end.isoformat(),
                "room_count": len(room_ids),
                "available_room_ids": available,
                "occupied_room_ids": sorted(occupied),
                "available_count": len(available),
            }

    def reserve_room(
        self,
        *,
        conversation_id: str,
        game_id: str | None,
        start_at: Any,
        end_at: Any,
        room_id: str | None,
        trace_id: str,
    ) -> tuple[RoomReservation, StateTransition]:
        start = parse_datetime_value(start_at)
        end = parse_datetime_value(end_at)
        if start is None or end is None or end <= start:
            raise ValueError("start_at and end_at must be valid datetimes with end_at after start_at")
        with self._write_transaction():
            availability = self.search_room_availability(start_at=start, end_at=end)
            if not availability["configured"]:
                raise ValueError("room inventory is not configured")
            chosen = str(room_id or "").strip()
            available = list(availability["available_room_ids"])
            if chosen and chosen not in available:
                raise ValueError(f"room is unavailable: {chosen}")
            if not chosen:
                if not available:
                    raise ValueError("no room is available for the requested interval")
                chosen = available[0]
            reservation = RoomReservation(
                reservation_id=new_id("room_reservation"),
                room_id=chosen,
                conversation_id=conversation_id,
                game_id=game_id,
                start_at=start,
                end_at=end,
                source_trace_id=trace_id,
            )
            self._connection.execute(
                """
                INSERT INTO runtime_room_reservations(
                    reservation_id, room_id, conversation_id, game_id, start_at, end_at, status, payload, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reservation.reservation_id,
                    reservation.room_id,
                    reservation.conversation_id,
                    reservation.game_id or "",
                    reservation.start_at.isoformat(),
                    reservation.end_at.isoformat(),
                    reservation.status,
                    _dumps(reservation.to_dict()),
                    reservation.updated_at.isoformat(),
                ),
            )
            transition = StateTransition(
                "room_reservation",
                reservation.reservation_id,
                None,
                reservation.status,
                "reserve_room",
                trace_id,
            )
            self._append_transition(transition)
            return reservation, transition

    def upsert_customer_relationship(self, relationship: CustomerRelationship) -> None:
        pair_key = relationship_pair_key(relationship.customer_a_id, relationship.customer_b_id)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_customer_relationships(pair_key, customer_a_id, customer_b_id, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pair_key) DO UPDATE SET
                    customer_a_id=excluded.customer_a_id,
                    customer_b_id=excluded.customer_b_id,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    pair_key,
                    relationship.customer_a_id,
                    relationship.customer_b_id,
                    _dumps(relationship.to_dict()),
                    relationship.updated_at.isoformat(),
                ),
            )

    def relationship_between(self, customer_id: str, other_customer_id: str) -> CustomerRelationship | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_customer_relationships WHERE pair_key = ?",
                (relationship_pair_key(customer_id, other_customer_id),),
            ).fetchone()
            if row is None:
                return None
            return _relationship_from_payload(_loads(row["payload"]))

    def relationship_context_for_sender(self, sender_id: str, games: list[Game]) -> list[dict[str, Any]]:
        return relationship_context_for_sender(
            sender_id=sender_id,
            games=games,
            customers=self.customers,
            relationship_lookup=self.relationship_between,
        )

    def append_user_turn(self, message, trace_id: str) -> None:
        self.append_turn(
            message.conversation_id,
            ConversationTurn(
                role=ConversationRole.USER,
                content=message.text,
                trace_id=trace_id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                metadata=dict(getattr(message, "metadata", {}) or {}),
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
        self.append_turn(
            conversation_id,
            ConversationTurn(
                role=ConversationRole.ASSISTANT,
                content=text,
                trace_id=trace_id,
                metadata=dict(metadata or {}),
            ),
        )

    def append_tool_turn(self, conversation_id: str, text: str, trace_id: str) -> None:
        self.append_turn(
            conversation_id,
            ConversationTurn(role=ConversationRole.TOOL, content=text, trace_id=trace_id),
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
            checkpoint = ConversationCheckpoint(
                conversation_id=conversation_id,
                summary=summary,
                facts=dict(facts),
                open_questions=list(open_questions),
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

    def record_task_memory(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        memory_type: str,
        field: str,
        value: Any,
        target_customer_id: str | None = None,
        evidence: str = "",
        confidence: float = 0.0,
        risk_level: str = "medium",
        scope: str = "current_task",
        metadata: dict[str, Any] | None = None,
        trace_id: str,
    ) -> tuple[TaskMemory, StateTransition]:
        with self._write_transaction():
            from .models import new_id

            memory = TaskMemory(
                memory_id=new_id("task_memory"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                memory_type=memory_type,
                field=field,
                value=value,
                target_customer_id=target_customer_id,
                evidence=evidence,
                confidence=float(confidence or 0.0),
                risk_level=risk_level or "medium",
                scope=scope or "current_task",
                source_trace_id=trace_id,
                metadata=dict(metadata or {}),
            )
            transition = StateTransition("task_memory", memory.memory_id, None, memory.status, "record_user_memory", trace_id)
            self._save_task_memory(memory)
            self._append_transition(transition)
            return memory, transition

    def record_pending_memory_candidate(
        self,
        *,
        conversation_id: str,
        customer_id: str,
        memory_type: str,
        field: str,
        value: Any,
        operation: str = "set",
        target_customer_id: str | None = None,
        evidence: str = "",
        confidence: float = 0.0,
        risk_level: str = "medium",
        scope: str = "long_term",
        metadata: dict[str, Any] | None = None,
        trace_id: str,
    ) -> tuple[PendingMemoryCandidate, StateTransition]:
        with self._write_transaction():
            from .models import new_id

            candidate = PendingMemoryCandidate(
                candidate_id=new_id("memory_candidate"),
                conversation_id=conversation_id,
                customer_id=customer_id,
                memory_type=memory_type,
                field=field,
                value=value,
                operation=operation or "set",
                target_customer_id=target_customer_id,
                evidence=evidence,
                confidence=float(confidence or 0.0),
                risk_level=risk_level or "medium",
                scope=scope or "long_term",
                source_trace_id=trace_id,
                metadata=dict(metadata or {}),
            )
            transition = StateTransition(
                "pending_memory_candidate",
                candidate.candidate_id,
                None,
                candidate.status,
                "record_user_memory",
                trace_id,
            )
            self._save_pending_memory_candidate(candidate)
            self._append_transition(transition)
            return candidate, transition

    def task_memory_context(self, conversation_id: str, customer_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            params: list[Any] = [conversation_id]
            condition = "conversation_id = ? AND status = 'active'"
            if customer_id:
                condition += " AND (customer_id = ? OR target_customer_id = ?)"
                params.extend([customer_id, customer_id])
            rows = self._connection.execute(
                f"""
                SELECT payload
                FROM runtime_task_memories
                WHERE {condition}
                ORDER BY updated_at DESC
                """,
                tuple(params),
            ).fetchall()
            return [_task_memory_from_payload(_loads(row["payload"])).to_dict() for row in rows]

    def pending_memory_candidates_for_context(
        self,
        conversation_id: str,
        customer_id: str | None = None,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        with self._lock:
            params: list[Any] = [conversation_id]
            condition = "conversation_id = ? AND status = 'pending_review'"
            if customer_id:
                condition += " AND (customer_id = ? OR target_customer_id = ?)"
                params.extend([customer_id, customer_id])
            rows = self._connection.execute(
                f"""
                SELECT payload
                FROM runtime_pending_memory_candidates
                WHERE {condition}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (*params, int(limit)),
            ).fetchall()
            return [_pending_memory_candidate_from_payload(_loads(row["payload"])).to_dict() for row in rows]

    def task_memory_excluded_customer_ids(
        self,
        conversation_id: str | None,
        anchor_ids: list[str] | set[str] | None,
    ) -> list[str]:
        if not conversation_id:
            return []
        anchors = {str(item) for item in anchor_ids or [] if str(item or "").strip()}
        if not anchors:
            return []
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload
                FROM runtime_task_memories
                WHERE conversation_id = ? AND status = 'active'
                """,
                (conversation_id,),
            ).fetchall()
            excluded: list[str] = []
            for row in rows:
                memory = _task_memory_from_payload(_loads(row["payload"]))
                if memory.customer_id not in anchors or not is_avoid_playing_memory(memory):
                    continue
                target_id = str(memory.target_customer_id or "")
                if target_id and target_id not in excluded:
                    excluded.append(target_id)
            return excluded

    def active_games(self, conversation_id: str | None = None) -> list[Game]:
        self._expire_stale_games(trace_id="system_lifecycle")
        games = [
            item
            for item in self.games.values()
            if item.status.value in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}
        ]
        if conversation_id:
            return [item for item in games if item.conversation_id == conversation_id]
        return games

    def idempotent_result(self, key: str | None) -> ToolResult | None:
        if not key:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload, created_at FROM runtime_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            result = _tool_result_from_payload(_loads(row["payload"]))
            if tool_result_is_in_progress(result):
                claimed_at = _datetime_from_payload(row["created_at"])
                if claimed_at <= now() - timedelta(seconds=IDEMPOTENCY_CLAIM_LEASE_SECONDS):
                    return None
            return result

    def claim_idempotent_result(self, key: str | None, claimed_result: ToolResult) -> tuple[bool, ToolResult | None]:
        if not key:
            return True, None
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO runtime_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (key, _dumps(claimed_result.to_dict()), _now_iso()),
            )
            if cursor.rowcount == 1:
                return True, None
            row = self._connection.execute(
                "SELECT payload, created_at FROM runtime_idempotency_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return False, None
            existing = _tool_result_from_payload(_loads(row["payload"]))
            claimed_at = _datetime_from_payload(row["created_at"])
            if tool_result_is_in_progress(existing) and claimed_at <= now() - timedelta(
                seconds=IDEMPOTENCY_CLAIM_LEASE_SECONDS
            ):
                cursor = self._connection.execute(
                    """
                    UPDATE runtime_idempotency_ledger
                    SET payload = ?, created_at = ?
                    WHERE idempotency_key = ? AND created_at = ?
                    """,
                    (_dumps(claimed_result.to_dict()), _now_iso(), key, row["created_at"]),
                )
                if cursor.rowcount == 1:
                    return True, None
            return False, existing

    def remember_result(self, key: str | None, result: ToolResult) -> None:
        if not key:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_idempotency_ledger(idempotency_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    payload=excluded.payload,
                    created_at=excluded.created_at
                """,
                (key, _dumps(result.to_dict()), _now_iso()),
            )

    def idempotent_message_result(self, message_id: str | None) -> AgentRuntimeResult | None:
        if not message_id:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_message_results WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                return None
            return _runtime_result_from_payload(_loads(row["payload"]))

    def remember_message_result(self, message_id: str | None, result: AgentRuntimeResult) -> None:
        if not message_id:
            return
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO runtime_message_results(message_id, conversation_id, trace_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO NOTHING
                """,
                (message_id, result.conversation_id, result.trace_id, _dumps(result.to_dict()), _now_iso()),
            )

    def upsert_pending_input_fragment(
        self,
        message,
        *,
        trace_id: str,
        quiet_deadline: datetime,
    ) -> tuple[PendingInputBatch, StateTransition | None, bool]:
        """Persist one fragment and reset the quiet deadline in one transaction."""

        key = pending_input_batch_key(message.conversation_id, message.sender_id)
        fragment = message.to_dict()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM runtime_pending_input_batches WHERE batch_key = ?",
                (key,),
            ).fetchone()
            existing = _pending_input_batch_from_payload(_loads(row["payload"])) if row else None
            message_id = str(fragment.get("message_id") or "")
            if existing is not None and message_id and any(
                str(item.get("message_id") or "") == message_id for item in existing.fragments
            ):
                return existing, None, False
            if existing is None or existing.status in {
                PendingInputBatchStatus.COMPLETED,
                PendingInputBatchStatus.IGNORED,
                PendingInputBatchStatus.FAILED,
            }:
                batch = PendingInputBatch(
                    batch_id=new_id("input_batch"),
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    fragments=[fragment],
                    quiet_deadline=quiet_deadline,
                    source_channel=str(message.metadata.get("channel") or ""),
                )
                previous_status = "absent"
            else:
                batch = existing
                previous_status = batch.status.value
                batch.fragments.append(fragment)
                batch.sender_name = message.sender_name or batch.sender_name
                batch.version += 1
                batch.status = PendingInputBatchStatus.PENDING
                batch.quiet_deadline = quiet_deadline
                batch.updated_at = now()
                batch.decision = {}
                if message.metadata.get("channel"):
                    batch.source_channel = str(message.metadata["channel"])
            self._save_pending_input_batch(batch)
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                previous_status,
                batch.status.value,
                "input_fragment_buffered",
                trace_id,
            )
            self._append_transition(transition)
            return batch, transition, True

    def pending_input_batch(self, conversation_id: str, sender_id: str) -> PendingInputBatch | None:
        key = pending_input_batch_key(conversation_id, sender_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_pending_input_batches WHERE batch_key = ?",
                (key,),
            ).fetchone()
            return _pending_input_batch_from_payload(_loads(row["payload"])) if row else None

    def due_pending_input_batches(self, *, at: datetime, limit: int = 100) -> list[PendingInputBatch]:
        with self._lock:
            stale_before = at - timedelta(seconds=PENDING_INPUT_PROCESSING_LEASE_SECONDS)
            rows = self._connection.execute(
                """
                SELECT payload FROM runtime_pending_input_batches
                WHERE (status = ? AND quiet_deadline <= ?)
                   OR (status = ? AND updated_at <= ?)
                ORDER BY quiet_deadline ASC
                LIMIT ?
                """,
                (
                    PendingInputBatchStatus.PENDING.value,
                    at.isoformat(),
                    PendingInputBatchStatus.PROCESSING.value,
                    stale_before.isoformat(),
                    max(1, int(limit)),
                ),
            ).fetchall()
            return [_pending_input_batch_from_payload(_loads(row["payload"])) for row in rows]

    def claim_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        trace_id: str,
    ) -> tuple[PendingInputBatch | None, StateTransition | None]:
        """Compare-and-set claim used by immediate and delayed dispatchers."""

        with self._lock, self._connection:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_pending_input_batches
                WHERE batch_id = ? AND version = ?
                """,
                (batch_id, int(expected_version)),
            ).fetchone()
            if row is None:
                return None, None
            batch = _pending_input_batch_from_payload(_loads(row["payload"]))
            stale_before = now() - timedelta(seconds=PENDING_INPUT_PROCESSING_LEASE_SECONDS)
            if batch.status != PendingInputBatchStatus.PENDING and not (
                batch.status == PendingInputBatchStatus.PROCESSING and batch.updated_at <= stale_before
            ):
                return None, None
            old = batch.status.value
            batch.status = PendingInputBatchStatus.PROCESSING
            batch.updated_at = now()
            cursor = self._connection.execute(
                """
                UPDATE runtime_pending_input_batches
                SET status = ?, payload = ?, updated_at = ?
                WHERE batch_id = ? AND version = ? AND status = ? AND updated_at = ?
                """,
                (
                    batch.status.value,
                    _dumps(batch.to_dict()),
                    batch.updated_at.isoformat(),
                    batch_id,
                    int(expected_version),
                    old,
                    _pending_input_batch_from_payload(_loads(row["payload"])).updated_at.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                return None, None
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                old,
                batch.status.value,
                "input_batch_claimed",
                trace_id,
            )
            self._append_transition(transition)
            return batch, transition

    def record_pending_input_decision(
        self,
        *,
        batch_id: str,
        expected_version: int,
        decision: dict[str, Any],
    ) -> PendingInputBatch | None:
        """Persist the model boundary decision while keeping the batch pending."""

        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM runtime_pending_input_batches WHERE batch_id = ? AND version = ?",
                (batch_id, int(expected_version)),
            ).fetchone()
            if row is None:
                return None
            batch = _pending_input_batch_from_payload(_loads(row["payload"]))
            batch.decision = dict(decision)
            batch.updated_at = now()
            self._save_pending_input_batch(batch)
            return batch

    def finish_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        status: PendingInputBatchStatus,
        trace_id: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[PendingInputBatch | None, StateTransition | None]:
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT payload FROM runtime_pending_input_batches WHERE batch_id = ? AND version = ?",
                (batch_id, int(expected_version)),
            ).fetchone()
            if row is None:
                return None, None
            batch = _pending_input_batch_from_payload(_loads(row["payload"]))
            old = batch.status.value
            batch.status = status
            batch.decision = dict(decision or {})
            batch.updated_at = now()
            self._save_pending_input_batch(batch)
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                old,
                status.value,
                "input_batch_finished",
                trace_id,
            )
            self._append_transition(transition)
            return batch, transition

    def register_message_reference(self, reference: MessageReference) -> None:
        if not reference.message_id:
            return
        with self._lock, self._connection:
            self._save_message_reference(reference)

    def link_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
        source_message_id: str | None = None,
        business_ref_type: str | None = None,
        business_ref_id: str | None = None,
        channel: str | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MessageReference:
        source = self._find_message_reference_source(
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            business_ref_type=business_ref_type,
            business_ref_id=business_ref_id,
        )
        if source is None:
            raise ValueError("source message reference not found")
        linked = MessageReference(
            message_id=str(message_id or ""),
            conversation_id=str(conversation_id or source.conversation_id),
            business_ref_type=source.business_ref_type,
            business_ref_id=source.business_ref_id,
            text=str(text or source.text or ""),
            channel=str(channel or source.channel or ""),
            sender_id=source.sender_id,
            sender_name=source.sender_name,
            recipient_id=source.recipient_id,
            recipient_name=source.recipient_name,
            metadata={
                **dict(source.metadata),
                **dict(metadata or {}),
                "linked_from_message_id": source.message_id,
                "linked_from_conversation_id": source.conversation_id,
            },
        )
        with self._lock, self._connection:
            self._save_message_reference(linked)
        return linked

    def _find_message_reference_source(
        self,
        *,
        conversation_id: str,
        source_message_id: str | None,
        business_ref_type: str | None,
        business_ref_id: str | None,
    ) -> MessageReference | None:
        if source_message_id:
            source = self.resolve_message_reference(conversation_id=conversation_id, message_id=source_message_id)
            if source is not None:
                return source
        if not business_ref_type or not business_ref_id:
            return None
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_message_references
                WHERE conversation_id = ? AND business_ref_type = ? AND business_ref_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (conversation_id, business_ref_type, business_ref_id),
            ).fetchone()
            if row is None:
                row = self._connection.execute(
                    """
                    SELECT payload FROM runtime_message_references
                    WHERE business_ref_type = ? AND business_ref_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (business_ref_type, business_ref_id),
                ).fetchone()
            if row is None:
                return None
            return _message_reference_from_payload(_loads(row["payload"]))

    def resolve_message_reference(
        self,
        *,
        conversation_id: str,
        message_id: str,
    ) -> MessageReference | None:
        if not message_id:
            return None
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_message_references
                WHERE conversation_id = ? AND message_id = ?
                """,
                (conversation_id, message_id),
            ).fetchone()
            if row is None:
                row = self._connection.execute(
                    """
                    SELECT payload FROM runtime_message_references
                    WHERE message_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (message_id,),
                ).fetchone()
            if row is None:
                return None
            return _message_reference_from_payload(_loads(row["payload"]))

    def clear_runtime_state(
        self,
        *,
        include_customers: bool = False,
        include_badcases: bool = False,
    ) -> dict[str, int]:
        tables = [
            ("games", "runtime_games"),
            ("invite_drafts", "runtime_invite_drafts"),
            ("outbound_message_drafts", "runtime_outbound_message_drafts"),
            ("room_reservations", "runtime_room_reservations"),
            ("state_transitions", "runtime_state_transitions"),
            ("conversation_turns", "runtime_conversation_turns"),
            ("conversation_checkpoints", "runtime_conversation_checkpoints"),
            ("conversation_versions", "runtime_conversation_versions"),
            ("idempotency_ledger", "runtime_idempotency_ledger"),
            ("message_results", "runtime_message_results"),
            ("message_references", "runtime_message_references"),
            ("task_memories", "runtime_task_memories"),
            ("pending_memory_candidates", "runtime_pending_memory_candidates"),
            ("pending_input_batches", "runtime_pending_input_batches"),
        ]
        if include_customers:
            tables.append(("customers", "runtime_customers"))
            tables.append(("customer_relationships", "runtime_customer_relationships"))
        else:
            tables.append(("customers", ""))
            tables.append(("customer_relationships", ""))
        if include_badcases:
            tables.append(("badcases", "runtime_badcases"))
        else:
            tables.append(("badcases", ""))
        with self._lock, self._connection:
            deleted: dict[str, int] = {}
            for key, table in tables:
                if not table:
                    deleted[key] = 0
                    continue
                row = self._connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
                deleted[key] = int(row["count"] if row else 0)
            for _, table in tables:
                if table:
                    self._connection.execute(f"DELETE FROM {table}")
            return deleted

    def search_current_games(
        self,
        requirement: dict[str, Any],
        limit: int = 8,
        *,
        sender_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        requirement = normalize_requirement(requirement)
        scored: list[dict[str, Any]] = []
        requested_seats = requested_seat_count_from_search_requirement(requirement, default=1)
        anchor_ids = task_memory_anchor_ids(requirement, sender_id=sender_id)
        task_excluded = set(self.task_memory_excluded_customer_ids(conversation_id, anchor_ids))
        for game in self.active_games():
            if game.remaining_seats() <= 0:
                continue
            if task_excluded and any(game_contains_customer(game, customer_id) for customer_id in task_excluded):
                continue
            score, reasons = score_requirement(requirement, game.requirement)
            if requirement and score <= 0:
                continue
            scored.append(
                {
                    "game": game_for_model_context(game, self.customers),
                    "score": score,
                    "reasons": reasons or ["active_open_game"],
                    "join_projection": join_projection(game, sender_id=sender_id, requested_seats=requested_seats),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: int(limit)]

    def search_customers(
        self,
        requirement: dict[str, Any],
        *,
        exclude_customer_ids: list[str] | None = None,
        limit: int = 8,
        sender_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        requirement = normalize_requirement(requirement)
        excluded = set(exclude_customer_ids or [])
        anchor_ids = task_memory_anchor_ids(requirement, sender_id=sender_id, excluded_customer_ids=excluded)
        excluded.update(self.task_memory_excluded_customer_ids(conversation_id, anchor_ids))
        scored: list[dict[str, Any]] = []
        for customer in self.customers.values():
            if customer.no_contact or customer.customer_id in excluded:
                continue
            if self.active_game_for_customer(customer.customer_id):
                continue
            score, reasons = score_customer(requirement, customer)
            relationship_score, relationship_reasons, blocked = score_customer_relationships(
                customer.customer_id,
                anchor_ids,
                self.relationship_between,
            )
            if blocked:
                continue
            score += relationship_score
            reasons.extend(relationship_reasons)
            if score <= 0:
                continue
            scored.append({"customer": customer.to_model_context(), "score": score, "reasons": reasons})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: int(limit)]

    def active_game_for_customer(self, customer_id: str) -> Game | None:
        self._expire_stale_games(trace_id="system_lifecycle")
        for game in self.games.values():
            if game.status.value not in {GameStatus.FORMING.value, GameStatus.INVITING.value, GameStatus.READY.value}:
                continue
            if any(item.customer_id == customer_id and item.status in {"joined", "confirmed"} for item in game.participants):
                return game
        return None

    def create_game(
        self,
        *,
        conversation_id: str,
        organizer_id: str,
        organizer_name: str,
        requirement: dict[str, Any],
        known_players: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[Game, StateTransition]:
        with self._write_transaction():
            from .models import new_id

            duplicate = next(
                (
                    item
                    for item in self.games.values()
                    if item.conversation_id == conversation_id
                    and item.organizer_id == organizer_id
                    and item.status in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"active game already exists: {duplicate.game_id}")
            normalized_requirement = normalize_requirement(requirement)
            default_requester_seat_count = seat_count_from_payload(normalized_requirement, default=1)
            participants = normalize_game_participants(
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                known_players=known_players,
                default_requester_seat_count=default_requester_seat_count,
            )
            parties = normalize_game_parties(participants)
            claimed_seats = sum(
                max(1, int(item.seat_count))
                for item in participants
                if item.status in {"joined", "confirmed"}
            )
            if claimed_seats > 4:
                raise ValueError(f"initial participants exceed table capacity: {claimed_seats}>4")

            game = Game(
                game_id=new_id("game"),
                conversation_id=conversation_id,
                organizer_id=organizer_id,
                organizer_name=organizer_name,
                requirement=normalize_requirement_with_party(normalized_requirement, parties),
                participants=participants,
                parties=parties,
            )
            apply_game_lifecycle(game)
            transition = StateTransition("game", game.game_id, None, game.status.value, "create_game", trace_id)
            self._save_game(game)
            self._append_transition(transition)
            return game, transition

    def _expire_stale_games(self, *, trace_id: str) -> list[StateTransition]:
        with self._lock, self._connection:
            stamp = datetime.now(DEFAULT_TZ)
            transitions: list[StateTransition] = []
            for game in self.games.values():
                transition = expire_game_if_stale(game, at=stamp, trace_id=trace_id)
                if transition is None:
                    continue
                transitions.append(transition)
                self._save_game(game)
                self._append_transition(transition)
                released = self._release_room_reservations_for_game(
                    game.game_id,
                    trace_id=trace_id,
                    reason="game_lifecycle_closed",
                )
                transitions.extend(released)
            return transitions

    def _release_room_reservations_for_game(
        self,
        game_id: str,
        *,
        trace_id: str,
        reason: str,
    ) -> list[StateTransition]:
        rows = self._connection.execute(
            """
            SELECT payload
            FROM runtime_room_reservations
            WHERE game_id = ? AND status IN ('held', 'confirmed')
            """,
            (game_id,),
        ).fetchall()
        transitions: list[StateTransition] = []
        for row in rows:
            reservation = _room_reservation_from_payload(_loads(row["payload"]))
            old = reservation.status
            reservation.status = "released"
            reservation.updated_at = datetime.now(DEFAULT_TZ)
            self._connection.execute(
                """
                UPDATE runtime_room_reservations
                SET status = ?, payload = ?, updated_at = ?
                WHERE reservation_id = ?
                """,
                (
                    reservation.status,
                    _dumps(reservation.to_dict()),
                    reservation.updated_at.isoformat(),
                    reservation.reservation_id,
                ),
            )
            transition = StateTransition(
                "room_reservation",
                reservation.reservation_id,
                old,
                reservation.status,
                reason,
                trace_id,
            )
            transitions.append(transition)
            self._append_transition(transition)
        return transitions

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraft], list[StateTransition]]:
        self._expire_stale_games(trace_id=trace_id)
        with self._lock, self._connection:
            from .models import new_id, now

            game = self.require_game(game_id)
            if game.status not in {GameStatus.FORMING, GameStatus.INVITING}:
                raise ValueError(f"game does not accept invitations in status={game.status.value}: {game_id}")
            requested_customer_ids = [
                str(item.get("customer_id") or "").strip()
                for item in invitations
                if isinstance(item, dict)
            ]
            if any(not customer_id for customer_id in requested_customer_ids):
                raise ValueError("every invitation requires customer_id")
            if len(requested_customer_ids) != len(set(requested_customer_ids)):
                raise ValueError("duplicate customer_id in invitation request")
            open_customer_ids = {
                draft.customer_id
                for draft in self.invite_drafts.values()
                if draft.game_id == game_id and draft.status in OPEN_INVITE_STATUSES
            }
            duplicated = sorted(open_customer_ids.intersection(requested_customer_ids))
            if duplicated:
                raise ValueError(f"customer already has an open invitation for this game: {','.join(duplicated)}")
            transitions: list[StateTransition] = []
            if game.status == GameStatus.FORMING:
                old = game.status.value
                game.status = GameStatus.INVITING
                game.updated_at = now()
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "create_invite_drafts", trace_id))
                self._save_game(game)
            drafts: list[InviteDraft] = []
            for raw in invitations:
                if not isinstance(raw, dict):
                    continue
                draft = InviteDraft(
                    draft_id=new_id("draft"),
                    game_id=game_id,
                    customer_id=str(raw.get("customer_id") or ""),
                    display_name=str(raw.get("display_name") or raw.get("customer_id") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                drafts.append(draft)
                self._save_message_reference(
                    MessageReference(
                        message_id=draft.draft_id,
                        conversation_id=game.conversation_id,
                        business_ref_type="invite_draft",
                        business_ref_id=draft.draft_id,
                        text=draft.message_text,
                        channel=str(draft.metadata.get("channel") or "internal"),
                        recipient_id=draft.customer_id,
                        recipient_name=draft.display_name,
                        metadata={"source": "create_invite_drafts", "game_id": game_id},
                    )
                )
                transitions.append(StateTransition("invite_draft", draft.draft_id, None, draft.status.value, "create_invite_drafts", trace_id))
                self._save_invite(draft)
            for transition in transitions:
                self._append_transition(transition)
            return drafts, transitions

    def create_outbound_message_drafts(
        self,
        *,
        conversation_id: str,
        drafts: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[OutboundMessageDraft], list[StateTransition]]:
        with self._lock, self._connection:
            from .models import new_id

            created: list[OutboundMessageDraft] = []
            transitions: list[StateTransition] = []
            for raw in drafts:
                if not isinstance(raw, dict):
                    continue
                draft = OutboundMessageDraft(
                    draft_id=new_id("outbound"),
                    conversation_id=conversation_id,
                    recipient_id=str(raw.get("recipient_id") or ""),
                    recipient_name=str(raw.get("recipient_name") or raw.get("recipient_id") or ""),
                    channel=str(raw.get("channel") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    purpose=str(raw.get("purpose") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                created.append(draft)
                self._save_message_reference(
                    MessageReference(
                        message_id=draft.draft_id,
                        conversation_id=draft.conversation_id,
                        business_ref_type="outbound_message_draft",
                        business_ref_id=draft.draft_id,
                        text=draft.message_text,
                        channel=draft.channel,
                        recipient_id=draft.recipient_id,
                        recipient_name=draft.recipient_name,
                        metadata={"source": "create_outbound_message_drafts", "purpose": draft.purpose},
                    )
                )
                transitions.append(
                    StateTransition(
                        "outbound_message_draft",
                        draft.draft_id,
                        None,
                        draft.status.value,
                        "create_outbound_message_drafts",
                        trace_id,
                    )
                )
                self._save_outbound_message_draft(draft)
            for transition in transitions:
                self._append_transition(transition)
            return created, transitions

    def update_invite_delivery_status(
        self,
        *,
        draft_id: str,
        status: InviteStatus | str,
        trace_id: str,
        reason: str,
    ) -> tuple[InviteDraft, StateTransition]:
        """Persist an approved send outcome in the same SQLite transaction."""

        with self._write_transaction():
            draft = self.invite_drafts.get(draft_id)
            if draft is None:
                raise ValueError(f"invite draft not found: {draft_id}")
            target = status if isinstance(status, InviteStatus) else InviteStatus(str(status))
            allowed = {
                InviteStatus.PENDING_APPROVAL: {InviteStatus.SENT, InviteStatus.SUPERSEDED},
                InviteStatus.SENT: {InviteStatus.SENT},
            }
            if target not in allowed.get(draft.status, set()):
                raise ValueError(f"invalid invite delivery transition: {draft.status.value}->{target.value}")
            old = draft.status.value
            draft.status = target
            draft.updated_at = now()
            transition = StateTransition("invite_draft", draft_id, old, target.value, reason, trace_id)
            self._save_invite(draft)
            self._append_transition(transition)
            return draft, transition

    def record_candidate_reply(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        status: str,
        seat_count: int = 1,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        with self._write_transaction():
            game = self.require_game(game_id)
            transitions: list[StateTransition] = []
            normalized_status = status.strip()
            normalized_seat_count = max(1, min(4, int(seat_count or 1)))
            existing_participant = next((item for item in game.participants if item.customer_id == customer_id), None)
            if normalized_status in CONFIRMED_CANDIDATE_STATUSES:
                claimed_by_others = sum(
                    max(1, int(item.seat_count))
                    for item in game.participants
                    if item.customer_id != customer_id and item.status in {"joined", "confirmed"}
                )
                available_seats = max(0, game.seats_total - claimed_by_others)
                if normalized_seat_count > available_seats:
                    raise ValueError(
                        f"seat capacity exceeded for game {game_id}: requested={normalized_seat_count}, "
                        f"available={available_seats}"
                    )
            for draft in self.invite_drafts.values():
                if draft.game_id == game_id and draft.customer_id == customer_id:
                    old = draft.status.value
                    draft.status = invite_status_from_candidate_status(normalized_status)
                    draft.updated_at = datetime.now(DEFAULT_TZ)
                    transitions.append(StateTransition("invite_draft", draft.draft_id, old, draft.status.value, "record_candidate_reply", trace_id))
                    self._save_invite(draft)
            if normalized_status in CONFIRMED_CANDIDATE_STATUSES:
                if existing_participant is None:
                    game.participants.append(
                        GameParticipant(
                            customer_id=customer_id,
                            display_name=display_name or customer_id,
                            status="confirmed",
                            source="candidate_reply",
                            seat_count=normalized_seat_count,
                        )
                    )
                    transitions.append(
                        StateTransition(
                            "game_participant",
                            f"{game.game_id}:{customer_id}",
                            None,
                            "confirmed",
                            "record_candidate_reply",
                            trace_id,
                        )
                    )
                else:
                    old_status = existing_participant.status
                    old_seat_count = max(1, int(existing_participant.seat_count))
                    existing_participant.status = "confirmed"
                    existing_participant.seat_count = normalized_seat_count
                    if old_status != existing_participant.status or old_seat_count != normalized_seat_count:
                        transitions.append(
                            StateTransition(
                                "game_participant",
                                f"{game.game_id}:{customer_id}",
                                f"{old_status}:seats={old_seat_count}",
                                f"{existing_participant.status}:seats={normalized_seat_count}",
                                "record_candidate_reply",
                                trace_id,
                            )
                        )
            elif normalized_status in UNCONFIRMED_CANDIDATE_STATUSES and existing_participant is not None:
                old_status = existing_participant.status
                old_seat_count = max(1, int(existing_participant.seat_count))
                existing_participant.status = normalized_status
                existing_participant.seat_count = normalized_seat_count
                if old_status != existing_participant.status or old_seat_count != normalized_seat_count:
                    transitions.append(
                        StateTransition(
                            "game_participant",
                            f"{game.game_id}:{customer_id}",
                            f"{old_status}:seats={old_seat_count}",
                            f"{existing_participant.status}:seats={normalized_seat_count}",
                            "record_candidate_reply",
                            trace_id,
                        )
                    )
            game.parties = normalize_game_parties(game.participants)
            game.requirement = refresh_requirement_seat_snapshot(game.requirement, game.parties, game.remaining_seats())
            if game.remaining_seats() == 0 and game.status != GameStatus.READY:
                old = game.status.value
                game.status = GameStatus.READY
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "seats_full", trace_id))
            elif game.remaining_seats() > 0 and game.status == GameStatus.READY:
                old = game.status.value
                game.status = (
                    GameStatus.INVITING
                    if any(draft.game_id == game.game_id for draft in self.invite_drafts.values())
                    else GameStatus.FORMING
                )
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "seats_reopened", trace_id))
            game.updated_at = datetime.now(DEFAULT_TZ)
            self._save_game(game)
            for transition in transitions:
                self._append_transition(transition)
            return game, transitions

    def update_game_status(self, *, game_id: str, status: str, reason: str, trace_id: str) -> tuple[Game, StateTransition]:
        with self._lock, self._connection:
            game = self.require_game(game_id)
            target = GameStatus(status)
            old = game.status.value
            allowed = ALLOWED_GAME_TRANSITIONS.get(old, set())
            if target.value != old and target.value not in allowed:
                raise ValueError(f"illegal game status transition: {old}->{target.value}")
            game.status = target
            if target in {GameStatus.CANCELLED, GameStatus.FINISHED}:
                game.closed_reason = reason or target.value
            game.updated_at = datetime.now(DEFAULT_TZ)
            transition = StateTransition("game", game.game_id, old, target.value, reason or "update_game_status", trace_id)
            self._save_game(game)
            self._append_transition(transition)
            if target in {GameStatus.CANCELLED, GameStatus.FINISHED}:
                self._release_room_reservations_for_game(
                    game.game_id,
                    trace_id=trace_id,
                    reason="game_status_closed",
                )
            return game, transition

    def record_badcase(self, payload: dict[str, Any], *, trace_id: str, conversation_id: str) -> dict[str, Any]:
        with self._lock, self._connection:
            from .models import new_id

            record = {"badcase_id": new_id("badcase"), "trace_id": trace_id, "conversation_id": conversation_id, **dict(payload)}
            self._connection.execute(
                """
                INSERT INTO runtime_badcases(badcase_id, trace_id, conversation_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record["badcase_id"], trace_id, conversation_id, _dumps(record), _now_iso()),
            )
            return record

    def require_game(self, game_id: str) -> Game:
        with self._lock:
            row = self._connection.execute("SELECT payload FROM runtime_games WHERE game_id = ?", (game_id,)).fetchone()
            if row is None:
                raise ValueError(f"game not found: {game_id}")
            return _game_from_payload(_loads(row["payload"]))

    def _save_game(self, game: Game) -> None:
        self._connection.execute(
            """
            INSERT INTO runtime_games(game_id, conversation_id, status, payload, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(game_id) DO UPDATE SET
                conversation_id=excluded.conversation_id,
                status=excluded.status,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (game.game_id, game.conversation_id, game.status.value, _dumps(game.to_dict()), game.updated_at.isoformat()),
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

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runtime_customers(
                customer_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_customer_relationships(
                pair_key TEXT PRIMARY KEY,
                customer_a_id TEXT NOT NULL,
                customer_b_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_games(
                game_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_invite_drafts(
                draft_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_outbound_message_drafts(
                draft_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_rooms(
                room_id TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_room_reservations(
                reservation_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                game_id TEXT NOT NULL DEFAULT '',
                start_at TEXT NOT NULL,
                end_at TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_state_transitions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_conversation_turns(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                role TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_conversation_checkpoints(
                conversation_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_conversation_versions(
                conversation_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_idempotency_ledger(
                idempotency_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_message_results(
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_message_references(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                business_ref_type TEXT NOT NULL,
                business_ref_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(conversation_id, message_id)
            );
            CREATE TABLE IF NOT EXISTS runtime_task_memories(
                memory_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                target_customer_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_pending_memory_candidates(
                candidate_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                target_customer_id TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_pending_input_batches(
                batch_key TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                quiet_deadline TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_badcases(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                badcase_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_runtime_turns_conversation_id ON runtime_conversation_turns(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_runtime_customer_relationships_a ON runtime_customer_relationships(customer_a_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_customer_relationships_b ON runtime_customer_relationships(customer_b_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_games_status ON runtime_games(status);
            CREATE INDEX IF NOT EXISTS idx_runtime_invites_game_id ON runtime_invite_drafts(game_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_outbound_conversation_id ON runtime_outbound_message_drafts(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_room_reservation_window ON runtime_room_reservations(room_id, status, start_at, end_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_checkpoints_updated_at ON runtime_conversation_checkpoints(updated_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_message_references_message_id ON runtime_message_references(message_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_message_references_business ON runtime_message_references(business_ref_type, business_ref_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_task_memories_conversation ON runtime_task_memories(conversation_id, status);
            CREATE INDEX IF NOT EXISTS idx_runtime_task_memories_customer ON runtime_task_memories(customer_id, target_customer_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_pending_memory_candidates_conversation ON runtime_pending_memory_candidates(conversation_id, status);
            CREATE INDEX IF NOT EXISTS idx_runtime_pending_memory_candidates_customer ON runtime_pending_memory_candidates(customer_id, target_customer_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_pending_input_due ON runtime_pending_input_batches(status, quiet_deadline);
            CREATE INDEX IF NOT EXISTS idx_runtime_pending_input_conversation ON runtime_pending_input_batches(conversation_id, sender_id);
            """
        )
        self._connection.commit()


def _customer_from_payload(payload: dict[str, Any]) -> CustomerProfile:
    return CustomerProfile(
        customer_id=str(payload.get("customer_id") or ""),
        display_name=str(payload.get("display_name") or ""),
        public_name=str(payload.get("public_name") or "") or None,
        private_remark=str(payload.get("private_remark") or ""),
        gender=payload.get("gender"),
        preferred_games=[str(item) for item in payload.get("preferred_games") or []],
        preferred_stakes=[str(item) for item in payload.get("preferred_stakes") or []],
        preferred_time_tags=[str(item) for item in payload.get("preferred_time_tags") or []],
        profile_facts=[str(item) for item in payload.get("profile_facts") or []],
        smoke_preference=payload.get("smoke_preference"),
        response_score=float(payload.get("response_score") or 0.5),
        fatigue_score=float(payload.get("fatigue_score") or 0.0),
        no_contact=bool(payload.get("no_contact")),
        notes=str(payload.get("notes") or ""),
    )


def _relationship_from_payload(payload: dict[str, Any]) -> CustomerRelationship:
    return CustomerRelationship(
        customer_a_id=str(payload.get("customer_a_id") or ""),
        customer_b_id=str(payload.get("customer_b_id") or ""),
        played_together_count=int(payload.get("played_together_count") or 0),
        avoid_playing=bool(payload.get("avoid_playing")),
        notes=str(payload.get("notes") or ""),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _task_memory_from_payload(payload: dict[str, Any]) -> TaskMemory:
    return TaskMemory(
        memory_id=str(payload.get("memory_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        memory_type=str(payload.get("memory_type") or ""),
        field=str(payload.get("field") or ""),
        value=payload.get("value"),
        target_customer_id=str(payload.get("target_customer_id") or "") or None,
        evidence=str(payload.get("evidence") or ""),
        confidence=float(payload.get("confidence") or 0.0),
        risk_level=str(payload.get("risk_level") or "medium"),
        scope=str(payload.get("scope") or "current_task"),
        status=str(payload.get("status") or "active"),
        source_trace_id=payload.get("source_trace_id"),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _pending_memory_candidate_from_payload(payload: dict[str, Any]) -> PendingMemoryCandidate:
    return PendingMemoryCandidate(
        candidate_id=str(payload.get("candidate_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        memory_type=str(payload.get("memory_type") or ""),
        field=str(payload.get("field") or ""),
        value=payload.get("value"),
        operation=str(payload.get("operation") or "set"),
        target_customer_id=str(payload.get("target_customer_id") or "") or None,
        evidence=str(payload.get("evidence") or ""),
        confidence=float(payload.get("confidence") or 0.0),
        risk_level=str(payload.get("risk_level") or "medium"),
        scope=str(payload.get("scope") or "long_term"),
        status=str(payload.get("status") or "pending_review"),
        source_trace_id=payload.get("source_trace_id"),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _pending_input_batch_from_payload(payload: dict[str, Any]) -> PendingInputBatch:
    return PendingInputBatch(
        batch_id=str(payload.get("batch_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        sender_id=str(payload.get("sender_id") or ""),
        sender_name=str(payload.get("sender_name") or ""),
        fragments=[dict(item) for item in payload.get("fragments") or [] if isinstance(item, dict)],
        version=int(payload.get("version") or 1),
        status=PendingInputBatchStatus(str(payload.get("status") or PendingInputBatchStatus.PENDING.value)),
        quiet_deadline=_datetime_from_payload(payload.get("quiet_deadline")),
        source_channel=str(payload.get("source_channel") or ""),
        decision=dict(payload.get("decision") or {}) if isinstance(payload.get("decision"), dict) else {},
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _turn_from_payload(payload: dict[str, Any]) -> ConversationTurn:
    return ConversationTurn(
        role=ConversationRole(str(payload.get("role") or ConversationRole.USER.value)),
        content=str(payload.get("content") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        sender_id=payload.get("sender_id"),
        sender_name=payload.get("sender_name"),
        metadata=dict(payload.get("metadata") or {}),
        occurred_at=_datetime_from_payload(payload.get("occurred_at")),
    )


def _checkpoint_from_payload(payload: dict[str, Any]) -> ConversationCheckpoint:
    return ConversationCheckpoint(
        conversation_id=str(payload.get("conversation_id") or ""),
        summary=str(payload.get("summary") or ""),
        facts=dict(payload.get("facts") or {}) if isinstance(payload.get("facts"), dict) else {},
        open_questions=[str(item) for item in payload.get("open_questions") or []],
        source_trace_id=payload.get("source_trace_id"),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _game_from_payload(payload: dict[str, Any]) -> Game:
    participants = normalize_game_participants(
        organizer_id=str(payload.get("organizer_id") or ""),
        organizer_name=str(payload.get("organizer_name") or ""),
        known_players=list(payload.get("participants") or []),
    )
    parties = [
        Party(
            party_id=str(item.get("party_id") or f"party_{item.get('contact_id') or item.get('customer_id') or ''}"),
            contact_id=str(item.get("contact_id") or item.get("customer_id") or ""),
            contact_name=str(item.get("contact_name") or item.get("display_name") or item.get("contact_id") or ""),
            seat_count=int(item.get("seat_count") or 1),
            known_member_ids=[str(member) for member in item.get("known_member_ids") or []],
            anonymous_seat_count=int(item.get("anonymous_seat_count") or 0),
            status=str(item.get("status") or "joined"),
            source=str(item.get("source") or "requester"),
        )
        for item in payload.get("parties") or []
        if isinstance(item, dict)
    ]
    return Game(
        game_id=str(payload.get("game_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        organizer_id=str(payload.get("organizer_id") or ""),
        organizer_name=str(payload.get("organizer_name") or ""),
        requirement=dict(payload.get("requirement") or {}),
        status=GameStatus(str(payload.get("status") or GameStatus.FORMING.value)),
        participants=participants,
        parties=parties,
        seats_total=int(payload.get("seats_total") or 4),
        planned_start_at=_optional_datetime_from_payload(payload.get("planned_start_at")),
        planned_end_at=_optional_datetime_from_payload(payload.get("planned_end_at")),
        expires_at=_optional_datetime_from_payload(payload.get("expires_at") or payload.get("lifecycle_expires_at")),
        closed_reason=str(payload.get("closed_reason") or ""),
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _invite_from_payload(payload: dict[str, Any]) -> InviteDraft:
    return InviteDraft(
        draft_id=str(payload.get("draft_id") or ""),
        game_id=str(payload.get("game_id") or ""),
        customer_id=str(payload.get("customer_id") or ""),
        display_name=str(payload.get("display_name") or ""),
        message_text=str(payload.get("message_text") or ""),
        status=InviteStatus(str(payload.get("status") or InviteStatus.PENDING_APPROVAL.value)),
        metadata=dict(payload.get("metadata") or {}),
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _outbound_message_draft_from_payload(payload: dict[str, Any]) -> OutboundMessageDraft:
    return OutboundMessageDraft(
        draft_id=str(payload.get("draft_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        recipient_id=str(payload.get("recipient_id") or ""),
        recipient_name=str(payload.get("recipient_name") or ""),
        channel=str(payload.get("channel") or ""),
        message_text=str(payload.get("message_text") or ""),
        purpose=str(payload.get("purpose") or ""),
        status=OutboundDraftStatus(str(payload.get("status") or OutboundDraftStatus.PENDING_APPROVAL.value)),
        metadata=dict(payload.get("metadata") or {}),
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _room_reservation_from_payload(payload: dict[str, Any]) -> RoomReservation:
    return RoomReservation(
        reservation_id=str(payload.get("reservation_id") or ""),
        room_id=str(payload.get("room_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        game_id=str(payload.get("game_id") or "") or None,
        start_at=_datetime_from_payload(payload.get("start_at")),
        end_at=_datetime_from_payload(payload.get("end_at")),
        status=str(payload.get("status") or "held"),
        source_trace_id=str(payload.get("source_trace_id") or "") or None,
        created_at=_datetime_from_payload(payload.get("created_at")),
        updated_at=_datetime_from_payload(payload.get("updated_at")),
    )


def _message_reference_from_payload(payload: dict[str, Any]) -> MessageReference:
    return MessageReference(
        message_id=str(payload.get("message_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        business_ref_type=str(payload.get("business_ref_type") or ""),
        business_ref_id=str(payload.get("business_ref_id") or ""),
        text=str(payload.get("text") or ""),
        channel=payload.get("channel"),
        sender_id=payload.get("sender_id"),
        sender_name=payload.get("sender_name"),
        recipient_id=payload.get("recipient_id"),
        recipient_name=payload.get("recipient_name"),
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {},
        created_at=_datetime_from_payload(payload.get("created_at")),
    )


def _transition_from_payload(payload: dict[str, Any]) -> StateTransition:
    return StateTransition(
        entity_type=str(payload.get("entity_type") or ""),
        entity_id=str(payload.get("entity_id") or ""),
        from_status=payload.get("from_status"),
        to_status=str(payload.get("to_status") or ""),
        reason=str(payload.get("reason") or ""),
        trace_id=str(payload.get("trace_id") or ""),
        occurred_at=_datetime_from_payload(payload.get("occurred_at")),
    )


def _tool_call_from_payload(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name=str(payload.get("name") or ""),
        arguments=dict(payload.get("arguments") or {}),
        reason=str(payload.get("reason") or ""),
        idempotency_key=payload.get("idempotency_key"),
    )


def _action_from_payload(payload: dict[str, Any]) -> AgentAction:
    return AgentAction(
        goal=str(payload.get("goal") or ""),
        objective_status=str(payload.get("objective_status") or "unknown"),
        reasoning_summary=str(payload.get("reasoning_summary") or ""),
        reply_to_user=str(payload.get("reply_to_user") or ""),
        tool_calls=[
            _tool_call_from_payload(item)
            for item in payload.get("tool_calls") or []
            if isinstance(item, dict)
        ],
        needs_human=bool(payload.get("needs_human")),
        stop_reason=dict(payload.get("stop_reason") or {}) if isinstance(payload.get("stop_reason"), dict) else {},
        badcase=payload.get("badcase") if isinstance(payload.get("badcase"), dict) else None,
    )


def _tool_result_from_payload(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        name=str(payload.get("name") or ""),
        called=bool(payload.get("called")),
        allowed=bool(payload.get("allowed")),
        result=dict(payload.get("result") or {}),
        error=payload.get("error"),
        idempotency_key=payload.get("idempotency_key"),
        deduplicated=bool(payload.get("deduplicated")),
        state_transitions=[
            _transition_from_payload(item)
            for item in payload.get("state_transitions") or []
            if isinstance(item, dict)
        ],
    )


def _runtime_result_from_payload(payload: dict[str, Any]) -> AgentRuntimeResult:
    return AgentRuntimeResult(
        trace_id=str(payload.get("trace_id") or ""),
        conversation_id=str(payload.get("conversation_id") or ""),
        final_reply=str(payload.get("final_reply") or ""),
        actions=[
            _action_from_payload(item)
            for item in payload.get("actions") or []
            if isinstance(item, dict)
        ],
        tool_results=[
            _tool_result_from_payload(item)
            for item in payload.get("tool_results") or []
            if isinstance(item, dict)
        ],
        state_transitions=[
            _transition_from_payload(item)
            for item in payload.get("state_transitions") or []
            if isinstance(item, dict)
        ],
    )


def _datetime_from_payload(value: Any) -> datetime:
    if value:
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=DEFAULT_TZ)
    return datetime.now(DEFAULT_TZ)


def _optional_datetime_from_payload(value: Any) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=DEFAULT_TZ)


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _loads(payload: str) -> dict[str, Any]:
    raw = json.loads(payload)
    return raw if isinstance(raw, dict) else {}


def _now_iso() -> str:
    return datetime.now(DEFAULT_TZ).isoformat()
