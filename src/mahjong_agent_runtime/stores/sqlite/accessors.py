"""SQLite accessors store operations."""

from __future__ import annotations

from typing import Any
from ...models import (
    ConversationCheckpoint,
    ConversationTaskContext,
    CustomerProfile,
    CustomerRelationship,
    Game,
    GameParticipant,
    InviteDraft,
    OutboundMessageDraft,
    PendingInputBatch,
    PendingMemoryCandidate,
    RoomReservation,
    ScheduledAgentTask,
    StateTransition,
    TaskMemory,
)
from ...store import normalize_game_parties
from .serialization import (
    _checkpoint_from_payload,
    _customer_from_payload,
    _game_from_payload,
    _game_participant_from_row,
    _invite_from_payload,
    _loads,
    _outbound_message_draft_from_payload,
    _pending_input_batch_from_payload,
    _pending_memory_candidate_from_payload,
    _relationship_from_payload,
    _room_reservation_from_payload,
    _scheduled_agent_task_from_payload,
    _task_context_from_payload,
    _task_memory_from_payload,
    _transition_from_payload,
)

class SQLiteAccessorsStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

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
            participant_rows = self._connection.execute(
                """
                SELECT game_id, customer_id, display_name, status, source, seat_count,
                       party_id, known_member_ids, anonymous_seat_count, joined_at
                FROM runtime_game_participants
                ORDER BY game_id, joined_at, customer_id
                """
            ).fetchall()
            participants_by_game: dict[str, list[GameParticipant]] = {}
            for participant_row in participant_rows:
                participants_by_game.setdefault(str(participant_row["game_id"]), []).append(
                    _game_participant_from_row(participant_row)
                )
            games: dict[str, Game] = {}
            for row in rows:
                game = _game_from_payload(_loads(row["payload"]))
                game.participants = participants_by_game.get(game.game_id, [])
                game.parties = normalize_game_parties(game.participants)
                games[game.game_id] = game
            return games

    def _load_game_participants(self, game_id: str) -> list[GameParticipant]:
        """Load the normalized participant rows for one game in join order."""

        rows = self._connection.execute(
            """
            SELECT game_id, customer_id, display_name, status, source, seat_count,
                   party_id, known_member_ids, anonymous_seat_count, joined_at
            FROM runtime_game_participants
            WHERE game_id = ?
            ORDER BY joined_at, customer_id
            """,
            (game_id,),
        ).fetchall()
        return [_game_participant_from_row(row) for row in rows]

    def _hydrate_game(self, payload: dict[str, Any]) -> Game:
        """Build the aggregate from its base row plus normalized participants."""

        game = _game_from_payload(payload)
        game.participants = self._load_game_participants(game.game_id)
        game.parties = normalize_game_parties(game.participants)
        return game

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
    def task_contexts(self) -> dict[str, ConversationTaskContext]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM runtime_task_contexts").fetchall()
            return {
                item.task_context_id: item
                for item in (_task_context_from_payload(_loads(row["payload"])) for row in rows)
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

    @property
    def scheduled_tasks(self) -> dict[str, ScheduledAgentTask]:
        with self._lock:
            rows = self._connection.execute("SELECT task_id, payload FROM runtime_scheduled_agent_tasks").fetchall()
            return {
                str(row["task_id"]): _scheduled_agent_task_from_payload(_loads(row["payload"]))
                for row in rows
            }
