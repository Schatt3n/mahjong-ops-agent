"""SQLite migration store operations."""

from __future__ import annotations

from ...models import now
from ...store import (
    normalize_game_participants,
    normalize_game_parties,
)
from .serialization import (
    _dumps,
    _game_from_payload,
    _game_storage_payload,
    _loads,
    _optional_datetime_from_payload,
)

class SQLiteMigrationStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

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
            CREATE TABLE IF NOT EXISTS runtime_game_participants(
                game_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                seat_count INTEGER NOT NULL DEFAULT 1,
                party_id TEXT,
                known_member_ids TEXT NOT NULL DEFAULT '[]',
                anonymous_seat_count INTEGER NOT NULL DEFAULT 0,
                joined_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(game_id, customer_id),
                FOREIGN KEY(game_id) REFERENCES runtime_games(game_id) ON DELETE CASCADE
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
            CREATE TABLE IF NOT EXISTS runtime_task_contexts(
                task_context_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
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
            CREATE TABLE IF NOT EXISTS runtime_scheduled_agent_tasks(
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                due_at TEXT NOT NULL,
                lease_until TEXT,
                idempotency_key TEXT NOT NULL UNIQUE,
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
            CREATE INDEX IF NOT EXISTS idx_runtime_game_participants_customer ON runtime_game_participants(customer_id, status);
            CREATE INDEX IF NOT EXISTS idx_runtime_game_participants_game_status ON runtime_game_participants(game_id, status);
            CREATE INDEX IF NOT EXISTS idx_runtime_invites_game_id ON runtime_invite_drafts(game_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_outbound_conversation_id ON runtime_outbound_message_drafts(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_room_reservation_window ON runtime_room_reservations(room_id, status, start_at, end_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_checkpoints_updated_at ON runtime_conversation_checkpoints(updated_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_task_contexts_current ON runtime_task_contexts(conversation_id, customer_id, status, updated_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_message_references_message_id ON runtime_message_references(message_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_message_references_business ON runtime_message_references(business_ref_type, business_ref_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_task_memories_conversation ON runtime_task_memories(conversation_id, status);
            CREATE INDEX IF NOT EXISTS idx_runtime_task_memories_customer ON runtime_task_memories(customer_id, target_customer_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_pending_memory_candidates_conversation ON runtime_pending_memory_candidates(conversation_id, status);
            CREATE INDEX IF NOT EXISTS idx_runtime_pending_memory_candidates_customer ON runtime_pending_memory_candidates(customer_id, target_customer_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_pending_input_due ON runtime_pending_input_batches(status, quiet_deadline);
            CREATE INDEX IF NOT EXISTS idx_runtime_pending_input_conversation ON runtime_pending_input_batches(conversation_id, sender_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_scheduled_agent_due ON runtime_scheduled_agent_tasks(status, due_at, lease_until);
            CREATE INDEX IF NOT EXISTS idx_runtime_scheduled_agent_aggregate ON runtime_scheduled_agent_tasks(task_type, aggregate_id);
            """
        )
        self._migrate_embedded_game_participants()
        self._connection.commit()

    def _migrate_embedded_game_participants(self) -> None:
        """Backfill legacy embedded participants and remove them from game JSON.

        The migration is idempotent. Once normalized rows exist, they are the
        authority and a stale legacy payload can never overwrite them.
        """

        rows = self._connection.execute(
            "SELECT game_id, payload, updated_at FROM runtime_games"
        ).fetchall()
        for row in rows:
            game_id = str(row["game_id"])
            payload = _loads(row["payload"])
            participant_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) AS count FROM runtime_game_participants WHERE game_id = ?",
                    (game_id,),
                ).fetchone()["count"]
            )
            if participant_count == 0:
                legacy_participant_payloads = {
                    str(item.get("customer_id") or ""): item
                    for item in payload.get("participants") or []
                    if isinstance(item, dict) and str(item.get("customer_id") or "")
                }
                participants = normalize_game_participants(
                    organizer_id=str(payload.get("organizer_id") or ""),
                    organizer_name=str(payload.get("organizer_name") or ""),
                    known_players=list(payload.get("participants") or []),
                )
                legacy_joined_at = _optional_datetime_from_payload(payload.get("created_at")) or now()
                for participant in participants:
                    raw = legacy_participant_payloads.get(participant.customer_id, {})
                    participant.display_name = str(raw.get("display_name") or participant.display_name)
                    participant.source = str(raw.get("source") or participant.source)
                    participant.party_id = str(raw.get("party_id") or participant.party_id or "") or None
                    participant.known_member_ids = [
                        str(item) for item in raw.get("known_member_ids") or participant.known_member_ids
                    ]
                    participant.anonymous_seat_count = max(
                        0,
                        int(raw.get("anonymous_seat_count", participant.anonymous_seat_count) or 0),
                    )
                    participant.joined_at = (
                        _optional_datetime_from_payload(raw.get("joined_at"))
                        if isinstance(raw, dict)
                        else None
                    ) or legacy_joined_at
                migration_game = _game_from_payload(payload)
                migration_game.participants = participants
                migration_game.parties = normalize_game_parties(participants)
                self._save_game_participants(migration_game)

            normalized_payload = _game_storage_payload(_game_from_payload(payload))
            if normalized_payload != payload:
                self._connection.execute(
                    "UPDATE runtime_games SET payload = ?, updated_at = ? WHERE game_id = ?",
                    (_dumps(normalized_payload), str(row["updated_at"]), game_id),
                )
