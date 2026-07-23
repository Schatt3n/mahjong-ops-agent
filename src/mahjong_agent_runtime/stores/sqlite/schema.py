"""SQLite schema initialization and strict version validation."""

from __future__ import annotations

CURRENT_SCHEMA_VERSION = 1


class SQLiteSchemaStoreMixin:
    """Create only the current schema and reject older database layouts."""

    __slots__ = ()

    def _initialize_schema(self) -> None:
        existing_tables = {
            str(row["name"])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        schema_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if existing_tables and schema_version != CURRENT_SCHEMA_VERSION:
            raise RuntimeError(
                "unsupported SQLite schema version "
                f"{schema_version}; expected {CURRENT_SCHEMA_VERSION}. "
                "Rebuild the development database; automatic legacy migration is intentionally disabled."
            )
        if not existing_tables and schema_version not in {0, CURRENT_SCHEMA_VERSION}:
            raise RuntimeError(
                "unsupported SQLite schema version "
                f"{schema_version}; expected {CURRENT_SCHEMA_VERSION}"
            )
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
                task_context_id TEXT NOT NULL DEFAULT '',
                source_message_id TEXT NOT NULL DEFAULT '',
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_conversation_checkpoints(
                conversation_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_task_context_checkpoints(
                task_context_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
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
            CREATE TABLE IF NOT EXISTS runtime_agent_runs(
                run_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                run_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                lease_owner TEXT NOT NULL DEFAULT '',
                lease_until TEXT,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS waiting_demands(
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT,
                demand JSON NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                matched_game_id TEXT
            );
            CREATE TABLE IF NOT EXISTS runtime_badcases(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                badcase_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_channel_identities(
                channel TEXT NOT NULL,
                external_user_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                can_private_message INTEGER NOT NULL DEFAULT 0,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(channel, external_user_id)
            );
            CREATE TABLE IF NOT EXISTS runtime_group_room_policies(
                room_id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                managed INTEGER NOT NULL DEFAULT 1,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_group_board_states(
                room_id TEXT PRIMARY KEY,
                source_message_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_game_conversation_links(
                link_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                room_id TEXT NOT NULL,
                customer_id TEXT NOT NULL DEFAULT '',
                link_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(game_id, conversation_id, customer_id, link_type),
                FOREIGN KEY(game_id) REFERENCES runtime_games(game_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS runtime_group_board_snapshots(
                snapshot_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                external_message_id TEXT NOT NULL,
                rendered_text TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(room_id, external_message_id)
            );
            CREATE TABLE IF NOT EXISTS runtime_group_board_items(
                snapshot_id TEXT NOT NULL,
                item_no INTEGER NOT NULL,
                game_id TEXT NOT NULL,
                rendered_text TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, item_no),
                FOREIGN KEY(snapshot_id) REFERENCES runtime_group_board_snapshots(snapshot_id) ON DELETE CASCADE,
                FOREIGN KEY(game_id) REFERENCES runtime_games(game_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS runtime_group_game_claims(
                claim_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                source_conversation_id TEXT NOT NULL,
                source_message_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(source_conversation_id, source_message_id),
                FOREIGN KEY(game_id) REFERENCES runtime_games(game_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS runtime_channel_switches(
                switch_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                private_conversation_id TEXT NOT NULL,
                status TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runtime_channel_observations(
                channel TEXT NOT NULL,
                source_message_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                room_id TEXT NOT NULL,
                room_topic TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                message_text TEXT NOT NULL,
                message_type TEXT NOT NULL,
                is_room INTEGER NOT NULL DEFAULT 0,
                self_message INTEGER NOT NULL DEFAULT 0,
                route_status TEXT NOT NULL,
                route_mode TEXT NOT NULL,
                route_reason TEXT NOT NULL,
                semantic_action TEXT NOT NULL,
                semantic_category TEXT NOT NULL,
                semantic_confidence REAL NOT NULL DEFAULT 0,
                business_message_detected INTEGER NOT NULL DEFAULT 0,
                payload TEXT NOT NULL,
                received_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(channel, source_message_id)
            );
            CREATE INDEX IF NOT EXISTS idx_runtime_turns_conversation_id ON runtime_conversation_turns(conversation_id, id);
            CREATE INDEX IF NOT EXISTS idx_runtime_turns_task_context
                ON runtime_conversation_turns(conversation_id, task_context_id, id);
            CREATE INDEX IF NOT EXISTS idx_runtime_turns_source_message
                ON runtime_conversation_turns(conversation_id, source_message_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_task_checkpoints_conversation
                ON runtime_task_context_checkpoints(conversation_id, updated_at);
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
            CREATE INDEX IF NOT EXISTS idx_runtime_agent_runs_recovery
                ON runtime_agent_runs(status, lease_until, updated_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_agent_runs_conversation
                ON runtime_agent_runs(conversation_id, run_version, status);
            CREATE INDEX IF NOT EXISTS idx_waiting_demands_active_expiry ON waiting_demands(status, expires_at);
            CREATE INDEX IF NOT EXISTS idx_waiting_demands_sender ON waiting_demands(conversation_id, sender_id, status);
            CREATE INDEX IF NOT EXISTS idx_runtime_channel_identity_customer
                ON runtime_channel_identities(customer_id, channel, can_private_message);
            CREATE INDEX IF NOT EXISTS idx_runtime_game_links_room
                ON runtime_game_conversation_links(room_id, game_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_board_snapshots_room
                ON runtime_group_board_snapshots(room_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_group_claims_game
                ON runtime_group_game_claims(game_id, customer_id);
            CREATE INDEX IF NOT EXISTS idx_runtime_channel_switch_active
                ON runtime_channel_switches(customer_id, room_id, status, expires_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_channel_observations_room_time
                ON runtime_channel_observations(room_id, received_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_channel_observations_topic_time
                ON runtime_channel_observations(room_topic, received_at);
            CREATE INDEX IF NOT EXISTS idx_runtime_channel_observations_semantic_time
                ON runtime_channel_observations(semantic_action, received_at);
            """
        )
        self._connection.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
        self._connection.commit()
