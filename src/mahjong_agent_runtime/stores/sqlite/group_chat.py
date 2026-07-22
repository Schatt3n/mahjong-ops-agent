"""SQLite persistence for public-room routing, boards, and atomic claims."""

from __future__ import annotations

from datetime import datetime

from ...domains import game_schedule_sort_key
from ...models import (
    GameStatus,
    ScheduledAgentTask,
    ScheduledTaskStatus,
    StateTransition,
    new_id,
    now,
)
from .serialization import _dumps, _loads, _scheduled_agent_task_from_payload


class SQLiteGroupChatStoreMixin:
    __slots__ = ()

    def upsert_channel_identity(self, identity):
        identity.updated_at = now()
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_channel_identities WHERE channel = ? AND external_user_id = ?",
                (identity.channel, identity.external_user_id),
            ).fetchone()
            if row is not None:
                identity.created_at = self._channel_identity_from_payload(_loads(row["payload"])).created_at
            self._connection.execute(
                """
                INSERT INTO runtime_channel_identities(
                    channel, external_user_id, customer_id, can_private_message, payload, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel, external_user_id) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    can_private_message=excluded.can_private_message,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    identity.channel,
                    identity.external_user_id,
                    identity.customer_id,
                    int(identity.can_private_message),
                    _dumps(identity.to_dict()),
                    identity.updated_at.isoformat(),
                ),
            )
            return identity

    def get_channel_identity(self, channel: str, external_user_id: str):
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_channel_identities WHERE channel = ? AND external_user_id = ?",
                (channel, external_user_id),
            ).fetchone()
            return self._channel_identity_from_payload(_loads(row["payload"])) if row else None

    def get_channel_identity_for_customer(self, customer_id: str, channel: str = "wechaty"):
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_channel_identities
                WHERE customer_id = ? AND channel = ?
                ORDER BY can_private_message DESC, updated_at DESC
                LIMIT 1
                """,
                (customer_id, channel),
            ).fetchone()
            return self._channel_identity_from_payload(_loads(row["payload"])) if row else None

    def upsert_group_room_policy(self, policy):
        policy.updated_at = now()
        with self._write_transaction():
            self._connection.execute(
                """
                INSERT INTO runtime_group_room_policies(room_id, channel, managed, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(room_id) DO UPDATE SET
                    channel=excluded.channel,
                    managed=excluded.managed,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    policy.room_id,
                    policy.channel,
                    int(policy.managed),
                    _dumps(policy.to_dict()),
                    policy.updated_at.isoformat(),
                ),
            )
            return policy

    def get_group_room_policy(self, room_id: str):
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_group_room_policies WHERE room_id = ?",
                (room_id,),
            ).fetchone()
            return self._group_room_policy_from_payload(_loads(row["payload"])) if row else None

    def upsert_group_board_state(self, board_state):
        """Persist one replaceable owner-authored board, separate from outbound snapshots."""

        with self._write_transaction():
            self._connection.execute(
                """
                INSERT INTO runtime_group_board_states(room_id, source_message_id, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(room_id) DO UPDATE SET
                    source_message_id=excluded.source_message_id,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    board_state.room_id,
                    board_state.source_message_id,
                    _dumps(board_state.to_dict()),
                    board_state.last_published_at.isoformat(),
                ),
            )
            return board_state

    def get_group_board_state(self, room_id: str):
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_group_board_states WHERE room_id = ?",
                (room_id,),
            ).fetchone()
            return self._group_board_state_from_payload(_loads(row["payload"])) if row else None

    def link_game_conversation(self, link):
        with self._write_transaction():
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_game_conversation_links
                WHERE game_id = ? AND conversation_id = ? AND customer_id = ? AND link_type = ?
                """,
                (link.game_id, link.conversation_id, link.customer_id or "", link.link_type),
            ).fetchone()
            if row is not None:
                return self._game_conversation_link_from_payload(_loads(row["payload"]))
            self._connection.execute(
                """
                INSERT INTO runtime_game_conversation_links(
                    link_id, game_id, conversation_id, room_id, customer_id, link_type, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link.link_id,
                    link.game_id,
                    link.conversation_id,
                    link.room_id,
                    link.customer_id or "",
                    link.link_type,
                    _dumps(link.to_dict()),
                    link.created_at.isoformat(),
                ),
            )
            return link

    def game_conversation_links(self, game_id: str | None = None, room_id: str | None = None):
        clauses: list[str] = []
        values: list[str] = []
        if game_id:
            clauses.append("game_id = ?")
            values.append(game_id)
        if room_id:
            clauses.append("room_id = ?")
            values.append(room_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT payload FROM runtime_game_conversation_links {where} ORDER BY created_at, link_id",
                tuple(values),
            ).fetchall()
            return [self._game_conversation_link_from_payload(_loads(row["payload"])) for row in rows]

    def get_board_eligible_games(self, room_id: str):
        game_ids = {item.game_id for item in self.game_conversation_links(room_id=room_id)}
        games = [
            game
            for game_id, game in self.games.items()
            if game_id in game_ids
            and game.status in {GameStatus.FORMING, GameStatus.INVITING}
            and game.remaining_seats() > 0
        ]
        return sorted(games, key=game_schedule_sort_key)

    def save_board_snapshot(self, snapshot):
        with self._write_transaction():
            self._connection.execute(
                """
                INSERT INTO runtime_group_board_snapshots(
                    snapshot_id, room_id, conversation_id, external_message_id,
                    rendered_text, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.room_id,
                    snapshot.conversation_id,
                    snapshot.external_message_id,
                    snapshot.rendered_text,
                    _dumps(snapshot.to_dict()),
                    snapshot.created_at.isoformat(),
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO runtime_group_board_items(snapshot_id, item_no, game_id, rendered_text)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (snapshot.snapshot_id, item.item_no, item.game_id, item.rendered_text)
                    for item in snapshot.items
                ],
            )
            return snapshot

    def get_board_snapshot(self, snapshot_id: str):
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runtime_group_board_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            return self._board_snapshot_from_payload(_loads(row["payload"])) if row else None

    def get_latest_board_snapshot(self, room_id: str):
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_group_board_snapshots
                WHERE room_id = ? ORDER BY rowid DESC LIMIT 1
                """,
                (room_id,),
            ).fetchone()
            return self._board_snapshot_from_payload(_loads(row["payload"])) if row else None

    def get_board_snapshot_by_message_id(self, room_id: str, external_message_id: str):
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_group_board_snapshots
                WHERE room_id = ? AND external_message_id = ? LIMIT 1
                """,
                (room_id, external_message_id),
            ).fetchone()
            return self._board_snapshot_from_payload(_loads(row["payload"])) if row else None

    def get_game_claim_by_source(self, source_conversation_id: str, source_message_id: str):
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_group_game_claims
                WHERE source_conversation_id = ? AND source_message_id = ?
                """,
                (source_conversation_id, source_message_id),
            ).fetchone()
            return self._game_claim_from_payload(_loads(row["payload"])) if row else None

    def atomic_claim_seat(
        self,
        *,
        room_id: str,
        game_id: str,
        customer_id: str,
        display_name: str,
        source_conversation_id: str,
        source_message_id: str,
        trace_id: str,
    ):
        from ...group_chat.models import GameClaim

        with self._write_transaction():
            row = self._connection.execute(
                """
                SELECT payload FROM runtime_group_game_claims
                WHERE source_conversation_id = ? AND source_message_id = ?
                """,
                (source_conversation_id, source_message_id),
            ).fetchone()
            if row is not None:
                claim = self._game_claim_from_payload(_loads(row["payload"]))
                return claim, self.require_game(claim.game_id), [], True
            game, transitions = self._record_candidate_reply_in_transaction(
                game_id=game_id,
                customer_id=customer_id,
                display_name=display_name,
                status="confirmed",
                seat_count=1,
                trace_id=trace_id,
            )
            claim = GameClaim(
                claim_id=new_id("claim"),
                room_id=room_id,
                game_id=game_id,
                customer_id=customer_id,
                source_conversation_id=source_conversation_id,
                source_message_id=source_message_id,
            )
            self._connection.execute(
                """
                INSERT INTO runtime_group_game_claims(
                    claim_id, room_id, game_id, customer_id, source_conversation_id,
                    source_message_id, status, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.claim_id,
                    room_id,
                    game_id,
                    customer_id,
                    source_conversation_id,
                    source_message_id,
                    claim.status,
                    _dumps(claim.to_dict()),
                    claim.created_at.isoformat(),
                ),
            )
            return claim, game, transitions, False

    def record_channel_switch(self, switch):
        with self._write_transaction():
            self._connection.execute(
                """
                INSERT INTO runtime_channel_switches(
                    switch_id, room_id, customer_id, private_conversation_id,
                    status, expires_at, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    switch.switch_id,
                    switch.room_id,
                    switch.customer_id,
                    switch.private_conversation_id,
                    switch.status,
                    switch.expires_at.isoformat(),
                    _dumps(switch.to_dict()),
                    switch.created_at.isoformat(),
                ),
            )
            return switch

    def get_recent_active_channel_switch(
        self,
        customer_id: str,
        *,
        room_id: str | None = None,
        at: datetime | None = None,
    ):
        stamp = at or now()
        clauses = ["customer_id = ?", "status = 'active'", "expires_at >= ?"]
        values: list[str] = [customer_id, stamp.isoformat()]
        if room_id:
            clauses.append("room_id = ?")
            values.append(room_id)
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT payload FROM runtime_channel_switches
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC LIMIT 1
                """,
                tuple(values),
            ).fetchone()
            return self._channel_switch_from_payload(_loads(row["payload"])) if row else None

    def ensure_group_board_publish_task(
        self,
        *,
        room_id: str,
        due_at: datetime,
        trace_id: str,
        urgent: bool = False,
    ):
        base_task_id = f"group_board_publish:{room_id}"
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT payload FROM runtime_scheduled_agent_tasks WHERE task_id = ?",
                (base_task_id,),
            ).fetchone()
            existing = _scheduled_agent_task_from_payload(_loads(row["payload"])) if row else None
            if existing is not None and existing.status == ScheduledTaskStatus.PENDING:
                if urgent or due_at < existing.due_at:
                    existing.due_at = due_at
                    existing.updated_at = now()
                    self._save_scheduled_agent_task(existing)
                return existing, None
            task_id = base_task_id
            if existing is not None and existing.status == ScheduledTaskStatus.PROCESSING:
                task_id = new_id(base_task_id)
            task = existing if task_id == base_task_id and existing is not None else ScheduledAgentTask(
                task_id=task_id,
                task_type="publish_group_board",
                aggregate_type="group_room",
                aggregate_id=room_id,
                conversation_id=f"wechaty:room:{room_id}",
                subject_id="system",
                subject_name="system",
                due_at=due_at,
                idempotency_key=task_id,
                payload={"event_type": "publish_group_board", "room_id": room_id},
            )
            old_status = existing.status.value if existing is not None else None
            task.status = ScheduledTaskStatus.PENDING
            task.due_at = due_at
            task.lease_until = None
            task.completed_at = None
            task.last_error = ""
            task.updated_at = now()
            self._save_scheduled_agent_task(task)
            transition = StateTransition(
                "scheduled_agent_task",
                task.task_id,
                old_status,
                task.status.value,
                "group_board_publish_scheduled",
                trace_id,
            )
            self._append_transition(transition)
            return task, transition

    @staticmethod
    def _channel_identity_from_payload(payload):
        from ...group_chat.models import ChannelIdentity
        from .serialization import datetime_from_payload

        return ChannelIdentity(
            channel=str(payload.get("channel") or "wechaty"),
            external_user_id=str(payload.get("external_user_id") or ""),
            customer_id=str(payload.get("customer_id") or ""),
            public_name=str(payload.get("public_name") or ""),
            private_conversation_id=str(payload.get("private_conversation_id") or ""),
            can_private_message=bool(payload.get("can_private_message")),
            is_friend=bool(payload.get("is_friend")),
            created_at=datetime_from_payload(payload.get("created_at")),
            updated_at=datetime_from_payload(payload.get("updated_at")),
        )

    @staticmethod
    def _group_room_policy_from_payload(payload):
        from ...group_chat.models import GroupRoomPolicy
        from .serialization import datetime_from_payload

        return GroupRoomPolicy(
            room_id=str(payload.get("room_id") or ""),
            channel=str(payload.get("channel") or "wechaty"),
            managed=bool(payload.get("managed", True)),
            board_enabled=bool(payload.get("board_enabled", True)),
            outbound_enabled=bool(payload.get("outbound_enabled", True)),
            merge_window_seconds=int(payload.get("merge_window_seconds") or 30),
            updated_at=datetime_from_payload(payload.get("updated_at")),
        )

    @staticmethod
    def _game_conversation_link_from_payload(payload):
        from ...group_chat.models import GameConversationLink
        from .serialization import datetime_from_payload

        return GameConversationLink(
            link_id=str(payload.get("link_id") or ""),
            game_id=str(payload.get("game_id") or ""),
            conversation_id=str(payload.get("conversation_id") or ""),
            room_id=str(payload.get("room_id") or ""),
            customer_id=str(payload.get("customer_id") or "") or None,
            link_type=str(payload.get("link_type") or "origin"),
            created_at=datetime_from_payload(payload.get("created_at")),
        )

    @staticmethod
    def _board_snapshot_from_payload(payload):
        from ...group_chat.models import BoardSnapshot, BoardSnapshotItem
        from .serialization import datetime_from_payload

        return BoardSnapshot(
            snapshot_id=str(payload.get("snapshot_id") or ""),
            room_id=str(payload.get("room_id") or ""),
            conversation_id=str(payload.get("conversation_id") or ""),
            external_message_id=str(payload.get("external_message_id") or ""),
            rendered_text=str(payload.get("rendered_text") or ""),
            items=[
                BoardSnapshotItem(
                    item_no=int(item.get("item_no") or 0),
                    game_id=str(item.get("game_id") or ""),
                    rendered_text=str(item.get("rendered_text") or ""),
                )
                for item in payload.get("items") or []
                if isinstance(item, dict)
            ],
            created_at=datetime_from_payload(payload.get("created_at")),
        )

    @staticmethod
    def _group_board_state_from_payload(payload):
        from ...group_chat.models import BoardItem, BoardState
        from .serialization import datetime_from_payload

        return BoardState(
            room_id=str(payload.get("room_id") or ""),
            items=[
                BoardItem(
                    id=str(item.get("id") or ""),
                    display_no=int(item.get("display_no") or 0),
                    game_type=str(item.get("game_type") or ""),
                    table_id=str(item.get("table_id") or ""),
                    time=str(item.get("time") or "") or None,
                    smoking=str(item.get("smoking") or "") or None,
                    stakes=str(item.get("stakes") or ""),
                    special_rules=str(item.get("special_rules") or "") or None,
                    status=str(item.get("status") or "waiting"),
                    slots_total=int(item.get("slots_total") or 4),
                    slots_filled=int(item.get("slots_filled") or 0),
                    participants=[str(value) for value in item.get("participants") or []],
                )
                for item in payload.get("items") or []
                if isinstance(item, dict)
            ],
            source_message_id=str(payload.get("source_message_id") or ""),
            last_published_at=datetime_from_payload(payload.get("last_published_at")),
        )

    @staticmethod
    def _game_claim_from_payload(payload):
        from ...group_chat.models import GameClaim
        from .serialization import datetime_from_payload

        return GameClaim(
            claim_id=str(payload.get("claim_id") or ""),
            room_id=str(payload.get("room_id") or ""),
            game_id=str(payload.get("game_id") or ""),
            customer_id=str(payload.get("customer_id") or ""),
            source_conversation_id=str(payload.get("source_conversation_id") or ""),
            source_message_id=str(payload.get("source_message_id") or ""),
            status=str(payload.get("status") or "claimed"),
            created_at=datetime_from_payload(payload.get("created_at")),
        )

    @staticmethod
    def _channel_switch_from_payload(payload):
        from ...group_chat.models import ChannelSwitch
        from .serialization import datetime_from_payload

        return ChannelSwitch(
            switch_id=str(payload.get("switch_id") or ""),
            room_id=str(payload.get("room_id") or ""),
            customer_id=str(payload.get("customer_id") or ""),
            source_conversation_id=str(payload.get("source_conversation_id") or ""),
            source_message_id=str(payload.get("source_message_id") or ""),
            private_conversation_id=str(payload.get("private_conversation_id") or ""),
            trigger_summary=str(payload.get("trigger_summary") or ""),
            status=str(payload.get("status") or "active"),
            created_at=datetime_from_payload(payload.get("created_at")),
            expires_at=datetime_from_payload(payload.get("expires_at")),
        )


__all__ = ["SQLiteGroupChatStoreMixin"]
