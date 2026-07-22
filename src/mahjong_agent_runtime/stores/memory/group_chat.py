"""In-memory persistence for group routing, boards, and claims."""

from __future__ import annotations

from datetime import datetime

from ...domains import game_schedule_sort_key
from ...models import GameStatus, ScheduledAgentTask, ScheduledTaskStatus, StateTransition, new_id, now


class InMemoryGroupChatStoreMixin:
    __slots__ = ()

    @staticmethod
    def _channel_identity_key(channel: str, external_user_id: str) -> str:
        return f"{channel}\x1f{external_user_id}"

    def upsert_channel_identity(self, identity):
        with self._lock:
            key = self._channel_identity_key(identity.channel, identity.external_user_id)
            existing = self.channel_identities.get(key)
            if existing is not None:
                identity.created_at = existing.created_at
            identity.updated_at = now()
            self.channel_identities[key] = identity
            return identity

    def get_channel_identity(self, channel: str, external_user_id: str):
        with self._lock:
            return self.channel_identities.get(self._channel_identity_key(channel, external_user_id))

    def get_channel_identity_for_customer(self, customer_id: str, channel: str = "wechaty"):
        with self._lock:
            matches = [
                item
                for item in self.channel_identities.values()
                if item.customer_id == customer_id and item.channel == channel
            ]
            matches.sort(key=lambda item: item.updated_at, reverse=True)
            return matches[0] if matches else None

    def upsert_group_room_policy(self, policy):
        with self._lock:
            policy.updated_at = now()
            self.group_room_policies[policy.room_id] = policy
            return policy

    def get_group_room_policy(self, room_id: str):
        with self._lock:
            return self.group_room_policies.get(room_id)

    def upsert_group_board_state(self, board_state):
        """Replace the owner's latest live board for one room."""

        with self._lock:
            self.group_board_states[board_state.room_id] = board_state
            return board_state

    def get_group_board_state(self, room_id: str):
        with self._lock:
            return self.group_board_states.get(room_id)

    def link_game_conversation(self, link):
        with self._lock:
            duplicate = next(
                (
                    item
                    for item in self.game_conversation_link_records.values()
                    if item.game_id == link.game_id
                    and item.conversation_id == link.conversation_id
                    and item.customer_id == link.customer_id
                    and item.link_type == link.link_type
                ),
                None,
            )
            if duplicate is not None:
                return duplicate
            self.game_conversation_link_records[link.link_id] = link
            return link

    def game_conversation_links(self, game_id: str | None = None, room_id: str | None = None):
        with self._lock:
            records = list(self.game_conversation_link_records.values())
            if game_id:
                records = [item for item in records if item.game_id == game_id]
            if room_id:
                records = [item for item in records if item.room_id == room_id]
            return sorted(records, key=lambda item: item.created_at)

    def get_board_eligible_games(self, room_id: str):
        with self._lock:
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
        with self._lock:
            self.board_snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    def get_board_snapshot(self, snapshot_id: str):
        with self._lock:
            return self.board_snapshots.get(snapshot_id)

    def get_latest_board_snapshot(self, room_id: str):
        with self._lock:
            return next(
                (
                    item
                    for item in reversed(tuple(self.board_snapshots.values()))
                    if item.room_id == room_id
                ),
                None,
            )

    def get_board_snapshot_by_message_id(self, room_id: str, external_message_id: str):
        with self._lock:
            return next(
                (
                    item
                    for item in self.board_snapshots.values()
                    if item.room_id == room_id and item.external_message_id == external_message_id
                ),
                None,
            )

    def get_game_claim_by_source(self, source_conversation_id: str, source_message_id: str):
        with self._lock:
            return self.game_claims.get(f"{source_conversation_id}\x1f{source_message_id}")

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

        source_key = f"{source_conversation_id}\x1f{source_message_id}"
        with self._lock:
            existing = self.game_claims.get(source_key)
            if existing is not None:
                return existing, self.require_game(existing.game_id), [], True
            game, transitions = self.record_candidate_reply(
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
            self.game_claims[source_key] = claim
            return claim, game, transitions, False

    def record_channel_switch(self, switch):
        with self._lock:
            self.channel_switches[switch.switch_id] = switch
            return switch

    def get_recent_active_channel_switch(
        self,
        customer_id: str,
        *,
        room_id: str | None = None,
        at: datetime | None = None,
    ):
        stamp = at or now()
        with self._lock:
            matches = [
                item
                for item in self.channel_switches.values()
                if item.customer_id == customer_id
                and item.status == "active"
                and item.expires_at >= stamp
                and (not room_id or item.room_id == room_id)
            ]
            matches.sort(key=lambda item: item.created_at, reverse=True)
            return matches[0] if matches else None

    def ensure_group_board_publish_task(
        self,
        *,
        room_id: str,
        due_at: datetime,
        trace_id: str,
        urgent: bool = False,
    ):
        base_task_id = f"group_board_publish:{room_id}"
        with self._lock:
            existing = self.scheduled_tasks.get(base_task_id)
            if existing is not None and existing.status == ScheduledTaskStatus.PENDING:
                if urgent or due_at < existing.due_at:
                    existing.due_at = due_at
                    existing.updated_at = now()
                return existing, None
            task_id = base_task_id
            if existing is not None and existing.status == ScheduledTaskStatus.PROCESSING:
                task_id = new_id(base_task_id)
            old_status = existing.status.value if existing is not None else None
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
            task.status = ScheduledTaskStatus.PENDING
            task.due_at = due_at
            task.lease_until = None
            task.completed_at = None
            task.last_error = ""
            task.updated_at = now()
            self.scheduled_tasks[task.task_id] = task
            transition = StateTransition(
                "scheduled_agent_task",
                task.task_id,
                old_status,
                task.status.value,
                "group_board_publish_scheduled",
                trace_id,
            )
            self.transitions.append(transition)
            return task, transition


__all__ = ["InMemoryGroupChatStoreMixin"]
