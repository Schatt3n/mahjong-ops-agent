"""实现内存存储层的群聊聊天数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta

from ...domains import game_schedule_sort_key
from ...models import GameStatus, ScheduledAgentTask, ScheduledTaskStatus, StateTransition, new_id, now
from ..board_arrangement import board_arrangement_projection, board_room_candidates


class InMemoryGroupChatStoreMixin:
    """为内存存储提供群会话、看板快照、身份映射与席位认领能力。"""

    __slots__ = ()

    @staticmethod
    def _channel_identity_key(channel: str, external_user_id: str) -> str:
        return f"{channel}\x1f{external_user_id}"

    def upsert_channel_identity(self, identity):
        """新增或更新通道身份映射，使重复写入收敛到同一条记录。"""
        with self._lock:
            key = self._channel_identity_key(identity.channel, identity.external_user_id)
            existing = self.channel_identities.get(key)
            if existing is not None:
                identity.created_at = existing.created_at
            identity.updated_at = now()
            self.channel_identities[key] = identity
            return identity

    def get_channel_identity(self, channel: str, external_user_id: str):
        """读取通道身份映射；返回值只反映当前存储或运行快照。"""
        with self._lock:
            return self.channel_identities.get(self._channel_identity_key(channel, external_user_id))

    def get_channel_identity_for_customer(self, customer_id: str, channel: str = "wechaty"):
        """读取通道身份映射for客户；返回值只反映当前存储或运行快照。"""
        with self._lock:
            matches = [
                item
                for item in self.channel_identities.values()
                if item.customer_id == customer_id and item.channel == channel
            ]
            matches.sort(key=lambda item: item.updated_at, reverse=True)
            return matches[0] if matches else None

    def upsert_group_room_policy(self, policy):
        """新增或更新群聊房间策略，使重复写入收敛到同一条记录。"""
        with self._lock:
            policy.updated_at = now()
            self.group_room_policies[policy.room_id] = policy
            return policy

    def get_group_room_policy(self, room_id: str):
        """读取群聊房间策略；返回值只反映当前存储或运行快照。"""
        with self._lock:
            return self.group_room_policies.get(room_id)

    def list_group_room_policies(self):
        """稳定读取全部群策略副本，调用方原地修改不会绕过Store的更新边界。"""
        with self._lock:
            return deepcopy(sorted(self.group_room_policies.values(), key=lambda item: item.room_id))

    def upsert_group_board_state(self, board_state):
        """新增或更新群聊看板状态，使重复写入收敛到同一条记录。

"""

        with self._lock:
            self.group_board_states[board_state.room_id] = board_state
            return board_state

    def get_group_board_state(self, room_id: str):
        """读取群聊看板状态；返回值只反映当前存储或运行快照。"""
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
                    and item.room_id == link.room_id
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
        """读取看板eligiblegames；返回值只反映当前存储或运行快照。"""
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

    def ensure_game_board_arrangement(self, game_id, *, trace_id, at=None, urgent=False, refresh=True):
        """在同一内存锁内原子登记群归属和刷新任务；失败恢复本次旁路写入。"""
        with self._lock:
            tasks = deepcopy(self.scheduled_tasks)
            links = dict(self.game_conversation_link_records)
            transition_count = len(self.transitions)
            try:
                return self._ensure_game_board_arrangement_locked(
                    game_id, trace_id=trace_id, at=at, urgent=urgent, refresh=refresh,
                )
            except Exception:
                self.scheduled_tasks.clear()
                self.scheduled_tasks.update(tasks)
                self.game_conversation_link_records.clear()
                self.game_conversation_link_records.update(links)
                del self.transitions[transition_count:]
                raise

    def _ensure_game_board_arrangement_locked(self, game_id, *, trace_id, at=None, urgent=False, refresh=True):
        """由建局或补偿入口持锁调用；群范围来自后端事实，好友关系不参与判断。"""
        from ...group_chat.models import GameConversationLink

        stamp = at or now()
        game = self.require_game(game_id)
        links = self.game_conversation_links(game_id=game_id)
        switch = self.get_recent_active_channel_switch(game.organizer_id, at=stamp)
        policies = board_room_candidates(game, self.group_room_policies.values(), links, switch)
        for policy in policies:
            newly_linked = not any(item.room_id == policy.room_id for item in links)
            if newly_linked:
                native = game.conversation_id.startswith(("group:", "wechaty:room:"))
                link_type = "origin" if native else (
                    "private_switch_created" if switch is not None and switch.room_id == policy.room_id
                    and switch.private_conversation_id == game.conversation_id else "private_board_created"
                )
                self.link_game_conversation(GameConversationLink(
                    link_id=new_id("game_conversation_link"), game_id=game_id,
                    conversation_id=game.conversation_id, room_id=policy.room_id,
                    customer_id=game.organizer_id, link_type=link_type,
                ))
            snapshot = self.get_latest_board_snapshot(policy.room_id)
            already_published = snapshot is not None and any(item.game_id == game_id for item in snapshot.items)
            active = any(item.task_type == "publish_group_board" and item.aggregate_id == policy.room_id
                         and item.status in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}
                         for item in self.scheduled_tasks.values())
            # 新归属本身改变了群的局池；正在处理的旧板可能已取完快照，不能算覆盖。
            # pending可合并，processing则须保留后继，即使本次只是幂等建局的补缺。
            if refresh or newly_linked or not already_published and not active:
                self.ensure_group_board_publish_task(
                    room_id=policy.room_id, due_at=stamp if urgent else stamp + timedelta(
                        seconds=max(0, int(policy.merge_window_seconds))), trace_id=trace_id, urgent=urgent,
                )
        return self.game_board_arrangement(game_id)

    def game_board_arrangement(self, game_id):
        """读取最小安全状态；安排、已发布与外发授权分别表达。"""
        with self._lock:
            self.require_game(game_id)
            links = self.game_conversation_links(game_id=game_id)
            return board_arrangement_projection(
                game_id, self.group_room_policies.values(), links, self.scheduled_tasks.values(),
                [self.get_latest_board_snapshot(room_id) for room_id in {item.room_id for item in links}],
            )

    def save_board_snapshot(self, snapshot):
        """保存看板快照，确保后续读取能够得到一致快照。"""
        with self._lock:
            self.board_snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    def get_board_snapshot(self, snapshot_id: str):
        """读取看板快照；返回值只反映当前存储或运行快照。"""
        with self._lock:
            return self.board_snapshots.get(snapshot_id)

    def get_latest_board_snapshot(self, room_id: str):
        """读取latest看板快照；返回值只反映当前存储或运行快照。"""
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
        """读取看板快照by消息id；返回值只反映当前存储或运行快照。"""
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
        """读取牌局claimbysource；返回值只反映当前存储或运行快照。"""
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
        """记录通道通道切换，形成后续决策和审计所需的持久证据。"""
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
        """读取recentactive通道通道切换；返回值只反映当前存储或运行快照。"""
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

    def ensure_group_board_publish_task(self, *, room_id, due_at, trace_id, urgent=False):
        """同锁合并待发责任；处理中的快照最多保留一个后继，重试沿用发布身份。"""
        with self._lock:
            base_task_id = f"group_board_publish:{room_id}"
            room_tasks = [item for item in self.scheduled_tasks.values()
                          if item.task_type == "publish_group_board" and item.aggregate_type == "group_room"
                          and item.aggregate_id == room_id]
            pending = next((item for item in sorted(room_tasks, key=lambda task: (task.created_at, task.task_id))
                            if item.status == ScheduledTaskStatus.PENDING), None)
            if pending is not None:
                if due_at < pending.due_at:
                    pending.due_at = due_at
                    pending.updated_at = now()
                    self.scheduled_tasks[pending.task_id] = pending
                return pending, None
            # 已完成、失败和处理中的旧代不复活，旧worker无法结束或延期新的发布责任。
            task_id = new_id(base_task_id) if room_tasks else base_task_id
            task = ScheduledAgentTask(
                task_id=task_id, task_type="publish_group_board", aggregate_type="group_room",
                aggregate_id=room_id, conversation_id=f"wechaty:room:{room_id}", subject_id="system",
                subject_name="system", due_at=due_at, idempotency_key=task_id,
                payload={"event_type": "publish_group_board", "room_id": room_id,
                         "publication_id": new_id("board_publication")},
            )
            self.scheduled_tasks[task.task_id] = task
            transition = StateTransition("scheduled_agent_task", task.task_id, None,
                                         task.status.value, "group_board_publish_scheduled", trace_id)
            self.transitions.append(transition)
            return task, transition


__all__ = ["InMemoryGroupChatStoreMixin"]
