"""让私聊组局、群看板、好友认领和回执沿用同一个真实牌局。"""

from __future__ import annotations

from datetime import datetime, timedelta
import json

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE
from mahjong_agent_runtime.group_chat.projections import public_group_game_summary
from mahjong_agent_runtime.group_chat.semantic_topic_router import SemanticTopicContractError
from mahjong_agent_runtime.knowledge import default_terminology_repository
from mahjong_agent_runtime.models import ScheduledTaskStatus, using_clock

from .venue_actors import ActorAction, ActorContext, MockActor, ModelActor
from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_dialogues import _live_actor_action
from .venue_friend_scenario import VenueFriendScenario


_BOARD_RETRY_TICK_LIMIT = 8


class VenueFriendGroupLinkScenario(VenueFriendScenario):
    """一位老板好友私聊新建牌局，另一位从群看板接续；用正式调度更新公开席位。"""

    def __init__(self, *, people, **options) -> None:
        self._visitors = tuple(people)
        super().__init__(people=self._visitors, **options)
        if len(self.sessions) != 1:
            raise ValueError("friend-group-link requires exactly one private organizer")

    def run(self) -> dict:
        """沿用空局池的私聊验收，再验证双向交付及公开投影。"""
        report = super().run()
        checks = report["checks"]
        checks.update(dict.fromkeys((
            "private_game_on_public_board", "board_recruited_another_friend",
            "board_action_delivered_private_receipt", "same_game_republished_with_updated_seats",
            "private_text_stayed_in_its_own_inbox", "board_visit_created_no_duplicate_game",
        ), False))
        report["friend_group_link"] = {
            "public_board_versions": 0, "board_visitors": 0, "private_receipts": 0,
            "claimed_seats_before": None, "claimed_seats_after": None, "entry_channel": None,
            "join_channel": None,
        }
        try:
            if checks["each_friend_created_own_new_game"] and checks["created_games_have_distinct_private_origins"]:
                self._link_game(report)
        finally:
            records = self.channel.events()
            with self.run_dir.joinpath("chat.jsonl").open("w", encoding="utf-8") as file:
                for row in records:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            report["chat_events"] = len(records)
        return report

    def _link_game(self, report: dict) -> None:
        checks, link = report["checks"], report["friend_group_link"]
        game, = self.store.games.values()
        anchor = max(datetime.fromisoformat(row["sent_at"]) for row in self.channel.events()).astimezone(self.start_at.tzinfo)
        first = self._board(game.game_id, after=anchor, phase="private-to-group")
        if first is None:
            return
        board, item, board_at = first
        baseline = public_group_game_summary(game)
        checks["private_game_on_public_board"] = baseline["seat_code"] in item["rendered_text"].split()
        link["public_board_versions"] = 1
        link["claimed_seats_before"] = baseline["claimed_seats"]

        group_cause = board["external_message_id"]
        guide_at = board_at + timedelta(seconds=1)
        self.channel.publish_owner(kind="group", recipient_id=self.channel.room_id, sent_at=guide_at,
            message_id="friend-link-board-guide", source_message_id=group_cause,
            text=f"看中哪桌就直接在群里引用这条看板报名，发桌号加「来」，比如「{item['item_no']}来」。"
                 "想问细节可以私聊，记得说群里哪条看板、几点和什么玩法；私聊只发桌号我不一定能对上。")
        visitors = self._compatible_friends(game)
        details = ("你今天只想看看群里这张现成桌能不能加，暂时不想请老板另外找人组一桌。"
                   "你实际看到了群里的看板和报名说明。想参加可以直接按说明在群里引用看板；"
                   "有疑问也可以私聊问，告诉老板是哪张看板和桌况，光说桌号对方可能对不上。"
                   "是否参加、是否先问细节、用哪个渠道由你自己决定。")
        for visitor in visitors[:2]:
            if not any(row["message_id"] == group_cause for row in self.channel.events_for(visitor.sender_id)):
                continue
            link["board_visitors"] += 1
            action_at = guide_at + timedelta(seconds=link["board_visitors"])
            first_touch = self._visit(visitor, group_cause, action_at, details, report,
                board_message_id=group_cause, item_no=item["item_no"])
            if first_touch is None:
                continue
            row, private = first_touch
            source_kind = row["channel"]
            if not self._joined(game.game_id, visitor.sender_id) and private:
                action_at += timedelta(seconds=1)
                follow = self._visit(visitor, private[-1]["message_id"], action_at,
                    "老板刚发来私聊。你只对原群看板上的这桌感兴趣。愿意参加可以在私聊把原桌说清楚，"
                    "也可以回到群里引用原看板按报名说明占位；有未确认的条件可继续私聊，或不参加。"
                    "回到群里时不要转述你和老板的私聊。",
                    report, board_message_id=group_cause, item_no=item["item_no"])
                if follow is not None:
                    row, private = follow
            if not self._joined(game.game_id, visitor.sender_id):
                continue
            checks["board_recruited_another_friend"] = self._seat_evidence(row, game.game_id, visitor.sender_id)
            checks["board_action_delivered_private_receipt"] = bool(row["successful_private_receipt_ids"])
            link["entry_channel"] = source_kind
            link["join_channel"] = row["channel"]
            link["private_receipts"] = len(row["successful_private_receipt_ids"])
            persisted = SQLiteAgentStore(self.store.path).require_game(game.game_id)
            after = public_group_game_summary(persisted)
            link["claimed_seats_after"] = after["claimed_seats"]
            refresh = self._board(game.game_id, after=action_at, phase="private-receipt-to-group-refresh")
            if refresh is not None:
                refreshed, refreshed_item, _ = refresh
                newer = refreshed["external_message_id"] != board["external_message_id"]
                link["public_board_versions"] += int(newer)
                checks["same_game_republished_with_updated_seats"] = (
                    newer and after["claimed_seats"] == baseline["claimed_seats"] + 1
                    and after["seat_code"] in refreshed_item["rendered_text"].split()
                )
            checks["private_text_stayed_in_its_own_inbox"] = self._privacy(game.organizer_id, visitor.sender_id)
            break
        checks["board_visit_created_no_duplicate_game"] = set(self.store.games) == {game.game_id}

    def _board(self, game_id: str, *, after: datetime, phase: str) -> tuple[dict, dict, datetime] | None:
        """沿同一发布代次的持久重试时间前进，直到可见快照与正式投递能对上。"""
        task = min(
            (item for item in self.store.scheduled_tasks.values()
             if item.task_type == BOARD_TASK_TYPE and item.aggregate_type == "group_room"
             and item.aggregate_id == self.channel.room_id and item.status == ScheduledTaskStatus.PENDING),
            key=lambda item: (item.due_at, item.task_id), default=None,
        )
        if task is None:
            return None
        publication_id = task.payload["publication_id"]
        at = max(task.due_at, after + timedelta(seconds=1))
        delivered = set()
        # 正式调度器保管失败次数；沙盒上限只防止异常状态让虚拟时钟无限推进。
        for tick in range(1, _BOARD_RETRY_TICK_LIMIT + 1):
            result = self.channel.publish_board(trace_id=f"friend-link:{phase}", sent_at=at)
            current = self.store.scheduled_tasks.get(task.task_id)
            self._record({
                "event_id": phase, "virtual_at": at.isoformat(), "publication_attempt": tick, "publication": result,
                "task_id": task.task_id, "publication_id": publication_id,
                "persisted_attempts": current.attempts if current else None,
                "persisted_status": current.status.value if current else None,
                "persisted_due_at": current.due_at.isoformat() if current else None,
                "persisted_last_error": current.last_error if current else None,
            })
            receipt = next((item for item in result["tasks"] if item["task_id"] == task.task_id), None)
            if receipt is None or current is None or current.payload.get("publication_id") != publication_id:
                return None
            # 前次已交付但快照落库失败时，重试可以证实旧回执，不能重新发送或提前唤醒顾客。
            delivered.update(row["message_id"] for row in result["archive_outputs"]
                             if row["kind"] == "group" and row["recipient_id"] == self.channel.room_id
                             and row["trace_id"] == receipt["trace_id"])
            snapshot = receipt.get("snapshot") or {}
            if (current.status == ScheduledTaskStatus.COMPLETED and receipt["status"] == "completed"
                    and snapshot.get("external_message_id") in delivered):
                item = next((entry for entry in snapshot["items"] if entry["game_id"] == game_id), None)
                return (snapshot, item, at) if item is not None else None
            if (current.status != ScheduledTaskStatus.PENDING or current.due_at <= at
                    or current.attempts <= task.attempts):
                return None
            task, at = current, current.due_at
        return None

    def _compatible_friends(self, game) -> list:
        """用分档与角色空闲期挑选唤醒对象；真实报名与发言仍由该角色决定。"""
        public = public_group_game_summary(game)["public_requirement"]
        target = public.get("game_type") or public.get("requested_game")
        stake = public.get("base_stake")
        if not isinstance(stake, (int, float)):
            stake = None
        terminology = default_terminology_repository()
        friends = self.channel.owner_friends
        period = "晚上" if (game.planned_start_at or self.start_at).hour >= 19 else "傍晚"
        eligible = []
        for person in self._visitors:
            if person.sender_id == game.organizer_id or person.sender_id not in friends:
                continue
            term = terminology.first_match(person.preferred_game, categories={"game_type"})
            if term is None or term.term.canonical.get("requested_game") != target:
                continue
            if stake is not None and not person.stake_min <= stake <= person.stake_max:
                continue
            if public.get("smoke_preference") == "smoking" and person.smoking == "non_smoking":
                continue
            # 避开角色明确不愿同桌的已知同伴；关系细节仍不传给正式 Agent。
            if any(edge.kind == "avoid_same_table" and edge.target_sender_id == game.organizer_id
                   for edge in person.relationships):
                continue
            eligible.append(person)
        return sorted(eligible, key=lambda person: (
            period not in person.available_periods, person.temperament.commitment != "ready_to_join",
            person.temperament.purpose != "find_table",
            public.get("smoke_preference") in (None, "any") and person.smoking == "non_smoking",
        ))

    def _visit(self, person, cause: str, at: datetime, intent: str, report: dict, *,
               board_message_id: str, item_no: int) -> tuple[dict, list[dict]] | None:
        """区分唤醒顾客的消息和报名时实际引用的公开快照，避免把私聊引用进群。"""
        history = self.channel.events_for(person.sender_id)[-24:]
        seen = {row["message_id"]: row for row in history}
        board = seen.get(board_message_id)
        if (cause not in seen or board is None or board["kind"] != "group"
                or board["recipient_id"] != self.channel.room_id or board["sender_id"] != AGENT_ID):
            return None
        first_touch = cause == board_message_id
        event_id = f"friend-board:{person.sender_id}:{'entry' if first_touch else 'followup'}"
        context = ActorContext(person, at.isoformat(), history=history,
            events=[{"event_id": event_id, "kind": "opportunity" if first_touch else "delivered",
                     "caused_by": cause, "at": at.isoformat(), "details": {
                         "case_goal": "try_join", "case_intent": intent, "board_no": item_no,
                     }}],
            private_targets=(AGENT_ID,), group_targets=(self.channel.room_id,),
            visible_groups=(self.channel.room_id,))
        row = {"event_id": event_id, "virtual_at": at.isoformat(), "actor_id": person.sender_id, "caused_by": cause}
        with using_clock(lambda: at):
            if isinstance(self.actor, MockActor) and first_touch:
                # 离线合约桩只衡量原生编号协议；真模式原样使用顾客模型的发言。
                action = ActorAction("send", f"{item_no}号我来", "group", self.channel.room_id)
            else:
                decision = self.actor.decide(context, trace_id=f"role:{event_id}")
                row["decision"] = decision.to_dict()
                action = decision.action
                if isinstance(self.actor, ModelActor):
                    live = _live_actor_action(row)
                    report["real_customer_actions"] += int(live)
                    report["checks"]["customer_decisions_from_model"] &= live
                if action is None:
                    report["failed_events"].append(event_id)
            if action is None or action.action != "send":
                self._record(row)
                return None
            kind = action.kind or "private"
            target = action.recipient_id or AGENT_ID
            message_id = f"customer:{event_id}"
            if self.fixture_agent is not None and kind == "private":
                self.fixture_agent.turns[message_id] = ("delivered", {})
            try:
                response = self.channel.send(MessageEnvelope(
                    message_id, person.sender_id, action.text, at, kind, target,
                    quoted_message_id=board_message_id if kind == "group" else None,
                ))
            except SemanticTopicContractError as exc:
                report["failed_events"].append(event_id)
                self._record({**row, "channel": kind, "message_id": message_id, "error_type": str(exc),
                              "routing_audit": exc.audit})
                return None
        conversation_id = self.channel.private_conversation_id(person.sender_id) if response.get("archive_outputs") else None
        private = [item for item in response.get("archive_outputs", ()) if item["kind"] == "private"
                   and item["sender_id"] == AGENT_ID and item["recipient_id"] == person.sender_id
                   and item["conversation_id"] == conversation_id
                   and item["source_message_id"] == message_id]
        runtime_results = (response,) if kind == "private" else response.get("group_processing", {}).get("runtime_results", ())
        # 正式占位与回执是独立结果。技术失败回复是可见聊天，但不构成成功回执。
        successful = {(item.get("trace_id"), item.get("final_reply")) for item in runtime_results
                      if item.get("conversation_id") == conversation_id and item.get("status") in {"completed", "waiting_user"}}
        row.update(message_id=message_id, channel=kind, runtime_status=response.get("status"),
                   trace_id=response.get("trace_id"), tool_results=response.get("tool_results", []),
                   routing=response.get("group_processing", {}).get("routing"), private_receipts=private,
                   successful_private_receipt_ids=[item["message_id"] for item in private
                                                   if (item["trace_id"], item["text"]) in successful])
        self._record(row)
        return row, private

    def _joined(self, game_id: str, customer_id: str) -> bool:
        return any(participant.customer_id == customer_id and participant.status in {"joined", "confirmed"}
                   for participant in self.store.require_game(game_id).participants)

    def _seat_evidence(self, row: dict, game_id: str, customer_id: str) -> bool:
        """编号协议核认领记录，自然群消息转私聊时核对实际工具和本人的原消息。"""
        if row["channel"] == "group":
            claim = self.store.get_game_claim_by_source(self.channel.group_conversation_id, row["message_id"])
            if claim is not None:
                return claim.status == "claimed" and claim.game_id == game_id and claim.customer_id == customer_id
            routing = row.get("routing") or {}
            topic_routing = (routing.get("detail") or {}).get("topic_routing") or {}
            source = topic_routing.get("message") or {}
            classification = source.get("classification") or {}
            if (routing.get("action") != "private_switch" or source.get("customer_id") != customer_id
                    or source.get("message_id") != row["message_id"] or classification.get("intent") != "claim"
                    or not classification.get("board_reference")):
                return False
        return any(tool.get("name") in {"join_game", "record_candidate_reply"}
                   and tool.get("called") and tool.get("allowed") and not tool.get("error")
                   and (tool.get("result", {}).get("game") or {}).get("game_id") == game_id
                   and tool.get("result", {}).get("recorded_status") in {"joined", "confirmed", "accepted"}
                   for tool in row["tool_results"])

    def _privacy(self, organizer_id: str, visitor_id: str) -> bool:
        """第三位好友仅看到公共事件；任何群出站都不引用私聊入站。"""
        records = self.channel.events()
        private = [row for row in records if row["kind"] == "private"]
        private_people = {member for row in private for member in (row["sender_id"], row["recipient_id"])}
        private_people.update((organizer_id, visitor_id))
        observer = next((person for person in self._visitors if person.sender_id not in private_people), None)
        if observer is None:
            return False
        private_ids = {row["message_id"] for row in private}
        group = [row for row in records if row["kind"] == "group"]
        public_projection = self.channel.events_for(observer.sender_id)
        return (public_projection == group and all(row["source_message_id"] not in private_ids for row in group)
                and all(self.channel.owner_friends[organizer_id] not in row["text"]
                        for row in group if row["direction"] == "outbound"))

    def _record(self, row: dict) -> None:
        """保留角色决策、发送回执和看板快照的本地完整依据。"""
        with self.run_dir.joinpath("events.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
