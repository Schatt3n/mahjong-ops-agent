"""把真实报名回执和角色的到离店决定接到滚动物理牌局。

运行器在恢复后和整波办结后调用 ``sync``。控制事件走 ``finish_control``，
到离店角色仍正常决定，``role_result`` 只接受本人可见证据支持的物理动作。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta

from .venue_channel import AGENT_ID
from .venue_dialogues import _live_actor_action
from .venue_role_journal import RoleEvent, VenueRoleJournal
from .venue_rolling_board import _written_games
from .venue_rolling_world import PhysicalActorEvent, VenueRollingWorld, _committed_ids, _four_hour_candidate, _venue_time
from .venue_scenario import ACTUAL_START_DELAY, PLAY_DURATION


PLAY_ROLE_KINDS = frozenset({"arrival", "departure"})
PLAY_TICK_KIND = "physical_tick"
_ARRIVE_BEFORE_START = timedelta(minutes=1)
_WAKE_AFTER_RECEIPT = timedelta(minutes=1)
_TICK_PRECISION = timedelta(microseconds=1)
_ACTOR_ACTIONS = {"arrival": "arrive", "departure": "leave"}
_ACTOR_INTENTS = {
    "arrival": "现在是你决定是否按此前沟通去店里的时机。按已收到的来话和自己的出行情况选择签到、等待或联系店家。"
               "没有明确依据时不要假设这局已开打或知道他人的报名情况。",
    "departure": "你此前确实参加了这一局。现在是你根据自己的实际情况决定是否离店、等一下或联系店家的时机。",
}


class VenueRollingPlayLink:
    """只聚合已办结的同局角色证据，库存开打和散场仍归物理世界组件执行。"""

    def __init__(self, world: VenueRollingWorld, channel, journal: VenueRoleJournal, *,
                 active_actors: Iterable[str], allow_controlled: bool = False) -> None:
        """运行器传入同一沙盒的库存、渠道账本和角色事件账本。"""
        if channel.runtime.store is not world.store:
            raise ValueError("physical role link must share the channel's sandbox business store")
        self.world, self.channel, self.journal = world, channel, journal
        self.active_actors = frozenset(active_actors)
        self.allow_controlled = allow_controlled
        self._has_plan = any(event.kind == PLAY_TICK_KIND for event in journal.events())

    def sync(self, through: datetime) -> None:
        """在持有角色账本的调度权时，恢复或同步当前整波已经办结的证据。"""
        until = _venue_time(through)
        ready = {game_id: game for game_id, game in self.world.store.games.items()
                 if _four_hour_candidate(game) and _committed_ids(game) <= self.active_actors}
        if not ready and not self._has_plan:
            return
        events = self.journal.events()
        controls = {(event.details["game_id"], event.details["phase"]): event
                    for event in events if event.kind == PLAY_TICK_KIND}
        plans = {game_id: event for (game_id, phase), event in controls.items() if phase == "arrivals"}
        self._observe_ready_games(until, events, {game_id: game for game_id, game in ready.items() if game_id not in plans})
        if not plans:
            return
        actions = defaultdict(dict)
        for event in events:
            if event.kind in PLAY_ROLE_KINDS and event.origin_id == event.details["game_id"]:
                actions[event.details["game_id"], event.kind][event.actor_id] = event
        for game_id, plan in plans.items():
            if plan.status == "finished" and (game_id, "start") not in controls:
                self._schedule_play(plan, actions[game_id, "arrival"])
        self._advance_play(until, plans, actions, controls)

    def finish_control(self, event: RoleEvent) -> None:
        """控制事件只推进时钟或确认原计划的四个私有唤醒，不计入角色决定。"""
        if event.kind != PLAY_TICK_KIND:
            raise ValueError("expected a physical control event")
        phase = event.details["phase"]
        event = self.journal.save_decision(event, {"phase": phase})
        wakes = []
        if phase == "arrivals":
            at = datetime.fromisoformat(event.details["start_at"]) - _ARRIVE_BEFORE_START
            wakes = [wakeup.event_id for wakeup in self._wake_players(event, kind="arrival", at=at)]
        self.journal.finish(event, {"phase": phase, "scheduled_wakes": wakes})

    def role_result(self, event: RoleEvent) -> dict | None:
        """物理动作必须来源于按本人回执唤醒的合格决定；其他回复照常从渠道处理。"""
        if event.kind not in PLAY_ROLE_KINDS:
            return None
        action = _ACTOR_ACTIONS[event.kind]
        if self._role_source(event, action) is not None:
            return {"runtime_status": action, "physical_action_recorded": True}
        return None

    def _observe_ready_games(self, at: datetime, events: tuple[RoleEvent, ...], candidates: dict) -> None:
        """从同局四个具名承诺及四人的实际私聊回执形成不可变到场计划。"""
        if not candidates:
            return
        proofs = self._participation_evidence(events, at, candidates)
        for game_id, game in candidates.items():
            participants = proofs[game_id]
            if not _committed_ids(game) <= participants.keys():
                continue
            start = max(_venue_time(game.planned_start_at) + ACTUAL_START_DELAY,
                        at + _ARRIVE_BEFORE_START + _WAKE_AFTER_RECEIPT)
            end = start + PLAY_DURATION
            if (start.date() != _venue_time(game.planned_start_at).date()
                    or game.expires_at is not None and start >= game.expires_at
                    or start.date() != end.date()
                    or not self.world.opens <= start.time() < end.time() <= self.world.closes):
                continue
            self._control(game_id, "arrivals", at, start_at=start.isoformat(),
                          participants={person: participants[person] for person in sorted(_committed_ids(game))})
            self._has_plan = True

    def _participation_evidence(self, events: tuple[RoleEvent, ...], at: datetime, candidates: dict) -> dict:
        """用角色决定、正式业务写入和消息账本联合证明每个参与者确实收到私聊。"""
        chat = {row["message_id"]: row for row in self.channel.events()}
        proofs = defaultdict(dict)
        for event in events:
            row = event.result or {}
            source = chat.get(row.get("message_id"))
            if (event.status != "finished" or event.due_at > at or event.actor_id not in self.active_actors
                    or not self._decision_source(event) or not row.get("channel_trace_id")
                    or row.get("runtime_status") not in {"completed", "waiting_user"} or source is None):
                continue
            if not self._matches_inbound(event, source):
                continue
            game_ids = self._participated_games(row, source, candidates)
            if game_ids and (receipt := self._private_receipt(event.actor_id, row, source, chat, at)) is not None:
                for game_id, route in game_ids.items():
                    proofs[game_id].setdefault(event.actor_id, {
                        "role_event_id": event.event_id, "participation_message_id": source["message_id"],
                        "private_receipt_id": receipt["message_id"], "receipt_at": receipt["sent_at"], "route": route,
                    })
        return proofs

    def _matches_inbound(self, event: RoleEvent, source: dict) -> bool:
        """参与者的实际来话必须与原角色决定、发送时间和对应会话相符。"""
        action = (event.decision["decision"].get("action") or {})
        if (action.get("action") != "send" or not event.decision.get("visible_cause_verified")
                or source["direction"] != "inbound" or source["sender_id"] != event.actor_id
                or (source["kind"], source["recipient_id"]) != (action.get("kind"), action.get("recipient_id"))
                or source["text"] != action.get("text") or datetime.fromisoformat(source["sent_at"]) != event.due_at):
            return False
        if source["kind"] == "group":
            return (source["recipient_id"] == self.channel.room_id
                    and source["conversation_id"] == self.channel.group_conversation_id)
        return (source["kind"] == "private" and source["recipient_id"] == AGENT_ID
                and source["conversation_id"] == self.channel.private_conversation_id(event.actor_id))

    def _participated_games(self, row: dict, source: dict, candidates: dict) -> dict[str, str]:
        """私聊建桌核对组织者，加入核对正式工具结果，群认领核对持久事务。"""
        games = {game_id: "join" for game_id in _written_games(
            row, {"join_game", "record_candidate_reply"}, joined_customer=source["sender_id"],
        ) if game_id in candidates}
        if source["kind"] == "group":
            claim = self.world.store.get_game_claim_by_source(source["conversation_id"], source["message_id"])
            if (claim and claim.status == "claimed" and claim.room_id == self.channel.room_id
                    and claim.customer_id == source["sender_id"] and claim.game_id in candidates):
                games[claim.game_id] = "group_claim"
        else:
            for game_id in _written_games(row, {"create_game"}) & candidates.keys():
                game = candidates[game_id]
                if (game.organizer_id, game.conversation_id) == (source["sender_id"], source["conversation_id"]):
                    games[game_id] = "private_create"
        return games

    def _private_receipt(self, actor_id: str, row: dict, source: dict, chat: dict, until: datetime) -> dict | None:
        """只接受随这条成功来话实际交付、归该角色本人的店家私聊。"""
        for output in row.get("outputs", ()):
            receipt = chat.get(output["message_id"])
            if (receipt and (receipt["direction"], receipt["kind"], receipt["sender_id"], receipt["recipient_id"])
                    == ("outbound", "private", AGENT_ID, actor_id)
                    and receipt["conversation_id"] == self.channel.private_conversation_id(actor_id)
                    and receipt["source_message_id"] == source["message_id"] and receipt["text"]
                    and receipt["sequence"] > source["sequence"]
                    and datetime.fromisoformat(source["sent_at"]) <= datetime.fromisoformat(receipt["sent_at"]) <= until):
                return receipt
        return None

    def _schedule_play(self, plan: RoleEvent, arrivals: dict[str, RoleEvent]) -> None:
        """只在原计划每人结束且所有角色实际签到后，把相同证据交给物理状态。"""
        people = plan.details["participants"]
        if not people.keys() <= arrivals.keys() or any(arrivals[actor_id].status != "finished" for actor_id in people):
            return
        proofs = [self._physical_action(arrivals[actor_id], plan=plan) for actor_id in people]
        if any(proof is None for proof in proofs):
            return
        game_id = plan.details["game_id"]
        at = datetime.fromisoformat(plan.details["start_at"])
        result = self.world.schedule(game_id, start_at=at, arrivals=proofs)
        if result["status"] == "scheduled":
            self._control(game_id, "start", at)

    def _advance_play(self, until: datetime, plans: dict[str, RoleEvent], actions: dict, controls: dict) -> None:
        """同一四小时终点先等待四位角色决定各自办结，再推进正式散场。"""
        physical = self.world.journal.events()
        self._wake_departures(plans, physical, actions, controls)
        departures = []
        for game_id, plan in plans.items():
            recent = actions.get((game_id, "departure"), {})
            departures.extend(proof for event in recent.values() if (proof := self._physical_action(event, plan=plan)) is not None)
            end = datetime.fromisoformat(plan.details["start_at"]) + PLAY_DURATION
            if end <= until and any(recent.get(actor_id) is None or recent[actor_id].status != "finished"
                                    for actor_id in plan.details["participants"]):
                if any(item.actor_id == game_id and item.kind == "finish" and item.status != "finished" for item in physical):
                    until = end - _TICK_PRECISION
        if any(event.status != "finished" and event.due_at <= until for event in physical):
            self.world.tick(until, departures=departures)
            self._wake_departures(plans, self.world.journal.events(), actions, controls)

    def _wake_departures(self, plans: dict[str, RoleEvent], physical: tuple[RoleEvent, ...], actions: dict, controls: dict) -> None:
        """只有正式开局成功后安排同一组参与者的四小时离店决定。"""
        for item in physical:
            result = item.result or {}
            if item.kind != "start" or result.get("status") != "started":
                continue
            plan = plans[item.actor_id]
            end = datetime.fromisoformat(result["expected_play_end_at"])
            if (item.actor_id, "before_end") not in controls:
                controls[item.actor_id, "before_end"] = self._control(item.actor_id, "before_end", end - timedelta(minutes=1))
            if not plan.details["participants"].keys() <= actions[item.actor_id, "departure"].keys():
                for event in self._wake_players(plan, kind="departure", at=end):
                    actions[item.actor_id, "departure"][event.actor_id] = event

    def _wake_players(self, plan: RoleEvent, *, kind: str, at: datetime) -> list[RoleEvent]:
        """每位角色的唤醒仅指向自己的已送达消息，局号保留在调度账本。"""
        game_id = plan.details["game_id"]
        return [self.journal.enqueue(actor_id=actor, kind=kind, origin_id=game_id, due_at=at,
                    caused_by=proof["private_receipt_id"], details={"game_id": game_id, "case_intent": _ACTOR_INTENTS[kind]})
                for actor, proof in plan.details["participants"].items()]

    def _control(self, game_id: str, phase: str, at: datetime, **details) -> RoleEvent:
        """用同局同阶段的稳定键推进共享虚拟时钟，不归计给任何顾客。"""
        return self.journal.enqueue(actor_id=AGENT_ID, kind=PLAY_TICK_KIND, origin_id=f"{phase}:{game_id}", due_at=at,
                                    details={"game_id": game_id, "phase": phase, **details})

    def _physical_action(self, event: RoleEvent, *, plan: RoleEvent | None = None) -> PhysicalActorEvent | None:
        """只从已持久办结并通过实际签到／离店边界的角色动作建立物理凭据。"""
        if event.status != "finished" or not (event.result or {}).get("physical_action_recorded"):
            return None
        if plan is not None:
            participant = plan.details["participants"].get(event.actor_id, {})
            start = datetime.fromisoformat(plan.details["start_at"])
            expected = start - _ARRIVE_BEFORE_START if event.kind == "arrival" else start + PLAY_DURATION
            if (event.origin_id != plan.details["game_id"] or event.caused_by != participant.get("private_receipt_id")
                    or event.due_at != expected):
                return None
        action = _ACTOR_ACTIONS[event.kind]
        source = self._role_source(event, action)
        if source is None:
            return None
        return PhysicalActorEvent(event.event_id, event.details["game_id"], event.actor_id, action, event.due_at, source)

    def _role_source(self, event: RoleEvent, action: str) -> str | None:
        """拒绝其他角色回执、错误事件类型、冒认签到和无实际模型回复的离线动作。"""
        row = event.decision or {}
        decision = row.get("decision") or {}
        context = decision.get("context") or {}
        wake = (context.get("events") or [{}])[-1]
        delivery = context.get("delivery") or {}
        if (event.actor_id not in self.active_actors or not row.get("visible_cause_verified")
                or (decision.get("action") or {}).get("action") != action
                or (wake.get("event_id"), wake.get("kind"), wake.get("caused_by")) != (event.event_id, event.kind, event.caused_by)
                or (delivery.get("message_id"), delivery.get("kind"), delivery.get("sender_id"), delivery.get("recipient_id"))
                    != (event.caused_by, "private", AGENT_ID, event.actor_id) or event.caused_by is None):
            return None
        return self._decision_source(event)

    def _decision_source(self, event: RoleEvent) -> str | None:
        """真实运行只信实际模型响应，离线合同替身须显式开放。"""
        if _live_actor_action(event.decision or {}):
            return "model"
        decision = (event.decision or {}).get("decision") or {}
        if self.allow_controlled and decision.get("source") == "mock" and decision.get("status") == "valid":
            return "controlled"
        return None

    def report(self) -> dict:
        """区分没有看到同局有效来话、已安排但缺少实际到场和完成物理局。"""
        events = self.journal.events()
        plans = {event.details["game_id"]: event for event in events
                 if event.kind == PLAY_TICK_KIND and event.details["phase"] == "arrivals"}
        actions = [proof for event in events if event.kind in PLAY_ROLE_KINDS
                   and (plan := plans.get(event.details["game_id"])) is not None
                   and (proof := self._physical_action(event, plan=plan)) is not None]
        arrivals = sum(proof.action == "arrive" for proof in actions)
        result = self.world.report()
        if result["status"] == "not_observed":
            if not plans:
                result["reason"] = "no_schedulable_same_game_with_verified_private_receipts"
            else:
                result["reason"] = "missing_actual_arrivals" if arrivals < 4 * len(plans) else "no_start_after_verified_arrivals"
        return {**result, "arrival_plans": len(plans),
                "verified_arrival_decisions": arrivals,
                "verified_departure_decisions": sum(proof.action == "leave" for proof in actions),
                "arrival_sources": dict(Counter(proof.source for proof in actions if proof.action == "arrive")),
                "departure_sources": dict(Counter(proof.source for proof in actions if proof.action == "leave")),
                "participation_routes": dict(Counter(participant["route"] for event in plans.values()
                                                    for participant in event.details["participants"].values())),
                "participation_evidence": [event.details for event in plans.values()]}
