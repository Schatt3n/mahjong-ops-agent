"""推进一座沙盒麻将馆的事件，保存模拟角色与真实业务的独立证据。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from heapq import heappop, heappush
import json
from pathlib import Path
import random
from typing import Any

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE
from mahjong_agent_runtime.models import GameStatus, using_clock

from .venue_actors import ActorContext, ModelActor, Persona
from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_dialogues import _live_actor_action


@dataclass(frozen=True, slots=True, order=True)
class VenueEvent:
    """同一时刻按入队顺序处理，可追溯到稳定的调度事件。"""

    at: datetime
    sequence: int
    kind: str = field(compare=False)
    actor_id: str | None = field(default=None, compare=False)
    caused_by: str | None = field(default=None, compare=False)

    @property
    def event_id(self) -> str:
        return f"venue-event-{self.sequence:04d}"


VENUE = {
    "name": "晚风棋牌室（模拟）",
    "opens": "12:00",
    "closes": "23:50",
    "rooms": [
        {"id": "东一包间", "seats": 4, "smoking": False, "hourly_price": 32, "equipment": ["自动麻将桌", "排风"]},
        {"id": "东二包间", "seats": 4, "smoking": False, "hourly_price": 36, "equipment": ["自动麻将桌", "空气净化器"]},
        {"id": "西侧包间", "seats": 4, "smoking": True, "hourly_price": 28, "equipment": ["自动麻将桌", "排风"]},
    ],
    "initial_blocks": [{"room_id": "东一包间", "starts": "18:30", "ends": "20:00", "reason": "设备检修"}],
}
PLAY_DURATION = timedelta(hours=4)
ACTUAL_START_DELAY = timedelta(minutes=3)
PROBE_DURATION = timedelta(minutes=30)


def select_roles(people: tuple[Persona, ...]) -> tuple[Persona, Persona, Persona, Persona, Persona, Persona]:
    """从种子产生的真实有向关系中选搭子、避桌者、补位者和新一局询问者。"""
    by_id = {person.persona_id: person for person in people}
    compatible = [person for person in people if person.preferred_game == "杭州麻将"
                  and person.stake_min <= 1 <= person.stake_max
                  and "晚上" in person.available_periods]
    compatible_ids = {person.persona_id for person in compatible}
    avoiders = {edge.target_id: person for person in compatible for edge in person.relationships
                if edge.kind == "avoid_same_table"}
    def role_fit(person):
        return (abs(person.temperament.preferred_duration_hours - 4),
                person.temperament.commitment == "ask_only")

    pairs = [(organizer, by_id[edge.target_id], avoiders[edge.target_id])
             for organizer in compatible for edge in organizer.relationships
             if edge.kind == "regular_partner" and edge.target_id in compatible_ids and edge.target_id in avoiders]
    pairs.sort(key=lambda pair: (role_fit(pair[0])[0] + role_fit(pair[1])[0],
                                 role_fit(pair[1]), role_fit(pair[0])))
    for organizer, partner, avoider in pairs:
        taken = {organizer.persona_id, partner.persona_id, avoider.persona_id}
        if len(taken) != 3:
            continue
        extras = [person for person in compatible if person.persona_id not in taken
                  and not any(e.kind == "avoid_same_table" and e.target_id in taken for e in person.relationships)]
        if len(extras) >= 3:
            first, second, *others = sorted(extras, key=role_fit)
            probe = min(others, key=lambda person: person.temperament.purpose != "ask_venue")
            return organizer, partner, avoider, first, second, probe
    raise ValueError("该种子无法生成有搭子、避桌和补位的兼容角色，请检查角色人口")


class VenueScenario:
    """被测 Agent 只收到普通消息，门店受控事件通过正式 Store 执行。"""

    def __init__(self, *, people, actor, judge, channel, store, run_dir: Path, start_at: datetime,
                 seed: int, background_events: int, fixture_agent=None) -> None:
        self.actor = actor
        self.judge = judge
        self.channel = channel
        self.store = store
        self.run_dir = run_dir
        self.people = {person.sender_id: person for person in people}
        self.organizer, self.partner, self.avoider, self.extra_a, self.extra_b, self.probe = select_roles(people)
        self.players = (self.organizer, self.partner, self.extra_a, self.extra_b)
        preference_people = [person for person in people if person.smoking == "non_smoking"
                             and person not in (*self.players, self.avoider, self.probe)]
        self.memory_person = min(preference_people, key=lambda person: person.temperament.purpose != "socialize")
        self.start_at = start_at
        self.expected_play_end_at: datetime | None = None
        self.clock = start_at - timedelta(minutes=70)
        self.terms = {"game_type": "杭州麻将", "stake": 1, "smoke": "non_smoking",
                      "duration_hours": 4, "start_time": start_at.strftime("%H:%M")}
        self.requirement = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                            "duration_kind": "fixed", "duration_hours": 4, "start_time_kind": "scheduled",
                            "start_time": self.terms["start_time"], "planned_start_at": start_at.isoformat()}
        self.fixture_agent = fixture_agent
        self.events: list[VenueEvent] = []
        self.sequence = self.opening_turns = 0
        self.game_id: str | None = None
        self.booking_id: str | None = None
        self.room_id: str | None = None
        self.social_message_id: str | None = None
        self.announcement_message_id: str | None = None
        self.observed_refusal = False
        self.opportunity_turns: dict[str, int] = {}
        self.confirmation_messages: dict[str, str] = {}
        self.real_customer_actions = 0
        self.controlled_actions_from_model = True
        self.arrivals: set[str] = set()
        self.departures: set[str] = set()
        self.checks: dict[str, bool | None] = dict.fromkeys((
            "create_via_agent", "board_delivered", "social_refusal", "peer_delivery_isolation", "memory_from_visible_message",
            "four_real_confirmations", "four_arrivals", "actual_start",
            "occupied_before_end", "overlap_denied", "agent_pre_end_read", "agent_post_end_read",
            "four_departures", "actual_four_hour_finish", "release_persisted", "same_room_rebookable",
        ))
        self.evidence: list[dict[str, Any]] = []
        self.failures: list[dict[str, str]] = []
        self.model_transport_errors: list[dict[str, Any]] = []
        self.timeline = run_dir / "events.jsonl"

        self._enqueue("need", start_at - timedelta(minutes=65), self.organizer)
        self._enqueue("peer_invite", start_at - timedelta(minutes=66), self.organizer)
        self._enqueue("preference", start_at - timedelta(minutes=68), self.memory_person)
        actual_start = start_at + ACTUAL_START_DELAY
        self._enqueue("start", actual_start)
        focus = {person.sender_id for person in (*self.players, self.avoider, self.probe, self.memory_person)}
        background = [person for person in people if person.sender_id not in focus]
        weights = [{"quiet": 1, "occasional": 3, "frequent": 7}[person.activity] for person in background]
        rng = random.Random(seed)
        # 活跃度只调整被唤醒概率；角色模型仍有等待及不发言的选择。
        spread = (start_at + timedelta(hours=4) - self.clock) / max(1, background_events)
        for index in range(background_events):
            person = rng.choices(background, weights=weights)[0]
            self._enqueue("activity", self.clock + spread * index, person)

    def _enqueue(self, kind: str, at: datetime, person: Persona | None = None, *, caused_by: str | None = None) -> None:
        self.sequence += 1
        heappush(self.events, VenueEvent(at, self.sequence, kind, person.sender_id if person else None, caused_by))

    def _record(self, event: VenueEvent, **payload: Any) -> dict[str, Any]:
        row = {"event_id": event.event_id, "virtual_at": event.at.isoformat(), "kind": event.kind,
               "caused_by": event.caused_by, **payload}
        with self.timeline.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def run(self) -> dict[str, Any]:
        """每个事件使用同一仿真时间；失败也留下可独立复盘的事件。"""
        with using_clock(lambda: self.clock):
            while self.events:
                event = heappop(self.events)
                self.clock = event.at
                try:
                    if event.actor_id:
                        self._act(event, self.people[event.actor_id])
                    else:
                        self._world(event)
                except (KeyError, ValueError, RuntimeError) as exc:
                    failure = {"event_id": event.event_id, "kind": event.kind, "reason": str(exc)}
                    self.failures.append(failure)
                    self._record(event, failure=failure)
        return self._report()

    def _actor_context(self, event: VenueEvent, person: Persona, visible: list[dict]) -> ActorContext:
        """给每次来话绑定实际受众和已见来源；这桌的时长计划只属于这一晚。"""
        kind = event.kind
        details = dict(self.terms) if kind in {"need", "peer_invite", "opportunity", "delivered", "share_joined"} else {}
        private_targets = (AGENT_ID,)
        groups = (self.channel.room_id,)
        group_targets = groups if kind in {"opportunity", "activity", "share_joined"} else ()
        if kind == "peer_invite":
            private_targets = (self.partner.sender_id,)
            details.update(peer_id=self.partner.sender_id, case_intent=(
                "你打算请店家组织今晚这桌，自己已留出完整四小时，想先问常搭档愿不愿一起。"
                "店家还没登记，不替他报名。"
            ))
        elif kind == "need":
            visible = [row for row in visible if row["kind"] == "private"
                       and AGENT_ID in (row["sender_id"], row["recipient_id"])]
            details.update(case_goal="organize_new", party_size=1, case_intent=(
                "今晚特意空出四小时和常搭档玩，开场和打满四小时都已列入你的个人安排。"
                "你这边只有自己确定参加，主动请店家新组这场并把自己算一位。搭档会自己报名。"
                "店家还未登记时可以先自然说来意，按实际追问补条件。这次只组局找人。"
            ))
        elif kind == "preference":
            details.update(case_intent=(
                "你一直不打有烟的桌。今天另有安排，晚上不会去店里。这一条只是顺便告诉店家"
                "以后推荐给你的局都要无烟，不需要对方查今晚的牌局。"
            ))
        elif kind in {"opportunity", "delivered"}:
            if person == self.avoider:
                details.update(case_intent=(
                    "原本在看这次群看板。刚收到熟悉群友的公开发言，看看是不是自己平常避开同桌的人。"
                    "如果这桌不去，你想回店家一声让他们继续找人，保留私人关系。"
                ))
                group_targets = ()
                if self.fixture_agent and event.caused_by == self.social_message_id:
                    details["known_attendees"] = [{"persona_id": self.partner.persona_id, "name": self.partner.name}]
            else:
                details.update(case_goal="try_join", party_size=1, case_intent=(
                    "今晚这一段你安排了完整四小时，想自己加入一桌。你看到群里有个合适的机会，"
                    "自己决定按公开看板报名还是私聊店家确认，已经看见的条件不必重问。"
                ))
        elif kind == "share_joined":
            private_targets = ()
            details.update(case_intent=(
                "刚拿到本人报名确认。你愿意在群里告诉同场熟人你会参加哪桌，只提自己的公开到场计划。"
            ))
        elif kind in {"arrival", "departure"}:
            private_targets = ()
            details["case_intent"] = (
                "你按已确认的约定来到店里，当前就在门口准备签到。" if kind == "arrival"
                else "你自己已经打完今天约好的四小时，现在收拾好准备离店。"
            )
        elif kind == "slot_query":
            details.update(start_time=self.clock.strftime("%H:%M"), duration_hours=0.5, party_size=4,
                           case_intent=(
                               "你和另外三个人在附近，有半小时空，想问从当前所给钟点起有没有空包间，先看看不订。"
                               if self.clock < self.expected_play_end_at else
                               "刚才还在等的人已经到齐，你们现在确定能玩半小时。"
                               "问店家当前所给钟点开始有没有空包间，先确认能不能打，不请店家预订。"
                           ))
        return ActorContext(person, self.clock.isoformat(), history=visible,
            events=[{"event_id": event.event_id, "kind": kind, "at": self.clock.isoformat(),
                     "caused_by": event.caused_by, "details": details}],
            private_targets=private_targets, group_targets=group_targets, visible_groups=groups)

    def _act(self, event: VenueEvent, person: Persona) -> None:
        if event.kind in {"opportunity", "delivered", "share_joined", "arrival", "departure", "slot_query"} and not self.game_id:
            self._record(event, skipped="尚无已登记牌局")
            return
        if event.kind == "share_joined" and person.sender_id not in self._confirmed():
            self._record(event, skipped="本人尚未获得实际的入局确认")
            return
        visible = self.channel.events_for(person.sender_id)[-24:]
        if event.caused_by and not any(row["message_id"] == event.caused_by for row in visible):
            self._record(event, skipped="本角色未收到触发这一轮的消息")
            return
        context = self._actor_context(event, person, visible)
        decision = self.actor.decide(context, trace_id=f"role:{event.event_id}")
        row = self._record(event, actor_id=person.sender_id, decision=decision.to_dict())
        verified = _live_actor_action(row)
        self.real_customer_actions += int(verified)
        if event.kind != "activity":
            self.evidence.append(row)
            self.controlled_actions_from_model &= verified
        action = decision.action
        if action is None:
            self.failures.append({"event_id": event.event_id, "kind": event.kind, "reason": "角色生成失败或动作合同不合格"})
            if decision.status == "transport_error":
                self.model_transport_errors.append({
                    "event_id": event.event_id, "model_stage": "role",
                    "http_statuses": [item["http_status"] for item in decision.diagnostics.get("attempts", [])
                                      if item.get("http_status") is not None],
                })
            return
        if action.action == "arrive":
            if event.kind != "arrival" or person.sender_id not in self._confirmed():
                raise ValueError("签到缺少到店事件或本人确认")
            self.arrivals.add(person.sender_id)
        if action.action == "leave":
            if event.kind != "departure" or person.sender_id not in self.arrivals:
                raise ValueError("离场缺少到期事件或本人签到")
            self.departures.add(person.sender_id)
        if action.action == "send":
            self._message(event, person, action)

    def _message(self, event: VenueEvent, person: Persona, action) -> None:
        kind = action.kind or "private"
        recipient = action.recipient_id or (self.channel.room_id if kind == "group" else AGENT_ID)
        message_id = f"customer:{event.event_id}"
        if self.fixture_agent:
            inputs = {"requirement": self.requirement, "start_at": self.clock.isoformat(),
                      "end_at": (self.clock + timedelta(minutes=30)).isoformat()}
            phase = "decline" if person == self.avoider and event.kind == "opportunity" else event.kind
            self.fixture_agent.turns[message_id] = (phase, inputs)
        quote = self.announcement_message_id if (
            kind == "group" and event.kind == "opportunity" and event.caused_by == self.announcement_message_id
        ) else None
        response = self.channel.send(MessageEnvelope(
            message_id, person.sender_id, action.text, self.clock, kind, recipient, quoted_message_id=quote,
        ))
        evidence = {
            "event_id": f"delivery:{event.event_id}", "message_id": message_id, "actor_id": person.sender_id,
            "trace_id": response.get("trace_id"), "tool_results": response.get("tool_results", []),
            "outputs": response.get("archive_outputs", []), "runtime_status": response.get("status"),
        }
        self._record(event, delivery=evidence)
        if event.kind != "activity":
            self.evidence.append(evidence)
        if event.kind == "need":
            self.opening_turns += 1
            self._after_need(event, person, evidence)
        if event.kind == "preference":
            self.checks["memory_from_visible_message"] = any(
                candidate.customer_id == person.sender_id and candidate.source_trace_id == evidence["trace_id"]
                and candidate.field == "smoke_preference" and candidate.status == "pending_review"
                and candidate.value in ("no_smoking", "no_smoke", "无烟") and candidate.evidence
                and candidate.evidence in action.text
                for candidate in self.store.pending_memory_candidates.values()
            )
        if event.kind == "peer_invite" and recipient == self.partner.sender_id and kind == "private":
            self.checks["peer_delivery_isolation"] = (
                any(row["message_id"] == message_id for row in self.channel.events_for(self.partner.sender_id))
                and all(not any(row["message_id"] == message_id for row in self.channel.events_for(user))
                        for user in (self.avoider.sender_id, AGENT_ID))
            )
        if event.kind in {"opportunity", "delivered"} and person in self.players and person != self.organizer:
            self._candidate_follow_up(event, person, evidence)
        if event.kind == "share_joined" and kind == "group":
            self.social_message_id = message_id
            self._enqueue("opportunity", event.at + timedelta(minutes=1), self.avoider, caused_by=message_id)
        if person == self.avoider and event.kind == "opportunity":
            self.observed_refusal = (kind == "private" and recipient == AGENT_ID
                                     and event.caused_by == self.social_message_id)
        if event.kind == "slot_query":
            key = "agent_pre_end_read" if self.clock < self.expected_play_end_at else "agent_post_end_read"
            relevant = "occupied_room_ids" if key == "agent_pre_end_read" else "available_room_ids"
            self.checks[key] = any(
                tool["name"] == "check_room_availability" and tool.get("called") and tool.get("allowed")
                and not tool.get("error") and self.room_id in tool["result"].get(relevant, [])
                and datetime.fromisoformat(tool["result"]["start_at"]) == self.clock
                and datetime.fromisoformat(tool["result"]["end_at"]) == self.clock + PROBE_DURATION
                for tool in evidence["tool_results"]
            ) and bool(evidence["outputs"])

    def _after_need(self, event: VenueEvent, person: Persona, evidence: dict[str, Any]) -> None:
        created = [tool["result"]["game"]["game_id"] for tool in evidence["tool_results"]
                   if tool.get("name") == "create_game" and tool.get("called")
                   and tool.get("allowed") and not tool.get("error")]
        if not created:
            delivered = self._visible_reply(person, evidence)
            if delivered and self.opening_turns < 4:
                self._enqueue("need", event.at + timedelta(minutes=1), person, caused_by=delivered)
            return

        self.game_id = created[0]
        game = self.store.require_game(self.game_id)
        self.checks["create_via_agent"] = (game.organizer_id == self.organizer.sender_id and len(created) == 1
            and evidence["runtime_status"] in {"completed", "waiting_user"} and bool(evidence["outputs"])
            and game.planned_start_at == self.start_at and game.planned_end_at == self.start_at + timedelta(hours=4))
        if not self.checks["create_via_agent"]:
            raise ValueError("正式建局结果的组织者、时间或时长与本轮请求不符")
        self._schedule_arrival(person, evidence)
        tasks = [task for task in self.store.scheduled_tasks.values() if task.task_type == BOARD_TASK_TYPE
                 and task.aggregate_id == self.channel.room_id and task.status.value == "pending"]
        if not tasks:
            raise ValueError("待组局缺少正式看板发布任务")
        self._enqueue("announce", max(event.at, min(task.due_at for task in tasks)))

    @staticmethod
    def _visible_reply(person: Persona, evidence: dict[str, Any]) -> str | None:
        return next((message["message_id"] for message in reversed(evidence["outputs"])
                     if message.get("recipient_id") == person.sender_id or message.get("kind") == "group"), None)

    def _candidate_follow_up(self, event: VenueEvent, person: Persona, evidence: dict[str, Any]) -> None:
        count = self.opportunity_turns.get(person.sender_id, 0) + 1
        self.opportunity_turns[person.sender_id] = count
        cause = self._visible_reply(person, evidence)
        if person.sender_id in self.confirmation_messages:
            return
        if self._schedule_arrival(person, evidence):
            if person == self.partner:
                self._enqueue("share_joined", event.at + timedelta(minutes=1), person, caused_by=cause)
        elif cause and count < 3:
            self._enqueue("delivered", event.at + timedelta(minutes=1), person, caused_by=cause)

    def _schedule_arrival(self, person: Persona, evidence: dict) -> bool:
        """只在本人已确认且收到回复后，安排其有真实对话来源的到店事件。"""
        cause = self._visible_reply(person, evidence)
        if (not cause or evidence["runtime_status"] not in {"completed", "waiting_user"}
                or person.sender_id in self.confirmation_messages or person.sender_id not in self._confirmed()):
            return False
        self.confirmation_messages[person.sender_id] = cause
        arrival_time = self.start_at - timedelta(minutes=5) + timedelta(seconds=self.players.index(person))
        if arrival_time >= self.clock:
            self._enqueue("arrival", arrival_time, person, caused_by=cause)
        return True

    def _confirmed(self) -> set[str]:
        return {person.customer_id for person in self.store.require_game(self.game_id).participants
                if person.status in ("joined", "confirmed")}

    def _world(self, event: VenueEvent) -> None:
        if event.kind == "announce":
            self._announce(event)
        elif event.kind == "start":
            self._start(event)
        elif event.kind == "occupied":
            self._occupancy(event)
        elif event.kind == "finish":
            self._finish(event)
        elif event.kind == "rebook":
            self._rebook(event)

    def _announce(self, event: VenueEvent) -> None:
        """执行已经排队的正式看板发布；顾客从送达公开看板开始行动。"""
        result = self.channel.publish_board(trace_id=event.event_id, sent_at=event.at)
        self.evidence.append(self._record(event, publication=result))
        published = {task["snapshot"]["external_message_id"] for task in result["tasks"] if task["snapshot"]
                     and task["status"] == "completed" and any(
                         item["game_id"] == self.game_id for item in task["snapshot"]["items"])}
        self.announcement_message_id = next(
            (row["message_id"] for row in result["archive_outputs"] if row["message_id"] in published), None)
        self.checks["board_delivered"] = bool(self.announcement_message_id)
        if not self.announcement_message_id:
            raise ValueError("正式看板没有发布包含本局的群消息")
        for delay, target in ((2, self.partner), (8, self.extra_a), (11, self.extra_b)):
            self._enqueue("opportunity", event.at + timedelta(minutes=delay), target, caused_by=self.announcement_message_id)

    def _start(self, event: VenueEvent) -> None:
        if not self.game_id:
            raise ValueError("没有形成待验证的牌局")
        expected = {person.sender_id for person in self.players}
        self.checks["social_refusal"] = bool(self.observed_refusal and self.game_id
            and self.avoider.sender_id not in self._confirmed())
        self.checks["four_real_confirmations"] = bool(self.game_id and self._confirmed() == expected)
        self.checks["four_arrivals"] = self.arrivals == expected
        if not (self.checks["four_real_confirmations"] and self.checks["four_arrivals"]):
            raise ValueError("到点未凑齐四名实际确认且签到的客户")

        end = event.at + PLAY_DURATION
        opening = datetime.strptime(VENUE["opens"], "%H:%M").time()
        closing = datetime.strptime(VENUE["closes"], "%H:%M").time()
        candidates = [room["id"] for room in VENUE["rooms"] if room["seats"] >= 4 and not room["smoking"]
                      and event.at.date() == end.date() and opening <= event.at.time() < end.time() <= closing]
        available = self.store.search_room_availability(start_at=event.at, end_at=end)["available_room_ids"]
        booked = [item.room_id for item in self.store.room_reservations.values() if item.game_id == self.game_id
                  and item.status in {"held", "confirmed"} and item.start_at <= event.at < item.end_at]
        self.room_id = next((name for name in candidates if name in booked), None) or next(
            (name for name in candidates if name in available), None)
        if self.room_id is None:
            raise ValueError("营业时间内没有可承接本局的四人无烟包间")
        game, transitions = self.store.start_game_play(
            self.game_id, attendee_ids=sorted(self.arrivals), room_id=self.room_id, at=event.at, trace_id=event.event_id)
        self.booking_id = game.play_reservation_id
        self.expected_play_end_at = game.expected_play_end_at
        self.checks["actual_start"] = (game.is_playing and game.actual_started_at == event.at
                                       and game.actual_started_at != game.planned_start_at
                                       and game.expected_play_end_at == end)
        self.evidence.append(self._record(event, game=game.to_dict(), transitions=[item.to_dict() for item in transitions]))
        self._enqueue("occupied", end - timedelta(minutes=1))
        self._enqueue("slot_query", end - timedelta(minutes=18), self.probe)
        for person in self.players:
            self._enqueue("departure", end, person)
        self._enqueue("finish", end)

    def _occupancy(self, event: VenueEvent) -> None:
        availability = self.store.search_room_availability(start_at=event.at, end_at=event.at + timedelta(minutes=2))
        self.checks["occupied_before_end"] = self.room_id in availability["occupied_room_ids"]
        before = set(self.store.room_reservations)
        try:
            self.store.reserve_room(conversation_id="venue-overlap-probe", game_id=None, start_at=event.at,
                end_at=event.at + timedelta(minutes=2), room_id=self.room_id, trace_id=event.event_id)
        except ValueError as exc:
            self.checks["overlap_denied"] = "room is unavailable" in str(exc) and set(self.store.room_reservations) == before
        else:
            self.checks["overlap_denied"] = False
        self.evidence.append(self._record(event, availability=availability, overlap_denied=self.checks["overlap_denied"]))

    def _finish(self, event: VenueEvent) -> None:
        self.checks["four_departures"] = self.departures == self.arrivals
        if not self.checks["four_departures"]:
            raise ValueError("缺少四位已到场客户的离店事件")
        game, changes = self.store.finish_game_play(self.game_id, at=event.at, trace_id=event.event_id)
        reopen = SQLiteAgentStore(self.run_dir / "agent.sqlite3")
        persisted = reopen.require_game(game.game_id)
        reservation = reopen.room_reservations[self.booking_id]
        self.checks["actual_four_hour_finish"] = bool(
            persisted.status == GameStatus.FINISHED and persisted.actual_ended_at and persisted.actual_started_at
            and persisted.actual_ended_at - persisted.actual_started_at == PLAY_DURATION)
        self.checks["release_persisted"] = reservation.status == "released" and any(
            step.entity_id == self.booking_id and step.to_status == "released" for step in changes)
        self.evidence.append(self._record(event, game=persisted.to_dict(), released=reservation.to_dict(),
            transitions=[step.to_dict() for step in changes]))
        if self.checks["actual_four_hour_finish"] and self.checks["release_persisted"]:
            self._enqueue("slot_query", event.at + timedelta(minutes=12), self.probe)
            self._enqueue("rebook", event.at + timedelta(minutes=14))

    def _rebook(self, event: VenueEvent) -> None:
        next_booking, _ = self.store.reserve_room(conversation_id="venue-next-booking", game_id=None, start_at=event.at,
            end_at=event.at + PROBE_DURATION, room_id=self.room_id, trace_id=event.event_id)
        persisted = SQLiteAgentStore(self.run_dir / "agent.sqlite3").room_reservations
        booking = persisted[next_booking.reservation_id]
        self.checks["same_room_rebookable"] = (
            booking.room_id == self.room_id and booking.status in {"held", "confirmed"}
            and booking.start_at == event.at and booking.end_at == event.at + PROBE_DURATION
            and persisted[self.booking_id].status == "released"
        )
        self.evidence.append(self._record(event, next_booking=next_booking.to_dict()))

    def _report(self) -> dict[str, Any]:
        records = self.channel.events()
        (self.run_dir / "chat.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        self.checks["population"] = len(self.people) == 500 == len(set(self.people))
        self.checks["conversation_evidence"] = any(row["direction"] == "inbound" for row in records) and any(
            row["direction"] == "outbound" for row in records)
        trace = self.run_dir / "agent_trace.log"
        if trace.exists():
            with trace.open(encoding="utf-8") as handle:
                self.checks["role_truth_not_in_agent_trace"] = bool(trace.stat().st_size) and not any(
                    "venue-persona-" in line for line in handle)
        else:
            self.checks["role_truth_not_in_agent_trace"] = None
        judge_input_exists = self.checks["conversation_evidence"] and self.evidence
        judged = None if self.judge is None or not judge_input_exists else self.judge.judge(
            "venue-four-hour-evening", self.evidence, trace_id="venue-independent-judge",
            checks=("角色发言符合本人的目标、偏好和可见信息", "拒绝及组局确认与可见对话一致",
                    "Agent 的业务承诺与工具结果和实际投递一致"),
        ).to_dict()
        if self.judge is not None and judged is None:
            judged = {"status": "not_run", "verdict": "inconclusive", "reason": "没有有效的双向聊天证据"}
        if isinstance(self.actor, ModelActor):
            self.checks["customer_decisions_from_model"] = bool(self.real_customer_actions and self.controlled_actions_from_model)
        return {"checks": self.checks, "failed_events": self.failures, "independent_judge": judged,
                "model_transport_errors": self.model_transport_errors,
                "game_id": self.game_id, "room_id": self.room_id, "chat_events": len(records),
                "people": len(self.people), "virtual_end": self.clock.isoformat(),
                "real_customer_actions": self.real_customer_actions}
