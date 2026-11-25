"""用真实沙盒私聊、看板、认领和角色账本检验滚动营业接线。"""

from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE
from mahjong_agent_runtime.models import GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_actors import ActorAction, ActorContext, MockActor, generate_personas
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import VenueFixtureAgent, _reply
from .venue_role_journal import VenueRoleJournal
from .venue_rolling_play_link import PLAY_ROLE_KINDS, PLAY_TICK_KIND, VenueRollingPlayLink
from .venue_rolling_roles import VenueRollingRoles
from .venue_rolling_world import VenueRollingWorld
from .venue_scenario import VENUE


_PLANNED = datetime.fromisoformat("2026-11-25T19:00:00+08:00")


class _VenueFixture(VenueFixtureAgent):
    """用离线话术完成已落库群认领和实际满员事件引发的正式通知。"""

    def complete(self, messages, *, trace_id, timeout_seconds):
        """确认与通知由正式服务实际触发；其他轮次复用原离线工具桩。"""
        trigger = self._context_payload(messages).get("system_trigger") or {}
        if trigger.get("trigger_type") in {"group_claim_confirmed", "group_game_full"}:
            return _reply("今晚七点这桌，给你留了一位。")
        return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)


class _VenueRoles(MockActor):
    """显式受控的本机角色可决定参加、等待或不离场。"""

    def __init__(self, idle=None):
        """指定单个角色在哪个私有物理事件等待，其余沿原角色桩运行。"""
        super().__init__(42)
        self.idle = idle

    def _choose(self, context, projected, event, rng):
        """发言和出入店动作仍须经过角色动作合同。"""
        if event["kind"] == "controlled_chat":
            details = context.events[-1]["details"]
            return ActorAction("send", details["text"], details["kind"], details["recipient"], details.get("quote"))
        if self.idle == (context.persona.sender_id, event["kind"]):
            return ActorAction("wait")
        return super()._choose(context, projected, event, rng)


def _open(tmp_path, people, *, allow_controlled):
    """从相同的独立运行目录重建渠道、正式库存和两个持久账本。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    store.configure_rooms([room["id"] for room in VENUE["rooms"]])
    fixture = _VenueFixture(seed=42)
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="rolling-room", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in people})
    journal = VenueRoleJournal(tmp_path / "role_events.sqlite3")
    link = VenueRollingPlayLink(VenueRollingWorld(store, tmp_path), channel, journal,
        active_actors=[person.sender_id for person in people], allow_controlled=allow_controlled)
    return link, fixture


def _decision(link, event, person, actor, *, extra=None, group_targets=()):
    """只把该角色真正读到的账本和本轮私人事件交给角色替身。"""
    history = [row for row in link.channel.events_for(person.sender_id)
               if datetime.fromisoformat(row["sent_at"]) <= event.due_at]
    context = ActorContext(person, event.due_at.isoformat(), history=history, events=[{
        "kind": "controlled_chat" if extra else event.kind, "event_id": event.event_id,
        "caused_by": event.caused_by, "at": event.due_at.isoformat(), "visible_to": (person.sender_id,),
        "details": extra or event.details,
    }], group_targets=group_targets, visible_groups=(link.channel.room_id,))
    action = actor.decide(context, trace_id=f"role:{event.event_id}")
    return link.journal.save_decision(event, {
        "actor_id": person.sender_id, "event_id": event.event_id, "decision": action.to_dict(),
        "virtual_at": event.due_at.isoformat(), "wake_kind": event.kind, "caused_by": event.caused_by,
        "visible_cause_verified": (action.context["delivery"] or {}).get("message_id") == event.caused_by,
    })


def _send(link, fixture, person, at, text, *, tool=None, quote=None, hide_receipt=False):
    """保存一个实际角色决定，调用正式通道并归档发送和私聊收据。"""
    journal, channel = link.journal, link.channel
    journal.enqueue(actor_id=person.sender_id, kind="controlled_intake", origin_id=at.isoformat(), due_at=at)
    event, = journal.claim_wave(1)
    message_id = f"customer:{event.event_id}"
    if tool is not None:
        fixture.turns[message_id] = tool
    kind, recipient = ("group", channel.room_id) if quote else ("private", AGENT_ID)
    event = _decision(link, event, person, _VenueRoles(), group_targets=(channel.room_id,),
                      extra={"text": text, "kind": kind, "recipient": recipient, "quote": quote})
    action = event.decision["decision"]["action"]
    with using_clock(lambda: at):
        response = channel.send(MessageEnvelope(message_id, person.sender_id, action["text"], at, action["kind"],
                                                action["recipient_id"], action.get("quote_message_id")))
    own = [row for row in response["archive_outputs"] if row["recipient_id"] == person.sender_id and row["kind"] == "private"]
    assert own and response["status"] in {"completed", "waiting_user"}, response
    outputs = ([row for row in response["archive_outputs"] if row["recipient_id"] != person.sender_id]
               if hide_receipt else own)
    journal.finish(event, {"message_id": message_id, "channel_trace_id": response["trace_id"],
                           "channel": kind,
                           "runtime_status": response["status"], "tool_results": response["tool_results"], "outputs": outputs})
    return response


def _filled_game(tmp_path, *, hide_receipt=False, allow_controlled=True):
    """四位角色分别经过正式建局、两次私聊加入和群看板认领，同一局真正落库满员。"""
    people = generate_personas(42)[:5]
    link, fixture = _open(tmp_path, people, allow_controlled=allow_controlled)
    store, channel = link.world.store, link.channel
    terms = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
             "start_time_kind": "scheduled", "planned_start_at": _PLANNED.isoformat(),
             "duration_kind": "fixed", "duration_hours": 4, "start_time": "19:00"}
    with link.journal.run_owner():
        at = _PLANNED - timedelta(hours=1)
        _send(link, fixture, people[0], at, "老板，今晚七点帮我组一桌无烟杭麻，一块底，打四小时，我先占一个。",
              tool=("need", {"requirement": terms}))
        game_id, = store.games
        for index in (1, 2):
            at += timedelta(minutes=1)
            _send(link, fixture, people[index], at, "老板，今晚七点有杭麻吗？",
                  tool=("cohort_offer", {"game_id": game_id}))
            at += timedelta(minutes=1)
            joined = _send(link, fixture, people[index], at, "老板，刚才说的今晚七点那桌杭麻，给我留个位吧。",
                           tool=("cohort_join", {"game_id": game_id}))
            assert store.require_game(game_id).seat_summary()["claimed_seats"] == index + 1, [
                (row["name"], row["error"]) for row in joined["tool_results"]]
        due = max(task.due_at for task in store.scheduled_tasks.values() if task.task_type == BOARD_TASK_TYPE)
        publication = channel.publish_board(trace_id="actual-board", sent_at=due)
        board, = publication["snapshots"]
        item, = board["items"]
        assert item["game_id"] == game_id
        at = max(at, due) + timedelta(minutes=1)
        claimed = _send(link, fixture, people[3], at, f"{item['item_no']}来", quote=board["external_message_id"], hide_receipt=hide_receipt)
        assert store.require_game(game_id).status == GameStatus.READY, claimed["group_processing"]["routing"]
        link.sync(at)
    return link, people, game_id


def _drive(link, people, game_id, *, idle=None, stop_after_start=False):
    """两位并发轮流交接，离场同刻拆成多波以验证不可提前办结。"""
    roster = {person.sender_id: person for person in people}
    with link.journal.run_owner():
        link.sync(link.journal.clock())
        while wave := link.journal.claim_wave(2):
            for event in wave:
                if event.kind == PLAY_TICK_KIND:
                    link.finish_control(event)
                    continue
                assert event.kind in PLAY_ROLE_KINDS
                person = roster[event.actor_id]
                event = _decision(link, event, person, _VenueRoles(idle))
                context = event.decision["decision"]["context"]
                assert context["delivery"]["recipient_id"] == person.sender_id
                assert "game_id" not in context["events"][-1]["details"]
                assert "confirmed" not in context["events"][-1]["details"]
                link.journal.finish(event, link.role_result(event) or {"runtime_status": "wait"})
            link.sync(wave[0].due_at)
            report = link.report()
            if any(event.kind == "departure" for event in wave) and report["verified_departure_decisions"] < 4:
                assert link.world.store.require_game(game_id).actual_ended_at is None
            if stop_after_start and report["actually_started"] == 1:
                return report
    return link.report()


def test_play_link_resumes_same_game_from_real_routes_through_release(tmp_path):
    """建局、私聊加入和群认领的本人回执都连接到同一真实四小时物理局。"""
    link, people, game_id = _filled_game(tmp_path)
    halfway = _drive(link, people, game_id, stop_after_start=True)
    assert halfway["actually_started"] == 1 and halfway["verified_arrival_decisions"] == 4
    plan, = halfway["participation_evidence"]
    assert plan["game_id"] == game_id
    routes = [plan["participants"][person.sender_id]["route"] for person in people[:4]]
    assert routes == ["private_create", "join", "join", "group_claim"]
    reopened, _ = _open(tmp_path, people, allow_controlled=True)
    report = _drive(reopened, people, game_id)
    assert report["status"] == "passed" and report["verified_departure_decisions"] == 4
    assert report["occupied_one_minute_before_end"] == report["four_hour_finishes"] == report["same_room_rebookable"] == 1
    assert reopened.world.store.require_game(game_id).status == GameStatus.FINISHED
    assert _drive(reopened, people, game_id) == report


def test_open_rolling_runner_drives_private_arrival_and_departure(tmp_path):
    """通过正常滚动入口消费已有正式报名，物理控制事件不计作顾客动作。"""
    link, people, game_id = _filled_game(tmp_path)
    start = _PLANNED - timedelta(minutes=10)
    scenario = VenueRollingRoles(people=people, actor=_VenueRoles(), judge=None, channel=link.channel,
        run_dir=tmp_path, start_at=start, seed=42, active_users=len(people), workers=2, dialogue_turns=2,
        judge_users=1, physical_play=True)
    report = scenario.run()
    assert all(report["checks"].values()), report["checks"]
    physical = report["rolling_play"]
    assert physical["status"] == "passed"
    assert physical["verified_arrival_decisions"] == physical["verified_departure_decisions"] == 4
    assert physical["arrival_sources"] == physical["departure_sources"] == {"controlled": 4}
    assert physical["participation_routes"] == {"private_create": 1, "join": 2, "group_claim": 1}
    assert physical["four_hour_finishes"] == physical["released_rooms"] == physical["same_room_rebookable"] == 1
    assert report["rolling_roles"]["wake_kinds"]["arrival"] == report["rolling_roles"]["wake_kinds"]["departure"] == 4
    assert PLAY_TICK_KIND not in report["rolling_roles"]["wake_kinds"]
    assert link.world.store.require_game(game_id).status == GameStatus.FINISHED


@pytest.mark.parametrize("hide_receipt,allow_controlled", [(True, True), (False, False)])
def test_play_link_rejects_missing_own_receipt_or_unproven_real_model(tmp_path, hide_receipt, allow_controlled):
    """正式满座不等于四位顾客均已收到本人回执，离线来源不能冒充真实角色。"""
    link, _, game_id = _filled_game(tmp_path, hide_receipt=hide_receipt, allow_controlled=allow_controlled)
    assert link.report()["status"] == "not_observed" and link.report()["arrival_plans"] == 0
    assert not any(event.kind in PLAY_ROLE_KINDS for event in link.journal.events())
    assert not link.world.store.require_game(game_id).is_playing


@pytest.mark.parametrize("kind", ["arrival", "departure"])
def test_play_link_a_wait_is_never_counted_as_arrival_or_departure(tmp_path, kind):
    """角色自由等待时，未签到的局不开始，未离场的局继续占房。"""
    link, people, game_id = _filled_game(tmp_path)
    report = _drive(link, people, game_id, idle=(people[3].sender_id, kind))
    if kind == "arrival":
        assert report["status"] == "not_observed" and report["reason"] == "missing_actual_arrivals"
        assert report["verified_arrival_decisions"] == 3 and report["actually_started"] == 0
    else:
        assert report["status"] == "blocked" and report["missing_departures_at_four_hours"] == 1
        assert report["verified_departure_decisions"] == 3 and report["released_rooms"] == 0
        assert link.world.store.require_game(game_id).is_playing
