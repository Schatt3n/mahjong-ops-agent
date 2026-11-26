"""验证同局改期、本人退出和跨群私聊查询。"""

import json
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock

from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, generate_personas
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import VenueFixtureAgent, _call, _reply
from .venue_role_journal import VenueRoleJournal
from .venue_rolling_board import VenueRollingBoard
from .venue_rolling_roles import VenueRollingRoles
from .venue_world import VenueWorld


class _RosterChangeAgent(VenueFixtureAgent):
    """离线模型只提出工具调用，按网关结果给到私聊回执。"""

    def __init__(self):
        """仅注册当前私聊轮次预期发给正式网关的工具合同。"""
        super().__init__(seed=42)
        self.mutations = {}
        self.followup_contexts = {}

    def complete(self, messages, *, trace_id, timeout_seconds):
        """保留工具是否实际执行的因果关系，失败时回复未生效。"""
        context = self._context_payload(messages)
        trigger = (context.get("system_trigger") or {}).get("trigger_type")
        if trigger == "group_claim_confirmed":
            return _reply("群看板那桌已经给你留了一位。")
        message_id = (context.get("current_message") or {}).get("message_id")
        if change := self.mutations.get(message_id):
            tool, arguments, confirmed, denied = change
            previous = context.get("previous_tool_results") or []
            if not previous:
                return _call(tool, arguments)
            result = previous[-1]
            if reference := result.get("result_reference"):
                result = context["turn_tool_evidence"][reference["index"]]
            return _reply(confirmed if result.get("allowed") and not result.get("error") else denied)
        if message_id in self.followup_contexts:
            self.followup_contexts[message_id] = context
            memberships = context["sender_active_game_memberships"]
            if not memberships:
                return _reply("我这里只能查你本人的报名。")
            membership, = memberships
            summary = next(game for game in context["active_game_visible_summaries"]
                           if game["game_id"] == membership["game_id"])
            start = summary["public_requirement"]["start_time"]
            if membership["participation_already_recorded"]:
                return _reply(f"这桌 {start} 开，你的 {membership['seat_count']} 个座位还在。")
            return _reply(f"这桌 {start} 开，我还没查到你的报名。")
        return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)


@pytest.mark.parametrize("decision", ["wait", "late", "withdraw"])
def test_private_overtime_replays_the_actors_choice_on_the_original_game(tmp_path, monkeypatch, decision):
    """加班只改变个人日程，暂停再跑不重选，所发原话走正式私聊和本人席位权限。"""
    start = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    open_at = start.replace(hour=20, minute=0)
    overtime_at = start + timedelta(minutes=15)
    world = VenueWorld(42, start)
    people = generate_personas(42, count=30)
    guest = next(person for person in people if world.availability(person).available_for(start, 3))
    organizer, patient, fourth, observer = [person for person in people if person != guest][:4]
    roster = (guest, organizer, patient, fourth, observer)
    busy_until = open_at + timedelta(minutes=5) if decision == "late" else world.availability(guest).free_until
    fixture = _RosterChangeAgent()

    class InterruptAfterDelivery(BaseException):
        """模拟业务和聊天已入档、角色事件尚未办结的中断。"""

    class OvertimeActor(MockActor):
        """按测试分支自主选择是否联系店家，其他唤醒只观察。"""

        def _choose(self, context, projected, event, rng):
            if event["kind"] != "overtime":
                return ActorAction("wait")
            assert context.persona == guest
            assert projected["delivery"] is None and projected["group_targets"] == []
            assert projected["private_targets"] == [AGENT_ID]
            assert event["details"]["work_busy_until"] == busy_until.isoformat()
            assert [row["message_id"] for row in projected["history"]] == ["overtime-seat-receipt"]
            assert len(calls) == 0 and archive.events() == confirmed_receipt
            assert archive.runtime.store.require_game(game_id).to_dict() == initial
            calls.append(event["event_id"])
            if decision == "wait":
                return ActorAction("wait")
            if decision == "withdraw":
                assert not event["details"]["can_arrive"] and "earliest_arrival" not in event["details"]
                return ActorAction("send", "老板，我加班赶不上今晚那桌了，麻烦把我这一位撤掉。")
            eta = datetime.fromisoformat(event["details"]["earliest_arrival"])
            assert eta == busy_until + timedelta(minutes=event["details"]["travel_minutes"])
            return ActorAction("send", f"老板，我临时加班，最快{eta:%H:%M}到。我原来报的那桌位置还在吗？")

    actor, calls = OvertimeActor(), []

    def open_scenario():
        store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
        runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
        channel = SandboxChannel(runtime, room_id="rolling-room", archive_path=tmp_path / "chat.sqlite3",
                                 identities={person.sender_id: person.name for person in roster})
        return VenueRollingRoles(people=roster, actor=actor, judge=None, channel=channel, run_dir=tmp_path,
            start_at=start, seed=42, active_users=len(roster), workers=2, dialogue_turns=2, judge_users=1)

    scenario = open_scenario()
    archive = scenario.channel
    store = archive.runtime.store
    with using_clock(lambda: start):
        game, _ = store.create_game(conversation_id=archive.private_conversation_id(organizer.sender_id),
            organizer_id=organizer.sender_id, organizer_name=organizer.name, trace_id="controlled-overtime-roster",
            requirement={"game_type": "hangzhou_mahjong", "stake": "1", "start_time_kind": "scheduled",
                         "planned_start_at": open_at.isoformat(), "start_time": "20:00", "duration_hours": 2},
            known_players=[])
        for person in (guest, patient, fourth):
            game, _ = store.join_game(game_id=game.game_id, customer_id=person.sender_id, display_name=person.name,
                                      trace_id="controlled-overtime-roster")
    game_id = game.game_id
    initial = store.require_game(game_id).to_dict()
    assert game.status.value == "ready" and game.remaining_seats() == 0
    delivered = archive.publish_owner(kind="private", recipient_id=guest.sender_id, sent_at=start,
        message_id="overtime-seat-receipt", text="今晚八点的杭麻已经给你留了一位。")
    assert delivered["status"] == "delivered"
    confirmed_receipt = archive.events()
    event_input = {"origin_id": "last-minute-shift", "at": overtime_at, "busy_until": busy_until}
    event = scenario.schedule_overtime(guest.sender_id, **event_input)
    assert scenario.schedule_overtime(guest.sender_id, **event_input).event_id == event.event_id
    with pytest.raises(ValueError, match="conflicts with an existing event"):
        scenario.schedule_overtime(guest.sender_id, **{**event_input, "busy_until": busy_until + timedelta(minutes=1)})
    assert store.require_game(game_id).to_dict() == initial and archive.events() == confirmed_receipt
    message_id = f"customer:{event.event_id}"
    if decision == "withdraw":
        fixture.mutations[message_id] = ("record_candidate_reply", {"game_id": game_id,
            "customer_id": guest.sender_id, "display_name": guest.name, "status": "declined"},
            "好，已经把你的位置撤掉。", "这次还没能撤位。")
    elif decision == "late":
        fixture.followup_contexts[message_id] = None

    finish = VenueRoleJournal.finish

    def stop(journal, current, result):
        if current.event_id == event.event_id:
            raise InterruptAfterDelivery
        return finish(journal, current, result)

    with monkeypatch.context() as patch:
        patch.setattr(VenueRoleJournal, "finish", stop)
        with pytest.raises(InterruptAfterDelivery):
            scenario.run()

    archived = archive.events()
    transition_count = len(store.transitions)
    reopened = open_scenario()
    report = reopened.run()
    assert all(report["checks"].values()) and report["rolling_roles"]["wake_kinds"]["overtime"] == 1
    final = reopened.journal.get(event.event_id)
    assert calls == [event.event_id] and final.status == "finished"
    assert archived == reopened.channel.events()
    reopened_store = reopened.channel.runtime.store
    assert transition_count == len(reopened_store.transitions)
    stored = reopened_store.require_game(game_id)
    assert stored.planned_start_at == open_at and set(reopened_store.games) == {game_id}
    assert reopened.channel.events_for(observer.sender_id) == []
    if decision == "wait":
        assert archived == confirmed_receipt and stored.to_dict() == initial and final.result["runtime_status"] == "wait"
        return

    inbound, reply = archived[len(confirmed_receipt):]
    assert (inbound["message_id"], inbound["kind"], inbound["sender_id"], inbound["recipient_id"]) == (
        message_id, "private", guest.sender_id, AGENT_ID)
    assert (reply["kind"], reply["recipient_id"], reply["source_message_id"]) == ("private", guest.sender_id, message_id)
    assert final.result["delivery_deduplicated"]
    followup, = [item for item in reopened.journal.events() if item.caused_by == reply["message_id"]]
    context = followup.decision["decision"]["context"]
    assert context["delivery"]["message_id"] == reply["message_id"]
    assert context["events"][0]["details"]["work_busy_until"] == busy_until.isoformat()
    trace = archive.runtime.trace_recorder.get_trace(final.result["trace_id"])
    assert any(entry.step == "context_built" for entry in trace)
    for entry in trace:
        assert "work_busy_until" not in json.dumps(entry.content, ensure_ascii=False)
    if decision == "withdraw":
        withdrawal, = [tool for tool in final.result["tool_results"] if tool["name"] == "record_candidate_reply"]
        assert withdrawal["called"] and withdrawal["allowed"] and stored.remaining_seats() == 1
        assert {person.customer_id for person in stored.participants if person.status == "declined"} == {guest.sender_id}
    else:
        assert not final.result["tool_results"] and stored.to_dict() == initial
        assert "座位还在" in reply["text"]


def _send(channel, message):
    """同一时钟用于渠道、权限、事务和发布任务。"""
    with using_clock(lambda: message.sent_at):
        return channel.send(message)


def _publish(board):
    """在真实发布任务的到期时刻归档并读取不可变看板。"""
    due = board.next_due()
    assert due is not None
    result = board.publish_due(due)
    assert result["status"] == "completed"
    snapshot, = result["snapshots"]
    receipt, = result["archive_outputs"]
    assert receipt["message_id"] == snapshot["external_message_id"]
    assert (receipt["kind"], receipt["direction"]) == ("group", "outbound")
    return due, receipt["message_id"]


def _mutation(channel, message, tool, *, allowed):
    """同时核对原工具结果与已交付本人私聊回执。"""
    result = _send(channel, message)
    effect, = [entry for entry in result["tool_results"] if entry["name"] == tool]
    assert effect["allowed"] is allowed
    assert effect["called"] is allowed
    assert bool(effect["error"]) == (not allowed)
    reply, = result["archive_outputs"]
    assert (reply["kind"], reply["sender_id"], reply["recipient_id"], reply["source_message_id"]) == (
        "private", AGENT_ID, message.sender_id, message.message_id,
    )
    assert reply["conversation_id"] == channel.private_conversation_id(message.sender_id)
    return effect, reply


def _setup_claimed_game(tmp_path):
    """发起人私聊组局，报名人只通过群看板报名。"""
    room = "venue-group"
    organizer, guest, observer = "organizer", "guest", "observer"
    members = {organizer: "阿文", guest: "阿远", observer: "阿明"}
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    fixture = _RosterChangeAgent()
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id=room, identities=members, archive_path=tmp_path / "chat.sqlite3")
    board = VenueRollingBoard(channel)
    at = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    original_start = at.replace(hour=20, minute=0)
    fixture.turns["create"] = ("need", {"requirement": {
        "game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
        "start_time_kind": "scheduled", "start_time": "20:00", "planned_start_at": original_start.isoformat(),
        "duration_kind": "fixed", "duration_hours": 2,
    }})
    created = _send(channel, MessageEnvelope(
        "create", organizer, "老板，帮我组个今晚八点的无烟杭麻局，一块底打两小时，我先占一位。", at, "private", OWNER_ID,
    ))
    write, = [entry for entry in created["tool_results"] if entry["name"] == "create_game"]
    assert write["allowed"] and write["called"] and not write["error"]
    game_id = write["result"]["game"]["game_id"]
    at, initial_board = _publish(board)

    joined = _send(channel, MessageEnvelope(
        "join", guest, "1来", at + timedelta(seconds=1), "group", room, initial_board,
    ))
    assert joined["group_processing"]["routing"]["detail"]["status"] == "claimed"
    claim_receipt, = joined["archive_outputs"]
    assert (claim_receipt["kind"], claim_receipt["recipient_id"], claim_receipt["source_message_id"]) == (
        "private", guest, "join",
    )
    claim = store.get_game_claim_by_source(channel.group_conversation_id, "join")
    assert (claim.game_id, claim.customer_id) == (game_id, guest)
    at, before_board = _publish(board)
    return channel, board, fixture, game_id, original_start, at, initial_board, claim_receipt, before_board


def test_reschedule_and_self_withdrawal_keep_the_same_game_and_refresh_board(tmp_path):
    """报名人无权改期，同行不能代确认后的客人退位，本人退出只释放自己的席位。"""
    organizer, guest, observer = "organizer", "guest", "observer"
    members = {organizer: "阿文", guest: "阿远", observer: "阿明"}
    channel, board, fixture, game_id, original_start, at, initial_board, claim_receipt, before_board = (
        _setup_claimed_game(tmp_path)
    )
    store = channel.runtime.store
    revised_start = original_start + timedelta(minutes=30)
    duration = timedelta(hours=2)
    before = store.require_game(game_id).to_dict()
    assert store.require_game(game_id).remaining_seats() == 2
    time_change = {
        "game_id": game_id, "requirement_patch": {"planned_start_at": revised_start.isoformat(), "start_time": "20:30"},
        "reason": "顾客提出将现有局改到八点半",
    }
    for name in ("guest-reschedule", "organizer-reschedule"):
        fixture.mutations[name] = ("update_game_requirement", time_change, "这桌改到今晚八点半开。",
                                   "改开局时间得由这桌发起人确认。")

    denied, denied_reschedule = _mutation(channel, MessageEnvelope(
        "guest-reschedule", guest, "把我刚报名这桌改成八点半开吧。", at + timedelta(seconds=1), "private", OWNER_ID,
    ), "update_game_requirement", allowed=False)
    assert "management requires its originating conversation" in denied["error"]
    assert store.require_game(game_id).to_dict() == before
    assert board.next_due() == original_start + timedelta(microseconds=1)

    rescheduled, reschedule_receipt = _mutation(channel, MessageEnvelope(
        "organizer-reschedule", organizer, "我发起的八点杭麻改成八点半开，还是打两小时。",
        at + timedelta(seconds=2), "private", OWNER_ID,
    ), "update_game_requirement", allowed=True)
    assert rescheduled["result"]["game"]["game_id"] == game_id
    updated = store.require_game(game_id)
    assert (updated.planned_start_at, updated.planned_end_at) == (revised_start, revised_start + duration)
    assert updated.remaining_seats() == 2
    assert {(p.customer_id, p.status) for p in updated.participants} == {
        (p["customer_id"], p["status"]) for p in before["participants"]
    }
    at, revised_board = _publish(board)

    fixture.mutations["reported-withdrawal"] = ("record_registered_member_withdrawal", {
        "game_id": game_id, "customer_id": guest, "reason": "发起人转述群报名人退出",
    }, "这位的席位已经取消。", "他是自己报名的，请他本人告诉我一声。")
    denial, denied_withdrawal = _mutation(channel, MessageEnvelope(
        "reported-withdrawal", organizer, "刚才群里报名的阿远不来了，替他取消那个位置。",
        at + timedelta(seconds=1), "private", OWNER_ID,
    ), "record_registered_member_withdrawal", allowed=False)
    assert "requires the original registration contact" in denial["error"]
    assert store.require_game(game_id).to_dict() == updated.to_dict()
    assert board.next_due() == revised_start + timedelta(microseconds=1)

    fixture.mutations["self-withdrawal"] = ("record_candidate_reply", {
        "game_id": game_id, "customer_id": guest, "display_name": members[guest], "status": "declined",
    }, "行，你的位置取消了，这桌还是八点半开。", "这个位置还没取消。")
    leave = MessageEnvelope("self-withdrawal", guest, "我八点半来不了，我的位置取消吧。",
                            at + timedelta(seconds=2), "private", OWNER_ID)
    withdrawn, withdrawal_receipt = _mutation(channel, leave, "record_candidate_reply", allowed=True)
    assert withdrawn["result"]["recorded_status"] == "declined"
    assert withdrawn["result"]["game"]["game_id"] == game_id
    _, final_board = _publish(board)
    delivered_before_retry = len(channel.events())
    state_before_retry = len(store.transitions)

    reopened = SQLiteAgentStore(store.path)
    reopened_runtime, _ = build_runtime(reopened, "mock", offline_client=fixture, local_coordination=True)
    archive = SandboxChannel(reopened_runtime, room_id=channel.room_id, identities=members,
                             archive_path=channel.archive_path)
    assert _send(archive, leave)["delivery_deduplicated"]
    assert len(archive.events()) == delivered_before_retry and len(reopened.transitions) == state_before_retry
    assert set(reopened.games) == {game_id}
    final = reopened.require_game(game_id)
    assert (final.planned_start_at, final.planned_end_at, final.remaining_seats()) == (
        revised_start, revised_start + duration, 3,
    )
    assert {p.customer_id: p.status for p in final.participants} == {
        organizer: next(p["status"] for p in before["participants"] if p["customer_id"] == organizer),
        guest: "declined",
    }
    assert [(change.entity_id, change.trace_id) for change in reopened.transitions
            if change.entity_type == "game_requirement" and change.entity_id == game_id] == [
        (game_id, reschedule_receipt["trace_id"]),
    ]
    assert [(change.entity_id, change.trace_id) for change in reopened.transitions
            if change.entity_type == "game_participant" and change.to_status.startswith("declined")] == [
        (f"{game_id}:{guest}", withdrawal_receipt["trace_id"]),
    ]
    assert len({initial_board, before_board, revised_board, final_board}) == 4
    for receipt_id, time_text, seats in ((initial_board, "20:00", "173"), (before_board, "20:00", "272"),
                                         (revised_board, "20:30", "272"), (final_board, "20:30", "173")):
        item, = reopened.get_board_snapshot_by_message_id(channel.room_id, receipt_id).items
        assert item.game_id == game_id
        assert time_text in item.rendered_text and item.rendered_text.endswith(seats)

    checked = [claim_receipt, denied_reschedule, reschedule_receipt, denied_withdrawal, withdrawal_receipt]
    for viewer in members:
        visible = {row["message_id"] for row in archive.events_for(viewer)}
        assert {row["message_id"] for row in checked} & visible == {
            row["message_id"] for row in checked if row["recipient_id"] == viewer
        }


def test_group_claimant_can_check_and_acknowledge_reschedule_in_first_owner_dm(tmp_path):
    """已报名的人主动私聊时读到改期后的同一个局及原席位，知悉无需再次报名。"""
    organizer, guest, observer = "organizer", "guest", "observer"
    channel, board, fixture, game_id, original_start, at, _, claim_receipt, _ = _setup_claimed_game(tmp_path)
    store = channel.runtime.store
    guest_dm = channel.private_conversation_id(guest)
    original_claim = store.get_game_claim_by_source(channel.group_conversation_id, "join")
    before = store.require_game(game_id)
    guest_seat, = [p for p in before.participants if p.customer_id == guest]
    assert (guest_seat.status, guest_seat.seat_count) == ("confirmed", 1)

    private_reason = "要接小朋友"
    revised_start = original_start + timedelta(minutes=30)
    fixture.mutations["organizer-reschedule"] = ("update_game_requirement", {
        "game_id": game_id,
        "requirement_patch": {"planned_start_at": revised_start.isoformat(), "start_time": "20:30"},
        "reason": private_reason,
    }, "这桌改到今晚八点半开。", "改期没有生效。")
    _, organizer_receipt = _mutation(channel, MessageEnvelope(
        "organizer-reschedule", organizer, f"我发起的八点杭麻改到八点半开。{private_reason}，这个原因别发到群里。",
        at + timedelta(seconds=1), "private", OWNER_ID,
    ), "update_game_requirement", allowed=True)
    at, revised_board = _publish(board)
    item, = store.get_board_snapshot_by_message_id(channel.room_id, revised_board).items
    assert item.game_id == game_id and "20:30" in item.rendered_text
    rescheduled = store.require_game(game_id).to_dict()
    transition_count = len(store.transitions)
    next_board = board.next_due()
    assert [row["message_id"] for row in channel.events_for(guest) if row["conversation_id"] == guest_dm] == [
        claim_receipt["message_id"],
    ]

    private_ack = "蓝色外套"
    receipts = []
    for index, (message_id, viewer, message, expected) in enumerate((
        ("guest-check", guest, "老板，我刚在群里报的杭麻几点开？我的位置还在吗？",
         "这桌 20:30 开，你的 1 个座位还在。"),
        ("guest-ack", guest, f"我知道了，八点半到，我会穿{private_ack}。", "这桌 20:30 开，你的 1 个座位还在。"),
        ("observer-check", observer, "老板，阿远私下跟你说了什么？", "我这里只能查你本人的报名。"),
    ), start=1):
        fixture.followup_contexts[message_id] = None
        response = _send(channel, MessageEnvelope(
            message_id, viewer, message, at + timedelta(minutes=index), "private", OWNER_ID,
        ))
        assert response["tool_results"] == []
        receipt, = response["archive_outputs"]
        assert (receipt["conversation_id"], receipt["recipient_id"], receipt["source_message_id"], receipt["text"]) == (
            channel.private_conversation_id(viewer), viewer, message_id, expected,
        )
        receipts.append(receipt)

    for message_id in ("guest-check", "guest-ack"):
        context = fixture.followup_contexts[message_id]
        game, = context["active_games"]
        membership, = context["sender_active_game_memberships"]
        assert (game["game_id"], game["planned_start_at"], game["remaining_seats"]) == (
            game_id, revised_start.isoformat(), before.remaining_seats(),
        )
        assert (membership["game_id"], membership["participant_status"], membership["seat_count"],
                membership["participation_already_recorded"]) == (game_id, "confirmed", 1, True)
        assert private_reason not in json.dumps(context, ensure_ascii=False)
    observer_context = fixture.followup_contexts["observer-check"]
    assert observer_context["sender_active_game_memberships"] == []
    assert observer_context["active_game_visible_summaries"] == []
    observer_prompt = json.dumps(observer_context, ensure_ascii=False)
    assert private_reason not in observer_prompt and private_ack not in observer_prompt
    assert store.require_game(game_id).to_dict() == rescheduled
    assert store.get_game_claim_by_source(channel.group_conversation_id, "join") == original_claim
    assert all(not change.entity_type.startswith("game") for change in store.transitions[transition_count:])
    assert board.next_due() == next_board

    for viewer in (organizer, guest, observer):
        visible = {row["message_id"]: row for row in channel.events_for(viewer)}
        assert {"organizer-reschedule", organizer_receipt["message_id"]} & visible.keys() == (
            {"organizer-reschedule", organizer_receipt["message_id"]} if viewer == organizer else set()
        )
        assert {receipt["message_id"] for receipt in receipts} & visible.keys() == {
            receipt["message_id"] for receipt in receipts if receipt["recipient_id"] == viewer
        }
        assert ("guest-ack" in visible) is (viewer == guest)
        assert private_reason not in visible[revised_board]["text"] and private_ack not in visible[revised_board]["text"]
