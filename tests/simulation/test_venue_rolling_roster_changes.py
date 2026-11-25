"""验证同局改期和本人退出，经正式渠道变更名单并刷新群看板。"""

from datetime import datetime, timedelta

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock

from .sim_adapter import build_runtime
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import VenueFixtureAgent, _call, _reply
from .venue_rolling_board import VenueRollingBoard


class _RosterChangeAgent(VenueFixtureAgent):
    """离线模型只提出工具调用，按网关结果给到私聊回执。"""

    def __init__(self):
        """仅注册当前私聊轮次预期发给正式网关的工具合同。"""
        super().__init__(seed=42)
        self.mutations = {}

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
        return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)


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


def test_reschedule_and_self_withdrawal_keep_the_same_game_and_refresh_board(tmp_path):
    """报名人无权改期，同行不能代确认后的客人退位，本人退出只释放自己的席位。"""
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
    revised_start = original_start + timedelta(minutes=30)
    duration = timedelta(hours=2)
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
    assert store.require_game(game_id).to_dict() == before and board.next_due() is None

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
    assert store.require_game(game_id).to_dict() == updated.to_dict() and board.next_due() is None

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
    archive = SandboxChannel(reopened_runtime, room_id=room, identities=members, archive_path=channel.archive_path)
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
        item, = reopened.get_board_snapshot_by_message_id(room, receipt_id).items
        assert item.game_id == game_id
        assert time_text in item.rendered_text and item.rendered_text.endswith(seats)

    checked = [claim_receipt, denied_reschedule, reschedule_receipt, denied_withdrawal, withdrawal_receipt]
    for viewer in members:
        visible = {row["message_id"] for row in archive.events_for(viewer)}
        assert {row["message_id"] for row in checked} & visible == {
            row["message_id"] for row in checked if row["recipient_id"] == viewer
        }
