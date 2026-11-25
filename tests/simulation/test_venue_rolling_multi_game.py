"""验证同群多局换号时，公开看板、报名事务与本人私聊回执均指向同一局。"""

from datetime import datetime, timedelta

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import DEFAULT_TZ, GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import VenueFixtureAgent, _call, _reply
from .venue_rolling_board import VenueRollingBoard


class _MultiGameAgent(VenueFixtureAgent):
    """确定性提交发起人的取消请求，仍由正式权限和状态机决定是否写入。"""

    def __init__(self):
        """取消目标从实际建局结果绑定，和其他桌的活动共用同一个运行时。"""
        super().__init__(seed=42)
        self.cancel_target = None

    def complete(self, messages, *, trace_id, timeout_seconds):
        """根据正式工具或通知输入为该场景提供离线话术。"""
        context = self._context_payload(messages)
        trigger = (context.get("system_trigger") or {}).get("trigger_type")
        if trigger == "group_claim_confirmed":
            return _reply("群看板那桌已经给你留了一位。")
        if trigger == "group_claim_rejected":
            return _reply("这个编号现在对不上了，这次没占上。")
        current = context.get("current_message") or {}
        if current.get("message_id") == "cancel-original":
            previous = context.get("previous_tool_results") or []
            if not previous:
                return _call("update_game_status", {
                    "game_id": self.cancel_target, "status": "cancelled", "reason": "发起人明确取消本局",
                })
            if previous[-1].get("allowed") and not previous[-1].get("error"):
                return _reply("好，已经取消这桌。")
        return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)


def _send(channel, message):
    """真实群认领和私聊工具读取同一可信演练时刻。"""
    with using_clock(lambda: message.sent_at):
        return channel.send(message)


def _create(channel, fixture, *, message_id, organizer, at, hour, game_type, alias, private_note):
    """通过好友私聊正式建局，并提取本轮成功工具返回的局号。"""
    start = at.replace(hour=hour, minute=0, second=0)
    fixture.turns[message_id] = ("need", {"requirement": {
        "game_type": game_type, "stake": "1", "smoke_preference": "no_smoking",
        "start_time_kind": "scheduled", "start_time": start.strftime("%H:%M"),
        "planned_start_at": start.isoformat(), "duration_kind": "fixed", "duration_hours": 2,
    }})
    response = _send(channel, MessageEnvelope(
        message_id, organizer, f"老板帮我新组一桌{alias}，{hour}点开始，打两小时，一块底，无烟，我自己占一位。{private_note}",
        at, "private", OWNER_ID,
    ))
    tools = response["tool_results"]
    created, = [tool for tool in tools if tool.get("name") == "create_game" and tool.get("called")
                and tool.get("allowed") and not tool.get("error")]
    reply, = response["archive_outputs"]
    assert reply["kind"] == "private" and reply["recipient_id"] == organizer
    game_id = created["result"]["game"]["game_id"]
    game = channel.runtime.store.require_game(game_id)
    assert game.organizer_id == organizer and game.conversation_id == channel.private_conversation_id(organizer)
    return game_id, {"message_id": message_id, "tool_results": tools}


def _publish(board):
    """让正式发布任务在应到时刻领取并核对实际的公开归档。"""
    due = board.next_due()
    assert due is not None
    response = board.publish_due(due)
    assert response["status"] == "completed"
    snapshot, = response["snapshots"]
    message, = response["archive_outputs"]
    assert message["message_id"] == snapshot["external_message_id"]
    assert message["conversation_id"] == board.channel.group_conversation_id
    assert message["direction"] == "outbound" and message["kind"] == "group"
    return due, message["message_id"], {item["item_no"]: item["game_id"] for item in snapshot["items"]}


def _claim_receipt(channel, response, request, game_id):
    """核对报名事务和确实送到此人私聊的确认。"""
    routing = response["group_processing"]["routing"]
    assert routing["action"] == "claim" and routing["game_id"] == game_id
    assert routing["detail"]["status"] == "claimed"
    receipt, = response["archive_outputs"]
    assert (receipt["sender_id"], receipt["kind"], receipt["recipient_id"], receipt["source_message_id"]) == (
        AGENT_ID, "private", request.sender_id, request.message_id,
    )
    assert receipt["conversation_id"] == channel.private_conversation_id(request.sender_id)
    return receipt["message_id"]


def test_same_group_renumbering_does_not_move_a_quoted_claim_to_another_game(tmp_path):
    """原引用跟随旧编号，取消后不落座；无引用按当前编号报另一桌。"""
    room = "venue-group"
    members = {"owner-a": "阿文", "owner-b": "阿文", "claim-a": "阿远", "claim-b": "阿禾", "observer": "阿明"}
    private_a, private_b = "我最近换了工作，别告诉群里。", "我刚搬家，这个只说给你。"
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    fixture = _MultiGameAgent()
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id=room, identities=members, archive_path=tmp_path / "chat.sqlite3")
    board = VenueRollingBoard(channel)
    at = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    game_a, created_a = _create(channel, fixture, message_id="create-original", organizer="owner-a", at=at,
                               hour=20, game_type="hangzhou_mahjong", alias="杭麻", private_note=private_a)
    at, first_board, first_mapping = _publish(board)
    assert first_mapping == {1: game_a}
    game_b, created_b = _create(channel, fixture, message_id="create-earlier", organizer="owner-b",
                               at=at + timedelta(seconds=1), hour=19, game_type="sichuan_mahjong", alias="川麻",
                               private_note=private_b)
    at, _, latest_mapping = _publish(board)
    assert latest_mapping == {1: game_b, 2: game_a} and game_a != game_b
    untouched_b = store.require_game(game_b).to_dict()

    original = MessageEnvelope("old-quote-valid", "claim-a", "1来", at + timedelta(seconds=1), "group", room, first_board)
    joined_a = _send(channel, original)
    receipt_a = _claim_receipt(channel, joined_a, original, game_a)
    assert store.require_game(game_a).remaining_seats() == 2
    assert store.require_game(game_b).to_dict() == untouched_b
    at, refreshed_a, refreshed_mapping = _publish(board)
    assert refreshed_mapping == latest_mapping

    fixture.cancel_target = game_a
    cancelled = _send(channel, MessageEnvelope(
        "cancel-original", "owner-a", "老板，麻烦取消我那桌今晚八点的杭麻。", at + timedelta(seconds=1), "private", OWNER_ID,
    ))
    cancellation, = [tool for tool in cancelled["tool_results"] if tool["name"] == "update_game_status"]
    assert cancellation["called"] and cancellation["allowed"] and not cancellation["error"]
    assert cancellation["result"]["game"]["game_id"] == game_a
    assert store.require_game(game_a).status == GameStatus.CANCELLED
    at, _, current_mapping = _publish(board)
    assert current_mapping == {1: game_b}
    cancelled_a = store.require_game(game_a).to_dict()

    stale = MessageEnvelope("old-quote-cancelled", "claim-b", "1来", at + timedelta(seconds=1), "group", room, first_board)
    rejected = _send(channel, stale)
    routing = rejected["group_processing"]["routing"]
    assert routing["action"] == "claim" and routing["game_id"] is None
    assert routing["detail"]["status"] == "rejected" and routing["detail"]["reason"] == "game_not_found"
    denial, = rejected["archive_outputs"]
    assert denial["kind"] == "private" and denial["recipient_id"] == stale.sender_id
    assert denial["source_message_id"] == stale.message_id and "没占上" in denial["text"]
    assert store.require_game(game_a).to_dict() == cancelled_a
    assert store.require_game(game_b).to_dict() == untouched_b

    current = MessageEnvelope("current-unquoted", "claim-b", "1来", at + timedelta(seconds=2), "group", room)
    joined_b = _send(channel, current)
    receipt_b = _claim_receipt(channel, joined_b, current, game_b)
    assert store.require_game(game_a).to_dict() == cancelled_a
    assert store.require_game(game_b).remaining_seats() == 2
    _, refreshed_b, updated_mapping = _publish(board)
    assert updated_mapping == {1: game_b}

    reopened = SQLiteAgentStore(store.path)
    reopened_runtime, _ = build_runtime(reopened, "mock", offline_client=fixture, local_coordination=True)
    archive = SandboxChannel(reopened_runtime, room_id=room, identities=members, archive_path=channel.archive_path)
    old_item, = reopened.get_board_snapshot_by_message_id(room, first_board).items
    new_item, = reopened.get_board_snapshot_by_message_id(room, refreshed_b).items
    assert old_item.game_id == game_a and old_item.rendered_text.endswith("173")
    assert new_item.game_id == game_b and new_item.rendered_text.endswith("272")
    assert reopened.require_game(game_a).status == GameStatus.CANCELLED
    assert reopened.require_game(game_b).remaining_seats() == 2
    for message, expected_game in ((original, game_a), (current, game_b)):
        claim = reopened.get_game_claim_by_source(archive.group_conversation_id, message.message_id)
        assert claim.game_id == expected_game and claim.customer_id == message.sender_id
    assert reopened.get_game_claim_by_source(archive.group_conversation_id, stale.message_id) is None

    records = {row["message_id"]: row for row in archive.events()}
    assert records[original.message_id]["quoted_message_id"] == records[stale.message_id]["quoted_message_id"] == first_board
    assert records[current.message_id]["quoted_message_id"] is None
    visible = {name: {row["message_id"] for row in archive.events_for(name)} for name in members}
    for name, own in (("claim-a", {receipt_a}), ("claim-b", {denial["message_id"], receipt_b})):
        for viewer, seen in visible.items():
            assert own & seen == (own if viewer == name else set())
    public = archive.events_for("observer")
    assert all(row["kind"] == "group" for row in public)
    assert not any(secret in row["text"] for row in public for secret in (private_a, private_b, "阿文", game_a, game_b))

    audit = VenueRollingBoard(archive).audit([
        created_a, created_b, {"message_id": original.message_id, "tool_results": joined_a["tool_results"]},
    ], archive.events())
    assert audit["private_games_on_public_board"] == 2
    assert audit["links"] == [{"game_id": game_a, "join_channel": "group", "source_board": first_board,
                               "participation_message": original.message_id, "private_receipt": receipt_a,
                               "updated_board": refreshed_a}]
