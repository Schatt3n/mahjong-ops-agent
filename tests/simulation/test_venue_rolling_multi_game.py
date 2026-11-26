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


def test_published_table_accepts_before_expiry_and_releases_hold_when_unfilled(tmp_path):
    """旧看板开场前可报名，晚到须确认，到期清理未开打的预约。"""
    room = "venue-group"
    members = {"host": "牌友甲", "early": "牌友乙", "late": "牌友丙"}
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    store.configure_rooms(["模拟包间"])
    fixture = _MultiGameAgent()
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id=room, identities=members, archive_path=tmp_path / "chat.sqlite3")
    opened = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    game_id, _ = _create(channel, fixture, message_id="create-expiring", organizer="host", at=opened,
                         hour=20, game_type="hangzhou_mahjong", alias="杭麻", private_note="")
    _, published_message, mapping = _publish(VenueRollingBoard(channel))
    assert mapping == {1: game_id}
    scheduled = store.require_game(game_id)
    deadline = scheduled.expires_at
    assert opened < scheduled.planned_start_at < deadline
    with using_clock(lambda: opened):
        reservation, _ = store.reserve_room(conversation_id=channel.private_conversation_id("host"), game_id=game_id,
            room_id="模拟包间", start_at=scheduled.planned_start_at, end_at=deadline, trace_id="expiry-case-reservation")

    before_start = scheduled.planned_start_at - timedelta(seconds=1)
    before = MessageEnvelope("before-start", "early", "1来", before_start, "group", room, published_message)
    _claim_receipt(channel, _send(channel, before), before, game_id)
    assert store.require_game(game_id).remaining_seats() == 2
    assert store.room_reservations[reservation.reservation_id].status == "held"

    after_start = MessageEnvelope("after-start-before-expiry", "late", "1来", deadline - timedelta(seconds=1),
                                  "group", room, published_message)
    stale = _send(channel, after_start)
    assert stale["group_processing"]["routing"]["game_id"] == game_id
    assert stale["group_processing"]["routing"]["detail"]["status"] == "rejected"
    assert store.get_game_claim_by_source(channel.group_conversation_id, after_start.message_id) is None
    assert store.require_game(game_id).remaining_seats() == 2
    assert store.room_reservations[reservation.reservation_id].status == "held"

    expired = MessageEnvelope("at-expiry", "late", "1来", deadline, "group", room, published_message)
    reply = _send(channel, expired)
    assert reply["group_processing"]["routing"]["detail"]["status"] == "rejected"
    receipt, = reply["archive_outputs"]
    assert (receipt["kind"], receipt["recipient_id"], receipt["source_message_id"]) == ("private", "late", expired.message_id)
    assert store.get_game_claim_by_source(channel.group_conversation_id, expired.message_id) is None
    with using_clock(lambda: deadline):
        assert store.active_games() == []

    reopened = SQLiteAgentStore(store.path)
    persisted = reopened.require_game(game_id)
    assert persisted.status == GameStatus.CANCELLED and persisted.closed_reason == "expired_without_full_table"
    assert persisted.remaining_seats() == 2 and persisted.actual_started_at is None and persisted.actual_ended_at is None
    assert reopened.room_reservations[reservation.reservation_id].status == "released"
    saved = reopened.get_board_snapshot_by_message_id(room, published_message)
    assert saved.items[0].game_id == game_id and saved.items[0].rendered_text.endswith("173")
    assert reopened.get_game_claim_by_source(channel.group_conversation_id, before.message_id).game_id == game_id
    assert receipt["message_id"] not in {row["message_id"] for row in channel.events_for("early")}


def test_board_clock_refresh_omits_expired_table_without_an_intervening_chat(tmp_path):
    """仅推进看板时钟也要按正式生命周期清理过期桌，避免重发已不能报名的号码。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    fixture = _MultiGameAgent()
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="venue-group", identities={"host": "牌友甲"},
                             archive_path=tmp_path / "chat.sqlite3")
    board = VenueRollingBoard(channel)
    opened = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    game_id, _ = _create(channel, fixture, message_id="create-quiet-expiry", organizer="host", at=opened,
                         hour=20, game_type="hangzhou_mahjong", alias="杭麻", private_note="")
    _, first_board, _ = _publish(board)
    deadline = store.require_game(game_id).expires_at
    with using_clock(lambda: deadline):
        store.ensure_game_board_arrangement(game_id, trace_id="pending-refresh-at-expiry", at=deadline, urgent=True)
    tick = board.next_due()
    assert tick is not None and tick <= deadline
    refreshed = board.publish_due(deadline)
    assert refreshed["status"] == "completed" and refreshed["snapshots"] == [] and refreshed["archive_outputs"] == []
    reopened = SQLiteAgentStore(store.path)
    assert reopened.require_game(game_id).closed_reason == "expired_without_full_table"
    assert reopened.get_board_snapshot_by_message_id(channel.room_id, first_board).items[0].game_id == game_id


def test_board_clock_republishes_unstarted_table_after_planned_time_without_chat(tmp_path):
    """早期业务刷新可提前，静默到原定开场后仍发布同桌的新状态，原看板不改。"""
    room = "venue-group"
    members = {"host": "牌友甲", "observer": "牌友乙"}
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    fixture = _MultiGameAgent()
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id=room, identities=members, archive_path=tmp_path / "chat.sqlite3")
    board = VenueRollingBoard(channel)
    opened = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    game_id, _ = _create(channel, fixture, message_id="create-quiet-boundary", organizer="host", at=opened,
                         hour=20, game_type="hangzhou_mahjong", alias="杭麻", private_note="")
    first_due, first_message_id, mapping = _publish(board)
    assert mapping == {1: game_id}
    game = store.require_game(game_id)
    planned = game.planned_start_at
    original_board = store.get_board_snapshot_by_message_id(room, first_message_id).to_dict()
    assert "回复编号即可认领" in original_board["rendered_text"]
    assert "晚到需先确认" not in original_board["rendered_text"]
    assert first_due < planned and board.next_due() == planned + timedelta(microseconds=1)

    # 普通业务进度可以把房间发布提前；发完这次进度后，仍保留到点的刷新。
    early_refresh = first_due + timedelta(minutes=5)
    with using_clock(lambda: early_refresh):
        store.ensure_game_board_arrangement(game_id, trace_id="room-progress", at=early_refresh, urgent=True)
    assert board.next_due() == early_refresh
    _, second_message_id, _ = _publish(board)
    assert second_message_id != first_message_id
    assert store.get_board_snapshot_by_message_id(room, second_message_id).rendered_text == original_board["rendered_text"]
    assert board.next_due() == planned + timedelta(microseconds=1)

    reopened_store = SQLiteAgentStore(store.path)
    reopened_runtime, _ = build_runtime(reopened_store, "mock", offline_client=fixture, local_coordination=True)
    reopened_channel = SandboxChannel(reopened_runtime, room_id=room, identities=members, archive_path=channel.archive_path)
    reopened_board = VenueRollingBoard(reopened_channel)
    assert reopened_board.publish_due(planned)["status"] == "no_due_task"
    _, third_message_id, refreshed_mapping = _publish(reopened_board)
    assert refreshed_mapping == mapping and len({first_message_id, second_message_id, third_message_id}) == 3
    refreshed = reopened_store.get_board_snapshot_by_message_id(room, third_message_id)
    assert "原定开场时间已过" in refreshed.rendered_text and "仍在凑人，晚到需先确认" in refreshed.rendered_text
    assert "回复编号即可认领" not in refreshed.rendered_text
    assert reopened_store.get_board_snapshot_by_message_id(room, first_message_id).to_dict() == original_board
    assert reopened_store.require_game(game_id).to_dict() == game.to_dict()
    assert len([row for row in reopened_channel.events() if row["direction"] == "inbound"]) == 1
    visible = reopened_channel.events_for("observer")
    assert [row["message_id"] for row in visible] == [first_message_id, second_message_id, third_message_id]
    assert visible[-1]["text"] == refreshed.rendered_text
