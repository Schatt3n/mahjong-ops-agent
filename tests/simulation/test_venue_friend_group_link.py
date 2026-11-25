"""在同一模拟门店验证好友私聊建局、群看板认领与私聊回执的完整主线。"""

from datetime import datetime, timedelta

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE
from mahjong_agent_runtime.models import DEFAULT_TZ, ScheduledTaskStatus, using_clock

from .sim_adapter import build_runtime
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import VenueFixtureAgent, _reply


class _LinkedVenueFixture(VenueFixtureAgent):
    """离线生成建局合同及正式占位通知所需的可见话术。"""

    def complete(self, messages, *, trace_id, timeout_seconds):
        """认领结果来自已执行的系统触发，后续私聊仍走正式入口。"""
        context = self._context_payload(messages)
        trigger = context.get("system_trigger") or {}
        if trigger.get("trigger_type") == "group_claim_confirmed":
            return _reply("群里看板上那桌已经给你留了一位。")
        if (context.get("current_message") or {}).get("message_id") == "claim-followup":
            return _reply("你刚在群里认领的那一席已经登记。")
        return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)


def test_one_world_links_private_friend_game_to_group_claim_and_back_to_private(tmp_path, monkeypatch):
    """从空档案走正式工具、发布任务、群认领及私聊交付，证明前后都是同一牌局。"""
    from scripts import agent_runtime_app

    def reject_network(*args, **kwargs):
        raise AssertionError("the sandbox must not contact the WeChat bridge")

    monkeypatch.setattr(agent_runtime_app, "request_local_json", reject_network)
    room = "venue-group"
    organizer, claimant, observer = "friend-000", "friend-001", "friend-002"
    members = {f"friend-{index:03}": f"牌友{index:03}" for index in range(500)}
    members[organizer] = members[claimant] = "同名牌友"
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    fixture = _LinkedVenueFixture(seed=42)
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id=room, identities=members, archive_path=tmp_path / "chat.sqlite3")
    opened_at = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    planned_at = opened_at.replace(hour=19)
    private_note = "这次是我刚换工作，不想在群里说。"

    assert len(channel.owner_friends) == 500
    assert not store.games and not channel.events()
    assert not store.customers[organizer].profile_facts and not store.customers[claimant].profile_facts
    fixture.turns["organizer-open"] = ("dialogue_open", {})
    with using_clock(lambda: opened_at):
        opening = channel.send(MessageEnvelope(
            "organizer-open", organizer, f"老板，能帮我新开一桌杭麻吗？{private_note}", opened_at, "private", OWNER_ID,
        ))
    first_reply, = opening["archive_outputs"]
    assert first_reply["recipient_id"] == organizer and first_reply["kind"] == "private"
    assert not store.games and channel.events_for(claimant) == []

    fixture.turns["organizer-create"] = ("need", {"requirement": {
        "game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
        "start_time_kind": "scheduled", "start_time": "19:30", "planned_start_at": planned_at.isoformat(),
        "duration_kind": "fixed", "duration_hours": 2,
    }})
    confirmed_at = opened_at + timedelta(minutes=1)
    with using_clock(lambda: confirmed_at):
        creation = channel.send(MessageEnvelope(
            "organizer-create", organizer, "七点半开始，打两小时，一块底无烟。我一个人，请帮我新开一桌。",
            confirmed_at, "private", OWNER_ID, first_reply["message_id"],
        ))
    created, = [tool for tool in creation["tool_results"] if tool["name"] == "create_game"]
    assert created["called"] and created["allowed"] and not created["error"]
    game_id = created["result"]["game"]["game_id"]
    original_game = store.require_game(game_id)
    assert original_game.conversation_id == channel.private_conversation_id(organizer)
    assert original_game.organizer_id == organizer and original_game.remaining_seats() == 3
    assert channel.events_for(observer) == []

    task, = [task for task in store.scheduled_tasks.values() if task.task_type == BOARD_TASK_TYPE
             and task.status == ScheduledTaskStatus.PENDING]
    assert task.aggregate_id == room
    assert channel.publish_board(trace_id="too-early", sent_at=task.due_at - timedelta(seconds=1))["status"] == "no_due_task"
    publication = channel.publish_board(trace_id="initial-board", sent_at=task.due_at)
    assert publication["status"] == "completed"
    board, = publication["archive_outputs"]
    snapshot, = publication["snapshots"]
    first_item, = snapshot["items"]
    assert first_item["game_id"] == game_id and first_item["rendered_text"].endswith("173")
    assert board["conversation_id"] == channel.group_conversation_id
    assert board["message_id"] == snapshot["external_message_id"]
    assert [event["message_id"] for event in channel.events_for(claimant)] == [board["message_id"]]

    claimed_at = task.due_at + timedelta(seconds=1)
    claim = MessageEnvelope("group-claim", claimant, f"{first_item['item_no']}来", claimed_at, "group", room, board["message_id"])
    claimed = channel.send(claim)
    assert claimed["group_processing"]["routing"]["detail"]["status"] == "claimed"
    owner_receipt, = claimed["archive_outputs"]
    assert owner_receipt["recipient_id"] == claimant and owner_receipt["kind"] == "private"
    assert owner_receipt["sender_id"] == AGENT_ID and owner_receipt["source_message_id"] == claim.message_id
    assert owner_receipt["conversation_id"] == channel.private_conversation_id(claimant)
    assert owner_receipt["conversation_id"] != original_game.conversation_id
    assert channel.send(claim)["delivery_deduplicated"]

    followup_at = claimed_at + timedelta(seconds=1)
    with using_clock(lambda: followup_at):
        followup = channel.send(MessageEnvelope(
            "claim-followup", claimant, "我刚才在群里报的是这桌吧？", followup_at, "private", OWNER_ID, owner_receipt["message_id"],
        ))
    claim_reply, = followup["archive_outputs"]
    assert claim_reply["recipient_id"] == claimant and claim_reply["source_message_id"] == "claim-followup"
    assert claim_reply["conversation_id"] == owner_receipt["conversation_id"]

    refresh_task, = [task for task in store.scheduled_tasks.values() if task.task_type == BOARD_TASK_TYPE
                     and task.status == ScheduledTaskStatus.PENDING]
    refreshed = channel.publish_board(trace_id="refreshed-board", sent_at=max(refresh_task.due_at, followup_at))
    assert refreshed["status"] == "completed"
    refreshed_board, = refreshed["archive_outputs"]
    refreshed_snapshot, = refreshed["snapshots"]
    updated_item, = refreshed_snapshot["items"]
    assert refreshed_board["message_id"] != board["message_id"]
    assert updated_item["game_id"] == game_id and updated_item["rendered_text"].endswith("272")

    reopened = SQLiteAgentStore(store.path)
    assert set(reopened.games) == {game_id}
    updated = reopened.require_game(game_id)
    assert updated.remaining_seats() == 2
    assert {member.customer_id for member in updated.participants} == {organizer, claimant}
    claim_record = reopened.get_game_claim_by_source(channel.group_conversation_id, claim.message_id)
    assert claim_record.game_id == game_id and claim_record.customer_id == claimant
    assert reopened.get_board_snapshot_by_message_id(room, board["message_id"]).items[0].rendered_text.endswith("173")
    assert reopened.get_board_snapshot_by_message_id(room, refreshed_board["message_id"]).items[0].game_id == game_id

    public_ids = [board["message_id"], claim.message_id, refreshed_board["message_id"]]
    assert [event["message_id"] for event in channel.events_for(observer)] == public_ids
    for event in channel.events():
        if event["kind"] == "group":
            assert event["message_id"] in public_ids
            assert not any(secret in event["text"] for secret in (private_note, owner_receipt["text"], "同名牌友", game_id))
    assert private_note not in "".join(event["text"] for event in channel.events_for(claimant))
    assert owner_receipt["message_id"] not in {event["message_id"] for event in channel.events_for(organizer)}
