"""在同一模拟门店验证好友私聊建局、群看板认领与私聊回执的完整主线。"""

from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.customer_visible_review import CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE, BoardEngine, GroupMessage
from mahjong_agent_runtime.models import DEFAULT_TZ, ScheduledTaskStatus, ToolResult, using_clock

from .group_transport import SimulationMessenger
from .sim_adapter import build_runtime
from .venue_actors import ActorAction, ActorDecision, MockActor, generate_personas
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_friend_group_link import VenueFriendGroupLinkScenario
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
    # 席位事务也读取业务时钟，回放报名和重投时都须使用同一个营业时间。
    with using_clock(lambda: claimed_at):
        claimed = channel.send(claim)
    assert claimed["group_processing"]["routing"]["detail"]["status"] == "claimed"
    owner_receipt, = claimed["archive_outputs"]
    assert owner_receipt["recipient_id"] == claimant and owner_receipt["kind"] == "private"
    assert owner_receipt["sender_id"] == AGENT_ID and owner_receipt["source_message_id"] == claim.message_id
    assert owner_receipt["conversation_id"] == channel.private_conversation_id(claimant)
    assert owner_receipt["conversation_id"] != original_game.conversation_id
    with using_clock(lambda: claimed_at):
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


@pytest.mark.parametrize("kind", ("group", "private"))
def test_board_followup_keeps_public_quote_separate_from_private_cause(tmp_path, kind):
    """私聊触发下一轮仍可以在群里引用公开看板，不能将私聊当成群引用或伪造跨会话引用。"""
    visitor = generate_personas(42)[0]
    board = {"message_id": "board-v1", "kind": "group", "recipient_id": "room", "sender_id": AGENT_ID,
             "text": "1、今晚七点三缺一"}
    private = {"message_id": "private-inbox", "kind": "private", "recipient_id": visitor.sender_id,
               "sender_id": AGENT_ID, "text": "你问的条件还没确认"}
    sent = []

    class Channel:
        room_id = "room"

        def events_for(self, customer_id):
            assert customer_id == visitor.sender_id
            return [board, private]

        def send(self, envelope):
            sent.append(envelope)
            return {"status": "completed"}

    class Actor:
        def decide(self, context, *, trace_id):
            assert context.to_dict()["delivery"]["message_id"] == private["message_id"]
            assert context.group_targets == ("room",)
            action = ActorAction("send", "1来" if kind == "group" else "群里那桌还能问一下吗", kind,
                                 "room" if kind == "group" else AGENT_ID)
            return ActorDecision(trace_id, "mock", "valid", action, None, context.to_dict(), {})

    scenario = VenueFriendGroupLinkScenario.__new__(VenueFriendGroupLinkScenario)
    scenario.channel, scenario.actor, scenario.fixture_agent, scenario.run_dir = Channel(), Actor(), None, tmp_path
    now = datetime(2026, 11, 25, 18, 45, tzinfo=DEFAULT_TZ)
    report = {"failed_events": []}
    assert scenario._visit(visitor, private["message_id"], now, "确认原看板", report,
                           board_message_id="unseen-board", item_no=1) is None
    row, _ = scenario._visit(visitor, private["message_id"], now, "确认原看板", report,
                             board_message_id=board["message_id"], item_no=1)
    assert row["caused_by"] == private["message_id"]
    assert sent[0].quoted_message_id == (board["message_id"] if kind == "group" else None)
    assert "你问的条件" not in sent[0].text


def _scheduled_board(tmp_path, fixture):
    """准备三个独立好友与生产调度生成的一次看板发布。"""
    people = generate_personas(42, count=3)
    organizer = people[0]
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="venue-group",
                             identities={person.sender_id: person.name for person in people},
                             archive_path=tmp_path / "chat.sqlite3")
    stamp = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    channel_id = store.get_group_room_policy(channel.room_id).channel
    messenger = SimulationMessenger(store, channel.room_id, channel=channel_id)
    board = BoardEngine(store=store, messenger=messenger, clock=lambda: stamp)
    with using_clock(lambda: stamp):
        game = board.import_game_from_post(GroupMessage(
            room_id=channel.room_id, conversation_id=channel.group_conversation_id, channel=channel_id,
            sender_external_id=organizer.sender_id, sender_name=organizer.name, sent_at=stamp,
            text="2026-11-25 19:30 0.5 无烟 173", message_id="original-post",
        ), trace_id="scope-setup")
    task, = store.scheduled_tasks.values()
    scenario = VenueFriendGroupLinkScenario.__new__(VenueFriendGroupLinkScenario)
    scenario.store, scenario.channel, scenario.run_dir = store, channel, tmp_path
    scenario.actor, scenario.fixture_agent = MockActor(42), fixture
    return scenario, runtime, task, game, people[1], stamp


@pytest.mark.parametrize(("review_failures", "statuses"), [
    (1, ("pending", "completed")),
    (3, ("pending", "pending", "failed")),
])
def test_board_follows_durable_review_retry_and_preserves_every_attempt(tmp_path, monkeypatch, review_failures, statuses):
    """调度器仍然负责失败预算和到期时间，仿真逐次保存未送达、重试和最终结果。"""
    scenario, runtime, task, game, _, stamp = _scheduled_board(tmp_path, _LinkedVenueFixture(seed=42))
    runtime.reply_self_review_enabled = True
    calls = []

    def board_review(_processor, **kwargs):
        assert kwargs["review_scope"] == "group_board"
        calls.append(kwargs["trace_id"])
        if len(calls) <= review_failures:
            return ToolResult(CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME, called=True, allowed=False,
                              error="TimeoutError: board review read timed out")
        return ToolResult(CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME, called=True, allowed=True,
                          result={"approved": True})

    monkeypatch.setattr(type(runtime.action_processor), "review_customer_visible_content", board_review)
    result = scenario._board(game.game_id, after=stamp, phase="retry-after-review")
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert tuple(row["persisted_status"] for row in rows) == statuses
    assert tuple(row["publication_attempt"] for row in rows) == tuple(range(1, len(statuses) + 1))
    assert tuple(row["persisted_attempts"] for row in rows) == tuple(range(1, len(statuses) + 1))
    assert {row["task_id"] for row in rows} == {task.task_id}
    assert {row["publication_id"] for row in rows} == {task.payload["publication_id"]}
    assert {row["publication"]["trace_id"] for row in rows} == {"friend-link:retry-after-review"}
    assert [row["publication"]["tasks"][0]["trace_id"] for row in rows] == calls
    assert len(set(calls)) == len(statuses)
    assert rows[0]["virtual_at"] == task.due_at.isoformat()
    assert all(previous["persisted_due_at"] == later["virtual_at"]
               for previous, later in zip(rows, rows[1:]))
    assert all(row["persisted_last_error"] and not row["publication"]["archive_outputs"] for row in rows[:-1])

    if statuses[-1] == "completed":
        snapshot, item, published_at = result
        assert item["game_id"] == game.game_id and published_at.isoformat() == rows[-1]["virtual_at"]
        assert scenario.channel.events()[0]["message_id"] == snapshot["external_message_id"]
        assert len(scenario.channel.events()) == 1
    else:
        assert result is None and not scenario.channel.events()
        assert not rows[-1]["publication"]["archive_outputs"]


def test_board_stops_when_paused_without_spending_an_attempt(tmp_path):
    """策略暂停仍保留待发任务，但沙盒不擅自修改权限，也不空转虚拟时钟。"""
    scenario, runtime, task, game, _, stamp = _scheduled_board(tmp_path, _LinkedVenueFixture(seed=42))
    policy = runtime.store.get_group_room_policy(scenario.channel.room_id)
    runtime.store.upsert_group_room_policy(replace(policy, board_enabled=False))

    assert scenario._board(game.game_id, after=stamp, phase="paused") is None
    row, = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert row["persisted_attempts"] == 0 and row["persisted_status"] == "pending"
    assert datetime.fromisoformat(row["persisted_due_at"]) > task.due_at
    assert not scenario.channel.events()


def test_group_claim_fallback_remains_in_transcript_without_counting_as_success_receipt(tmp_path):
    """已经真实占位但模型失败时，保留技术回复供审阅，不能算作成功通知。"""

    class TimedOutClaimNotice(_LinkedVenueFixture):
        def complete(self, messages, *, trace_id, timeout_seconds):
            context = self._context_payload(messages)
            if (context.get("system_trigger") or {}).get("trigger_type") == "group_claim_confirmed":
                raise TimeoutError("group claim notice read timed out")
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    scenario, _, _, game, visitor, stamp = _scheduled_board(tmp_path, TimedOutClaimNotice(seed=42))
    snapshot, item, published_at = scenario._board(game.game_id, after=stamp, phase="initial")
    row, private = scenario._visit(
        visitor, snapshot["external_message_id"], published_at + timedelta(seconds=1), "只报名看板这一桌",
        {"failed_events": []}, board_message_id=snapshot["external_message_id"], item_no=item["item_no"],
    )

    assert scenario._seat_evidence(row, game.game_id, visitor.sender_id)
    assert scenario._joined(game.game_id, visitor.sender_id)
    assert row["runtime_status"] == "needs_help"
    assert private == row["private_receipts"] and len(private) == 1
    assert row["successful_private_receipt_ids"] == []
