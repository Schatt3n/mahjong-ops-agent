"""群看板的真实适配链必须保留延期、重试、审查和不可变编号证据。"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BoardEngine, BoardPublicationError, GroupRoomPolicy
from mahjong_agent_runtime.models import ScheduledTaskStatus, now, using_clock
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("board_publication_app_test", ROOT / "scripts/agent_runtime_app.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class ReviewClient:
    """只代替审查模型传输，保留生产请求装配、解析与确定性末端过滤。"""

    def __init__(self, approved=True):
        self.approved = approved
        self.payloads = []

    def complete(self, messages, **kwargs):
        self.payloads.append(json.loads(messages[-1]["content"]))
        if hasattr(self, "on_review"):
            self.on_review()
        return json.dumps({"approved": self.approved, "needs_human": False,
            "reasoning_summary": "只检查公开看板", "violations": [] if self.approved else ["unsafe"],
            "item_reviews": [{"item_id": "group_board", "approved": self.approved,
                "suggested_safe_text": "", "reasoning_summary": "核对公开事实",
                "violations": [] if self.approved else ["unsafe"]}]})


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "delivery.sqlite3")
    yield value
    if isinstance(value, SQLiteAgentStore):
        value._connection.close()


def setup_delivery(store, monkeypatch, *, approved=True, requirement_overrides=None):
    """建立没有群转私聊记录的真实待组局，并走应用的生产发送与调度入口。"""
    review = ReviewClient(approved)
    runtime = AgentRuntime(llm_client=review, store=store, reply_self_review_enabled=True)
    store.upsert_group_room_policy(GroupRoomPolicy(room_id="room", merge_window_seconds=30))
    game, _ = store.create_game(conversation_id="private:requester", organizer_id="requester",
        organizer_name="PRIVATE_OWNER_CANARY", requirement={"game_type": "hangzhou_mahjong",
            "stake": "1", "stake_label": "1-256", "cap_score": 256, "smoke_preference": "any",
            "start_time_kind": "asap_when_full", "known_player_count": 1,
            "private_note": "PRIVATE_NOTE_CANARY", **(requirement_overrides or {})}, known_players=[], trace_id="create")
    monkeypatch.setattr(app, "get_runtime", lambda: runtime)
    scheduler = ScheduledAgentTaskScheduler(store=store, handler=app.handle_due_scheduled_agent_task,
        trace_recorder=runtime.trace_recorder, retry_delay_seconds=1)
    return runtime, review, game, scheduler


def health_and_receipt(path, *, payload=None, timeout_seconds=3.0):
    """提供带真实回执字段的假桥传输，不访问微信。"""
    return {"send_channel_enabled": True, "auto_send_reply": True} if path == "/health" else {"ok": True, "message_id": "external-board"}


def test_private_game_waits_for_merge_then_publishes_public_reviewed_board(store, monkeypatch):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    calls = []
    def request(path, **kwargs):
        calls.append((path, kwargs.get("payload")))
        return health_and_receipt(path, **kwargs)
    monkeypatch.setattr(app, "request_local_json", request)
    task = next(iter(store.scheduled_tasks.values()))
    assert scheduler.run_due_once(at=game.created_at) == 0
    assert calls == [] and store.get_latest_board_snapshot("room") is None
    assert scheduler.run_due_once(at=task.due_at) == 1
    snapshot = store.get_latest_board_snapshot("room")
    assert snapshot and [item.game_id for item in snapshot.items] == [game.game_id]
    assert "1-256" in snapshot.rendered_text and "173" in snapshot.rendered_text
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.COMPLETED
    assert len(review.payloads) == 1
    assert all("PRIVATE_" not in json.dumps(item) for item in review.payloads)
    assert "PRIVATE_" not in calls[-1][1]["text"]
    events = runtime.trace_recorder.get_trace(next(item["source_trace_id"] for path, item in calls if path == "/send"))
    assert any(event.step == "group_board_terminal_filter" and event.content["approved"] for event in events)


def test_board_start_boundary_refresh_survives_an_earlier_seat_update(store, monkeypatch):
    """一个普通群刷新被提前后，仍持久排期开场边界并发布真实编号快照。"""
    seed = now()
    starts_at = seed + timedelta(minutes=10)
    boundary = starts_at + timedelta(microseconds=1)
    with using_clock(lambda: seed):
        _, review, game, scheduler = setup_delivery(store, monkeypatch, requirement_overrides={
            "start_time_kind": "scheduled", "planned_start_at": starts_at.isoformat(), "duration_hours": 4,
        })
    sent = []

    def request(path, *, payload=None, **_kwargs):
        if path == "/health":
            return health_and_receipt(path)
        sent.append(payload)
        return {"ok": True, "message_id": f"board-{len(sent)}"}

    monkeypatch.setattr(app, "request_local_json", request)
    first_task, = store.scheduled_tasks.values()
    with using_clock(lambda: first_task.due_at):
        assert scheduler.run_due_once(at=first_task.due_at) == 1
    first_board = store.get_latest_board_snapshot("room")
    assert '如"1来"' in first_board.rendered_text
    successor, = (task for task in store.scheduled_tasks.values() if task.status == ScheduledTaskStatus.PENDING)
    assert successor.due_at == boundary

    earlier = starts_at - timedelta(minutes=2)
    with using_clock(lambda: earlier):
        store.join_game(game_id=game.game_id, customer_id="guest", display_name="顾客", trace_id="before-start-join")
        store.ensure_game_board_arrangement(game.game_id, trace_id="seat-refresh", at=earlier, urgent=True)
        assert store.scheduled_tasks[successor.task_id].due_at == earlier
        assert scheduler.run_due_once(at=earlier) == 1
    if isinstance(store, SQLiteAgentStore):
        reopened = SQLiteAgentStore(store.path)
        try:
            assert [item.due_at for item in reopened.due_scheduled_tasks(at=boundary)] == [boundary]
        finally:
            reopened._connection.close()
    with using_clock(lambda: starts_at):
        assert scheduler.run_due_once(at=starts_at) == 0
    with using_clock(lambda: boundary):
        assert scheduler.run_due_once(at=boundary) == 1
    fresh = store.get_latest_board_snapshot("room")
    assert "仍在凑人，晚到需先确认" in fresh.rendered_text
    assert '如"1来"' not in fresh.rendered_text
    assert fresh.items[0].game_id == first_board.items[0].game_id == game.game_id
    assert len(sent) == 3 and fresh.external_message_id != first_board.external_message_id
    assert store.get_board_snapshot(first_board.snapshot_id).rendered_text == first_board.rendered_text
    assert store.require_game(game.game_id).actual_started_at is None
    assert review.payloads[-1]["active_game_visible_summaries"][0]["board_registration"]["direct_claimable"] is False
    assert not any(task.status == ScheduledTaskStatus.PENDING for task in store.scheduled_tasks.values())


def test_review_crossing_planned_start_does_not_send_the_pre_start_board(store, monkeypatch):
    """如果审查中跨过开场，保留新刷新责任，旧可认领文案不能发送。"""
    stamp = now()
    start = stamp + timedelta(seconds=35)
    with using_clock(lambda: stamp):
        _, review, _, scheduler = setup_delivery(store, monkeypatch, requirement_overrides={
            "start_time_kind": "scheduled", "planned_start_at": start.isoformat(), "duration_hours": 4,
        })
    original, = store.scheduled_tasks.values()
    clock = [original.due_at]
    review.on_review = lambda: clock.__setitem__(0, start + timedelta(microseconds=1))
    paths = []
    monkeypatch.setattr(app, "request_local_json", lambda path, **kwargs:
        paths.append(path) or health_and_receipt(path, **kwargs))
    with using_clock(lambda: clock[0]):
        assert scheduler.run_due_once(at=original.due_at) == 1
    assert paths == ["/health"]
    assert store.get_latest_board_snapshot("room") is None
    assert "registration changed" in store.scheduled_tasks[original.task_id].last_error
    assert any(task.status == ScheduledTaskStatus.PENDING and task.due_at == clock[0]
               for task in store.scheduled_tasks.values())


def test_network_failure_preserves_task_and_reuses_delivery_identity_on_retry(store, monkeypatch):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    source_ids = []
    def request(path, *, payload=None, timeout_seconds=3.0):
        if path == "/health":
            return health_and_receipt(path)
        source_ids.append(payload["source_message_id"])
        if len(source_ids) == 1:
            raise URLError("network unavailable")
        return {"ok": True, "message_id": "external-retry"}
    monkeypatch.setattr(app, "request_local_json", request)
    task = next(iter(store.scheduled_tasks.values()))
    scheduler.run_due_once(at=task.due_at)
    pending = store.scheduled_tasks[task.task_id]
    assert pending.status == ScheduledTaskStatus.PENDING and pending.attempts == 1
    assert store.get_latest_board_snapshot("room") is None
    scheduler.run_due_once(at=pending.due_at)
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.COMPLETED
    assert len(source_ids) == 2 and len(set(source_ids)) == 1
    assert store.get_latest_board_snapshot("room").items[0].game_id == game.game_id


@pytest.mark.parametrize("pause", ["group_policy", "bridge_switch"])
def test_paused_permission_keeps_pending_without_consuming_failure_budget(store, monkeypatch, pause):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    requests = []
    if pause == "group_policy":
        store.upsert_group_room_policy(GroupRoomPolicy(room_id="room", outbound_enabled=False))
    def request(path, **kwargs):
        requests.append(path)
        return {"send_channel_enabled": False, "auto_send_reply": False}
    monkeypatch.setattr(app, "request_local_json", request)
    task = next(iter(store.scheduled_tasks.values()))
    scheduler.run_due_once(at=task.due_at)
    paused = store.scheduled_tasks[task.task_id]
    assert paused.status == ScheduledTaskStatus.PENDING and paused.attempts == 0
    assert store.get_latest_board_snapshot("room") is None and "/send" not in requests
    assert review.payloads == []
    store.upsert_group_room_policy(GroupRoomPolicy(room_id="room"))
    monkeypatch.setattr(app, "request_local_json", health_and_receipt)
    scheduler.run_due_once(at=paused.due_at)
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.COMPLETED


def test_review_rejection_never_reaches_send_and_keeps_retryable_task(store, monkeypatch):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch, approved=False)
    paths = []
    def request(path, **kwargs):
        paths.append(path)
        return health_and_receipt(path, **kwargs)
    monkeypatch.setattr(app, "request_local_json", request)
    task = next(iter(store.scheduled_tasks.values()))
    scheduler.run_due_once(at=task.due_at)
    assert paths == ["/health"]
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.PENDING
    assert store.get_latest_board_snapshot("room") is None


def test_permission_revoked_during_review_is_checked_before_send(store, monkeypatch):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    review.on_review = lambda: store.upsert_group_room_policy(GroupRoomPolicy(room_id="room", outbound_enabled=False))
    paths = []
    monkeypatch.setattr(app, "request_local_json", lambda path, **kwargs:
        paths.append(path) or health_and_receipt(path, **kwargs))
    task = next(iter(store.scheduled_tasks.values()))
    scheduler.run_due_once(at=task.due_at)
    assert paths == ["/health"] and len(review.payloads) == 1
    assert store.get_latest_board_snapshot("room") is None
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.PENDING
    assert store.scheduled_tasks[task.task_id].attempts == 0


def test_approved_model_cannot_bypass_terminal_filter(store, monkeypatch):
    """审查误批仍不能外发运行时专用字段；普通技术概念不属于此风险反例。"""
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake_label": "task_context_id"},
        reason="test unsafe public field", trace_id="unsafe")
    paths = []
    def request(path, **kwargs):
        paths.append(path)
        return health_and_receipt(path, **kwargs)
    monkeypatch.setattr(app, "request_local_json", request)
    task = next(item for item in store.scheduled_tasks.values() if item.status == ScheduledTaskStatus.PENDING)
    scheduler.run_due_once(at=task.due_at)
    assert paths == ["/health"] and len(review.payloads) == 1
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.PENDING
    assert store.get_latest_board_snapshot("room") is None


@pytest.mark.parametrize("credential", ["sk-test-abcdefghijklmnop123456", "Bearer abcdefghijklmnop123456", "CONFIGURED_TEST_CREDENTIAL_ONLY"])
def test_credential_filter_blocks_before_review_or_delivery(store, monkeypatch, credential):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    review.config = SimpleNamespace(api_key="CONFIGURED_TEST_CREDENTIAL_ONLY")
    store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake_label": credential},
        reason="credential filter test", trace_id="unsafe-key")
    paths = []
    monkeypatch.setattr(app, "request_local_json", lambda path, **kwargs:
        paths.append(path) or health_and_receipt(path, **kwargs))
    task = next(item for item in store.scheduled_tasks.values() if item.status == ScheduledTaskStatus.PENDING)
    scheduler.run_due_once(at=task.due_at)
    assert paths == ["/health"] and review.payloads == []
    assert store.get_latest_board_snapshot("room") is None
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.PENDING


def test_expired_publication_claim_cannot_send_after_review(store, monkeypatch):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    paths = []
    monkeypatch.setattr(app, "request_local_json", lambda path, **kwargs: paths.append(path) or health_and_receipt(path, **kwargs))
    task = next(iter(store.scheduled_tasks.values()))
    # 领取的是历史已过期租约，未发生接管也必须拒绝外发。
    claim = store.claim_scheduled_task(task.task_id, at=task.due_at)
    assert claim is not None
    class PastDateTime(datetime):
        @staticmethod
        def now():
            return claim.lease_until + timedelta(seconds=1)
    monkeypatch.setattr(app, "datetime", PastDateTime)
    with pytest.raises(BoardPublicationError, match="claim"):
        app.handle_due_scheduled_agent_task(claim, "stale")
    assert paths == ["/health"] and len(review.payloads) == 1


def test_startup_group_configuration_supports_first_private_message_and_revokes_removed_group(store, monkeypatch):
    store.upsert_group_room_policy(GroupRoomPolicy(room_id="old-room"))
    monkeypatch.setattr(app, "configured_wechaty_managed_rooms", lambda: {"room_ids": {"new-room"},
        "topic_keywords": {"unresolved-topic"}, "board_enabled": True, "outbound_enabled": False,
        "merge_window_seconds": 180})
    app.synchronize_managed_room_policies(store)
    assert not store.get_group_room_policy("old-room").managed
    assert store.get_group_room_policy("new-room").managed
    assert not store.get_group_room_policy("new-room").outbound_enabled
    game, _ = store.create_game(conversation_id="private:first", organizer_id="first", organizer_name="客户",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"}, known_players=[], trace_id="first")
    assert {link.room_id for link in store.game_conversation_links(game_id=game.game_id)} == {"new-room"}
    assert store.game_board_arrangement(game.game_id)["pending_task_count"] == 1


class LedgerMessenger:
    """模拟桥接的稳定业务键去重，区分调用次数和真正交付次数。"""

    def __init__(self):
        self.delivered = {}
        self.keys = []

    def send_group_message(self, room_id, text, *, metadata):
        key = metadata["source_message_id"]
        self.keys.append(key)
        self.delivered.setdefault(key, (text, f"external-{len(self.delivered)}"))
        return self.delivered[key][1]


def test_send_then_snapshot_failure_retries_stable_business_identity(store, monkeypatch):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    messenger = LedgerMessenger()
    engine = BoardEngine(store=store, messenger=messenger)
    original_save = type(store).save_board_snapshot
    attempts = []
    def interrupted_save(current_store, snapshot):
        attempts.append(snapshot)
        if len(attempts) == 1:
            raise OSError("interrupted after delivery")
        return original_save(current_store, snapshot)
    monkeypatch.setattr(type(store), "save_board_snapshot", interrupted_save)
    with pytest.raises(OSError):
        engine.publish("room", trace_id="first", publication_id="durable-publication")
    snapshot = engine.publish("room", trace_id="retry", publication_id="durable-publication")
    assert len(messenger.delivered) == 1 and len(messenger.keys) == 2
    assert messenger.keys[0] == messenger.keys[1]
    assert snapshot.items[0].game_id == game.game_id
    assert engine.publish("room", trace_id="completed-replay", publication_id="durable-publication") == snapshot
    assert len(messenger.keys) == 2


def test_identical_text_for_different_game_preserves_distinct_number_mapping(store, monkeypatch):
    runtime, review, first_game, scheduler = setup_delivery(store, monkeypatch)
    messenger = LedgerMessenger()
    engine = BoardEngine(store=store, messenger=messenger)
    first = engine.publish("room", trace_id="old", publication_id="same-job")
    store.update_game_status(game_id=first_game.game_id, status="cancelled", reason="customer cancelled", trace_id="cancel")
    second_game, _ = store.create_game(conversation_id="private:replacement", organizer_id="replacement",
        organizer_name="PRIVATE_REPLACEMENT", requirement={"game_type": "hangzhou_mahjong", "stake": "1",
            "stake_label": "1-256", "cap_score": 256, "smoke_preference": "any", "start_time_kind": "asap_when_full",
            "known_player_count": 1}, known_players=[], trace_id="replace")
    second = engine.publish("room", trace_id="new", publication_id="same-job")
    assert first.rendered_text == second.rendered_text
    assert first.snapshot_id != second.snapshot_id and len(messenger.delivered) == 2
    assert store.get_board_snapshot(first.snapshot_id).items[0].game_id == first_game.game_id
    assert store.get_board_snapshot(second.snapshot_id).items[0].game_id == second_game.game_id


@pytest.mark.parametrize("receipt", [{"ok": True}, {"ok": False}])
def test_missing_or_rejected_bridge_receipt_cannot_complete_publication(store, monkeypatch, receipt):
    runtime, review, game, scheduler = setup_delivery(store, monkeypatch)
    monkeypatch.setattr(app, "request_local_json", lambda path, **kwargs:
        health_and_receipt(path) if path == "/health" else receipt)
    task = next(iter(store.scheduled_tasks.values()))
    scheduler.run_due_once(at=task.due_at)
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.PENDING
    assert store.get_latest_board_snapshot("room") is None
