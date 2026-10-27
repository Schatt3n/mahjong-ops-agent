from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BoardEngine, ChannelIdentity, GroupMessage


TZ = ZoneInfo("Asia/Shanghai")


@pytest.fixture(autouse=True)
def fixed_participation_clock(monkeypatch):
    # 看板夹具的注入时钟是 7 月 22 日，入局事务的有效期校验也必须使用同一测试时间。
    # 保留原始报局时间与全部看板断言，不能让宿主日期决定此历史场景是否允许入局。
    monkeypatch.setattr("mahjong_agent_runtime.domains.game_participants.now", lambda: datetime(2026, 7, 22, 12, 0, tzinfo=TZ))


class FakeMessenger:
    def __init__(self) -> None:
        self.group_messages: list[tuple[str, str, str]] = []
        self.private_messages: list[tuple[str, str, str]] = []

    def send_group_message(self, room_id: str, text: str, *, metadata=None) -> str:
        message_id = f"group-output-{len(self.group_messages) + 1}"
        self.group_messages.append((room_id, text, message_id))
        return message_id

    def send_private_message(self, external_user_id: str, text: str, *, metadata=None) -> str:
        message_id = f"private-output-{len(self.private_messages) + 1}"
        self.private_messages.append((external_user_id, text, message_id))
        return message_id


def _engine():
    store = InMemoryAgentStore()
    messenger = FakeMessenger()
    clock = lambda: datetime(2026, 7, 22, 12, 0, tzinfo=TZ)
    return store, messenger, BoardEngine(store=store, messenger=messenger, clock=clock)


def _identity(store: InMemoryAgentStore, external_id: str = "external-a") -> None:
    store.upsert_channel_identity(
        ChannelIdentity(
            channel="wechaty",
            external_user_id=external_id,
            customer_id=f"customer:{external_id}",
            public_name="用户A",
            private_conversation_id=f"wechaty:contact:{external_id}",
            can_private_message=True,
            is_friend=True,
        )
    )


def _post(text: str = "14:00 0.5 无烟 371", *, message_id: str = "post-1") -> GroupMessage:
    return GroupMessage(
        room_id="room-1",
        conversation_id="wechaty:room:room-1",
        sender_external_id="external-a",
        sender_name="用户A",
        text=text,
        message_id=message_id,
        sent_at=datetime(2026, 7, 22, 12, 0, tzinfo=TZ),
    )


def test_new_game_schedules_one_durable_board_refresh() -> None:
    store, _, engine = _engine()
    _identity(store)

    engine.import_game_from_post(_post(), trace_id="trace-import")

    tasks = [item for item in store.scheduled_tasks.values() if item.task_type == "publish_group_board"]
    assert len(tasks) == 1
    assert tasks[0].due_at == datetime(2026, 7, 22, 12, 0, tzinfo=TZ) + timedelta(seconds=30)


def test_371_post_models_one_contact_representing_three_seats() -> None:
    store, _, engine = _engine()
    _identity(store)

    game = engine.import_game_from_post(_post(), trace_id="trace-import")

    assert game.remaining_seats() == 1
    assert len(game.participants) == 1
    assert game.participants[0].seat_count == 3
    assert game.participants[0].anonymous_seat_count == 2
    assert game.seat_summary() == {
        "seats_total": 4,
        "claimed_seats": 3,
        "remaining_seats": 1,
        "party_count": 1,
        "known_contact_count": 1,
        "anonymous_seat_count": 2,
    }


def test_multiple_changes_within_merge_window_share_one_pending_task() -> None:
    store, _, engine = _engine()

    first, _ = engine.on_game_event("room-1", "game_created", trace_id="trace-1")
    second, _ = engine.on_game_event("room-1", "game_updated", trace_id="trace-2")

    assert first.task_id == second.task_id
    assert len([item for item in store.scheduled_tasks.values() if item.task_type == "publish_group_board"]) == 1


def test_seat_claim_moves_pending_refresh_to_now() -> None:
    _, _, engine = _engine()

    engine.on_game_event("room-1", "game_created", trace_id="trace-1")
    task, _ = engine.on_game_event("room-1", "seat_claimed", trace_id="trace-2")

    assert task.due_at == datetime(2026, 7, 22, 12, 0, tzinfo=TZ)


def test_full_game_leaves_board_and_numbers_are_rebuilt() -> None:
    store, messenger, engine = _engine()
    _identity(store)
    first = engine.import_game_from_post(_post("14:00 0.5 无烟 371", message_id="post-1"), trace_id="trace-1")
    second = engine.import_game_from_post(_post("15:00 1 有烟 272", message_id="post-2"), trace_id="trace-2")
    engine.publish("room-1", trace_id="trace-board-1")
    store.join_game(game_id=first.game_id, customer_id="joiner", display_name="加入者", trace_id="trace-join")

    engine.publish("room-1", trace_id="trace-board-2")

    assert len(messenger.group_messages) == 2
    latest_text = messenger.group_messages[-1][1]
    assert "1、15:00" in latest_text
    assert "14:00" not in latest_text
    assert store.get_latest_board_snapshot("room-1").items[0].game_id == second.game_id


def test_quoted_board_number_resolves_exact_snapshot() -> None:
    store, _, engine = _engine()
    _identity(store)
    first = engine.import_game_from_post(_post("14:00 0.5 无烟 371", message_id="post-1"), trace_id="trace-1")
    engine.publish("room-1", trace_id="trace-board-1")
    first_board_message_id = store.get_latest_board_snapshot("room-1").external_message_id
    engine.import_game_from_post(_post("13:00 1 有烟 371", message_id="post-2"), trace_id="trace-2")
    engine.publish("room-1", trace_id="trace-board-2")

    resolved = engine.resolve_item_no("room-1", 1, quoted_message_id=first_board_message_id)

    assert resolved is not None
    assert resolved.game_id == first.game_id


def test_quoted_platform_message_id_resolves_through_transport_reference() -> None:
    store, _, engine = _engine()
    _identity(store)
    game = engine.import_game_from_post(_post(), trace_id="trace-import")
    snapshot = engine.publish("room-1", trace_id="trace-board")
    assert snapshot is not None
    source = next(
        reference
        for reference in store.message_references.values()
        if reference.business_ref_type == "group_board_snapshot"
        and reference.business_ref_id == snapshot.snapshot_id
        and reference.message_id.startswith("group_board:")
    )
    store.link_message_reference(
        conversation_id=snapshot.conversation_id,
        message_id="wechat-platform-message-1",
        source_message_id=source.message_id,
        channel="wechaty",
    )

    resolved = engine.resolve_item_no(
        "room-1",
        1,
        quoted_message_id="wechat-platform-message-1",
    )

    assert resolved is not None
    assert resolved.game_id == game.game_id


def test_unquoted_number_only_uses_latest_snapshot() -> None:
    store, _, engine = _engine()
    _identity(store)
    engine.import_game_from_post(_post("14:00 0.5 无烟 371", message_id="post-1"), trace_id="trace-1")
    engine.publish("room-1", trace_id="trace-board-1")
    latest = engine.import_game_from_post(_post("13:00 1 有烟 371", message_id="post-2"), trace_id="trace-2")
    engine.publish("room-1", trace_id="trace-board-2")

    resolved = engine.resolve_item_no("room-1", 1)

    assert resolved is not None
    assert resolved.game_id == latest.game_id


def test_stale_unquoted_number_never_falls_back_to_previous_snapshot() -> None:
    store, _, engine = _engine()
    _identity(store)
    game = engine.import_game_from_post(_post(), trace_id="trace-1")
    engine.publish("room-1", trace_id="trace-board-1")
    store.join_game(game_id=game.game_id, customer_id="joiner", display_name="加入者", trace_id="trace-join")
    engine.publish("room-1", trace_id="trace-board-2")

    assert engine.resolve_item_no("room-1", 1) is None


def test_no_active_games_does_not_publish_empty_board() -> None:
    _, messenger, engine = _engine()

    assert engine.publish("room-1", trace_id="trace-empty") is None
    assert messenger.group_messages == []


def test_sqlite_restart_restores_room_game_and_board_number(tmp_path) -> None:
    path = tmp_path / "group-board.sqlite3"
    first_store = SQLiteAgentStore(path)
    messenger = FakeMessenger()
    clock = lambda: datetime(2026, 7, 22, 12, 0, tzinfo=TZ)
    first_engine = BoardEngine(store=first_store, messenger=messenger, clock=clock)
    _identity(first_store)
    game = first_engine.import_game_from_post(_post(), trace_id="trace-import")
    snapshot = first_engine.publish("room-1", trace_id="trace-board")
    assert snapshot is not None

    restarted_store = SQLiteAgentStore(path)
    restarted_engine = BoardEngine(store=restarted_store, messenger=FakeMessenger(), clock=clock)
    restored = restarted_engine.resolve_item_no(
        "room-1",
        1,
        quoted_message_id=snapshot.external_message_id,
    )

    assert restored is not None
    assert restored.game_id == game.game_id
    assert restored.remaining_seats() == 1
