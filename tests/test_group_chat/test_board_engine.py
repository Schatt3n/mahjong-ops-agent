from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolGateway
from mahjong_agent_runtime.group_chat import BoardEngine, ChannelIdentity, GroupMessage, GroupRoomPolicy
from mahjong_agent_runtime.group_chat.topic_models import PUBLIC_TOPIC_FIELDS
from mahjong_agent_runtime.models import GameStatus, now


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
    assert "1、2026-07-22 15:00" in latest_text
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


@pytest.fixture(params=["memory", "sqlite"])
def self_recruitment_store(request, tmp_path):
    """自招注册同时覆盖两种权威 Store，SQLite 每例独立关闭连接。"""
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "self-recruitment.db")
    yield store
    if request.param == "sqlite":
        store._connection.close()


def _self_recruitment_setup(store, *, policy=True):
    """建立已知身份及显式托管群；登记入口不等于已经发消息。"""
    _identity(store)
    if policy:
        store.upsert_group_room_policy(GroupRoomPolicy(room_id="room-1"))
    messenger = FakeMessenger()
    engine = BoardEngine(store=store, messenger=messenger)
    message = _post("六点371，财敲，无烟，再来一个女生")
    message.sent_at = now()
    return engine, messenger, message


def _self_recruitment_receipt(store, message, *, constraints=None, intent="self_recruitment"):
    """固定已通过语义合同的来源，测试后端登记而非用桩判断自然语言。"""
    if constraints is None:
        constraints = {"game_type": "hangzhou_mahjong", "ruleset": "财敲", "smoking": "no_smoking",
                       "party_size": 3, "preferred_gender": "女", "time_window": {"start": "18:00", "end": None}}
    summary = {key: value for key, value in constraints.items() if key in PUBLIC_TOPIC_FIELDS}
    classification = {
        "intent": intent, "topic_action": "new", "topic_id": None,
        "goal_relation": "none", "task_context_id": None, "source_message_id": message.message_id,
        "constraints": {**constraints, "source_message_id": message.message_id}, "public_summary": summary,
    }
    return store.record_group_topic_message(
        channel=message.channel, room_id=message.room_id, customer_id="customer:external-a",
        message_id=message.message_id, text=message.text, sent_at=message.sent_at, intent=intent,
        public_summary=summary, classification=classification, trace_id="classified-source",
    )


def test_self_recruitment_known_three_registers_board_without_guessing_stake_or_contact(self_recruitment_store):
    store = self_recruitment_store
    engine, messenger, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message)

    game = engine.register_self_recruitment(message, receipt, trace_id="silent-register", tool_gateway=ToolGateway(store))

    assert game is not None and len(store.games) == 1
    assert game.conversation_id == "group:room-1:customer:customer:external-a"
    assert game.organizer_id == "customer:external-a"
    assert game.remaining_seats() == 1 and len(game.participants) == 1
    assert game.participants[0].seat_count == 3 and game.participants[0].anonymous_seat_count == 2
    assert game.requirement["game_variant"] == "caiqiao"
    assert game.requirement["preferred_gender"] == "女"
    assert game.requirement["recruitment_mode"] == "board_only"
    assert game.requirement["start_time"] == "18:00" and game.planned_start_at is None
    assert not {"stake", "base_stake", "stake_range", "date", "start_time_kind"} & game.requirement.keys()
    assert store.game_board_arrangement(game.game_id)["status"] == "scheduled"
    assert [link.room_id for link in store.game_conversation_links(game_id=game.game_id)] == ["room-1"]
    assert {task.task_type for task in store.scheduled_tasks.values()} == {"publish_group_board"}
    assert not store.invite_drafts and not store.outbound_message_drafts and not store.waiting_demands
    assert not messenger.group_messages and not messenger.private_messages
    assert store.get_group_topic_message(message.channel, message.room_id, game.organizer_id, message.message_id) == receipt["message"]

    snapshot = engine.publish(message.room_id, trace_id="next-board")
    assert snapshot is not None and snapshot.items[0].game_id == game.game_id
    assert "18:00" in snapshot.rendered_text and "档位待定" in snapshot.rendered_text and "371" in snapshot.rendered_text
    assert "群友自招（先联系发局人）" in snapshot.rendered_text
    assert "回复编号" not in snapshot.rendered_text
    assert all(secret not in snapshot.rendered_text for secret in ("女", "用户A", "external-a", "customer:"))


def test_self_recruitment_replay_reuses_original_and_does_not_reopen_closed_game(self_recruitment_store):
    store = self_recruitment_store
    engine, messenger, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message)
    gateway = ToolGateway(store)
    game = engine.register_self_recruitment(message, receipt, trace_id="first", tool_gateway=gateway)
    before = {key: task.to_dict() for key, task in store.scheduled_tasks.items()}
    replay = engine.register_self_recruitment(message, receipt, trace_id="retry", tool_gateway=gateway)
    assert replay is not None and replay.game_id == game.game_id and len(store.games) == 1
    assert {key: task.to_dict() for key, task in store.scheduled_tasks.items()} == before
    store.update_game_status(game_id=game.game_id, status="cancelled", reason="客户取消", trace_id="cancel")
    closed_before = {key: task.to_dict() for key, task in store.scheduled_tasks.items()}
    assert engine.register_self_recruitment(message, receipt, trace_id="late-retry", tool_gateway=gateway) is None
    assert len(store.games) == 1 and store.require_game(game.game_id).status == GameStatus.CANCELLED
    assert {key: task.to_dict() for key, task in store.scheduled_tasks.items()} == closed_before
    assert not messenger.group_messages and not messenger.private_messages


def test_new_self_recruitment_source_cannot_duplicate_or_change_existing_game(self_recruitment_store):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    gateway = ToolGateway(store)
    receipt = _self_recruitment_receipt(store, message)
    game = engine.register_self_recruitment(message, receipt, trace_id="first", tool_gateway=gateway)
    second = _post("八点371，国标，有烟", message_id="second-source")
    second.sent_at = message.sent_at + timedelta(seconds=1)
    updated_receipt = _self_recruitment_receipt(store, second, constraints={
        "game_type": "national_standard_mahjong", "party_size": 3,
        "smoking": "smoking", "time_window": {"start": "20:00", "end": None},
    })
    assert engine.register_self_recruitment(second, updated_receipt, trace_id="second", tool_gateway=gateway) is None
    assert len(store.games) == 1 and store.require_game(game.game_id).requirement["game_variant"] == "caiqiao"
    assert store.get_group_topic_message(second.channel, second.room_id, game.organizer_id, second.message_id) is not None


def test_self_recruitment_preference_updates_only_the_bound_original_game(self_recruitment_store):
    store = self_recruitment_store
    engine, messenger, original = _self_recruitment_setup(store)
    gateway = ToolGateway(store)
    tomorrow = (original.sent_at + timedelta(days=1)).replace(hour=19, minute=0)
    receipt = _self_recruitment_receipt(store, original, constraints={
        "game_type": "hangzhou_mahjong", "ruleset": "财敲", "party_size": 3,
        "planned_start_at": tomorrow.isoformat(), "smoking": "no_smoking",
    })
    game = engine.register_self_recruitment(original, receipt, trace_id="register", tool_gateway=gateway)
    topic_id = receipt["topic"]["topic_id"]
    store.bind_group_topic_customer(topic_id=topic_id, customer_id=game.organizer_id,
        conversation_id=game.conversation_id, game_ids=[game.game_id], source_message_id=original.message_id,
        at=original.sent_at)

    def supplement(source_id, gender, offset, *, external_id="external-a"):
        """造同话题的有效分类收据，区分后端可接受的写入来源。"""
        message = _post(f"最好再来个{gender}生", message_id=source_id)
        message.sender_external_id = external_id
        message.sent_at = original.sent_at + timedelta(seconds=offset)
        topic = store.get_group_topic(topic_id)
        decision = {"intent": "self_recruitment", "topic_action": "existing", "topic_id": topic_id,
            "goal_relation": "none", "task_context_id": None, "source_message_id": source_id,
            "constraints": {"preferred_gender": gender}, "public_summary": {}}
        return message, store.record_group_topic_message(channel=message.channel, room_id=message.room_id,
            customer_id=f"customer:{external_id}", message_id=source_id, text=message.text, sent_at=message.sent_at,
            intent=decision["intent"], topic_id=topic_id, expected_version=topic["version"], classification=decision)

    _identity(store, "external-b")
    foreign, foreign_receipt = supplement("foreign-supplement", "男", 1, external_id="external-b")
    assert engine.register_self_recruitment(foreign, foreign_receipt, trace_id="foreign", tool_gateway=gateway) is None

    current, current_receipt = supplement("new-supplement", "女", 3)
    updated = engine.register_self_recruitment(current, current_receipt, trace_id="current", tool_gateway=gateway)
    assert updated is not None and updated.game_id == game.game_id and updated.requirement["preferred_gender"] == "女"
    store.bind_group_topic_customer(topic_id=topic_id, customer_id=updated.organizer_id,
        conversation_id=updated.conversation_id, game_ids=[updated.game_id], source_message_id=current.message_id,
        at=current.sent_at)
    stale, stale_receipt = supplement("stale-supplement", "男", 2)
    assert engine.register_self_recruitment(stale, stale_receipt, trace_id="stale", tool_gateway=gateway) is None
    replay = engine.register_self_recruitment(current, current_receipt, trace_id="retry", tool_gateway=gateway)
    assert replay.game_id == game.game_id and replay.requirement["preferred_gender"] == "女"
    assert replay.planned_start_at == tomorrow and replay.remaining_seats() == 1 and len(store.games) == 1
    assert [transition.trace_id for transition in store.transitions if transition.entity_type == "game_requirement"] == ["current"]
    assert not messenger.group_messages and not messenger.private_messages


@pytest.mark.parametrize("constraints", [
    {"game_type": "hangzhou_mahjong"},
    {"party_size": 3},
    {"game_type": "hangzhou_mahjong", "party_size": 0},
    {"game_type": "hangzhou_mahjong", "party_size": 4},
    {"game_type": "hangzhou_mahjong", "party_size": True},
    {"game_type": "hangzhou_mahjong", "party_size": 3, "planned_start_at": "2026-07-22T18:00:00"},
])
def test_insufficient_self_recruitment_keeps_only_source_observation(self_recruitment_store, constraints):
    store = self_recruitment_store
    engine, messenger, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message, constraints=constraints)
    assert engine.register_self_recruitment(message, receipt, trace_id="insufficient", tool_gateway=ToolGateway(store)) is None
    assert not store.games and not store.scheduled_tasks and not store.outbound_message_drafts
    assert not messenger.group_messages and not messenger.private_messages
    assert store.get_group_topic_message(message.channel, message.room_id, "customer:external-a", message.message_id) == receipt["message"]


@pytest.mark.parametrize("policy_changes", [None, {"managed": False}, {"board_enabled": False}, {"channel": "console"}])
def test_self_recruitment_requires_explicit_managed_board_policy(self_recruitment_store, policy_changes):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store, policy=False)
    if policy_changes is not None:
        store.upsert_group_room_policy(GroupRoomPolicy(room_id=message.room_id, **policy_changes))
    receipt = _self_recruitment_receipt(store, message)
    assert engine.register_self_recruitment(message, receipt, trace_id="policy-blocked", tool_gateway=ToolGateway(store)) is None
    assert not store.games and not store.scheduled_tasks


def test_self_recruitment_registration_does_not_require_outbound_permission(self_recruitment_store):
    store = self_recruitment_store
    engine, messenger, message = _self_recruitment_setup(store)
    store.upsert_group_room_policy(GroupRoomPolicy(room_id=message.room_id, outbound_enabled=False))
    receipt = _self_recruitment_receipt(store, message)
    game = engine.register_self_recruitment(message, receipt, trace_id="outbound-off", tool_gateway=ToolGateway(store))
    assert game is not None and not messenger.group_messages and not messenger.private_messages


@pytest.mark.parametrize("time_fields", [{}, {"start_when_full": True}, {"planned_start_at": "2026-11-10T18:00:00+08:00"}])
def test_self_recruitment_preserves_only_typed_known_start_time(self_recruitment_store, time_fields):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    # 固定 ISO 来源取当前消息后的日期，避免测试寿命受宿主年月日影响。
    if "planned_start_at" in time_fields:
        time_fields = {"planned_start_at": (message.sent_at + timedelta(days=1)).isoformat()}
    receipt = _self_recruitment_receipt(store, message, constraints={
        "game_type": "hangzhou_mahjong", "party_size": 1, **time_fields,
    })
    game = engine.register_self_recruitment(message, receipt, trace_id="typed-time", tool_gateway=ToolGateway(store))
    assert game is not None and game.remaining_seats() == 3
    if not time_fields:
        assert game.planned_start_at is None and "start_time" not in game.requirement
        assert "时间待定" in engine._render_game(1, game)
    elif "planned_start_at" in time_fields:
        assert game.planned_start_at == datetime.fromisoformat(time_fields["planned_start_at"])
    else:
        assert game.planned_start_at is None and game.requirement["start_time_kind"] == "asap_when_full"
        assert "人齐开" in engine._render_game(1, game)


def test_self_recruitment_rejects_unpersisted_or_replaced_receipt_and_gateway_without_write(self_recruitment_store):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message)
    forged = {**receipt, "message": {**receipt["message"], "text": "伪造的另一条来源"}}
    gateway = ToolGateway(store)
    assert engine.register_self_recruitment(message, forged, trace_id="forged", tool_gateway=gateway) is None
    assert engine.register_self_recruitment(message, {}, trace_id="absent", tool_gateway=gateway) is None
    readonly = ToolGateway(store, allowed_execution_modes={"read_only"})
    assert engine.register_self_recruitment(message, receipt, trace_id="readonly", tool_gateway=readonly) is None
    assert not store.games and not store.scheduled_tasks


def test_self_recruitment_requires_current_authenticated_identity(self_recruitment_store):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message)
    message.sender_external_id = "unbound-outsider"
    assert engine.register_self_recruitment(message, receipt, trace_id="unbound", tool_gateway=ToolGateway(store)) is None
    assert not store.games and not store.scheduled_tasks


def test_member_discussion_receipt_cannot_create_self_recruitment_game(self_recruitment_store):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message, constraints={}, intent="member_discussion")
    assert engine.register_self_recruitment(message, receipt, trace_id="discussion", tool_gateway=ToolGateway(store)) is None
    assert not store.games and not store.scheduled_tasks


def test_mixed_board_claim_hint_only_points_to_ordinary_game(self_recruitment_store):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message)
    self_posted = engine.register_self_recruitment(message, receipt, trace_id="self-post", tool_gateway=ToolGateway(store))
    ordinary, _ = store.create_game(
        conversation_id="group:room-1:customer:another-owner", organizer_id="another-owner",
        organizer_name="另一位发起人", requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[], trace_id="ordinary-fixture",
    )
    snapshot = engine.publish(message.room_id, trace_id="mixed-board")
    assert snapshot is not None and len(snapshot.items) == 2
    own_row = next(item for item in snapshot.items if item.game_id == self_posted.game_id)
    ordinary_row = next(item for item in snapshot.items if item.game_id == ordinary.game_id)
    assert "群友自招（先联系发局人）" in own_row.rendered_text
    assert "群友自招" not in ordinary_row.rendered_text
    assert f'普通组局可回复编号认领，如"{ordinary_row.item_no}来"' in snapshot.rendered_text
    assert f'如"{own_row.item_no}来"' not in snapshot.rendered_text


@pytest.mark.parametrize("explicit_end", [True, False])
def test_self_recruitment_explicit_five_hour_interval_controls_lifecycle(self_recruitment_store, explicit_end):
    """五小时不能被默认四小时覆盖；未给结束时刻时不把默认生命周期宣称为约定时长。"""
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    message.text = "七点cq 371，无烟，再来一个女生  5h" if explicit_end else "七点cq 371，无烟，再来一个女生"
    start = (message.sent_at + timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=5)
    constraints = {"game_type": "hangzhou_mahjong", "ruleset": "财敲", "party_size": 3,
                   "smoking": "no_smoking", "preferred_gender": "女", "planned_start_at": start.isoformat()}
    if explicit_end:
        constraints["end_at"] = end.isoformat()
    receipt = _self_recruitment_receipt(store, message, constraints=constraints)
    game = engine.register_self_recruitment(message, receipt, trace_id="five-hour-source", tool_gateway=ToolGateway(store))
    assert game is not None and game.planned_start_at == start
    if explicit_end:
        assert game.requirement["duration_hours"] == 5 and game.requirement["duration_kind"] == "fixed"
        assert end.hour == 0 and end.date() == start.date() + timedelta(days=1)
        assert game.planned_end_at == end and game.expires_at == end
        assert game.requirement["end_at"] == game.requirement["planned_end_at"] == end.isoformat()
    else:
        assert "duration_hours" not in game.requirement and "duration_kind" not in game.requirement
        assert game.planned_end_at == start + timedelta(hours=4)
        assert game.expires_at == game.planned_end_at


@pytest.mark.parametrize("end_offset_hours", [0, -1, 25])
def test_self_recruitment_invalid_explicit_interval_does_not_fall_back_to_default(self_recruitment_store, end_offset_hours):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    start = message.sent_at + timedelta(days=1)
    receipt = _self_recruitment_receipt(store, message, constraints={
        "game_type": "hangzhou_mahjong", "party_size": 3, "planned_start_at": start.isoformat(),
        "end_at": (start + timedelta(hours=end_offset_hours)).isoformat(),
    })
    assert engine.register_self_recruitment(message, receipt, trace_id="invalid-interval", tool_gateway=ToolGateway(store)) is None
    assert not store.games and not store.scheduled_tasks


def test_self_recruitment_end_without_start_does_not_invent_duration(self_recruitment_store):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    end = message.sent_at + timedelta(hours=5)
    receipt = _self_recruitment_receipt(store, message, constraints={
        "game_type": "hangzhou_mahjong", "party_size": 3, "end_at": end.isoformat(),
    })
    game = engine.register_self_recruitment(message, receipt, trace_id="end-only", tool_gateway=ToolGateway(store))
    assert game is not None and game.planned_start_at is None and game.planned_end_at is None
    assert game.requirement["end_at"] == end.isoformat() and "duration_hours" not in game.requirement
    assert game.expires_at == game.created_at + timedelta(hours=4)


@pytest.mark.parametrize("genders", [
    {"required_gender": "女"},
    {"required_gender": "男"},
    {"preferred_gender": "女", "required_gender": "女"},
])
def test_self_recruitment_explicit_required_gender_stays_private(self_recruitment_store, genders):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message, constraints={
        "game_type": "hangzhou_mahjong", "party_size": 3, **genders,
    })
    game = engine.register_self_recruitment(message, receipt, trace_id="required-gender", tool_gateway=ToolGateway(store))
    assert game is not None and all(game.requirement[field] == value for field, value in genders.items())
    snapshot = engine.publish(message.room_id, trace_id="public-board")
    assert all(value not in snapshot.rendered_text for value in ("男", "女", "required_gender", "preferred_gender"))


def test_conflicting_required_and_preferred_gender_keeps_only_observation(self_recruitment_store):
    store = self_recruitment_store
    engine, _, message = _self_recruitment_setup(store)
    receipt = _self_recruitment_receipt(store, message, constraints={
        "game_type": "hangzhou_mahjong", "party_size": 3, "required_gender": "男", "preferred_gender": "女",
    })
    assert engine.register_self_recruitment(message, receipt, trace_id="gender-conflict", tool_gateway=ToolGateway(store)) is None
    assert not store.games and not store.scheduled_tasks
