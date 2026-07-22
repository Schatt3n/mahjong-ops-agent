from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import threading
from zoneinfo import ZoneInfo

from mahjong_agent_runtime import AgentContextBuilder, AgentRuntimeResult, InMemoryAgentStore, SQLiteAgentStore, ToolGateway, UserMessage
from mahjong_agent_runtime.group_chat import (
    BoardEngine,
    ChannelIdentity,
    ClaimHandler,
    GroupMessage,
    GroupMessageHandler,
    L1RuleEngine,
    L2IntentRouter,
    NotifyDispatcher,
)


TZ = ZoneInfo("Asia/Shanghai")


class FakeMessenger:
    def __init__(self) -> None:
        self.group_messages: list[tuple[str, str, str]] = []
        self.private_messages: list[tuple[str, str, str]] = []

    def send_group_message(self, room_id: str, text: str, *, metadata=None) -> str:
        message_id = f"group-{len(self.group_messages) + 1}"
        self.group_messages.append((room_id, text, message_id))
        return message_id

    def send_private_message(self, external_user_id: str, text: str, *, metadata=None) -> str:
        message_id = f"private-{len(self.private_messages) + 1}"
        self.private_messages.append((external_user_id, text, message_id))
        return message_id


class FakeRuntime:
    def __init__(self) -> None:
        self.user_messages = []
        self.triggers = []

    def handle_user_message(self, message, *, trace_id=None):
        self.user_messages.append(message)
        return AgentRuntimeResult(trace_id=trace_id or "trace", conversation_id=message.conversation_id, final_reply="还差一位")

    def handle_system_trigger(self, trigger, *, trace_id=None):
        self.triggers.append(trigger)
        return AgentRuntimeResult(trace_id=trace_id or "trace", conversation_id=trigger.conversation_id, final_reply="占上了")


def _identity(store, external: str, *, friend: bool, public_name: str) -> ChannelIdentity:
    identity = ChannelIdentity(
        channel="wechaty",
        external_user_id=external,
        customer_id=f"customer:{external}",
        public_name=public_name,
        private_conversation_id=f"wechaty:contact:{external}",
        can_private_message=friend,
        is_friend=friend,
    )
    store.upsert_channel_identity(identity)
    return identity


def _group_message(text: str, sender: str, *, message_id: str, quote: str | None = None) -> GroupMessage:
    return GroupMessage(
        room_id="room-1",
        conversation_id="wechaty:room:room-1",
        sender_external_id=sender,
        sender_name=sender,
        text=text,
        message_id=message_id,
        quoted_message_id=quote,
        sent_at=datetime(2026, 7, 22, 12, 0, tzinfo=TZ),
    )


def _stack(store):
    messenger = FakeMessenger()
    runtime = FakeRuntime()
    clock = lambda: datetime(2026, 7, 22, 12, 0, tzinfo=TZ)
    board = BoardEngine(store=store, messenger=messenger, clock=clock)
    notifier = NotifyDispatcher(store=store, messenger=messenger, runtime=runtime)
    claims = ClaimHandler(board_engine=board, store=store, notify_dispatcher=notifier, clock=clock)
    handler = GroupMessageHandler(
        rule_engine=L1RuleEngine(bot_id="bot"),
        intent_router=L2IntentRouter(store, ack_picker=lambda _: "私聊回你"),
        board_engine=board,
        claim_handler=claims,
        notify_dispatcher=notifier,
        messenger=messenger,
        runtime=runtime,
        clock=clock,
    )
    return messenger, runtime, board, claims, handler


def _open_game(store, board: BoardEngine):
    _identity(store, "poster", friend=True, public_name="发布者")
    game = board.import_game_from_post(
        _group_message("14:00 0.5 无烟 371", "poster", message_id="post"),
        trace_id="trace-import",
    )
    board.publish("room-1", trace_id="trace-board")
    snapshot = store.get_latest_board_snapshot("room-1")
    return game, snapshot


def test_friend_claims_open_seat_and_receives_private_confirmation() -> None:
    store = InMemoryAgentStore()
    messenger, runtime, board, claims, _ = _stack(store)
    game, snapshot = _open_game(store, board)
    identity = _identity(store, "friend", friend=True, public_name="好友")

    result = claims.process_claim(
        _group_message("1来", "friend", message_id="claim", quote=snapshot.external_message_id),
        1,
        trace_id="trace-claim",
    )

    assert result.status == "claimed"
    assert any(item.customer_id == identity.customer_id for item in store.require_game(game.game_id).participants)
    assert runtime.triggers[-1].conversation_id == identity.private_conversation_id
    assert messenger.private_messages[-1][1] == "占上了"


def test_customer_notification_context_uses_public_projection_only() -> None:
    store = InMemoryAgentStore()
    _, runtime, board, claims, _ = _stack(store)
    game, snapshot = _open_game(store, board)
    identity = _identity(store, "friend", friend=True, public_name="好友")

    claims.process_claim(
        _group_message("1来", "friend", message_id="claim", quote=snapshot.external_message_id),
        1,
        trace_id="trace-claim",
    )

    serialized = json.dumps([trigger.payload for trigger in runtime.triggers], ensure_ascii=False)
    assert game.organizer_id not in serialized
    assert game.organizer_name not in serialized
    assert "requesting_party" not in serialized
    assert "seat_claims" not in serialized
    assert identity.customer_id not in serialized


def test_non_friend_claims_open_seat_and_gets_minimal_public_notifications() -> None:
    store = InMemoryAgentStore()
    messenger, _, board, claims, _ = _stack(store)
    _, snapshot = _open_game(store, board)
    _identity(store, "stranger", friend=False, public_name="群昵称")

    result = claims.process_claim(
        _group_message("1来", "stranger", message_id="claim", quote=snapshot.external_message_id),
        1,
        trace_id="trace-claim",
    )

    assert result.status == "claimed"
    public_texts = [item[1] for item in messenger.group_messages]
    assert "@群昵称 占上了" in public_texts
    assert "@群昵称 人齐了，14:00开" in public_texts


def test_game_full_notifies_non_friend_with_minimal_public_message() -> None:
    store = InMemoryAgentStore()
    messenger, _, board, claims, _ = _stack(store)
    _identity(store, "poster", friend=True, public_name="发布者")
    board.import_game_from_post(
        _group_message("14:00 0.5 无烟 272", "poster", message_id="post-272"),
        trace_id="trace-import",
    )
    board.publish("room-1", trace_id="trace-board")
    snapshot = store.get_latest_board_snapshot("room-1")
    _identity(store, "friend", friend=True, public_name="好友")
    _identity(store, "stranger", friend=False, public_name="群昵称")

    claims.process_claim(
        _group_message("1来", "friend", message_id="claim-friend", quote=snapshot.external_message_id),
        1,
        trace_id="trace-friend",
    )
    result = claims.process_claim(
        _group_message("1来", "stranger", message_id="claim-stranger", quote=snapshot.external_message_id),
        1,
        trace_id="trace-stranger",
    )

    assert result.status == "claimed"
    assert ("room-1", "@群昵称 人齐了，14:00开") in [item[:2] for item in messenger.group_messages]


def test_full_and_duplicate_claims_are_rejected_or_deduplicated() -> None:
    store = InMemoryAgentStore()
    _, _, board, claims, _ = _stack(store)
    _, snapshot = _open_game(store, board)
    _identity(store, "friend", friend=True, public_name="好友")
    message = _group_message("1来", "friend", message_id="claim", quote=snapshot.external_message_id)

    first = claims.process_claim(message, 1, trace_id="trace-first")
    second = claims.process_claim(message, 1, trace_id="trace-second")

    assert first.status == "claimed"
    assert second.status == "claimed"
    assert second.deduplicated is True


def test_two_people_racing_for_last_seat_only_one_succeeds(tmp_path) -> None:
    store = SQLiteAgentStore(tmp_path / "claim-race.sqlite3")
    _, _, board, claims, _ = _stack(store)
    _, snapshot = _open_game(store, board)
    _identity(store, "one", friend=True, public_name="一号")
    _identity(store, "two", friend=True, public_name="二号")
    barrier = threading.Barrier(2)

    def run(external: str):
        barrier.wait()
        return claims.process_claim(
            _group_message("1来", external, message_id=f"claim-{external}", quote=snapshot.external_message_id),
            1,
            trace_id=f"trace-{external}",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["one", "two"]))

    assert sorted(item.status for item in results) == ["claimed", "rejected"]


def test_time_conflict_rejects_claim() -> None:
    store = InMemoryAgentStore()
    _, _, board, claims, _ = _stack(store)
    _, snapshot = _open_game(store, board)
    identity = _identity(store, "busy", friend=True, public_name="忙碌用户")
    store.create_game(
        conversation_id="private-busy",
        organizer_id=identity.customer_id,
        organizer_name="忙碌用户",
        requirement={
            "start_time_kind": "scheduled",
            "planned_start_at": datetime(2026, 7, 22, 14, 30, tzinfo=TZ).isoformat(),
            "duration_hours": 4,
            "known_player_count": 1,
        },
        known_players=[],
        trace_id="trace-busy-game",
    )

    result = claims.process_claim(
        _group_message("1来", "busy", message_id="claim-busy", quote=snapshot.external_message_id),
        1,
        trace_id="trace-claim",
    )

    assert result.status == "rejected"
    assert result.reason == "time_conflict"


def test_private_switch_carries_only_sender_request_and_profile() -> None:
    store = InMemoryAgentStore()
    messenger, runtime, _, _, handler = _stack(store)
    _identity(store, "friend", friend=True, public_name="好友")

    result = handler.handle(
        _group_message("帮我约个今晚0.5无烟的", "friend", message_id="request"),
        trace_id="trace-switch",
    )

    assert result.action == "private_switch"
    assert messenger.group_messages[-1][1] == "私聊回你"
    trigger = runtime.triggers[-1]
    assert trigger.payload["user_original_text"] == "帮我约个今晚0.5无烟的"
    serialized = json.dumps(trigger.payload, ensure_ascii=False)
    assert "private_remark" not in serialized
    assert "群里其他人的话" not in serialized


def test_group_agent_loop_receives_structured_reply_constraints() -> None:
    store = InMemoryAgentStore()
    messenger, runtime, _, _, handler = _stack(store)
    _identity(store, "friend", friend=True, public_name="好友")

    result = handler.handle(_group_message("现在齐了吗", "friend", message_id="query"), trace_id="trace-query")

    assert result.action == "agent_loop"
    assert runtime.user_messages[-1].metadata["reply_constraints"] == {
        "max_length": 20,
        "no_private_info": True,
    }
    assert messenger.group_messages[-1][1] == "还差一位"


def test_context_builder_accepts_only_backend_group_reply_constraints() -> None:
    store = InMemoryAgentStore()
    builder = AgentContextBuilder(store, ToolGateway(store))
    message = UserMessage(
        conversation_id="group:room-1:customer:customer-a",
        sender_id="customer-a",
        sender_name="用户A",
        text="齐了吗",
        metadata={
            "source": "group",
            "reply_constraints": {"max_length": 20, "no_private_info": True},
        },
    )

    built = builder.build(message, trace_id="trace-context")

    assert built.payload["reply_constraints"] == {"max_length": 20, "no_private_info": True}
    assert "不得超过 20 个中文字符" in built.messages[1]["content"]
    assert "其他客户" in built.messages[1]["content"]


def test_group_context_includes_public_room_board_without_participant_identity() -> None:
    store = InMemoryAgentStore()
    _, _, board, _, _ = _stack(store)
    game, _ = _open_game(store, board)
    # This test exercises visibility, not lifecycle expiry; keep the fixture active independently of wall time.
    game.expires_at = datetime(2099, 7, 22, 18, 0, tzinfo=TZ)
    builder = AgentContextBuilder(store, ToolGateway(store))
    message = UserMessage(
        conversation_id="group:room-1:customer:outsider",
        sender_id="outsider",
        sender_name="群友",
        text="现在齐了吗",
        metadata={
            "source": "group",
            "room_id": "room-1",
            "reply_constraints": {"max_length": 20, "no_private_info": True},
        },
    )

    built = builder.build(message, trace_id="trace-room-context")

    assert built.payload["group_room_board_games"][0]["game_id"] == game.game_id
    assert built.payload["group_room_board_games"][0]["seat_code"] == "371"
    serialized = json.dumps(built.payload["group_room_board_games"], ensure_ascii=False)
    assert game.organizer_id not in serialized
    assert game.organizer_name not in serialized
    assert "requesting_party" not in serialized
    assert "seat_claims" not in serialized


def test_group_reply_after_private_switch_is_redirected_without_leaking_context() -> None:
    store = InMemoryAgentStore()
    messenger, _, _, _, handler = _stack(store)
    _identity(store, "friend", friend=True, public_name="好友")
    handler.handle(_group_message("帮我组个局", "friend", message_id="request"), trace_id="trace-switch")

    result = handler.handle(_group_message("0.5无烟", "friend", message_id="continued"), trace_id="trace-leak")

    assert result.action == "redirect_private"
    assert messenger.group_messages[-1][1] == "@好友 私聊回你了，看下哈"
