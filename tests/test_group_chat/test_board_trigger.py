from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mahjong_agent_runtime import HookEvent, InMemoryAgentStore
from mahjong_agent_runtime.group_chat import (
    ChannelSwitch,
    GameConversationLink,
    GroupBoardTrigger,
    GroupRoomPolicy,
)


TZ = ZoneInfo("Asia/Shanghai")
STAMP = datetime(2026, 7, 22, 12, 0, tzinfo=TZ)


def _game(store: InMemoryAgentStore, *, conversation_id: str = "wechaty:contact:user-a"):
    game, transition = store.create_game(
        conversation_id=conversation_id,
        organizer_id="customer-a",
        organizer_name="用户A",
        requirement={
            "requested_game": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "start_time_kind": "asap_when_full",
            "known_player_count": 1,
        },
        known_players=[],
        trace_id="trace-create",
    )
    return game, transition


def _event(tool_name: str, game, transition, *, arguments=None, called=True, error=None) -> HookEvent:
    return HookEvent(
        name="after_tool_execute",
        trace_id="trace-hook",
        payload={
            "call": {"name": tool_name, "arguments": dict(arguments or {})},
            "result": {
                "name": tool_name,
                "called": called,
                "allowed": True,
                "deduplicated": False,
                "error": error,
                "result": {"game": {"game_id": game.game_id}},
                "state_transitions": [transition.to_dict()],
            },
            "conversation_id": game.conversation_id,
            "sender_id": "customer-a",
        },
    )


def test_create_game_after_private_switch_links_room_and_schedules_board(monkeypatch) -> None:
    monkeypatch.setattr("mahjong_agent_runtime.stores.memory.group_chat.now", lambda: STAMP)
    store = InMemoryAgentStore()
    store.upsert_group_room_policy(
        GroupRoomPolicy(room_id="room-1", merge_window_seconds=30)
    )
    store.record_channel_switch(
        ChannelSwitch(
            switch_id="switch-1",
            room_id="room-1",
            customer_id="customer-a",
            source_conversation_id="wechaty:room:room-1",
            source_message_id="message-1",
            private_conversation_id="wechaty:contact:user-a",
            trigger_summary="群里请求组局",
            created_at=STAMP,
            expires_at=STAMP + timedelta(minutes=10),
        )
    )
    game, transition = _game(store)
    trigger = GroupBoardTrigger(store=store, clock=lambda: STAMP)

    trigger(
        _event(
            "create_game",
            game,
            transition,
            arguments={"organizer_id": "customer-a"},
        )
    )

    links = store.game_conversation_links(game_id=game.game_id)
    assert len(links) == 1
    assert links[0].room_id == "room-1"
    assert links[0].link_type == "private_switch_created"
    task = next(item for item in store.scheduled_tasks.values() if item.task_type == "publish_group_board")
    assert task.due_at == STAMP + timedelta(seconds=30)


def test_linked_game_seat_change_schedules_urgent_board_refresh() -> None:
    store = InMemoryAgentStore()
    store.upsert_group_room_policy(
        GroupRoomPolicy(room_id="room-1", merge_window_seconds=30)
    )
    game, transition = _game(store)
    store.link_game_conversation(
        GameConversationLink(
            link_id="link-1",
            game_id=game.game_id,
            conversation_id=game.conversation_id,
            room_id="room-1",
            customer_id="customer-a",
            link_type="origin",
        )
    )
    trigger = GroupBoardTrigger(store=store, clock=lambda: STAMP)

    trigger(
        _event(
            "record_candidate_reply",
            game,
            transition,
            arguments={"status": "confirmed"},
        )
    )

    task = next(item for item in store.scheduled_tasks.values() if item.task_type == "publish_group_board")
    assert task.due_at == STAMP


def test_failed_tool_result_does_not_schedule_board_refresh() -> None:
    store = InMemoryAgentStore()
    store.upsert_group_room_policy(GroupRoomPolicy(room_id="room-1"))
    game, transition = _game(store)
    store.link_game_conversation(
        GameConversationLink(
            link_id="link-1",
            game_id=game.game_id,
            conversation_id=game.conversation_id,
            room_id="room-1",
            customer_id="customer-a",
            link_type="origin",
        )
    )
    trigger = GroupBoardTrigger(store=store, clock=lambda: STAMP)

    before = {key: value.to_dict() for key, value in store.scheduled_tasks.items()}
    transition_count = len(store.transitions)

    trigger(_event("join_game", game, transition, called=False, error="seat unavailable"))

    assert {key: value.to_dict() for key, value in store.scheduled_tasks.items()} == before
    assert len(store.transitions) == transition_count
