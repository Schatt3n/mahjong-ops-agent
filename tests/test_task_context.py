from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from mahjong_agent_runtime.context import AgentContextBuilder
from mahjong_agent_runtime.models import CustomerProfile, CustomerRelationship, UserMessage, now
from mahjong_agent_runtime.sqlite_store import SQLiteAgentStore
from mahjong_agent_runtime.store import InMemoryAgentStore
from mahjong_agent_runtime.task_context import TaskContextManager
from mahjong_agent_runtime.tools import ToolGateway


def seed_finished_morning_task(store, *, conversation_id: str, customer_id: str):
    morning = now() - timedelta(hours=8)
    manager = TaskContextManager(store)
    first = manager.prepare(
        UserMessage(
            conversation_id=conversation_id,
            sender_id=customer_id,
            sender_name="A",
            text="上午十点0.5无烟，帮我组一个",
            sent_at=morning,
        ),
        trace_id="trace_morning_context",
    )
    store.append_user_turn(
        UserMessage(
            conversation_id=conversation_id,
            sender_id=customer_id,
            sender_name="A",
            text="上午十点0.5无烟，帮我组一个",
            sent_at=morning,
        ),
        "trace_morning_user",
    )
    store.append_assistant_turn(conversation_id, "好，我帮你问问。", "trace_morning_reply")
    store.upsert_conversation_checkpoint(
        conversation_id=conversation_id,
        summary="上午十点0.5无烟局正在组。",
        facts={"start_time": "10:00", "stake": "0.5", "smoke_preference": "no_smoke"},
        open_questions=[],
        trace_id="trace_morning_checkpoint",
    )
    memory, _ = store.record_task_memory(
        conversation_id=conversation_id,
        customer_id=customer_id,
        memory_type="requirement",
        field="smoke_preference",
        value="no_smoke",
        evidence="这次无烟",
        confidence=0.99,
        trace_id="trace_morning_memory",
    )
    game, _ = store.create_game(
        conversation_id=conversation_id,
        organizer_id=customer_id,
        organizer_name="A",
        requirement={"game_type": "hangzhou_mahjong", "stake": "0.5", "start_time": "10:00"},
        known_players=[],
        trace_id="trace_morning_game",
    )
    store.update_game_status(
        game_id=game.game_id,
        status="inviting",
        reason="started_inviting",
        trace_id="trace_morning_inviting",
    )
    store.update_game_status(
        game_id=game.game_id,
        status="finished",
        reason="morning_game_completed",
        trace_id="trace_morning_finished",
    )
    return first.context, memory, game


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_finished_morning_task_is_excluded_from_afternoon_context(tmp_path: Path, backend: str) -> None:
    store = (
        InMemoryAgentStore()
        if backend == "memory"
        else SQLiteAgentStore(tmp_path / "task_context.sqlite3")
    )
    conversation_id = "wechat:A"
    customer_id = "A"
    store.upsert_customer(
        CustomerProfile(
            customer_id=customer_id,
            display_name="A",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5"],
        )
    )
    store.upsert_customer_relationship(
        CustomerRelationship(
            customer_a_id=customer_id,
            customer_b_id="long_term_avoid",
            avoid_playing=True,
            notes="已审核的长期关系约束",
        )
    )
    morning_context, morning_memory, _ = seed_finished_morning_task(
        store,
        conversation_id=conversation_id,
        customer_id=customer_id,
    )

    afternoon_message = UserMessage(
        conversation_id=conversation_id,
        sender_id=customer_id,
        sender_name="A",
        text="下午再帮我组一场",
        sent_at=now() + timedelta(hours=1),
    )
    prepared = TaskContextManager(store).prepare(afternoon_message, trace_id="trace_afternoon_context")
    store.append_user_turn(afternoon_message, "trace_afternoon_user")
    built = AgentContextBuilder(store, ToolGateway(store)).build(
        afternoon_message,
        trace_id="trace_afternoon_user",
    )

    assert prepared.reset_applied is True
    assert prepared.reason == "previous_related_game_terminal"
    assert prepared.context.task_context_id != morning_context.task_context_id
    assert store.task_contexts[morning_context.task_context_id].status == "closed"
    assert store.task_memories[morning_memory.memory_id].status == "archived"
    assert [item["content"] for item in built.payload["recent_conversation"]] == ["下午再帮我组一场"]
    assert built.payload["conversation_checkpoint"] is None
    assert built.payload["task_memories"] == []
    assert built.payload["active_games"] == []
    assert built.payload["task_context_window"]["task_context_id"] == prepared.context.task_context_id
    assert built.payload["sender_profile"]["preferred_stakes"] == ["0.5"]
    assert store.relationship_between(customer_id, "long_term_avoid").avoid_playing is True
    assert built.audit["omitted_before_task_context"] >= 2
    assert built.audit["checkpoint_excluded_by_task_context"] is True


def test_active_game_keeps_context_even_after_long_idle_gap() -> None:
    store = InMemoryAgentStore()
    manager = TaskContextManager(store, idle_reset_seconds=60)
    morning = now() - timedelta(hours=3)
    morning_message = UserMessage(
        conversation_id="active_game_chat",
        sender_id="A",
        sender_name="A",
        text="晚上七点帮我组一个",
        sent_at=morning,
    )
    first = manager.prepare(morning_message, trace_id="trace_active_morning")
    store.append_user_turn(morning_message, "trace_active_morning")
    store.create_game(
        conversation_id="active_game_chat",
        organizer_id="A",
        organizer_name="A",
        requirement={"game_type": "hangzhou_mahjong", "stake": "0.5", "start_time": "19:00"},
        known_players=[],
        trace_id="trace_active_game",
    )

    later_message = UserMessage(
        conversation_id="active_game_chat",
        sender_id="A",
        sender_name="A",
        text="现在几个人了",
        sent_at=now(),
    )
    later = manager.prepare(later_message, trace_id="trace_active_later")

    assert later.reset_applied is False
    assert later.context.task_context_id == first.context.task_context_id
    assert later.reason == "continue_current_task"


def test_idle_conversation_without_active_game_starts_new_task_context() -> None:
    store = InMemoryAgentStore()
    manager = TaskContextManager(store, idle_reset_seconds=60)
    first_message = UserMessage(
        conversation_id="idle_chat",
        sender_id="A",
        sender_name="A",
        text="我先看看",
        sent_at=now() - timedelta(minutes=10),
    )
    first = manager.prepare(first_message, trace_id="trace_idle_first")
    store.append_user_turn(first_message, "trace_idle_first")

    later = manager.prepare(
        UserMessage(
            conversation_id="idle_chat",
            sender_id="A",
            sender_name="A",
            text="下午帮我组一个",
            sent_at=now(),
        ),
        trace_id="trace_idle_later",
    )

    assert later.reset_applied is True
    assert later.reason == "idle_task_timeout"
    assert later.context.task_context_id != first.context.task_context_id


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_temporary_candidate_exclusion_does_not_leak_into_next_task(tmp_path: Path, backend: str) -> None:
    store = (
        InMemoryAgentStore()
        if backend == "memory"
        else SQLiteAgentStore(tmp_path / "task_candidate_isolation.sqlite3")
    )
    conversation_id = "wechat:A:candidate"
    manager = TaskContextManager(store)
    morning_message = UserMessage(
        conversation_id=conversation_id,
        sender_id="A",
        sender_name="A",
        text="这一局不和B打",
        sent_at=now() - timedelta(hours=6),
    )
    manager.prepare(morning_message, trace_id="trace_candidate_morning")
    store.append_user_turn(morning_message, "trace_candidate_morning")
    memory, _ = store.record_task_memory(
        conversation_id=conversation_id,
        customer_id="A",
        memory_type="relationship",
        field="avoid_playing",
        value=True,
        target_customer_id="B",
        evidence="这一局不和B打",
        confidence=0.99,
        trace_id="trace_candidate_memory",
    )
    assert store.task_memory_excluded_customer_ids(conversation_id, ["A"]) == ["B"]

    game, _ = store.create_game(
        conversation_id=conversation_id,
        organizer_id="A",
        organizer_name="A",
        requirement={"game_type": "hangzhou_mahjong", "stake": "0.5", "start_mode": "asap_when_full"},
        known_players=[],
        trace_id="trace_candidate_game",
    )
    store.update_game_status(
        game_id=game.game_id,
        status="cancelled",
        reason="morning_request_closed",
        trace_id="trace_candidate_cancelled",
    )

    afternoon = UserMessage(
        conversation_id=conversation_id,
        sender_id="A",
        sender_name="A",
        text="下午再组一局",
        sent_at=now(),
    )
    prepared = manager.prepare(afternoon, trace_id="trace_candidate_afternoon")

    assert prepared.reset_applied is True
    assert store.task_memories[memory.memory_id].status == "archived"
    assert store.task_memory_excluded_customer_ids(conversation_id, ["A"]) == []
