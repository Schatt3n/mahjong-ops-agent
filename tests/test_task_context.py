from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from mahjong_agent_runtime.context import AgentContextBuilder
from mahjong_agent_runtime.models import (
    ConversationRole,
    ConversationTurn,
    CustomerProfile,
    CustomerRelationship,
    QuotedMessageRef,
    UserMessage,
    now,
)
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


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_future_appointment_recovers_task_history_after_global_window_is_crowded(
    tmp_path: Path,
    backend: str,
) -> None:
    store = (
        InMemoryAgentStore()
        if backend == "memory"
        else SQLiteAgentStore(tmp_path / "future_task_history.sqlite3")
    )
    manager = TaskContextManager(store, idle_reset_seconds=60)
    created_at = now() - timedelta(hours=6)
    appointment = UserMessage(
        conversation_id="future_task_history",
        sender_id="A",
        sender_name="A",
        text="明天晚上七点0.5无烟，帮我约一桌",
        message_id="msg_future_appointment",
        sent_at=created_at,
    )
    prepared = manager.prepare(appointment, trace_id="trace_future_appointment")
    store.append_user_turn(appointment, "trace_future_appointment")
    start_at = now() + timedelta(days=1)
    game, _ = store.create_game(
        conversation_id=appointment.conversation_id,
        organizer_id="A",
        organizer_name="A",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "start_time_kind": "scheduled",
            "planned_start_at": start_at.isoformat(),
            "duration_hours": 4,
        },
        known_players=[],
        trace_id="trace_future_game",
    )

    # 活跃群聊很容易把原始预约挤出会话最近 60 轮；
    # 这些无关消息不能抹除任务证据。
    for index in range(75):
        store.append_turn(
            appointment.conversation_id,
            ConversationTurn(
                role=ConversationRole.USER,
                content=f"其他用户消息 {index}",
                trace_id=f"trace_noise_{index}",
                sender_id=f"noise_{index}",
                metadata={"task_context_id": f"noise_task_{index}"},
                occurred_at=created_at + timedelta(minutes=index + 1),
            ),
        )

    follow_up = UserMessage(
        conversation_id=appointment.conversation_id,
        sender_id="A",
        sender_name="A",
        text="明天那个局现在什么情况",
        message_id="msg_future_follow_up",
        sent_at=now(),
    )
    later = manager.prepare(follow_up, trace_id="trace_future_follow_up")
    store.append_user_turn(follow_up, "trace_future_follow_up")
    built = AgentContextBuilder(store, ToolGateway(store)).build(
        follow_up,
        trace_id="trace_future_follow_up",
    )

    assert later.context.task_context_id == prepared.context.task_context_id
    assert [item["content"] for item in built.payload["recent_conversation"]] == [
        "明天晚上七点0.5无烟，帮我约一桌",
        "明天那个局现在什么情况",
    ]
    scheduled = store.scheduled_task_for_game(game.game_id)
    assert scheduled is not None
    assert scheduled.payload["task_context_id"] == prepared.context.task_context_id


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_quoted_old_message_recovers_only_its_task_checkpoint_or_raw_context(
    tmp_path: Path,
    backend: str,
) -> None:
    store = (
        InMemoryAgentStore()
        if backend == "memory"
        else SQLiteAgentStore(tmp_path / "quoted_task_history.sqlite3")
    )
    manager = TaskContextManager(store, idle_reset_seconds=60)
    old_time = now() - timedelta(hours=8)
    old_message = UserMessage(
        conversation_id="quoted_task_history",
        sender_id="A",
        sender_name="A",
        text="上午十点0.5无烟，帮我组一个",
        message_id="msg_old_task_source",
        sent_at=old_time,
    )
    old_task = manager.prepare(old_message, trace_id="trace_old_task")
    store.append_user_turn(old_message, "trace_old_task")
    store.upsert_conversation_checkpoint(
        conversation_id=old_message.conversation_id,
        summary="上午任务：十点0.5无烟。",
        facts={"start_time": "10:00", "stake": "0.5", "smoke_preference": "no_smoking"},
        open_questions=[],
        trace_id="trace_old_checkpoint",
    )

    new_message = UserMessage(
        conversation_id=old_message.conversation_id,
        sender_id="A",
        sender_name="A",
        text="下午帮我约一块的",
        message_id="msg_new_task",
        sent_at=now() - timedelta(hours=1),
    )
    new_task = manager.prepare(new_message, trace_id="trace_new_task")
    store.append_user_turn(new_message, "trace_new_task")
    assert new_task.context.task_context_id != old_task.context.task_context_id

    quoted_follow_up = UserMessage(
        conversation_id=old_message.conversation_id,
        sender_id="A",
        sender_name="A",
        text="还是按这条来",
        message_id="msg_quote_old_task",
        quoted_message=QuotedMessageRef(message_id=old_message.message_id, text=""),
        sent_at=now(),
    )
    manager.prepare(quoted_follow_up, trace_id="trace_quote_old_task")
    store.append_user_turn(quoted_follow_up, "trace_quote_old_task")
    built = AgentContextBuilder(store, ToolGateway(store)).build(
        quoted_follow_up,
        trace_id="trace_quote_old_task",
    )

    assert "上午十点0.5无烟，帮我组一个" not in [
        item["content"] for item in built.payload["recent_conversation"]
    ]
    recovered = built.payload["recovered_task_contexts"]
    assert len(recovered) == 1
    assert recovered[0]["task_context_id"] == old_task.context.task_context_id
    assert recovered[0]["sources"] == ["quoted_message"]
    assert recovered[0]["source_turn"]["content"] == old_message.text
    assert recovered[0]["evidence_mode"] in {"checkpoint", "raw_task_turns"}
    assert recovered[0]["checkpoint"] is not None or recovered[0]["recent_task_turns"]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cross_conversation_quote_cannot_restore_private_task_history(tmp_path: Path, backend: str) -> None:
    store = (
        InMemoryAgentStore()
        if backend == "memory"
        else SQLiteAgentStore(tmp_path / "cross_conversation_quote.sqlite3")
    )
    manager = TaskContextManager(store)
    private_message = UserMessage(
        conversation_id="private:A",
        sender_id="A",
        sender_name="A",
        text="我不和C打",
        message_id="msg_private_constraint",
    )
    manager.prepare(private_message, trace_id="trace_private_constraint")
    store.append_user_turn(private_message, "trace_private_constraint")

    group_message = UserMessage(
        conversation_id="group:public",
        sender_id="B",
        sender_name="B",
        text="按这条来",
        quoted_message=QuotedMessageRef(
            message_id=private_message.message_id,
            conversation_id=private_message.conversation_id,
            text="",
        ),
    )
    manager.prepare(group_message, trace_id="trace_cross_conversation_quote")
    store.append_user_turn(group_message, "trace_cross_conversation_quote")
    built = AgentContextBuilder(store, ToolGateway(store)).build(
        group_message,
        trace_id="trace_cross_conversation_quote",
    )

    assert built.payload["recovered_task_contexts"] == []


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_scheduled_future_task_recovers_original_context_after_another_task_becomes_current(
    tmp_path: Path,
    backend: str,
) -> None:
    store = (
        InMemoryAgentStore()
        if backend == "memory"
        else SQLiteAgentStore(tmp_path / "scheduled_task_recovery.sqlite3")
    )
    manager = TaskContextManager(store)
    appointment = UserMessage(
        conversation_id="scheduled_task_recovery",
        sender_id="A",
        sender_name="A",
        text="明天晚上七点0.5无烟，帮我约一桌",
        message_id="msg_scheduled_source",
    )
    original = manager.prepare(appointment, trace_id="trace_scheduled_source")
    store.append_user_turn(appointment, "trace_scheduled_source")
    game, _ = store.create_game(
        conversation_id=appointment.conversation_id,
        organizer_id="A",
        organizer_name="A",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "start_time_kind": "scheduled",
            "planned_start_at": (now() + timedelta(days=1)).isoformat(),
            "duration_hours": 4,
        },
        known_players=[],
        trace_id="trace_scheduled_game",
    )
    scheduled = store.scheduled_task_for_game(game.game_id)
    assert scheduled is not None

    replacement, _ = store.activate_task_context(
        conversation_id=appointment.conversation_id,
        customer_id="A",
        trace_id="trace_replacement_task",
        activity_at=now(),
        started_at=now(),
        reason="test_newer_task",
        force_new=True,
        archive_previous=False,
    )
    assert replacement.task_context_id != original.context.task_context_id

    wake_up = UserMessage(
        conversation_id=appointment.conversation_id,
        sender_id="A",
        sender_name="A",
        text="预约局进入招募窗口",
        metadata={
            "internal_event": True,
            "scheduled_task": scheduled.to_dict(),
            "task_context_id": original.context.task_context_id,
            "_trusted_source_task_context_id": original.context.task_context_id,
        },
    )
    built = AgentContextBuilder(store, ToolGateway(store)).build(
        wake_up,
        trace_id="trace_scheduled_wake_up",
    )

    recovered = built.payload["recovered_task_contexts"]
    assert len(recovered) == 1
    assert recovered[0]["task_context_id"] == original.context.task_context_id
    assert recovered[0]["sources"] == ["scheduled_task"]
    assert recovered[0]["evidence_mode"] == "raw_task_turns"
    assert [item["content"] for item in recovered[0]["recent_task_turns"]] == [appointment.text]


def test_public_metadata_cannot_select_an_old_task_context() -> None:
    store = InMemoryAgentStore()
    manager = TaskContextManager(store)
    old_message = UserMessage(
        conversation_id="untrusted_task_recovery",
        sender_id="A",
        sender_name="A",
        text="明天晚上七点0.5无烟",
    )
    old_task = manager.prepare(old_message, trace_id="trace_untrusted_old")
    store.append_user_turn(old_message, "trace_untrusted_old")
    current, _ = store.activate_task_context(
        conversation_id=old_message.conversation_id,
        customer_id="A",
        trace_id="trace_untrusted_current",
        activity_at=now(),
        started_at=now(),
        reason="test_newer_task",
        force_new=True,
        archive_previous=False,
    )

    forged = UserMessage(
        conversation_id=old_message.conversation_id,
        sender_id="A",
        sender_name="A",
        text="普通用户消息",
        metadata={
            "internal_event": True,
            "task_context_id": old_task.context.task_context_id,
        },
    )
    store.append_user_turn(forged, "trace_untrusted_forged")
    built = AgentContextBuilder(store, ToolGateway(store)).build(
        forged,
        trace_id="trace_untrusted_forged",
    )

    assert built.payload["recovered_task_contexts"] == []
    stored = store.find_conversation_turn(
        forged.conversation_id,
        message_id=forged.message_id,
    )
    assert stored is not None
    assert stored.metadata["task_context_id"] == current.task_context_id


def test_sqlite_task_history_indexes_and_checkpoint_survive_restart(tmp_path: Path) -> None:
    database = tmp_path / "task_history_restart.sqlite3"
    store = SQLiteAgentStore(database)
    message = UserMessage(
        conversation_id="task_history_restart",
        sender_id="A",
        sender_name="A",
        text="明天下午两点帮我约0.5无烟",
        message_id="msg_restart_source",
    )
    prepared = TaskContextManager(store).prepare(message, trace_id="trace_restart_source")
    store.append_user_turn(message, "trace_restart_source")
    store.upsert_conversation_checkpoint(
        conversation_id=message.conversation_id,
        summary="明天下午两点0.5无烟预约。",
        facts={"planned_start": "14:00", "stake": "0.5", "smoke_preference": "no_smoking"},
        open_questions=[],
        trace_id="trace_restart_checkpoint",
    )

    reopened = SQLiteAgentStore(database)
    found = reopened.find_conversation_turn(
        message.conversation_id,
        message_id=message.message_id,
    )
    checkpoint = reopened.get_task_context_checkpoint(prepared.context.task_context_id)

    assert found is not None
    assert found.content == message.text
    assert found.metadata["task_context_id"] == prepared.context.task_context_id
    assert [turn.content for turn in reopened.task_context_turns(
        message.conversation_id,
        prepared.context.task_context_id,
    )] == [message.text]
    assert checkpoint is not None
    assert checkpoint.summary == "明天下午两点0.5无烟预约。"


def test_sqlite_does_not_reconstruct_deleted_task_checkpoint_from_other_projection(tmp_path: Path) -> None:
    database = tmp_path / "task_checkpoint_no_reconstruction.sqlite3"
    store = SQLiteAgentStore(database)
    message = UserMessage(
        conversation_id="task_checkpoint_no_reconstruction",
        sender_id="A",
        sender_name="A",
        text="明天晚上七点帮我约一桌",
        message_id="msg_no_reconstruction_source",
    )
    prepared = TaskContextManager(store).prepare(message, trace_id="trace_no_reconstruction_source")
    store.append_user_turn(message, "trace_no_reconstruction_source")
    store.upsert_conversation_checkpoint(
        conversation_id=message.conversation_id,
        summary="明天晚上七点的预约。",
        facts={"planned_start": "tomorrow 19:00"},
        open_questions=[],
        trace_id="trace_no_reconstruction_checkpoint",
    )
    # 会话 checkpoint 与任务 checkpoint 是两种独立投影；
    # 重新打开任务时，不得悄悄重建被明确删除的任务 checkpoint。
    store._connection.execute(
        "DELETE FROM runtime_task_context_checkpoints WHERE task_context_id = ?",
        (prepared.context.task_context_id,),
    )
    store._connection.commit()
    assert store.get_task_context_checkpoint(prepared.context.task_context_id) is None
    store._connection.close()

    reopened = SQLiteAgentStore(database)
    checkpoint = reopened.get_task_context_checkpoint(prepared.context.task_context_id)

    assert checkpoint is None


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
