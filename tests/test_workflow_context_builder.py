from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mahjong_agent.context_builder import WorkflowContextBuilder, WorkflowContextBuilderConfig
from mahjong_agent.core import AgentCore
from mahjong_agent.memory import InMemoryShortTermMemoryStore, ShortTermMemoryRecord
from mahjong_agent.models import (
    ChannelType,
    CustomerProfile,
    GameRequest,
    GameStatus,
    Message,
    PlayPreference,
)
from mahjong_agent.workflow_models import GameRequirement, SlotSource, SlotValue, UserMessage


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 30, 16, 0, tzinfo=TZ)


def make_message(
    text: str,
    message_id: str,
    conversation_id: str = "group_a",
    sender_id: str = "zhang",
    sent_at: datetime | None = None,
) -> Message:
    return Message(
        text=text,
        sender_id=sender_id,
        sender_name="张哥" if sender_id == "zhang" else "王姐",
        channel_id=conversation_id,
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=sent_at or NOW,
        id=message_id,
        metadata={"conversation_id": conversation_id},
    )


def confirmed_slot(name: str, value, source: SlotSource = SlotSource.EXPLICIT) -> SlotValue:
    return SlotValue(
        name=name,
        value=value,
        source=source,
        confidence=0.9,
        confirmed=True,
        needs_confirmation=False,
    )


def test_short_term_memory_store_scopes_by_conversation_and_sender() -> None:
    store = InMemoryShortTermMemoryStore(ttl_seconds=60)
    first = ShortTermMemoryRecord(
        conversation_id="group_a",
        sender_id="zhang",
        user_message=UserMessage(
            text="通宵0.5有人吗",
            sender_id="zhang",
            sender_name="张哥",
            conversation_id="group_a",
            trace_id="trace_a",
        ),
        system_reply="0.5的暂时没有诶。要组一个吗？",
        created_at=NOW,
    )
    other_sender = ShortTermMemoryRecord(
        conversation_id="group_a",
        sender_id="wang",
        user_message=UserMessage(
            text="下午有人吗",
            sender_id="wang",
            sender_name="王姐",
            conversation_id="group_a",
            trace_id="trace_b",
        ),
        created_at=NOW,
    )
    store.append(first, now=NOW)
    store.append(other_sender, now=NOW)

    assert store.load("group_a", "zhang", now=NOW + timedelta(seconds=30)) == [first]
    assert store.load("group_a", "wang", now=NOW + timedelta(seconds=30)) == [other_sender]
    assert store.load("group_a", "zhang", now=NOW + timedelta(seconds=90)) == []


def test_workflow_context_builder_includes_followup_memory_for_llm() -> None:
    core = AgentCore()
    memory = InMemoryShortTermMemoryStore()
    requirement = GameRequirement()
    requirement.set_slot(confirmed_slot("stake", "0.5"))
    requirement.set_slot(confirmed_slot("duration_mode", "overnight"))
    previous_user = UserMessage(
        text="通宵0.5有人吗",
        sender_id="zhang",
        sender_name="张哥",
        conversation_id="group_a",
        trace_id="trace_prev",
        message_id="msg_prev",
    )
    memory.append(
        ShortTermMemoryRecord(
            conversation_id="group_a",
            sender_id="zhang",
            user_message=previous_user,
            system_reply="0.5的暂时没有诶。要组一个吗？",
            game_requirement=requirement,
            created_at=NOW - timedelta(seconds=20),
        ),
        now=NOW,
    )

    result = WorkflowContextBuilder(core, memory).build(
        make_message("组", "msg_current"),
        now=NOW,
        trace_id="trace_current",
    )
    prompt = result.context.to_prompt_dict()

    assert result.used_short_memory is True
    assert prompt["current_message"]["text"] == "组"
    assert prompt["previous_system_reply"] == "0.5的暂时没有诶。要组一个吗？"
    assert prompt["followup_context"]["current_message_may_answer_previous_reply"] is True
    assert prompt["followup_context"]["previous_game_requirement"]["slots"]["stake"]["value"] == "0.5"
    assert "要组一个吗" in prompt["memory_summary"]


def test_workflow_context_builder_structures_profile_history_and_open_games() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="zhang",
            display_name="张哥",
            preferred_levels=["0.5", "1"],
            smoke_free_preference=False,
            usual_party_size=1,
            usual_party_size_confidence=0.9,
            play_preferences=[
                PlayPreference(
                    game_type="hangzhou_mahjong",
                    preferred_variants=["caiqiao"],
                    preferred_play_options=["财敲"],
                )
            ],
        )
    )
    core.store.messages["msg_old"] = make_message(
        "老板，今天下班有人打麻将吗",
        "msg_old",
        sent_at=NOW - timedelta(minutes=2),
    )
    core.store.messages["msg_other"] = make_message(
        "另一个群消息",
        "msg_other",
        conversation_id="group_b",
        sent_at=NOW - timedelta(minutes=1),
    )
    core.store.games["game_open"] = GameRequest(
        id="game_open",
        organizer_id="ran",
        organizer_name="冉姐",
        channel_id="group_a",
        status=GameStatus.OPEN,
        game_type="hangzhou_mahjong",
        variant="caiqiao",
        current_player_count=3,
        missing_count=1,
        level="0.5",
        start_at=NOW + timedelta(hours=2),
        duration_hours=4,
        rules=["杭麻", "无烟"],
        play_options=["财敲"],
    )

    result = WorkflowContextBuilder(
        core,
        config=WorkflowContextBuilderConfig(max_recent_turns=4),
    ).build(make_message("现在还有0.5吗", "msg_current"), now=NOW, trace_id="trace_current")
    prompt = result.context.to_prompt_dict()

    profile_slots = prompt["customer_profile"]["preferred_slots"]
    assert profile_slots["stake_preferences"]["value"] == ["0.5", "1"]
    assert profile_slots["game_type_preferences"]["value"] == ["hangzhou_mahjong"]
    assert profile_slots["variant_preferences"]["value"] == ["caiqiao"]
    assert profile_slots["party_size"]["value"] == 1

    recent_texts = [turn["user_message"]["text"] for turn in prompt["recent_turns"]]
    assert "老板，今天下班有人打麻将吗" in recent_texts
    assert "另一个群消息" not in recent_texts

    assert len(prompt["open_games"]) == 1
    open_game_slots = prompt["open_games"][0]["slots"]
    assert open_game_slots["stake"]["value"] == "0.5"
    assert open_game_slots["missing_count"]["value"] == 1
    assert open_game_slots["smoke"]["value"] == "no_smoke"
    assert open_game_slots["duration_hours"]["value"] == 4
