from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mahjong_agent.context_builder import (
    TrialShortMemoryTextMerger,
    TrialWorkflowFollowupContextBuilder,
    WorkflowContextBuilder,
    WorkflowContextBuilderConfig,
)
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


def parse_dt(value):
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    return datetime.fromisoformat(str(value))


def test_trial_workflow_followup_context_builder_packages_legacy_memory() -> None:
    builder = TrialWorkflowFollowupContextBuilder(
        parse_datetime=parse_dt,
        text_normalizer=lambda value: str(value or "").replace("0。5", "0.5"),
        memory_ttl_seconds=120,
    )
    memory = [
        {
            "trace_id": "trace_prev",
            "at": (NOW - timedelta(seconds=30)).isoformat(),
            "text": "通宵0。5有人吗",
            "effective_text": "通宵0.5有人吗",
            "parsed": {"intent_action": "inquire_existing_game", "user_intent": "咨询现有局"},
            "missing_fields": [],
            "suggested_reply": {"text": "0.5的暂时没有诶。要组一个吗？", "reasoning_summary": "无匹配局"},
            "game": {"level": "0.5"},
            "tool_results": {"search_current_open_games": {"called": True, "result_count": 0}},
        }
    ]

    followup = builder.build(memory, "可以", NOW)

    assert followup["schema_version"] == "trial_workflow_followup_context.v1"
    assert followup["previous_trace_id"] == "trace_prev"
    assert followup["previous_system_suggested_reply"] == "0.5的暂时没有诶。要组一个吗？"
    assert followup["previous_intent_action"] == "inquire_existing_game"
    assert followup["previous_user_intent"] == "咨询现有局"
    assert followup["previous_game"] == {"level": "0.5"}
    assert followup["current_user_text"] == "可以"
    assert builder.is_grouping_confirmation_followup(followup, "组") is True
    assert builder.is_grouping_confirmation_followup(followup, "不组") is False


def test_trial_workflow_followup_context_builder_ignores_expired_or_non_followup_memory() -> None:
    builder = TrialWorkflowFollowupContextBuilder(
        parse_datetime=parse_dt,
        text_normalizer=lambda value: str(value or ""),
        memory_ttl_seconds=120,
    )
    memory = [
        {
            "trace_id": "trace_old",
            "at": (NOW - timedelta(seconds=121)).isoformat(),
            "text": "有人吗",
            "suggested_reply": {"text": "现在没有诶，要组一个吗？"},
        }
    ]

    assert builder.build(memory, "可以", NOW) == {}
    assert builder.build(
        [
            {
                "trace_id": "trace_prev",
                "at": (NOW - timedelta(seconds=30)).isoformat(),
                "text": "有人吗",
                "suggested_reply": {"text": "现在没有诶，要组一个吗？"},
            }
        ],
        "这是一段比较长的全新问题，不应该当成上一轮短回复继续处理",
        NOW,
    ) == {}


def test_trial_short_memory_text_merger_merges_recent_fragments_and_deduplicates() -> None:
    merger = TrialShortMemoryTextMerger(
        parse_datetime=parse_dt,
        is_pool_inquiry_text=lambda text: False,
        is_explicit_grouping_request=lambda source_text, effective_text, game: False,
        merge_window_seconds=600,
        critical_fields={"known_players", "start_time"},
    )
    memory = [
        {"at": (NOW - timedelta(seconds=60)).isoformat(), "text": "老板"},
        {"at": (NOW - timedelta(seconds=30)).isoformat(), "text": "老板"},
        {"at": (NOW - timedelta(seconds=20)).isoformat(), "text": "今天下午"},
    ]

    assert merger.build(memory, "有没有打麻将的", NOW) == "老板\n今天下午\n有没有打麻将的"


def test_trial_short_memory_text_merger_keeps_pending_goal_beyond_merge_window() -> None:
    merger = TrialShortMemoryTextMerger(
        parse_datetime=parse_dt,
        is_pool_inquiry_text=lambda text: False,
        is_explicit_grouping_request=lambda source_text, effective_text, game: False,
        merge_window_seconds=120,
        critical_fields={"known_players"},
    )
    memory = [
        {
            "at": (NOW - timedelta(seconds=300)).isoformat(),
            "text": "下午两点 0.5 无烟杭麻，帮我组一桌",
            "effective_text": "下午两点 0.5 无烟杭麻，帮我组一桌",
            "missing_fields": ["known_players"],
            "action": "ask_clarification",
        }
    ]

    assert (
        merger.build(memory, "我这边两个人", NOW)
        == "下午两点 0.5 无烟杭麻，帮我组一桌\n我这边两个人"
    )


def test_trial_short_memory_text_merger_keeps_new_pool_query_separate_from_previous_grouping() -> None:
    merger = TrialShortMemoryTextMerger(
        parse_datetime=parse_dt,
        is_pool_inquiry_text=lambda text: "有人吗" in text,
        is_explicit_grouping_request=lambda source_text, effective_text, game: "帮我组一桌" in source_text,
        merge_window_seconds=600,
        critical_fields={"known_players"},
    )
    memory = [
        {
            "at": (NOW - timedelta(seconds=60)).isoformat(),
            "text": "下午两点 0.5 无烟杭麻，帮我组一桌",
            "missing_fields": [],
            "action": "ask_clarification",
        }
    ]

    assert merger.build(memory, "通常局有人吗", NOW) == "通常局有人吗"
    assert merger.should_merge(["下午两点 0.5 无烟杭麻，帮我组一桌"], "通常局有人吗") is False


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
    assert prompt["followup_context"]["schema_version"] == "followup_context.v1"
    assert prompt["followup_context"]["unresolved_questions"] == ["create_confirmation"]
    assert prompt["followup_context"]["expected_answer_type"] == "yes_no_confirmation"
    assert prompt["followup_context"]["current_message_response_type"] == "short_ack"
    assert prompt["followup_context"]["should_treat_current_message_as_followup"] is True
    assert prompt["followup_context"]["previous_turn"]["message_id"] == "msg_prev"
    assert prompt["followup_context"]["previous_game_requirement"]["slots"]["stake"]["value"] == "0.5"
    assert "要组一个吗" in prompt["memory_summary"]


def test_workflow_context_builder_marks_slot_fill_followup_for_llm() -> None:
    core = AgentCore()
    memory = InMemoryShortTermMemoryStore()
    previous_user = UserMessage(
        text="老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可",
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
            system_reply="可以，我先确认下：大概几点能到？你这边几个人？",
            created_at=NOW - timedelta(seconds=20),
        ),
        now=NOW,
    )

    result = WorkflowContextBuilder(core, memory).build(
        make_message("六点，我这边两个人", "msg_current"),
        now=NOW,
        trace_id="trace_current",
    )
    followup = result.context.to_prompt_dict()["followup_context"]

    assert followup["schema_version"] == "followup_context.v1"
    assert followup["unresolved_questions"] == ["start_time", "party_size"]
    assert followup["expected_answer_type"] == "slot_fill"
    assert followup["current_message_response_type"] == "slot_fill"
    assert followup["should_treat_current_message_as_followup"] is True
    assert followup["signals"]["current_message_is_slot_fill"] is True
    assert followup["signals"]["previous_reply_asked_clarification"] is True


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
            metadata={
                "controlled_profile_observations": [
                    {
                        "field": "smoke_preference",
                        "value": "any",
                        "confidence": 0.82,
                        "evidence": "用户说有烟无烟都行",
                    }
                ]
            },
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
    assert "画像观察：smoke_preference=any；证据：用户说有烟无烟都行" in prompt["customer_profile"]["recent_facts"]

    recent_texts = [turn["user_message"]["text"] for turn in prompt["recent_turns"]]
    assert "老板，今天下班有人打麻将吗" in recent_texts
    assert "另一个群消息" not in recent_texts

    assert len(prompt["open_games"]) == 1
    open_game_slots = prompt["open_games"][0]["slots"]
    assert open_game_slots["stake"]["value"] == "0.5"
    assert open_game_slots["missing_count"]["value"] == 1
    assert open_game_slots["smoke"]["value"] == "no_smoke"
    assert open_game_slots["duration_hours"]["value"] == 4
