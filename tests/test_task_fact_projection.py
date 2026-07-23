from __future__ import annotations

from datetime import datetime

from mahjong_agent_runtime import AgentContextBuilder, InMemoryAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.domains.context_builders.task_facts import project_explicit_task_facts
from mahjong_agent_runtime.group_chat.parsing import parse_explicit_need
from mahjong_agent_runtime.tool_consistency import validate_explicit_task_fact_consistency


def test_parse_explicit_need_understands_chinese_seat_structure() -> None:
    parsed = parse_explicit_need("三缺一，帮我约个无烟局")

    assert parsed["seat_format"] == "371"
    assert parsed["known_player_count"] == 3
    assert parsed["needed_seats"] == 1


def test_half_stake_transport_variants_project_to_same_explicit_fact() -> None:
    for text in ("五点0.5，173，无烟", "五点0，5，173，无烟", "五点0、5，173，无烟", "五点0 5，173，无烟"):
        parsed = parse_explicit_need(text, anchor=datetime.fromisoformat("2026-07-04T12:00:00+08:00"))

        assert parsed["stake"] == "0.5"
        assert parsed["base_stake"] == 0.5
        assert parsed["seat_format"] == "173"


def test_latest_explicit_party_size_overrides_earlier_seat_structure() -> None:
    projection = project_explicit_task_facts(
        recent_conversation=[
            {"role": "user", "content": "三缺一，先帮我看看"},
            {"role": "assistant", "content": "现在几个人？"},
            {"role": "user", "content": "我这边两个人"},
        ],
        current_message={"text": "杭麻1块无烟，人齐开"},
        checkpoint=None,
    )

    assert projection["facts"]["known_player_count"] == 2
    assert projection["facts"]["needed_seats"] == 2
    assert projection["facts"]["seat_format"] == "272"
    assert projection["facts"]["game_type"] == "hangzhou_mahjong"
    assert projection["facts"]["stake"] == "1"
    assert projection["facts"]["smoke_preference"] == "no_smoking"


def test_projection_keeps_prior_explicit_count_when_current_turn_only_adds_slots() -> None:
    projection = project_explicit_task_facts(
        recent_conversation=[
            {"role": "user", "content": "三缺一，帮我约个无烟局"},
            {"role": "assistant", "content": "打什么麻将、什么档位？"},
        ],
        current_message={"text": "杭麻1块无烟马上开"},
        checkpoint=None,
    )

    assert projection["facts"] == {
        "game_type": "hangzhou_mahjong",
        "stake": "1",
        "base_stake": 1.0,
        "stake_label": "1",
        "smoke_preference": "no_smoking",
        "known_player_count": 3,
        "needed_seats": 1,
        "seat_format": "371",
    }
    assert projection["binding_fields"] == [
        "game_type",
        "stake",
        "base_stake",
        "stake_label",
        "smoke_preference",
        "known_player_count",
        "needed_seats",
        "seat_format",
    ]


def test_context_builder_exposes_explicit_task_facts_to_the_model() -> None:
    store = InMemoryAgentStore()
    conversation_id = "explicit_fact_context_case"
    store.append_user_turn(
        UserMessage(
            conversation_id=conversation_id,
            sender_id="requester",
            sender_name="发起人",
            text="三缺一，先帮我看看",
            message_id="explicit_fact_message_1",
        ),
        "trace_explicit_fact_1",
    )
    store.append_assistant_turn(
        conversation_id,
        "打什么麻将、什么档位？",
        "trace_explicit_fact_1",
    )

    built = AgentContextBuilder(store, ToolGateway(store)).build(
        UserMessage(
            conversation_id=conversation_id,
            sender_id="requester",
            sender_name="发起人",
            text="杭麻1块无烟，人齐开",
            message_id="explicit_fact_message_2",
        ),
        trace_id="trace_explicit_fact_2",
    )

    facts = built.payload["explicit_task_facts"]
    assert facts["facts"]["game_type"] == "hangzhou_mahjong"
    assert facts["facts"]["stake"] == "1"
    assert facts["facts"]["smoke_preference"] == "no_smoking"
    assert facts["facts"]["known_player_count"] == 3
    assert facts["facts"]["needed_seats"] == 1
    assert facts["facts"]["seat_format"] == "371"
    assert facts["evidence"]["known_player_count"] == "recent_user_turn"
    assert built.audit["explicit_task_fact_count"] >= 6


def test_ambiguous_clock_uses_message_timestamp_as_semantic_anchor() -> None:
    afternoon = datetime.fromisoformat("2026-07-04T15:42:00+08:00")

    parsed = parse_explicit_need("帮我约个6.30无烟的", anchor=afternoon)

    assert parsed["start_time"] == "18:30"
    assert parsed["planned_start_at"] == "2026-07-04T18:30:00+08:00"


def test_ambiguous_four_oclock_resolves_to_same_day_afternoon_when_sent_at_noon() -> None:
    afternoon = datetime.fromisoformat("2026-07-04T13:59:00+08:00")

    parsed = parse_explicit_need("4点无烟0.5，173", anchor=afternoon)

    assert parsed["start_time"] == "16:00"
    assert parsed["planned_start_at"] == "2026-07-04T16:00:00+08:00"


def test_explicit_daypart_still_wins_over_nearest_time_inference() -> None:
    evening = datetime.fromisoformat("2026-07-04T20:00:00+08:00")

    parsed = parse_explicit_need("明天下午两点打", anchor=evening)

    assert parsed["start_time"] == "14:00"
    assert parsed["planned_start_at"] == "2026-07-05T14:00:00+08:00"


def test_task_fact_projection_uses_current_message_sent_at() -> None:
    projection = project_explicit_task_facts(
        recent_conversation=[],
        current_message={
            "text": "帮我约个6.30无烟的",
            "sent_at": "2026-07-04T15:42:00+08:00",
        },
        checkpoint=None,
    )

    assert projection["facts"]["start_time"] == "18:30"
    assert projection["facts"]["planned_start_at"] == "2026-07-04T18:30:00+08:00"


def test_explicit_task_fact_guard_rejects_time_drift_in_tool_arguments() -> None:
    call = ToolCall(
        name="search_current_games",
        arguments={
            "requirement": {
                "start_time_kind": "scheduled",
                "start_time": "06:30",
                "planned_start_at": "2026-07-04T06:30:00+08:00",
            }
        },
        reason="查局",
    )
    context = {
        "explicit_task_facts": {
            "facts": {
                "start_time_kind": "scheduled",
                "start_time": "18:30",
                "planned_start_at": "2026-07-04T18:30:00+08:00",
            }
        }
    }

    error, reference = validate_explicit_task_fact_consistency(call, context)

    assert error is not None
    assert "start_time" in error
    assert reference["start_time"] == "18:30"


def test_explicit_task_fact_guard_accepts_matching_tool_arguments() -> None:
    call = ToolCall(
        name="search_current_games",
        arguments={
            "requirement": {
                "start_time_kind": "scheduled",
                "start_time": "18:30",
                "planned_start_at": "2026-07-04T18:30:00+08:00",
            }
        },
        reason="查局",
    )
    context = {
        "explicit_task_facts": {
            "facts": {
                "start_time_kind": "scheduled",
                "start_time": "18:30",
                "planned_start_at": "2026-07-04T18:30:00+08:00",
            }
        }
    }

    error, _ = validate_explicit_task_fact_consistency(call, context)

    assert error is None
