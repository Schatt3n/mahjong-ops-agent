from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mahjong_agent import (
    AgentCore,
    ChannelType,
    ContextBuilder,
    ContextBuilderConfig,
    CustomerProfile,
    GameRequest,
    GameStatus,
    Message,
    PlayPreference,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)


def make_message(
    text: str,
    message_id: str,
    channel_id: str = "group-a",
    sender_id: str = "wxid_customer_001",
    sent_at: datetime | None = None,
) -> Message:
    return Message(
        text=text,
        sender_id=sender_id,
        sender_name="张哥",
        channel_id=channel_id,
        channel_type=ChannelType.WECHAT_GROUP,
        sent_at=sent_at or NOW,
        id=message_id,
    )


def test_context_builder_redacts_sensitive_content_and_uses_stable_refs() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="wxid_customer_001",
            display_name="张哥 13812345678",
            aliases=["微信号 wxid_secret888"],
            preferred_levels=["0.5"],
            tags=["无烟"],
            usual_party_size=1,
            usual_party_size_confidence=0.9,
            play_preferences=[
                PlayPreference(
                    game_type="hangzhou_mahjong",
                    preferred_levels=["0.5"],
                    preferred_variants=["caiqiao"],
                    preferred_play_options=["财敲"],
                )
            ],
        )
    )
    message = make_message(
        "老板 我电话13812345678 微信wxid_secret999 今晚0.5三缺一",
        "msg-sensitive",
    )

    result = ContextBuilder(core).build(message, now=NOW)
    dumped = json.dumps(result.context, ensure_ascii=False)

    assert "13812345678" not in dumped
    assert "wxid_secret" not in dumped
    assert "wxid_customer_001" not in dumped
    assert "[手机号]" in dumped
    assert "[微信号]" in dumped
    assert result.context["runtime"]["message_ref"].startswith("message_")
    assert result.context["audit"]["context_digest"].startswith("ctx_")
    assert result.context_digest == result.context["audit"]["context_digest"]
    assert result.context["customer_profile_summary"]["customer_ref"].startswith("customer_")
    assert result.context["privacy"]["identity_policy"] == "stable_refs_only"
    assert result.context["privacy"]["redaction_counts"]["phone"] >= 1


def test_context_builder_keeps_conversations_isolated() -> None:
    core = AgentCore()
    a1 = make_message("老板", "msg-a1", channel_id="group-a", sent_at=NOW)
    b1 = make_message("这是另一个群的秘密话题", "msg-b1", channel_id="group-b", sent_at=NOW + timedelta(seconds=1))
    a2 = make_message("今晚0.5三缺一", "msg-a2", channel_id="group-a", sent_at=NOW + timedelta(seconds=2))
    for message in [a1, b1]:
        core.store.messages[message.id] = message

    context = ContextBuilder(core).build(a2, now=NOW + timedelta(seconds=2)).context
    dumped = json.dumps(context["conversation_summary"], ensure_ascii=False)

    assert "老板" in dumped
    assert "今晚0.5三缺一" in dumped
    assert "另一个群" not in dumped
    assert context["conversation_summary"]["conversation_ref"].startswith("conversation_")


def test_context_builder_includes_bounded_operational_snapshots() -> None:
    core = AgentCore()
    core.configure_room_capacity(2)
    core.add_room_hold(
        start_at=NOW + timedelta(hours=1),
        end_at=NOW + timedelta(hours=5),
        room_id="room-1",
        source="room_schedule",
    )
    for index in range(4):
        game = GameRequest(
            organizer_id=f"host-{index}",
            organizer_name=f"发起人{index}",
            channel_id="group-a",
            status=GameStatus.OPEN,
            current_player_count=3,
            missing_count=1,
            level="0.5",
            start_at=NOW + timedelta(hours=index + 1),
        )
        core.store.games[game.id] = game

    context = ContextBuilder(
        core,
        ContextBuilderConfig(max_open_games=2, max_room_holds=1),
    ).build(make_message("还有人吗", "msg-current"), now=NOW).context

    assert len(context["game_state_snapshot"]["recent_open_games"]) == 2
    assert len(context["room_state_snapshot"]["active_holds"]) == 1
    assert context["room_state_snapshot"]["capacity"] == 2
    assert context["tool_policy"]["tool_calling_enabled"] is False
    assert context["allowed_tools"] == []


def test_context_builder_trims_history_when_context_budget_is_tight() -> None:
    core = AgentCore()
    for index in range(20):
        message = make_message(
            f"第{index}条历史消息，内容比较长，用来触发上下文预算裁剪",
            f"msg-history-{index}",
            sent_at=NOW + timedelta(seconds=index),
        )
        core.store.messages[message.id] = message

    current = make_message("今晚7点还有0.5吗", "msg-current", sent_at=NOW + timedelta(minutes=1))
    context = ContextBuilder(
        core,
        ContextBuilderConfig(max_context_chars=3500, max_recent_messages=20),
    ).build(current, now=NOW + timedelta(minutes=1)).context

    assert "conversation_summary.recent_messages" in context["context_budget"]["trimmed_sections"]
    assert len(context["conversation_summary"]["recent_messages"]) < 20
    assert context["context_budget"]["estimated_chars"] > 0


def test_context_builder_keeps_trace_id_read_only_when_server_provides_it() -> None:
    core = AgentCore()
    message = make_message("今晚有人打麻将吗", "msg-trace")
    message.metadata["trace_id"] = "trace-prod-20260620-001"

    context = ContextBuilder(core).build(message, now=NOW).context

    assert context["runtime"]["trace_id"] == "trace-prod-20260620-001"
    assert "trace_id" in context["runtime"]["read_only_fields"]
    assert any("trace_id" in boundary for boundary in context["safety_boundaries"])
