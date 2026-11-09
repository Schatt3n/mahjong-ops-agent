from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent_runtime.group_chat import GroupMessage, L1RuleEngine
from mahjong_agent_runtime.group_chat.parsing import parse_game_post


TZ = ZoneInfo("Asia/Shanghai")


def _message(text: str, *, sender_id: str = "user-a") -> GroupMessage:
    return GroupMessage(
        room_id="room-1",
        conversation_id="wechaty:room:room-1",
        sender_external_id=sender_id,
        sender_name="用户A",
        text=text,
        message_id=f"message:{text}",
        sent_at=datetime(2026, 7, 22, 13, 0, tzinfo=TZ),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("14:00 无烟 371 0.5", {"start_time": "14:00", "stake": "0.5", "smoke_preference": "no_smoking", "known_player_count": 3, "needed_seats": 1}),
        ("cq272 13.00 无烟 1块", {"start_time": "13:00", "stake": "1", "smoke_preference": "no_smoking", "known_player_count": 2, "needed_seats": 2}),
        ("川麻 2-16 人齐开 173", {"start_time_kind": "asap_when_full", "stake": "2", "cap_score": 16.0, "known_player_count": 1, "needed_seats": 3}),
    ],
)
def test_standard_game_post_is_strictly_parsed(text: str, expected: dict) -> None:
    """完整字段仍需结合语义与既有委托判断，解析成功本身不授予登记权限。"""
    message = _message(text)
    parsed = parse_game_post(text, anchor=message.sent_at)

    assert parsed is not None
    for key, value in expected.items():
        assert parsed[key] == value
    assert L1RuleEngine(bot_id="bot").process(message).action == "pass_to_L2"


@pytest.mark.parametrize("text", ["今天0.5涨到1了", "371路公交堵死了", "3楼灯坏了"])
def test_numbers_without_a_complete_game_post_are_not_imported(text: str) -> None:
    assert L1RuleEngine(bot_id="bot").process(_message(text)).action == "pass_to_L2"


@pytest.mark.parametrize("text", ["3来", "3我来", "3这个我可以", "3 加我"])
def test_standard_claim_resolves_item_number(text: str) -> None:
    result = L1RuleEngine(bot_id="bot").process(_message(text))

    assert result.action == "claim"
    assert result.item_no == 3


def test_non_claim_starting_with_number_is_not_misclassified() -> None:
    assert L1RuleEngine(bot_id="bot").process(_message("3楼灯坏了")).action != "claim"


def test_natural_language_request_requires_semantic_routing() -> None:
    result = L1RuleEngine(bot_id="bot").process(_message("老板，14:00 无烟371 0.5，帮我挂上"))

    assert result.action == "pass_to_L2"


def test_plain_game_post_requires_semantic_context() -> None:
    result = L1RuleEngine(bot_id="bot").process(_message("14:00 无烟371 0.5"))

    assert result.action == "pass_to_L2"


def test_bot_echo_is_ignored() -> None:
    assert L1RuleEngine(bot_id="bot").process(_message("14:00 无烟371 0.5", sender_id="bot")).action == "ignore"
