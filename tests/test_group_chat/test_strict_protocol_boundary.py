from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent_runtime import Game, InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.group_chat import BoardEngine, GroupMessage, L1RuleEngine
from mahjong_agent_runtime.group_chat.parsing import parse_explicit_need, parse_game_post
from mahjong_agent_runtime.group_chat.projections import public_game_start_display


ANCHOR = datetime(2026, 11, 5, 11, 20, tzinfo=ZoneInfo("Asia/Shanghai"))


def message(text):
    return GroupMessage(room_id="room", conversation_id="room-chat", sender_external_id="customer",
                        sender_name="陈星", text=text, message_id="original", sent_at=ANCHOR)


@pytest.mark.parametrize("text", [
    "明晚七点半杭麻0.5我这边两个人，先看看有没有能一起坐的。",
    "老板，帮我找明天下午两点国标1块无烟的，我一个人。",
    "老板，帮我组明晚七点半杭麻0.5无烟的局，我一个人。",
    "14:00 无烟 371 0.5 还有吗？",
    "不是14:00 无烟 371 0.5",
    "他说14:00 无烟 371 0.5",
    "“14:00 无烟 371 0.5”",
    "老板，14:00 无烟371 0.5，帮我挂上",
    "明晚19:30 杭麻0.5 272",
    "下周四19:30 杭麻0.5 272",
    "14:00 无烟 371 0.5 15:00",
    "14:00 无烟 371 272 0.5",
    "2026-02-30 14:00 无烟 371 0.5",
    "2026-11-06 人齐开 无烟 371 0.5",
])
def test_open_language_and_ambiguous_fields_require_semantic_routing(text):
    assert parse_game_post(text, anchor=ANCHOR) is None
    assert L1RuleEngine(bot_id="bot").process(message(text)).action == "pass_to_L2"


@pytest.mark.parametrize("text", ["不是3我来", "他说3我来", "“3我来”", "3我来吗？"])
def test_quoted_negated_or_questioned_claim_is_not_protocol_claim(text):
    result = L1RuleEngine(bot_id="bot").process(message(text))
    assert result.action == "pass_to_L2"
    assert result.item_no is None


@pytest.mark.parametrize("text", ["好", "收到", "ok", "👍"])
def test_short_acknowledgements_require_personal_topic_context(text):
    assert L1RuleEngine(bot_id="bot").process(message(text)).action == "pass_to_L2"


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_query_cannot_create_game_or_publish_task_even_when_import_called_directly(kind, tmp_path):
    store = InMemoryAgentStore() if kind == "memory" else SQLiteAgentStore(tmp_path / "query.sqlite3")
    board = BoardEngine(store=store, messenger=object(), clock=lambda: ANCHOR)
    with pytest.raises(ValueError, match="not an explicit game post"):
        board.import_game_from_post(message("明晚七点半杭麻0.5我这边两个人，先看看有没有能一起坐的。"), trace_id="query-only")
    assert not store.games
    assert not store.scheduled_tasks


@pytest.mark.parametrize("text,expected", [
    ("2026-11-06 19:30 杭麻 0.5 无烟 272", "2026-11-06T19:30:00+08:00"),
    ("07:30 国标 1块 无烟 371", "2026-11-05T07:30:00+08:00"),
])
def test_complete_protocol_uses_explicit_date_or_literal_24_hour_clock(text, expected):
    parsed = parse_game_post(text, anchor=ANCHOR)
    assert parsed is not None
    assert parsed["planned_start_at"] == expected


@pytest.mark.parametrize("hour", [0, 11, 20, 23])
def test_tomorrow_evening_equivalent_phrases_keep_date_at_all_anchor_hours(hour):
    anchor = ANCHOR.replace(hour=hour)
    for text in ("明晚七点半杭麻0.5", "明天晚上七点半杭麻0.5"):
        assert parse_explicit_need(text, anchor=anchor)["planned_start_at"] == "2026-11-06T19:30:00+08:00"


def test_public_display_distinguishes_same_clock_on_different_dates():
    values = []
    for day in (5, 6):
        game = Game(game_id=f"game-{day}", conversation_id="private", organizer_id="a", organizer_name="A",
                    requirement={"start_time_kind": "scheduled"}, planned_start_at=ANCHOR.replace(day=day, hour=19, minute=30))
        values.append(public_game_start_display(game))
    assert values == ["2026-11-05 19:30", "2026-11-06 19:30"]


def test_public_display_keeps_asap_semantics():
    game = Game(game_id="asap", conversation_id="private", organizer_id="a", organizer_name="A",
                requirement={"start_time_kind": "asap_when_full"}, planned_start_at=ANCHOR)
    assert public_game_start_display(game) == "人齐开"
