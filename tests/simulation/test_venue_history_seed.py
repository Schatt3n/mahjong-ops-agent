"""核对匿名素材的历史前缀与真实后续互相独立。"""

from dataclasses import asdict
from datetime import date, time
import json

import pytest

from .venue_history_seed import DEFAULT_HISTORY_DATASET, load_historical_group_seed


def test_first_historical_cut_keeps_relative_time_identity_and_quotes():
    """前四条消息按已发生顺序投影，未来发言者未进入前缀名单。"""
    seed = load_historical_group_seed()
    assert seed.capture_date == date(2026, 11, 18)
    assert seed.timeline_start_local == time(20, 0)
    assert seed.cutoff == "m04"
    assert seed.held_out_count == 22
    assert seed.actor_aliases == ("customer_1", "customer_2", "customer_3")
    assert [message.message_alias for message in seed.prefix] == ["m01", "m02", "m03", "m04"]
    assert [message.offset_seconds for message in seed.prefix] == [0, 3, 772, 897]
    assert [message.quoted_message_alias for message in seed.prefix] == [None, None, "m01", "m03"]


def test_evaluation_and_future_turn_text_never_enter_seed(tmp_path):
    """同一份素材可留出真人后续，角色可见字段不带评测备注或未发送消息。"""
    case = json.loads(DEFAULT_HISTORY_DATASET.read_text(encoding="utf-8"))
    case["messages"][4]["text"] = "HELD_OUT_MESSAGE"
    case["messages"][2]["quoted_text"] = "RUNTIME_MUST_RESOLVE_THIS_FROM_THE_EARLIER_MESSAGE"
    case["candidate_expectation"]["probe"] = "EVALUATOR_ONLY"
    path = tmp_path / DEFAULT_HISTORY_DATASET.name
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    data = repr(asdict(load_historical_group_seed(path)))
    assert "HELD_OUT_MESSAGE" not in data
    assert "RUNTIME_MUST_RESOLVE_THIS_FROM_THE_EARLIER_MESSAGE" not in data
    assert "EVALUATOR_ONLY" not in data


def test_prefix_fails_if_quote_depends_on_future_held_out_message(tmp_path):
    """引用未来群消息会失败，防止运行器意外为角色补入真实后续。"""
    case = json.loads(DEFAULT_HISTORY_DATASET.read_text(encoding="utf-8"))
    case["messages"][3]["target_message_alias"] = "m05"
    path = tmp_path / DEFAULT_HISTORY_DATASET.name
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="m04 引用了尚未出现的消息"):
        load_historical_group_seed(path)


def test_later_cut_keeps_public_mention_and_rejects_unknown_cut():
    """切点可以移动，公开提及按当时的消息保留。"""
    seed = load_historical_group_seed(cutoff="m24")
    assert seed.prefix[-1].mentioned_actor_aliases == ("customer_3",)
    assert seed.held_out_count == 2
    with pytest.raises(ValueError, match="时间线没有切点"):
        load_historical_group_seed(cutoff="m99")
