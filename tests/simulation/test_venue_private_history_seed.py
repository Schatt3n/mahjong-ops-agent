"""验证真人私聊切点的时间、角色和素材隔离。"""

from dataclasses import asdict
from datetime import datetime
import json

import pytest

from .venue_private_history_seed import (
    DEFAULT_PRIVATE_HISTORY_CASE_ID,
    DEFAULT_PRIVATE_HISTORY_DATASET,
    load_historical_private_seed,
)


def test_private_prefix_preserves_provenance_time_and_roles():
    """截图只记录同一分钟，保留原始时间和真实轮次顺序。"""
    seed = load_historical_private_seed()
    assert seed.case_id == DEFAULT_PRIVATE_HISTORY_CASE_ID
    assert seed.episode_id == "availability_then_smoke_preference"
    assert seed.cutoff_turn == 2 and seed.held_out_count == 2
    assert seed.started_at == seed.cutoff_at == datetime.fromisoformat("2026-07-05T12:29:00+08:00")
    assert [message.turn for message in seed.prefix] == [1, 2]
    assert [message.role for message in seed.prefix] == ["customer", "boss"]
    assert [message.offset_seconds for message in seed.prefix] == [0, 0]
    assert [message.source_message_id for message in seed.prefix] == [
        f"history-dm:{DEFAULT_PRIVATE_HISTORY_CASE_ID}:{turn}" for turn in (1, 2)
    ]


def test_private_prefix_can_select_a_later_episode_without_earlier_tasks():
    """其他营业日的片段可独立续演，前序任务不会被一并加载。"""
    seed = load_historical_private_seed(episode_id="existing_game_join_then_duration_cancel", cutoff_turn=7)
    assert seed.started_at == datetime.fromisoformat("2026-07-10T16:52:00+08:00")
    assert [message.turn for message in seed.prefix] == [5, 6, 7]
    assert seed.held_out_count == 11
    with pytest.raises(ValueError, match="后面确有顾客续轮"):
        load_historical_private_seed(episode_id="existing_game_join_then_duration_cancel", cutoff_turn=6)


def test_private_held_out_reply_outcome_and_screenshot_do_not_leave_loader(tmp_path):
    """角色只收到已投递正文，评测内容和截图路径仍在源资料。"""
    record = json.loads(DEFAULT_PRIVATE_HISTORY_DATASET.read_text(encoding="utf-8"))
    record["messages"][0]["source_image"] = "PRIVATE_SCREENSHOT_SENTINEL"
    record["messages"][2]["text"] = "FUTURE_CUSTOMER_SENTINEL"
    record["episodes"][0]["outcome"] = "EVALUATOR_SENTINEL"
    path = tmp_path / DEFAULT_PRIVATE_HISTORY_DATASET.name
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    seed = load_historical_private_seed(path)
    assert "SENTINEL" not in repr(asdict(seed))
    assert seed.held_out_count == 2


@pytest.mark.parametrize("key,value,error", [
    ("observed_at", "2026-07-05T12:29:00", "缺少时区"),
    ("observed_at", "2026-07-05T12:28:00+08:00", "时间倒序"),
    ("text", "联系 wxid_TEST_ONLY", "仍包含联系方式"),
])
def test_private_prefix_rejects_missing_source_time_and_unredacted_contact(tmp_path, key, value, error):
    """时间与匿名边界失效时，加载器不产生可供角色消费的前缀。"""
    record = json.loads(DEFAULT_PRIVATE_HISTORY_DATASET.read_text(encoding="utf-8"))
    record["messages"][1][key] = value
    path = tmp_path / DEFAULT_PRIVATE_HISTORY_DATASET.name
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_historical_private_seed(path)
