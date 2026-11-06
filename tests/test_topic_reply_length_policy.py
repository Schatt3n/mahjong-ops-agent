"""用户确认的普通话题200字上限，只修订长度，不替代语义或持久状态验收。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_topic_policy as policy
import run_real_owner_chat_live_eval as owner


# full v6 原第1轮的原文；原报告来源及哈希保存在追加政策中，不改原始评分。
OBSERVED_REPLY = "哈哈，天天都得盯着，谁有空、谁缺人、时间对不对得上，全靠人记。真有个靠谱的 AI 帮我盯这些，我肯定用啊——不嫌累，还记得住谁跟谁打过、谁不爱抽烟。不过最后拍板还得人来，客人临时变卦、谁跟谁合不合得来，这些它不一定拿得准。"
CASUAL_IDS = policy.topic_policy()["casual_scenario_ids"]


def validate_reply(tmp_path, scenario_id, reply, *, semantic_passed=True):
    """用真实SQLite前态检查长度与相邻硬约束；语义桩不冒充真实判定成绩。"""
    scene = next(item for item in owner.live_eval_scenarios() if item.scenario_id == scenario_id)
    store = owner.SQLiteAgentStore(tmp_path / "length.sqlite3")
    try:
        scene.setup(store)
        result = SimpleNamespace(final_reply=reply, status="completed", tool_results=[])
        return owner.validate_result(result, scene, [], store, {"passed": semantic_passed, "error": None})
    finally:
        store._connection.close()


@pytest.mark.parametrize("scenario_id", CASUAL_IDS)
@pytest.mark.parametrize("length,passed", [(80, True), (111, True), (199, True), (200, True), (201, False)])
def test_ordinary_topic_length_boundary_keeps_actual_length(tmp_path, scenario_id, length, passed):
    # 重复字符仅用于确定性计数边界，是否自然相关仍由独立语义判定负责。
    validation = validate_reply(tmp_path, scenario_id, "嗯" * length)
    check = next(item for item in validation["checks"] if item["name"] == "final_reply_should_be_short_owner_wechat_style")
    assert check["max_reply_chars"] == 200
    assert check["actual_length"] == length
    assert check["passed"] is passed
    assert validation["passed"] is passed


def test_original_111_character_reply_passes_new_length_only(tmp_path):
    assert len(OBSERVED_REPLY) == policy.topic_reply_length_policy()["observed_failure"]["actual_reply_chars"] == 111
    validation = validate_reply(tmp_path, CASUAL_IDS[0], OBSERVED_REPLY)
    assert validation["passed"]
    assert len(OBSERVED_REPLY) > policy.topic_reply_length_policy()["previous_max_reply_chars"]


@pytest.mark.parametrize("reply,semantic_passed,failed_check", [
    ("嗯，今晚来打吗？", False, "reply_should_match_expected_meaning"),
    ("内部 task_context_id 已更新。", True, "final_reply_should_pass_customer_visible_contract"),
    ("嗯。\n确实。", True, "final_reply_should_be_single_message"),
])
def test_longer_limit_does_not_bypass_semantics_privacy_or_message_contract(tmp_path, reply, semantic_passed, failed_check):
    validation = validate_reply(tmp_path, CASUAL_IDS[0], reply, semantic_passed=semantic_passed)
    assert not validation["passed"]
    assert not next(item for item in validation["checks"] if item["name"] == failed_check)["passed"]


def test_length_policy_preserves_other_scenarios_explicit_limits_and_semantics():
    original = owner.live_eval_scenarios(apply_current_topic_policy=False)
    current = {scene.scenario_id: scene for scene in owner.live_eval_scenarios()}
    for scene in original:
        assert current[scene.scenario_id].max_reply_chars == (200 if scene.scenario_id in CASUAL_IDS else scene.max_reply_chars)
    for explicit in (12, 40, None):
        scene = replace(current[CASUAL_IDS[0]], max_reply_chars=explicit)
        assert policy.apply_topic_policy_to_scenarios([scene])[0].max_reply_chars == explicit
        snapshot = {"max_reply_chars": explicit, "expected_active_game_count": 1}
        assert policy.apply_topic_policy_to_hard_snapshot(snapshot, CASUAL_IDS[0]) == snapshot
    before = owner.golden_reply_cases(apply_current_topic_policy=False)
    after = policy.apply_topic_policy_to_cases(before)
    # 新追加长度政策只增加版本追溯；它不向独立语义判定输入宽松的业务标准。
    for case in after:
        if case["scenario_id"] in CASUAL_IDS:
            assert policy.topic_reply_length_policy()["revision_id"] in case["product_contract_revision_ids"]
    assert before == owner.golden_reply_cases(apply_current_topic_policy=False)


def test_length_revision_binds_immutable_predecessor_and_source(monkeypatch, tmp_path):
    declaration = policy.topic_reply_length_policy()
    assert declaration["previous_policy_sha256"] == hashlib.sha256(policy.POLICY_PATH.read_bytes()).hexdigest()
    assert declaration["original_asset_sha256"] == hashlib.sha256((policy.ROOT / declaration["original_asset"]).read_bytes()).hexdigest()
    snapshot = {"max_reply_chars": 80, "expected_active_game_count": 1}
    saved = deepcopy(snapshot)
    assert policy.apply_topic_policy_to_hard_snapshot(snapshot, CASUAL_IDS[0]) == {**snapshot, "max_reply_chars": 200}
    assert snapshot == saved
    changed = tmp_path / "changed.json"
    changed.write_bytes(policy.LENGTH_POLICY_PATH.read_bytes() + b" ")
    monkeypatch.setattr(policy, "LENGTH_POLICY_PATH", changed)
    with pytest.raises(ValueError, match="length policy changed"):
        policy.topic_reply_length_policy()
