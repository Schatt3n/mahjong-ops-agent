"""将用户通用话题政策应用于当前验收视图，旧黄金和原始失败保持不可变。"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "eval/handoffs/20261105_general_topic_policy.json"
POLICY_SHA256 = "f3f03a6ec7f0f0a42c4ddf9a69b7daa277bd238a6fff67feae27cb1f96baaaac"
LENGTH_POLICY_PATH = ROOT / "eval/handoffs/20261105_topic_reply_length_policy.json"
LENGTH_POLICY_SHA256 = "e6cae27d65e8e9e65f5b6ab1daaa0c1b3fa97758537e0b7dd84837265ea455a8"


def topic_policy() -> dict:
    """原资产哈希绑定政策来源，不能按本次模型输出重写历史标准。"""
    raw = POLICY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != POLICY_SHA256:
        raise ValueError("topic policy source changed without review")
    policy = json.loads(raw)
    source = ROOT / policy["original_asset"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != policy["original_asset_sha256"]:
        raise ValueError("topic policy original golden changed")
    return policy


def topic_reply_length_policy() -> dict:
    """追加用户确认的普通话题上限，绑定旧政策和黄金来源而不重写原证据。"""
    raw = LENGTH_POLICY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LENGTH_POLICY_SHA256:
        raise ValueError("topic reply length policy changed without review")
    policy, previous = json.loads(raw), topic_policy()
    if (policy["previous_policy"] != str(POLICY_PATH.relative_to(ROOT))
            or policy["previous_policy_sha256"] != POLICY_SHA256
            or policy["original_asset"] != previous["original_asset"]
            or policy["original_asset_sha256"] != previous["original_asset_sha256"]
            or policy["scenario_ids"] != previous["casual_scenario_ids"]):
        raise ValueError("topic reply length policy source mismatch")
    return policy


def apply_topic_policy_to_scenarios(scenarios: list) -> list:
    """撤下泛词与强制闲聊措辞；业务工具、人数、条件、持久化断言全部保留。"""
    from mahjong_agent_runtime.customer_visible_contract import FORBIDDEN_INTERNAL_IDENTIFIERS, FORBIDDEN_CUSTOMER_SERVICE_PHRASES

    policy, length_policy = topic_policy(), topic_reply_length_policy()
    result = []
    for scenario in scenarios:
        forbidden = [term for term in scenario.forbidden_reply_contains if term not in policy["removed_generic_terms"]]
        changes = {"forbidden_reply_contains": list(dict.fromkeys([*forbidden, *FORBIDDEN_INTERNAL_IDENTIFIERS]))}
        if scenario.scenario_id in policy["casual_scenario_ids"]:
            changes["required_reply_any"] = []
        # 仅修订普通话题继承的旧默认上限，显式更严格的业务约束不能被覆盖。
        if (scenario.scenario_id in length_policy["scenario_ids"]
                and scenario.max_reply_chars == length_policy["previous_max_reply_chars"]):
            changes["max_reply_chars"] = length_policy["max_reply_chars"]
        if scenario.scenario_id == "quoted_correction_should_not_pollute_business_state":
            # 是否强拉业务由独立语义审查；不以出现“七点”等片段代替指代判断。
            changes["forbidden_reply_contains"] = [*FORBIDDEN_INTERNAL_IDENTIFIERS, *FORBIDDEN_CUSTOMER_SERVICE_PHRASES, "请问还有什么可以帮"]
        result.append(replace(scenario, **changes))
    return result


def apply_topic_policy_to_cases(cases: list[dict]) -> list[dict]:
    """只修订泛词禁项和引用解释边界；原始事实、禁止变更及隐私语义原样保留。"""
    policy, length_policy = topic_policy(), topic_reply_length_policy()
    result = deepcopy(cases)
    for case in result:
        expected = case["expected"]
        previous = deepcopy(expected)
        if "forbidden_reply_contains" in expected:
            expected["forbidden_reply_contains"] = [word for word in expected["forbidden_reply_contains"]
                if word not in policy["removed_generic_terms"]]
        if expected.pop("customer_visible_reply_must_not_claim_system_identity", None) is not None:
            expected["must_preserve_privacy_boundary"] = policy["current_privacy_rule"]
        for item in expected.get("forbidden_meaning", []):
            if case["case_id"] == "synthetic_live_quoted_correction_should_not_pollute_business_state" and item["id"] == "system_exposition":
                item["description"] = policy["current_privacy_rule"] + "引用更正只需自然回应，无需展开未被询问的内容。"
        if expected != previous:
            case.setdefault("product_contract_revision_ids", []).append(policy["revision_id"])
        if case.get("scenario_id") in length_policy["scenario_ids"]:
            case.setdefault("product_contract_revision_ids", []).append(length_policy["revision_id"])
    return result


def apply_topic_policy_to_hard_snapshot(snapshot: dict, scenario_id: str) -> dict:
    """历史快照先完成原修订链验证，再按同版政策投影当前可执行字段。"""
    from mahjong_agent_runtime.customer_visible_contract import FORBIDDEN_INTERNAL_IDENTIFIERS, FORBIDDEN_CUSTOMER_SERVICE_PHRASES

    policy, result = topic_policy(), deepcopy(snapshot)
    if "forbidden_reply_contains" in result:
        words = [word for word in result["forbidden_reply_contains"] if word not in policy["removed_generic_terms"]]
        result["forbidden_reply_contains"] = list(dict.fromkeys([*words, *FORBIDDEN_INTERNAL_IDENTIFIERS]))
        if scenario_id == "quoted_correction_should_not_pollute_business_state":
            result["forbidden_reply_contains"] = [*FORBIDDEN_INTERNAL_IDENTIFIERS, *FORBIDDEN_CUSTOMER_SERVICE_PHRASES, "请问还有什么可以帮"]
    if scenario_id in policy["casual_scenario_ids"] and "required_reply_any" in result:
        result["required_reply_any"] = []
    length_policy = topic_reply_length_policy()
    if (scenario_id in length_policy["scenario_ids"]
            and result.get("max_reply_chars") == length_policy["previous_max_reply_chars"]):
        result["max_reply_chars"] = length_policy["max_reply_chars"]
    return result
