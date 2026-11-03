"""评测目录的确定性冲突检查；不按自然语言关键词替代语义判定。"""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_real_owner_chat_live_eval as owner
from run_inquiry_grounding_eval import _semantic_criteria_ids
from mahjong_agent_runtime.domains.tools.registry import default_tool_definitions


@pytest.mark.parametrize('scenario',owner.live_eval_scenarios(),ids=lambda s:s.scenario_id)
def test_effective_catalog_has_no_conflicting_tool_or_empty_semantic_contract(scenario):
    registry=default_tool_definitions(None)
    required=set(scenario.required_tool_names);forbidden=set(scenario.forbidden_tool_names)
    assert not required&forbidden
    assert required|forbidden|set(scenario.required_tool_name_any)<=set(registry)
    assert not scenario.required_tool_name_any or set(scenario.required_tool_name_any)-forbidden
    assert scenario.required_reply_any==[] or all(group and all(isinstance(s,str) and s for s in group) for group in scenario.required_reply_any)
    assert all(type(cap) is int and cap>=0 for cap in scenario.max_tool_call_counts.values())
    assert not set(scenario.required_reply_contains)&set(scenario.forbidden_reply_contains)
    assert not scenario.required_reply_any or all(set(group)-set(scenario.forbidden_reply_contains) for group in scenario.required_reply_any)
    expected=owner.reply_semantic_expectation(scenario)
    assert _semantic_criteria_ids(expected) and expected['golden_sources']
