"""个人匹配默认实际进入查询参数；开放类型协议不替模型选玩法或取得写入授权。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.domains.context_builders.contracts import planning_contract
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract
from mahjong_agent_runtime.domains.tools.validation import validate_schema

ROOT = Path(__file__).resolve().parents[1]


def query_spec(gateway):
    return next(spec for spec in gateway.tool_specs_for_prompt() if spec["name"] == "search_current_games")


def test_model_tool_schema_exposes_optional_open_game_type_and_keeps_read_scope():
    spec = query_spec(ToolGateway(store=InMemoryAgentStore()))
    requirement = spec["schema"]["properties"]["requirement"]
    game = requirement["properties"]["game_type"]
    assert game["type"] == "string" and "enum" not in game
    assert "game_type" not in requirement.get("required", [])
    assert requirement["additionalProperties"] is True
    assert validate_schema({"requirement": {}}, spec["schema"]) is None
    assert validate_schema({"requirement": {"game_type": "unregistered_custom_game_v42"}}, spec["schema"]) is None
    assert "个人匹配传" in game["description"]
    assert "可靠默认" in game["description"]
    assert "盘点按显式范围" in game["description"]
    assert spec["execution_mode"] == "read_only" and spec["risk_level"] == "low" and spec["parallel_safe"]


@pytest.mark.parametrize("value", [None, 3, True, [], {"game": "custom"}])
def test_query_game_type_is_typed_without_backend_repair_or_state_change(value):
    store = InMemoryAgentStore()
    gateway = ToolGateway(store=store)
    call = ToolCall(name="search_current_games", arguments={"requirement": {"game_type": value}})
    result = gateway.execute(call, trace_id="typed-game", conversation_id="query", sender_id="customer", sender_name="客户", step_index=1)
    assert not result.called and not result.allowed and "game_type must be string" in result.error
    assert call.arguments["requirement"]["game_type"] == value
    assert store.games == {} and not store.invite_drafts


def test_missing_game_type_is_not_automatically_filled_from_customer_profile():
    store = InMemoryAgentStore()
    store.upsert_customer(CustomerProfile(customer_id="customer", display_name="常客", preferred_games=["national_standard_mahjong"]))
    gateway = ToolGateway(store=store)
    call = ToolCall(name="search_current_games", arguments={"requirement": {"stake": "1"}})
    result = gateway.execute(call, trace_id="no-defaulting", conversation_id="query", sender_id="customer", sender_name="常客", step_index=1)
    assert result.called and result.allowed and result.error is None
    assert "game_type" not in call.arguments["requirement"] and "game_type" not in result.result["requirement"]
    assert store.games == {} and not store.invite_drafts


def test_query_parameter_guidance_separates_scope_defaults_time_kind_and_authorization():
    spec = query_spec(ToolGateway(store=InMemoryAgentStore()))
    description = spec["description"]
    assert "所选条件须写入参数" in description
    assert "planning_contract.game_query_strategy_contract" in description
    strategy = planning_contract()["game_query_strategy_contract"]
    assert strategy == game_query_strategy_contract()
    assert "复用本次有效原话、同一任务事实及无冲突的可靠画像" in strategy["known_fact_reuse"]
    assert "不擅自追加个人过滤" in strategy["known_fact_reuse"]
    assert "多值或可靠来源冲突不擅选" in strategy["unknown_fact_boundary"]
    assert "不阻断公开库存查询" in strategy["unknown_fact_boundary"]
    assert "不授权建局、占座、招募或外发" in strategy["side_effect_boundary"]
    assert "start_time_kind 是开场方式，不代表现在时段" in description
    assert "现在不等于人齐开" in description
    hints = planning_contract()["tool_selection_hints"]["known_fact_retention"]
    assert "默认须实际写入 requirement.game_type" in hints
    assert "本次新改优先" in hints and "盘点按其显式范围" in hints and "多值或真实冲突不硬选" in hints
    assert "查询默认不授权加入、成局或外发" in hints


def test_system_reuses_query_defaults_without_promoting_them_to_user_confirmation():
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    assert "须作为默认实际写入 requirement.game_type" in prompt
    assert "不能只在已知事实或计划里参考而省参数" in prompt
    assert "默认召回不冒充客户本轮明确确认" in prompt
    assert "检索本身不授权写入" in prompt
    assert "本次新改玩法优先于长期默认" in prompt
    assert "浏览全店不追加个人画像过滤，定向盘点保留其显式限制" in prompt
    assert "未实际提交的玩法过滤不能靠后续 known_facts 补成已查条件" in prompt
    assert "画像单一可靠玩法可承接一般约局" not in prompt
    assert "默认偏好与召回参考" not in prompt
