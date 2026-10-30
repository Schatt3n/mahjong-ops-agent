"""查询合同显式声明后端已有条件；用混合局池验证实参而非模型自述决定检索范围。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.domains.customer_domain import CURRENT_GAME_SCORE_FIELDS
from mahjong_agent_runtime.domains.tools.validation import validate_schema


@pytest.fixture(params=["memory", "sqlite", "sqlite_reopened"])
def mixed_query_gateway(request, tmp_path):
    """同一玩法和底注下混合烟况、封顶、时长，不能靠唯一一桌巧合掩盖漏条件。"""
    path = tmp_path / "query-filters.sqlite3"
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(path)
    store.upsert_customer(CustomerProfile(customer_id="requester", display_name="客户",
        preferred_games=["club_custom_mahjong"], preferred_stakes=["1"], smoke_preference="no_smoke"))
    ids = {}
    for name, smoke, cap, duration in [
        ("smoking_fixed", "smoking", 256, "fixed"),
        ("no_smoke_fixed", "no_smoke", 256, "fixed"),
        ("no_smoke_overnight", "no_smoke", 256, "overnight"),
        ("no_smoke_other_cap", "no_smoke", 128, "fixed"),
    ]:
        game, _ = store.create_game(conversation_id=f"pool-{name}", organizer_id=f"owner-{name}", organizer_name="牌友",
            requirement={"game_type": "club_custom_mahjong", "stake": "1", "smoke_preference": smoke,
                "cap_score": cap, "start_time_kind": "asap_when_full", "duration_kind": duration},
            known_players=[{"customer_id": f"owner-{name}", "display_name": "牌友", "seat_count": 1}], trace_id=f"seed-{name}")
        ids[name] = game.game_id
    if request.param == "sqlite_reopened":
        store._connection.close()
        store = SQLiteAgentStore(path)
    yield store, ToolGateway(store), ids
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def execute_query(fixture, requirement):
    """直接执行模型应提交的合同参数，不从画像或说明文字补烟况。"""
    store, gateway, _ = fixture
    before = {key: game.to_dict() for key, game in store.games.items()}
    original = deepcopy(requirement)
    call = ToolCall(name="search_current_games", arguments={"requirement": requirement, "limit": 20},
        reason="按本次已知无烟等条件检索；这段说明不能代替参数。")
    result = gateway.execute(call, trace_id="query-filter-trace", conversation_id="request-chat",
        sender_id="requester", sender_name="客户", step_index=1)
    assert result.called and result.allowed and result.error is None
    assert call.arguments["requirement"] == original
    assert {key: game.to_dict() for key, game in store.games.items()} == before
    assert not store.invite_drafts and not store.outbound_message_drafts and not store.scheduled_tasks
    return result


def matched_ids(result):
    return {row["game"]["game_id"] for row in result.result["matches"]}


def test_model_query_schema_declares_every_canonical_scored_field_without_new_required_slots():
    """显式列出实际能力，不能让开放附加字段掩盖参数说明缺失。"""
    gateway = ToolGateway(InMemoryAgentStore())
    spec = next(item for item in gateway.tool_specs_for_prompt() if item["name"] == "search_current_games")
    query = spec["schema"]["properties"]["requirement"]
    canonical = {item[0] for item in CURRENT_GAME_SCORE_FIELDS} | {"cap_score"}
    assert canonical <= query["properties"].keys()
    assert query["additionalProperties"] is True and not query.get("required")
    assert "逐项写入参数" in query["description"] and "未知条件可省略" in query["description"]
    assert "待确认备选" in query["properties"]["smoke_preference"]["description"]
    assert "排序" in query["properties"]["duration_kind"]["description"]
    assert "不保证排除" in query["properties"]["duration_kind"]["description"]
    assert validate_schema({"requirement": {}}, spec["schema"]) is None
    # 只声明已有规范化协议，不借 schema 变更禁用此前可用的多值表达。
    assert validate_schema({"requirement": {"smoke_preference": ["no_smoke", "smoking"],
        "duration_kind": ["fixed", "overnight"], "cap_score": "256"}}, spec["schema"]) is None
    assert spec["execution_mode"] == "read_only" and spec["parallel_safe"] is True


@pytest.mark.parametrize("smoke", ["no_smoking", "no_smoke", "无烟", ["no_smoking"]])
def test_explicit_smoke_filter_reaches_real_mixed_inventory(mixed_query_gateway, smoke):
    _, _, ids = mixed_query_gateway
    result = execute_query(mixed_query_gateway, {"game_type": "club_custom_mahjong", "stake": "1", "smoke_preference": smoke})
    assert matched_ids(result) == {ids["no_smoke_fixed"], ids["no_smoke_overnight"], ids["no_smoke_other_cap"]}
    assert result.result["alternatives"] == []
    assert result.result["requirement"]["smoke_preference"] == smoke
    assert "smoke_preference" in result.result["customer_reply_contract"]["evidence_scope"]["applied_filter_fields"]


@pytest.mark.parametrize("smoke", ["smoking", "有烟", "烟"])
def test_explicit_smoking_is_not_overwritten_by_no_smoke_profile(mixed_query_gateway, smoke):
    _, _, ids = mixed_query_gateway
    result = execute_query(mixed_query_gateway, {"game_type": "club_custom_mahjong", "stake": "1", "smoke_preference": smoke})
    assert matched_ids(result) == {ids["smoking_fixed"]}
    assert result.result["requirement"]["smoke_preference"] == smoke


@pytest.mark.parametrize("extra", [{}, {"smoke_preference": "any"}, {"smoke_preference": ["no_smoke", "smoking"]}])
def test_omitted_or_unrestricted_smoke_does_not_silently_apply_profile(mixed_query_gateway, extra):
    _, _, ids = mixed_query_gateway
    result = execute_query(mixed_query_gateway, {"game_type": "club_custom_mahjong", "stake": "1", **deepcopy(extra)})
    assert matched_ids(result) == set(ids.values())
    if not extra:
        assert "smoke_preference" not in result.result["requirement"]
        assert "smoke_preference" not in result.result["customer_reply_contract"]["evidence_scope"]["applied_filter_fields"]
    else:
        assert result.result["requirement"]["smoke_preference"] == extra["smoke_preference"]


@pytest.mark.parametrize("cap", [256, "256"])
def test_cap_score_is_independent_of_base_stake_and_excludes_known_conflict(mixed_query_gateway, cap):
    _, _, ids = mixed_query_gateway
    result = execute_query(mixed_query_gateway, {"game_type": "club_custom_mahjong", "stake": "1",
        "smoke_preference": "no_smoking", "cap_score": cap})
    assert matched_ids(result) == {ids["no_smoke_fixed"], ids["no_smoke_overnight"]}
    assert result.result["requirement"]["stake"] == "1"
    assert result.result["requirement"]["cap_score"] == 256
    assert "stake_range" not in result.result["requirement"]
    assert "cap_score" in result.result["customer_reply_contract"]["evidence_scope"]["applied_filter_fields"]


@pytest.mark.parametrize("duration", ["fixed", ["fixed"]])
def test_duration_category_ranks_but_does_not_claim_a_hard_hour_filter(mixed_query_gateway, duration):
    _, _, ids = mixed_query_gateway
    result = execute_query(mixed_query_gateway, {"game_type": "club_custom_mahjong", "stake": "1",
        "smoke_preference": "no_smoking", "cap_score": 256, "duration_kind": duration})
    matches = result.result["matches"]
    assert [item["game"]["game_id"] for item in matches] == [ids["no_smoke_fixed"], ids["no_smoke_overnight"]]
    assert matches[0]["score"] > matches[1]["score"]
    assert "duration_kind_matched" in matches[0]["reasons"]
    assert "duration_kind_matched" not in matches[1]["reasons"]
    assert result.result["requirement"]["duration_kind"] == duration
    assert "duration_hours" not in result.result["requirement"]


def test_filter_descriptions_do_not_change_requester_capacity_scope(mixed_query_gateway):
    _, _, ids = mixed_query_gateway
    result = execute_query(mixed_query_gateway, {"game_type": "club_custom_mahjong", "stake": "1",
        "smoke_preference": "no_smoking", "requesting_party": {"seat_count": 3}})
    assert matched_ids(result) == {ids["no_smoke_fixed"], ids["no_smoke_overnight"], ids["no_smoke_other_cap"]}
    assert all(row["join_projection"]["requested_seats"] == 3 and row["join_projection"]["would_fill_game"]
        for row in result.result["matches"])
    assert "known_player_count" not in result.result["requirement"]
    assert "needed_seats" not in result.result["requirement"]
