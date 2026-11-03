"""容量过滤的证据范围必须与真实检索一致，不能把多人无位误报成全店空池。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway


@pytest.fixture(params=["memory", "sqlite"])
def capacity_store(request, tmp_path):
    """双实现使用独立事实源，读取合同不能改变局况或创建外发与跟进任务。"""
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "capacity-scope.sqlite3")
    yield store
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def _seed_open_game(store, label, remaining):
    """按真实参与者占座建立局池，避免直接改数据库人数伪造容量。"""
    owner = f"owner-{label}"
    game, _ = store.create_game(
        conversation_id=f"owner-chat-{label}", organizer_id=owner, organizer_name="牌友",
        requirement={"game_type": "national_standard_mahjong", "stake": "1", "start_time_kind": "asap_when_full"},
        known_players=[{"customer_id": owner, "display_name": "牌友", "seat_count": 4 - remaining}],
        trace_id=f"seed-{label}",
    )
    return game


def _query(store, requirement):
    """走真实 Gateway 验证合同与结果，同时确认参数和业务事实均未被补写。"""
    original = deepcopy(requirement)
    games_before = {key: game.to_dict() for key, game in store.games.items()}
    result = ToolGateway(store).execute(
        ToolCall(name="search_current_games", arguments={"requirement": requirement}, reason="核对容量查询证据"),
        trace_id="capacity-scope", conversation_id="request-chat", sender_id="requester", sender_name="客户", step_index=1,
    )
    assert result.called and result.allowed and result.error is None
    assert requirement == original
    assert {key: game.to_dict() for key, game in store.games.items()} == games_before
    assert not store.invite_drafts and not store.outbound_message_drafts and not store.scheduled_tasks
    return result.result


@pytest.mark.parametrize("requirement", [
    {"requesting_party": {"seat_count": 2}},
    {"seat_count": 2}, {"seats": 2}, {"party_size": 2}, {"player_count": 2},
])
def test_capacity_only_empty_result_is_not_unfiltered_empty_pool(capacity_store, requirement):
    """池内仍有三缺一；两人无位只证明本次容量范围为空，不证明全店无局。"""
    game = _seed_open_game(capacity_store, "one-seat", 1)
    result = _query(capacity_store, requirement)
    assert result["matches"] == [] and result["alternatives"] == []
    assert [item["game"]["game_id"] for item in capacity_store.search_current_games({})] == [game.game_id]
    contract = result["customer_reply_contract"]
    scope = contract["evidence_scope"]
    assert scope["kind"] == "submitted_filters_only"
    assert scope["applied_filter_fields"] == ["requesting_party.seat_count"]
    assert scope["unfiltered_available_pool_empty"] is False
    assert scope["supports_narrower_empty_claim"] is False
    assert contract["search_result_semantics"]["status"] == "no_actionable_match"
    assert "unfiltered inventory read" not in contract["search_result_semantics"]["instruction"]


def test_capacity_filtered_query_can_still_prove_genuinely_empty_available_pool(capacity_store):
    """容量查询的作用域不变，额外未过滤空池读取仍可提供真实子范围缺失证明。"""
    result = _query(capacity_store, {"requesting_party": {"seat_count": 2}})
    contract = result["customer_reply_contract"]
    scope = contract["evidence_scope"]
    assert result["matches"] == [] and result["alternatives"] == []
    assert scope["kind"] == "submitted_filters_only"
    assert scope["applied_filter_fields"] == ["requesting_party.seat_count"]
    assert scope["unfiltered_available_pool_empty"] is True
    assert scope["supports_narrower_empty_claim"] is True
    assert contract["search_result_semantics"]["status"] == "no_actionable_match"


@pytest.mark.parametrize("requirement", [
    {}, {"requesting_party": {}}, {"requesting_party": {"contact_id": "requester"}},
    {"requesting_party": {"seat_count": 1}}, {"seat_count": 1}, {"party_size": 1},
    {"known_player_count": 3, "needed_seats": 1}, {"current_player_count": 3},
    {"requesting_party": {"seat_count": 1}, "seat_count": 2},
    {"seat_count": "unknown"}, {"unrecognized_party_size": 2},
])
@pytest.mark.parametrize("populated", [False, True])
def test_baseline_and_non_effective_capacity_fields_remain_unfiltered(capacity_store, requirement, populated):
    """缺省与单人不收窄可用局池；桌况、未知字段及失效别名不能冒充已应用容量条件。"""
    if populated:
        _seed_open_game(capacity_store, "one-seat", 1)
    result = _query(capacity_store, requirement)
    contract = result["customer_reply_contract"]
    scope = contract["evidence_scope"]
    assert len(result["matches"]) == int(populated)
    assert scope["kind"] == "unfiltered_inventory"
    assert scope["applied_filter_fields"] == []
    assert scope["unfiltered_available_pool_empty"] is (not populated)
    assert scope["supports_narrower_empty_claim"] is (not populated)
    assert contract["search_result_semantics"]["status"] == ("unfiltered_pool_listing" if populated else "unfiltered_pool_empty")
    assert contract["search_result_semantics"]["actionable_match_count"] == 0


@pytest.mark.parametrize("requirement", [
    {"requesting_party": {"seat_count": 2}},
    {"seat_count": 2}, {"seats": 2}, {"party_size": 2}, {"player_count": 2},
    {"requesting_party": {"seat_count": 2}, "seat_count": 1},
])
def test_nonempty_capacity_query_reports_only_real_capacity_matches(capacity_store, requirement):
    """混合局池排除一空位桌并保留两空位桌，证据字段与真实投影共同证明过滤已执行。"""
    _seed_open_game(capacity_store, "one-seat", 1)
    available = _seed_open_game(capacity_store, "two-seats", 2)
    result = _query(capacity_store, requirement)
    assert [item["game"]["game_id"] for item in result["matches"]] == [available.game_id]
    assert result["matches"][0]["join_projection"]["requested_seats"] == 2
    contract = result["customer_reply_contract"]
    assert contract["evidence_scope"]["kind"] == "submitted_filters_only"
    assert contract["evidence_scope"]["applied_filter_fields"] == ["requesting_party.seat_count"]
    assert contract["evidence_scope"]["supports_narrower_empty_claim"] is False
    assert contract["search_result_semantics"]["status"] == "actionable_matches"
    assert contract["search_result_semantics"]["actionable_match_count"] == 1


def test_capacity_and_scoring_fields_are_both_retained_in_evidence_scope(capacity_store):
    """容量硬条件不能挤掉已有玩法评分条件，两者均以实际执行的规范字段声明。"""
    _seed_open_game(capacity_store, "one-seat", 1)
    result = _query(capacity_store, {"game_type": "national_standard_mahjong", "party_size": 2})
    contract = result["customer_reply_contract"]
    assert result["matches"] == [] and result["alternatives"] == []
    assert contract["evidence_scope"]["applied_filter_fields"] == ["game_type", "requesting_party.seat_count"]
    assert contract["search_result_semantics"]["status"] == "no_actionable_match"
