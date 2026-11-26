"""容量过滤的证据范围必须与真实检索一致，不能把多人无位误报成全店空池。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from mahjong_agent_runtime import (
    AgentAction, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_payload
from mahjong_agent_runtime.services import ToolExecutionService


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


@pytest.mark.parametrize("seat_hint", [1, 3, None], ids=["conflict", "same-seats", "unknown-seats"])
def test_table_seat_reference_conflict_is_scoped_to_named_lookup_and_reply_review(capacity_store, seat_hint):
    """实际候选缺三人不能证明顾客指的缺一桌，同轮另查仍返回自己的检索结果。"""
    game = _seed_open_game(capacity_store, "board-row", 3)
    reference = "刚才群里那桌" if seat_hint is None else f"刚才群里差{seat_hint}人那桌"
    message = UserMessage("request-chat", "requester", "客户", f"{reference}是国标吗？另有国标能打吗？")
    requirement = {"game_type": "national_standard_mahjong"}
    if seat_hint is not None:
        requirement["needed_seats"] = seat_hint
    service = ToolExecutionService(capacity_store, ToolGateway(capacity_store), InMemoryTraceRecorder())
    action = AgentAction(goal="分别查原桌和其他可打的桌", objective_status="needs_tool", reasoning_summary="", tool_calls=[
        ToolCall("search_current_games", {"requirement": requirement, "lookup_scope": scope}, call_id=scope, depends_on=[])
        for scope in ("specific_table", "personal")
    ])
    results = service.execute_tool_calls(action, message=message, trace_id="table-reference",
        previous_step_tool_results=[], step_index=1, run_id="table-reference",
        run_version=capacity_store.conversation_version(message.conversation_id)).tool_results
    named, personal = results
    for result in results:
        assert result.called and result.allowed and result.error is None
        (match,) = result.result["matches"]
        assert match["game"]["game_id"] == game.game_id
        assert match["game"]["seat_summary"]["remaining_seats"] == 3
    assert named.result["requirement"].get("needed_seats") == seat_hint
    conflict = {"requested_remaining_seats": 1,
                "conflicting_candidates": [{"game_id": game.game_id, "remaining_seats": 3}]}
    if seat_hint == 1:
        assert named.result["table_reference_evidence"] == conflict
        assert compact_tool_payload(named.result)["table_reference_evidence"] == conflict
    else:
        assert "table_reference_evidence" not in named.result
        assert "table_reference_evidence" not in compact_tool_payload(named.result)
    assert "table_reference_evidence" not in personal.result
    assert "table_reference_evidence" not in compact_tool_payload(personal.result)

    reply = "你指的是群里哪一桌？另有国标，还缺三人。"
    action = AgentAction(goal="核对原桌并告知另查结果", objective_status="waiting_user", reasoning_summary="",
                         reply_to_user=reply)
    payload = build_reply_self_review_payload(message=message, action=action, review_scope="reply_to_user",
        context_payload={"turn_tool_evidence": [result.to_dict() for result in results]},
        review_items=[{"item_id": "reply_to_user", "recipient_id": message.sender_id, "text": reply}])
    evidence = payload["semantic_fidelity_contract"]["game_availability_evidence"]
    assert evidence["searches_by_scope"] == {
        "specific_table": {"status": "completed", "matched_count": 1, "call_id": "specific_table",
                           **({"table_reference_evidence": conflict} if seat_hint == 1 else {})},
        "personal": {"status": "completed", "matched_count": 1, "call_id": "personal"},
    }
    assert evidence["latest_search"] == {"status": "completed", "matched_count": 1}


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
