"""精确时长是可核验的查询条件，生命周期默认值不能冒充客户约定。"""

from copy import deepcopy
from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentContextBuilder, CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_payload
from mahjong_agent_runtime.group_chat.projections import public_group_game_summary
from mahjong_agent_runtime.models import now, using_clock


@pytest.fixture(params=['memory', 'sqlite_reopened'])
def duration_store(request, tmp_path):
    """两种事实存储共用完全相同的检索及落库断言。"""
    path = tmp_path / 'duration.sqlite3'
    store = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(path)
    store.upsert_customer(CustomerProfile(customer_id='requester', display_name='客户',
        preferred_games=['hangzhou_mahjong'], preferred_stakes=['1']))
    games = {}
    for label, duration in [('four', 4), ('five', 5), ('unknown', None)]:
        requirement = {'game_type': 'hangzhou_mahjong', 'stake': '1', 'smoke_preference': 'no_smoking',
            'start_time_kind': 'scheduled', 'planned_start_at': (now() + timedelta(days=1)).isoformat()}
        if duration is not None:
            requirement['duration_hours'] = duration
        game, _ = store.create_game(conversation_id=f'pool-{label}', organizer_id=f'host-{label}',
            organizer_name='牌友', requirement=requirement,
            known_players=[{'customer_id': f'host-{label}', 'display_name': '牌友'}], trace_id=f'seed-{label}')
        games[label] = game.game_id
    if request.param == 'sqlite_reopened':
        store._connection.close()
        store = SQLiteAgentStore(path)
    yield store, games
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def query(fixture, requirement, *, limit=8):
    """只通过工具网关查询，同时检查输入、牌局与外发状态保持。"""
    store, _ = fixture
    before = {key: value.to_dict() for key, value in store.games.items()}
    original = deepcopy(requirement)
    result = ToolGateway(store).execute(ToolCall(name='search_current_games',
        arguments={'requirement': requirement, 'limit': limit}), trace_id='duration-search',
        conversation_id='request-chat', sender_id='requester', sender_name='客户', step_index=1)
    assert result.called and result.allowed and not result.error, result
    assert requirement == original
    assert {key: value.to_dict() for key, value in store.games.items()} == before
    assert not store.invite_drafts and not store.outbound_message_drafts
    return result.result


def test_max_duration_excludes_long_game_and_does_not_claim_unknown_as_exact(duration_store):
    store, ids = duration_store
    payload = query(duration_store, {'game_type': 'hangzhou_mahjong', 'stake': '1', 'max_duration_hours': 4})
    assert [row['game']['game_id'] for row in payload['matches']] == [ids['four']]
    assert [row['game']['game_id'] for row in payload['alternatives']] == [ids['unknown']]
    unknown = payload['alternatives'][0]
    assert unknown['match_kind'] == 'duration_confirmation_required'
    assert unknown['decision_required_fields'] == ['duration_hours']
    projected, = compact_tool_payload(payload)['alternatives']
    assert projected['match_kind'] == 'duration_confirmation_required'
    assert projected['decision_required_fields'] == ['duration_hours']
    assert projected['game']['game_id'] == ids['unknown'] and 'planned_end_at' not in projected['game']
    assert store.require_game(ids['unknown']).planned_end_at is not None
    assert 'duration_hours' not in store.require_game(ids['unknown']).requirement
    assert 'max_duration_hours' in payload['customer_reply_contract']['evidence_scope']['applied_filter_fields']


@pytest.mark.parametrize('maximum, expected', [(3.99, []), (4, ['four']), (5, ['four', 'five'])])
def test_maximum_boundary_is_inclusive(duration_store, maximum, expected):
    _, ids = duration_store
    payload = query(duration_store, {'max_duration_hours': maximum})
    assert {row['game']['game_id'] for row in payload['matches']} == {ids[name] for name in expected}
    assert {row['game']['game_id'] for row in payload['alternatives']} == {ids['unknown']}
    assert payload['customer_reply_contract']['evidence_scope']['supports_narrower_empty_claim'] is False


def test_exact_duration_and_limit_are_applied_before_results_are_truncated(duration_store):
    _, ids = duration_store
    payload = query(duration_store, {'duration_hours': 4}, limit=1)
    assert [row['game']['game_id'] for row in payload['matches']] == [ids['four']]
    assert not payload['alternatives']
    assert payload['customer_reply_contract']['evidence_scope']['applied_filter_fields'] == ['duration_hours']


def test_unknown_duration_requires_confirmation_even_with_supporting_profile(duration_store):
    _, ids = duration_store
    payload = query(duration_store, {'game_type': 'hangzhou_mahjong', 'max_duration_hours': 2})
    assert not payload['matches']
    assert [row['game']['game_id'] for row in payload['alternatives']] == [ids['unknown']]
    assert 'unknown' not in payload['customer_reply_contract']['matched_result_summaries']


def test_inconsistent_exact_and_maximum_cannot_promote_unknown_duration(duration_store):
    payload = query(duration_store, {'duration_hours': 4, 'max_duration_hours': 3})
    assert not payload['matches'] and not payload['alternatives']
    assert payload['customer_reply_contract']['evidence_scope']['supports_narrower_empty_claim'] is False


def test_authoritative_maximum_is_preserved_without_claiming_exact_duration(duration_store):
    store, ids = duration_store
    game, _ = store.create_game(conversation_id='bounded-goal', organizer_id='bounded-host',
        organizer_name='牌友', requirement={'game_type': 'hangzhou_mahjong', 'stake': '1',
            'start_time_kind': 'asap_when_full', 'max_duration_hours': 3.5},
        known_players=[{'customer_id': 'bounded-host', 'display_name': '牌友'}], trace_id='bounded-seed')
    payload = query(duration_store, {'max_duration_hours': 4})
    matched = next(row for row in payload['matches'] if row['game']['game_id'] == game.game_id)
    assert matched['game']['requirement']['max_duration_hours'] == 3.5
    assert 'duration_hours' not in matched['game']['requirement']
    exact = query(duration_store, {'duration_hours': 4})
    assert game.game_id not in {row['game']['game_id'] for row in exact['matches']}
    assert game.game_id not in {row['game']['game_id'] for row in exact['alternatives']}
    # 同一局的权威上限不会被误判为候选人查询改写原局。
    result = ToolGateway(store).execute(ToolCall(name='search_customers', arguments={
        'game_id': game.game_id, 'requirement': dict(game.requirement), 'exclude_customer_ids': ['bounded-host']}),
        trace_id='bounded-followup', conversation_id='bounded-goal', sender_id='bounded-host',
        sender_name='牌友', step_index=1)
    assert result.called and result.allowed and not result.error, result


def test_hard_departure_uses_each_tables_actual_start_and_agreed_duration(duration_store):
    """同一个四小时上限不足以判断能否赶在固定时刻前结束，逐桌核对并隔离默认时长。"""
    store, _ = duration_store
    time = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    deadline = "2026-11-25T23:00:00+08:00"
    ids = {}
    with using_clock(lambda: time):
        for label, opening, terms in (
            ("late-end", "19:30", {"duration_hours": 4}),
            ("fits", "19:45", {"duration_hours": 2}),
            ("exact-edge", "20:00", {"duration_hours": 3}),
            ("known-ceiling", "20:00", {"max_duration_hours": 3}),
            ("too-wide-ceiling", "20:00", {"max_duration_hours": 4}),
            ("duration-unknown", "20:00", {}),
            ("begins-at-deadline", "23:00", {}),
            ("opening-unknown", None, {"duration_hours": 2}),
        ):
            host = f"hard-deadline-{label}"
            start = ({"planned_start_at": f"2026-11-25T{opening}:00+08:00"}
                     if opening else {"start_time_kind": "asap_when_full"})
            game, _ = store.create_game(
                conversation_id=host, organizer_id=host, organizer_name="牌友",
                requirement={"game_type": "sichuan_mahjong", **start, **terms},
                known_players=[{"customer_id": host, "display_name": "牌友"}], trace_id=host,
            )
            ids[label] = game.game_id
        result = query(duration_store, {"game_type": "sichuan_mahjong", "must_end_by": deadline})
        match_ids = {row["game"]["game_id"] for row in result["matches"]}
        conditional_ids = {row["game"]["game_id"] for row in result["alternatives"]
                           if row["match_kind"] == "duration_confirmation_required"}
        assert match_ids == {ids[label] for label in ("fits", "exact-edge", "known-ceiling")}
        assert conditional_ids == {ids[label] for label in ("too-wide-ceiling", "duration-unknown", "opening-unknown")}
        match_projection = {row["game"]["game_id"]: row["game"] for row in compact_tool_payload(result)["matches"]}
        assert "planned_end_at" not in match_projection[ids["known-ceiling"]]
        assert match_projection[ids["exact-edge"]]["planned_end_at"] == deadline
        assert result["customer_reply_contract"]["evidence_scope"]["applied_filter_fields"] == ["game_type", "must_end_by"]
        assert store.require_game(ids["late-end"]).planned_end_at.isoformat() == "2026-11-25T23:30:00+08:00"
        by_start = query(duration_store, {"game_type": "sichuan_mahjong", "start_time_range": {
            "start_at": "2026-11-25T19:30:00+08:00", "end_at": "2026-11-25T19:30:00+08:00",
        }})
        assert [row["game"]["game_id"] for row in by_start["matches"]] == [ids["late-end"]]


@pytest.mark.parametrize("deadline", ["23:00", "2026-11-25T23:00:00", "invalid", True])
def test_departure_filter_rejects_missing_timezone_or_wrong_type(duration_store, deadline):
    """查局截止时间必须是带偏移的绝对时刻，拒绝使无效条件被误记成已执行。"""
    store, _ = duration_store
    result = ToolGateway(store).execute(ToolCall("search_current_games", {"requirement": {"must_end_by": deadline}}),
        trace_id="invalid-cutoff", conversation_id="request-chat", sender_id="requester", sender_name="客户", step_index=1)
    assert not result.called and not result.allowed and "must_end_by" in result.error


@pytest.mark.parametrize('value', [True, 0, -1, '4', float('nan'), float('inf')])
def test_invalid_duration_is_rejected_by_gateway_before_business_execution(duration_store, value):
    store, _ = duration_store
    before = {key: item.to_dict() for key, item in store.games.items()}
    result = ToolGateway(store).execute(ToolCall(name='search_current_games',
        arguments={'requirement': {'max_duration_hours': value}}), trace_id='invalid-duration',
        conversation_id='request-chat', sender_id='requester', sender_name='客户', step_index=1)
    assert not result.called and not result.allowed and result.error
    assert {key: item.to_dict() for key, item in store.games.items()} == before


def test_search_model_context_preserves_agreed_end_but_omits_lifecycle_defaults(duration_store):
    """店家能看到约定过的结束时间，未约定时看不到可被误报的内部默认值。"""
    _, ids = duration_store
    payload = query(duration_store, {"game_type": "hangzhou_mahjong"})
    raw = {item["game"]["game_id"]: item["game"] for item in payload["matches"]}
    model = {item["game"]["game_id"]: item["game"] for item in compact_tool_payload(payload)["matches"]}
    assert model[ids["four"]]["planned_end_at"] == raw[ids["four"]]["planned_end_at"]
    assert model[ids["four"]]["requirement"]["duration_hours"] == 4
    unknown = model[ids["unknown"]]
    assert raw[ids["unknown"]]["planned_end_at"] and unknown["planned_start_at"]
    assert {"planned_end_at", "expires_at"}.isdisjoint(unknown)
    assert {"planned_end_at", "lifecycle_expires_at", "lifecycle_ttl_hours"}.isdisjoint(unknown["requirement"])


def test_unknown_table_end_is_hidden_in_follow_up_context_and_public_projection(duration_store):
    """顾客23点离开时，先前仍未约定时长的20:45桌没有可用来判冲突的默认00:45。"""
    store, _ = duration_store
    observed = datetime.fromisoformat("2026-11-25T19:12:00+08:00")
    start_at = "2026-11-25T20:45:00+08:00"
    default_end = "2026-11-26T00:45:00+08:00"
    with using_clock(lambda: observed):
        gateway = ToolGateway(store)
        created = gateway.execute(ToolCall(name="create_game", arguments={
            "organizer_id": "time-table-host", "organizer_name": "发起人",
            "known_players": [{"customer_id": "time-table-host", "display_name": "发起人"}],
            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "planned_start_at": start_at},
        }), trace_id="original-organizer", conversation_id="owner-creation-chat", sender_id="time-table-host",
            sender_name="发起人", step_index=1)
        assert created.called and created.allowed and not created.error
        raw = created.result["game"]
        table_id = raw["game_id"]
        # 顾客看到的是能接续的外部牌桌，已有本人关联，却没有本人占席证据。
        store.create_invite_drafts(game_id=table_id, invitations=[{
            "customer_id": "requester", "display_name": "客户", "message_text": "20:45的杭麻有空位。",
        }], trace_id="earlier-related-game")
        store.append_assistant_turn("request-chat", "20:45那桌时长还没定。你能打到几点？", "prior-duration-question")
        message = UserMessage(conversation_id="request-chat", sender_id="requester", sender_name="客户",
                              text="老板，能打到十一点。20:45那桌1块位的，合适就留我一个。", sent_at=observed)
        context = AgentContextBuilder(store, gateway).build(message, trace_id="duration-follow-up").payload
        unknown = next(row for row in context["active_games"] if row["game_id"] == table_id)
        assert unknown["planned_start_at"] == start_at and unknown["remaining_seats"] == 3
        assert {"planned_end_at", "expires_at"}.isdisjoint(unknown)
        assert {"planned_end_at", "lifecycle_expires_at", "lifecycle_ttl_hours"}.isdisjoint(unknown["requirement"])
        text = "这桌还没定打到几点，得先核实能不能十一点先走。"
        action = AgentAction(goal="核对本人能否按时离场", objective_status="waiting_user", reasoning_summary="",
                             reply_to_user=text)
        review = build_reply_self_review_payload(message=message, action=action, context_payload=context,
            review_items=[{"item_id": "reply_to_user", "recipient_id": "requester", "text": text}], review_scope="reply_to_user")
        assert default_end not in json.dumps(context, ensure_ascii=False)
        assert default_end not in json.dumps(review, ensure_ascii=False)
        projection = compact_tool_payload(created.result)
        assert default_end not in json.dumps(projection, ensure_ascii=False)
        assert raw["planned_end_at"] == raw["expires_at"] == default_end
        assert store.require_game(table_id).planned_end_at.isoformat() == default_end
        assert all(p.customer_id != "requester" for p in store.require_game(table_id).participants)
        summary = public_group_game_summary(store.require_game(table_id))
        assert summary["planned_start_at"] == start_at and summary["planned_end_at"] is None


def test_opening_range_filters_by_authoritative_time_before_limit(duration_store):
    """跨日使用两个真实边界，保留区间中不在整点的局，且查询不会改写库存。"""
    from datetime import datetime
    from mahjong_agent_runtime.models import using_clock

    store = duration_store[0]
    now_at = datetime.fromisoformat("2026-11-25T22:30:00+08:00")
    opening_range = {"start_at": "2026-11-25T23:30:00+08:00", "end_at": "2026-11-26T01:00:00+08:00"}
    game_ids = {}
    with using_clock(lambda: now_at):
        for label, at in (("before", "2026-11-25T23:29:00+08:00"), ("middle", "2026-11-26T00:37:00+08:00"),
                          ("left", opening_range["start_at"]), ("right", opening_range["end_at"]),
                          ("after", "2026-11-26T01:01:00+08:00"), ("unsettled", None)):
            requirement = {"game_type": "red_center_mahjong"}
            if at:
                requirement["planned_start_at"] = at
            else:
                requirement["start_time_kind"] = "asap_when_full"
            game, _ = store.create_game(conversation_id=f"range-{label}", organizer_id=f"host-{label}", organizer_name="牌友",
                requirement=requirement, known_players=[{"customer_id": f"host-{label}", "display_name": "牌友"}],
                trace_id=f"start-range-{label}")
            game_ids[label] = game.game_id

        request = {"game_type": "red_center_mahjong", "start_time_range": opening_range}
        result = query(duration_store, request, limit=3)
        assert {entry["game"]["game_id"] for entry in result["matches"]} == {game_ids[key] for key in ("left", "middle", "right")}
        assert all("organizer_id" not in entry["game"] for entry in result["matches"])
        assert result["customer_reply_contract"]["evidence_scope"]["applied_filter_fields"] == [
            "game_type", "start_time_range",
        ]


@pytest.mark.parametrize("params", [
    {"start_time_range": {"start_at": "2026-11-25T20:00:00+08:00", "end_at": "2026-11-25T19:00:00+08:00"}},
    {"start_time_range": {"start_at": "2026-11-25T19:00:00", "end_at": "2026-11-25T21:00:00+08:00"}},
    {"start_time": "19:00", "start_time_range": {"start_at": "2026-11-25T19:00:00+08:00",
                                                  "end_at": "2026-11-25T21:00:00+08:00"}},
])
def test_invalid_opening_range_is_refused(duration_store, params):
    """拒绝错序、缺时区和与单点同时提交的区间。"""
    store, _ = duration_store
    result = ToolGateway(store).execute(ToolCall(name="search_current_games", arguments={"requirement": params}),
        trace_id="start-range-rejected", conversation_id="request-chat", sender_id="requester", sender_name="客户", step_index=1)
    assert result.error
    assert not result.allowed
