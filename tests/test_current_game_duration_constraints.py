"""精确时长是可核验的查询条件，生命周期默认值不能冒充客户约定。"""

from copy import deepcopy
from datetime import timedelta

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.models import now


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


@pytest.mark.parametrize('value', [True, 0, -1, '4', float('nan'), float('inf')])
def test_invalid_duration_is_rejected_by_gateway_before_business_execution(duration_store, value):
    store, _ = duration_store
    before = {key: item.to_dict() for key, item in store.games.items()}
    result = ToolGateway(store).execute(ToolCall(name='search_current_games',
        arguments={'requirement': {'max_duration_hours': value}}), trace_id='invalid-duration',
        conversation_id='request-chat', sender_id='requester', sender_name='客户', step_index=1)
    assert not result.called and not result.allowed and result.error
    assert {key: item.to_dict() for key, item in store.games.items()} == before
