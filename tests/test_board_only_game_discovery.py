"""群友自招局的查询候选需要桌主确认，公开事实之外的信息不能随工具结果传播。"""
from datetime import timedelta
import json

import pytest

from mahjong_agent_runtime import AgentAction, CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_payload
from mahjong_agent_runtime.group_chat import GroupRoomPolicy, L2IntentRouter
from mahjong_agent_runtime.group_chat.semantic_topic_router import SemanticTopicClassifier
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.tracing import InMemoryTraceRecorder
from tests.test_group_chat.test_claim_and_switch import _group_message, _identity, _stack
from tests.test_group_chat.test_semantic_topic_router import RecordingLLM, decision


@pytest.fixture(params=['memory', 'sqlite'])
def store(request, tmp_path):
    backend = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path / 'board-discovery.sqlite3')
    backend.upsert_customer(CustomerProfile('board-owner-private-id', '桌主私人姓名', public_name='桌主公开昵称',
                                           private_remark='桌主私密备注'))
    backend.upsert_customer(CustomerProfile('requester', '客户B', gender='男', preferred_stakes=['1', '3']))
    try:
        yield backend
    finally:
        if request.param == 'sqlite':
            backend._connection.close()


def create_board_game(store, **overrides):
    """保存与原故障相同的19点财敲三缺一，档位由各案例明确给出或省略。"""
    requirement = {'game_type': 'hangzhou_mahjong', 'game_variant': 'caiqiao', 'ruleset': '财敲',
                   'start_time': '19:00', 'smoke_preference': 'no_smoking', 'preferred_gender': '女',
                   'recruitment_mode': 'board_only', 'user_visible_summary': '不可跨会话使用的私密自由文本',
                   **overrides}
    game, _ = store.create_game(conversation_id='host-private-conversation', organizer_id='board-owner-private-id',
        organizer_name='桌主私人姓名', requirement=requirement,
        known_players=[{'customer_id': 'board-owner-private-id', 'display_name': '桌主私人姓名', 'seat_count': 3}],
        trace_id='register-board-game')
    return game


def search(store, **query):
    """经真实网关复现B的原查询参数，不能以模型自述代替成功检索。"""
    requirement = {'stake_range': {'min': 0.5, 'max': 2}, 'smoke_preference': 'any', **query}
    result = ToolGateway(store).execute(ToolCall('search_current_games', {'requirement': requirement}, '查现成局'),
        trace_id='requester-search', conversation_id='requester-private-conversation', sender_id='requester',
        sender_name='客户B', step_index=1)
    assert result.called and result.allowed and result.error is None
    return result.result


def self_recruitment_stack(store, llm, *, default_game_type=None):
    """走正式群入口和工具网关，模拟器只替代模型输出与微信交付。"""
    messenger, runtime, board, _, handler = _stack(store)
    board.clock = now
    runtime.tool_gateway = ToolGateway(store, trace_recorder=InMemoryTraceRecorder())
    handler.intent_router = L2IntentRouter(store, classifier=SemanticTopicClassifier(llm))
    store.upsert_group_room_policy(GroupRoomPolicy(
        room_id='room-1', self_recruitment_default_game_type=default_game_type))
    _identity(store, 'poster', friend=True, public_name='发帖人')
    return messenger, runtime, handler


def test_group_self_recruitment_recovers_then_supplements_and_private_search_finds_same_game(store):
    summary = {'game_type': 'hangzhou_mahjong', 'ruleset': '财敲', 'smoking': 'no_smoking'}
    first = decision('self_recruitment', goal_relation='none', public_summary=summary, constraints={'party_size': 3})
    corrected = {**first, 'constraints': {**summary, 'party_size': 3}}

    class NoSideEffectBeforeCorrection(RecordingLLM):
        def complete(self, messages, **kwargs):
            assert not store.games and not store.get_group_topic_message('wechaty', 'room-1', 'customer:poster', 'm1')
            return super().complete(messages, **kwargs)

    llm = NoSideEffectBeforeCorrection(first, corrected)
    messenger, runtime, handler = self_recruitment_stack(store, llm)
    post = _group_message('十点半，371，财敲，无烟', 'poster', message_id='m1')
    post.sent_at = now()
    registered = handler.handle(post, trace_id='self-recruitment-corrected')
    receipt = registered.detail['topic_routing']
    assert receipt['message']['classification']['audit']['contract_recovery_count'] == 1
    assert registered.game_id and list(store.games) == [registered.game_id]
    assert registered.detail['board_arrangement']['status'] != 'not_requested'

    supplement = decision('self_recruitment', topic_action='existing', topic_id=receipt['topic']['topic_id'],
                          goal_relation='none', source_message_id='m2', public_summary=summary,
                          constraints={'preferred_gender': '女'})
    handler.intent_router = L2IntentRouter(store, classifier=SemanticTopicClassifier(RecordingLLM(supplement)))
    followup = _group_message('最好来个女生', 'poster', message_id='m2')
    followup.sent_at = now()
    updated = handler.handle(followup, trace_id='self-recruitment-supplement')

    assert updated.game_id == registered.game_id and list(store.games) == [registered.game_id]
    game = store.require_game(registered.game_id)
    assert game.requirement['preferred_gender'] == '女' and 'required_gender' not in game.requirement
    assert game.planned_start_at is None and 'stake' not in game.requirement
    option, = search(store, game_type='hangzhou_mahjong')['alternatives']
    assert option['game']['game_id'] == registered.game_id
    assert {'stake', 'host_confirmation'} <= set(option['decision_required_fields'])
    assert not (messenger.group_messages or messenger.private_messages or runtime.user_messages or store.invite_drafts)


@pytest.mark.parametrize(('default_game_type', 'start_when_full', 'explicit_game_type', 'expected_type'), [
    ('hangzhou_mahjong', True, None, 'hangzhou_mahjong'),
    ('hangzhou_mahjong', False, None, 'hangzhou_mahjong'),
    ('hangzhou_mahjong', False, 'national_standard_mahjong', 'national_standard_mahjong'),
    (None, False, None, None),
])
def test_compact_self_recruitment_only_uses_configured_default_for_omitted_type(
        store, default_game_type, start_when_full, explicit_game_type, expected_type):
    summary = {'stakes': ['1'], 'smoking': 'smoking'}
    if explicit_game_type:
        summary['game_type'] = explicit_game_type
    constraints = {**summary, 'party_size': 3}
    if start_when_full:
        constraints['start_when_full'] = True
    llm = RecordingLLM(decision('self_recruitment', goal_relation='none', public_summary=summary, constraints=constraints))
    messenger, runtime, handler = self_recruitment_stack(store, llm, default_game_type=default_game_type)
    text = '国标1块有烟371' if explicit_game_type else '1块有烟371'
    post = _group_message(text + (' 人齐开' if start_when_full else ''), 'poster', message_id='m1')
    post.sent_at = now()

    result = handler.handle(post, trace_id='compact-self-recruitment')

    assert result.detail['topic_routing']['message']['classification']['constraints'] == constraints
    if expected_type is None:
        assert result.game_id is None and not store.games
        events = runtime.tool_gateway.trace_recorder.get_trace('compact-self-recruitment')
        assert [(event.step, event.content['fields']) for event in events] == [
            ('group_self_recruitment_not_registered', ['game_type']),
        ]
    else:
        game = store.require_game(result.game_id)
        assert game.requirement['game_type'] == expected_type and game.requirement['stake'] == '1'
        assert game.planned_start_at is None
        assert game.requirement.get('start_time') == ('人齐开' if start_when_full else None)
    assert not (messenger.group_messages or messenger.private_messages or runtime.user_messages or store.invite_drafts)


def test_unknown_stake_board_game_is_discoverable_but_requires_host_and_price_confirmation(store):
    game = create_board_game(store)
    result = search(store)
    assert result['matches'] == []
    option, = result['alternatives']
    assert option['match_kind'] == 'host_confirmation_required'
    assert set(option['decision_required_fields']) == {'host_confirmation', 'stake'}
    assert option['game']['game_id'] == game.game_id
    assert option['game']['requirement']['game_variant'] == 'caiqiao'
    assert option['game']['requirement']['ruleset'] == '财敲'
    assert option['game']['requirement']['start_time'] == '19:00'
    assert option['game']['requirement']['smoke_preference'] == 'no_smoking'
    assert option['game']['seat_summary']['remaining_seats'] == 1
    assert not {'stake', 'base_stake', 'stake_range', 'stake_label'} & set(option['game']['requirement'])
    semantics = result['customer_reply_contract']['search_result_semantics']
    assert semantics['status'] == 'decision_required_alternatives'
    assert semantics['actionable_match_count'] == 0 and semantics['host_confirmation_required_count'] == 1
    assert result['customer_reply_contract']['evidence_scope']['unfiltered_available_pool_empty'] is False
    visible = json.dumps(result, ensure_ascii=False)
    for private in ('board-owner-private-id', '桌主私人姓名', '桌主公开昵称', '桌主私密备注', 'host-private-conversation',
                    'preferred_gender', 'required_gender', '不可跨会话使用的私密自由文本'):
        assert private not in visible
    assert not store.invite_drafts and not store.outbound_message_drafts
    assert store.require_game(game.game_id).to_dict() == game.to_dict()


@pytest.mark.parametrize('overrides, query', [
    ({'stake': '3'}, {}),
    ({}, {'game_type': 'national_standard_mahjong'}),
    ({}, {'game_variant': 'other_variant'}),
    ({}, {'start_time': '20:00'}),
    ({}, {'requesting_party': {'seat_count': 2}}),
    ({}, {'smoke_preference': 'smoking'}),
    ({'duration_hours': 5}, {'max_duration_hours': 4}),
    ({'required_gender': '女'}, {}),
])
def test_board_only_does_not_override_known_hard_conflict(store, overrides, query):
    create_board_game(store, **overrides)
    result = search(store, **query)
    assert result['matches'] == result['alternatives'] == []


def test_known_price_and_matching_or_unknown_gender_still_require_host_confirmation(store):
    store.upsert_customer(CustomerProfile('requester', '客户B', gender='未知'))
    game = create_board_game(store, stake='1', required_gender='女')
    result = search(store)
    assert result['matches'] == []
    option, = result['alternatives']
    assert option['game']['game_id'] == game.game_id
    assert option['match_kind'] == 'host_confirmation_required'
    assert option['decision_required_fields'] == ['host_confirmation']
    assert option['game']['requirement']['stake'] == '1'
    assert 'required_gender' not in json.dumps(option, ensure_ascii=False)


def test_different_explicit_calendar_time_is_not_a_board_option(store):
    start = (now() + timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
    create_board_game(store, planned_start_at=start.isoformat(), start_time_kind='scheduled')
    result = search(store, planned_start_at=(start + timedelta(days=1)).isoformat(), start_time='19:00')
    assert result['matches'] == result['alternatives'] == []


def test_missing_board_smoke_remains_unknown_under_explicit_smoke_constraint(store):
    create_board_game(store, smoke_preference=None)
    result = search(store, smoke_preference='no_smoking')
    assert result['matches'] == []
    option, = result['alternatives']
    assert set(option['decision_required_fields']) == {'host_confirmation', 'stake', 'smoke_preference'}
    assert 'smoke_preference' not in option['game']['requirement']


def test_conditional_public_option_survives_compaction_and_reply_review_without_private_facts(store):
    create_board_game(store)
    result = search(store)
    compact = compact_tool_payload(result)
    option, = compact['alternatives']
    assert option['match_kind'] == 'host_confirmation_required'
    assert option['decision_required_fields'] == ['host_confirmation', 'stake']
    assert option['game'] == result['alternatives'][0]['game']
    message = UserMessage(conversation_id='requester-private-conversation', sender_id='requester', sender_name='客户B',
                          text='0.5到2块都行，有什么局？')
    reply = '七点有个财敲三缺一，档位和能不能上得问桌主。'
    review = build_reply_self_review_payload(message=message,
        action=AgentAction(goal='回答现成局咨询', objective_status='completed', reasoning_summary='依据真实查询',
                           reply_to_user=reply),
        review_items=[{'item_id': 'reply_to_user', 'recipient_id': 'requester', 'text': reply}],
        context_payload={'turn_tool_evidence': [{'name': 'search_current_games', 'called': True, 'allowed': True,
                                                'result': compact}]}, review_scope='reply_to_user')
    evidence = review['turn_tool_evidence'][0]['result']
    assert evidence['alternatives'] == [option]
    assert "pending the organizer's consent" in evidence['customer_reply_contract']['customer_visible_rule']
    serialized = json.dumps(review, ensure_ascii=False)
    assert all(value not in serialized for value in ('桌主私人姓名', '桌主公开昵称', 'preferred_gender', 'required_gender'))
