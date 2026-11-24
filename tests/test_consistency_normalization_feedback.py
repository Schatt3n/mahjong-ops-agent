"""规范化冲突须在Gateway前拒绝，完整来源反馈回灌后才允许有据纠正与真实执行。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mahjong_agent_runtime import (AgentAction, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder,
    SQLiteAgentStore, StaticAgentClient, ToolCall, ToolGateway, UserMessage)
from mahjong_agent_runtime.domains.game_domain import normalize_requirement
from mahjong_agent_runtime.group_chat.parsing import explicit_numeric_cap_candidates
from mahjong_agent_runtime.tool_consistency import validate_explicit_task_fact_consistency, validate_tool_call_consistency
from mahjong_agent_runtime.models import ToolResult


@pytest.fixture(params=['memory', 'sqlite'])
def store(request, tmp_path):
    """真实双Store验证工具反馈持久化和恢复，不替代主循环或事务。"""
    value = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path/'consistency.sqlite3')
    yield value
    if isinstance(value, SQLiteAgentStore):
        value._connection.close()


class TrackingGateway:
    """只记录真实Gateway入口，实际权限、规范化和Store写入仍交生产实现。"""
    def __init__(self, store, trace):
        self.delegate = ToolGateway(store=store, trace_recorder=trace)
        self.executed = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def execute(self, call, **kwargs):
        self.executed.append(call.name)
        return self.delegate.execute(call, **kwargs)


def action_payload(call=None, *, assertions=(), reply='', status='completed', known_facts=None):
    """仅提供模型协议，事实解释由固定正反断言输入表示，后端不能擅自挑选一项执行。"""
    return json.dumps(AgentAction(goal='按客户最新原话组局', objective_status='needs_tool' if call else status,
        reasoning_summary='受控恢复连线', reply_to_user=reply, tool_calls=[call] if call else [],
        objective_state={'fact_assertions': list(assertions), 'known_facts': known_facts or {}},
        stop_reason={'can_stop': call is None, 'why': '继续工具' if call else '登记已完成',
                     'pending_work': ['推进组局'] if call else [], 'depends_on_tool_results': True}).to_dict(), ensure_ascii=False)


def create_call(requirement, *, call_id):
    return ToolCall(name='create_game', call_id=call_id, arguments={
        'organizer_id': 'customer', 'organizer_name': '客户', 'requirement': requirement,
        'requesting_party': {'contact_id': 'customer', 'contact_name': '客户', 'seat_count': 1}}, reason='创建已授权组局')


def assert_no_business_effects(store, gateway):
    assert gateway.executed == []
    assert not store.games and not store.invite_drafts and not store.outbound_message_drafts and not store.scheduled_tasks


@pytest.mark.parametrize('text,expected', [
    ('这次底注1到3块都能打，256封顶，帮我组', [256]),
    ('封顶 １２８．５ 元。', [128.5]),
    ('有人说：256封顶。', []),
    ('他说“今晚，256封顶，能来吗”。', []),
    ('256封顶？', []),
    ('不接受256封顶。', []),
    ('上次256封顶，这次没约定上限。', []),
    ('底注1元到256元，没有约定封顶。', []),
    ('1,256封顶。', []),
    ('256封顶，128封顶。', [128, 256]),
])
def test_explicit_numeric_cap_candidates_preserve_literal_scope(text, expected):
    """保护完整数值，保留疑问、转述、歧义和其他金额语法给模型解释。"""
    assert explicit_numeric_cap_candidates(text) == expected


def test_source_cap_rejects_truncated_query_and_create_before_writing(store):
    """来源同样验查局和建局，多一步纠正不能把错误写入或缩短金额留下。"""
    message = UserMessage(conversation_id='cap-source-chat', sender_id='customer', sender_name='客户',
        text='这次底注1到3块都能打，256封顶，我一个人，烟不限，人齐开，帮我组', message_id='cap-source')
    requirement = {'game_type': 'national_standard_mahjong', 'stake_range': {'min': 1, 'max': 3},
                   'start_time_kind': 'asap_when_full', 'smoke_preference': 'any'}
    trace = InMemoryTraceRecorder()
    gateway = TrackingGateway(store, trace)

    def cap_facts(value):
        return [
            {'field': 'stake_range', 'value': requirement['stake_range'], 'role': 'current', 'source_message_id': message.message_id},
            {'field': 'cap_score', 'value': value, 'role': 'current', 'source_message_id': message.message_id},
        ]

    def search(value):
        return ToolCall('search_current_games', {'requirement': {**requirement, 'cap_score': value}},
                        call_id=f'search-{value}', reason='按本轮条件查现成局')

    class Main:
        """固定模型参数，实际拒绝、反馈和后续执行均经过生产运行时。"""

        calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            context = json.loads(messages[-1]['content'])
            if self.calls == 1:
                return action_payload(search(56), assertions=cap_facts(56))
            if self.calls in (2, 4):
                blocked = context['turn_tool_evidence'][-1]
                assert blocked['result']['reference_requirement']['cap_score'] == 256
                assert blocked['result']['reference_fact_sources']['cap_score']['source_message_id'] == message.message_id
                assert blocked['result']['validation_error']['code'] == 'explicit_task_fact_consistency_rejected'
                assert blocked['called'] is blocked['allowed'] is False and not blocked.get('state_transitions')
                assert not store.games and not store.invite_drafts and not store.outbound_message_drafts and not store.scheduled_tasks
            if self.calls == 2:
                return action_payload(search(256), assertions=cap_facts(256))
            if self.calls in (3, 4):
                value = 56 if self.calls == 3 else 256
                return action_payload(create_call({**requirement, 'cap_score': value}, call_id=f'create-{value}'),
                    assertions=cap_facts(value))
            if self.calls == 5:
                game_id = context['turn_tool_evidence'][-1]['result']['game']['game_id']
                return action_payload(ToolCall('search_customers', {'game_id': game_id}, call_id='candidate-read',
                                               reason='按已建局的条件找候选'))
            assert self.calls == 6
            return action_payload(reply='好，先帮你组。')

    runtime = AgentRuntime(llm_client=Main(), store=store, tool_gateway=gateway, trace_recorder=trace)
    result = runtime.handle_user_message(message, trace_id='source-cap-recovery')
    assert result.status == 'completed', [event.content for event in trace.get_trace(result.trace_id)
                                        if event.step in {'llm_error', 'action_contract_error', 'tool_explicit_fact_consistency_error'}]
    assert [tool.name for tool in result.tool_results if not tool.called and not tool.state_transitions] == ['search_current_games', 'create_game']
    assert gateway.executed == ['search_current_games', 'create_game', 'search_customers']
    assert next(tool for tool in result.tool_results if tool.name == 'search_current_games' and tool.called).result['requirement']['cap_score'] == 256
    assert len(store.games) == 1 and not store.invite_drafts and not store.outbound_message_drafts
    assert next(iter(store.games.values())).requirement['cap_score'] == 256


def test_explicit_cap_must_be_preserved_in_create_only_when_bound():
    """带封顶的当前事实不能在建局丢弃，未按个人条件过滤的只读查询仍可进行。"""
    facts = {'explicit_task_facts': {'facts': {'cap_score': 256}}}
    requirement = {'stake_range': {'min': 1, 'max': 3}}
    read = ToolCall('search_current_games', {'requirement': requirement})
    create = create_call(requirement, call_id='omit-cap')
    assert validate_explicit_task_fact_consistency(read, facts)[0] is None
    assert 'explicit current-task cap must be retained' in validate_explicit_task_fact_consistency(create, facts)[0]
    no_cap = {'explicit_task_facts': {'facts': {'cap_score': None}}}
    assert validate_explicit_task_fact_consistency(create, no_cap)[0] is None


@pytest.mark.parametrize('conflict_in', ['reference', 'proposal'])
@pytest.mark.parametrize('same_source', [False, True])
def test_conflicting_requirement_is_refused_and_model_can_correct_with_complete_feedback(store, conflict_in, same_source, tmp_path):
    """拒绝步无业务调用/写/外发；主模型收到持久反馈，修正角色或参数后完成创建与候选检索。"""
    trace = InMemoryTraceRecorder()
    gateway = TrackingGateway(store, trace)
    old = UserMessage(conversation_id='conflict-chat', sender_id='customer', sender_name='客户',
        text='可接受1到3块；具体档位等本次决定。', message_id='range-source')
    store.append_user_turn(old, 'old-source')
    current = UserMessage(conversation_id='conflict-chat', sender_id='customer', sender_name='客户',
        text='1到3块可以接受，这次固定2块，国标，烟不限，人齐开，我一个人，帮我组一桌。', message_id='fixed-source')
    requirement = {'game_type': 'national_standard_mahjong', 'stake': '2', 'smoke_preference': 'any',
                   'start_time_kind': 'asap_when_full', 'known_player_count': 1, 'needed_seats': 3}
    range_source = current.message_id if same_source else old.message_id
    range_assertion = {'field': 'stake_range', 'value': {'min': 1, 'max': 3}, 'role': 'current', 'source_message_id': range_source}
    fixed_assertion = {'field': 'stake', 'value': '2', 'role': 'current', 'source_message_id': current.message_id}
    bad_requirement = deepcopy(requirement)
    bad_assertions = [range_assertion, fixed_assertion] if conflict_in == 'reference' else [fixed_assertion]
    if conflict_in == 'proposal':
        bad_requirement['stake_range'] = {'min': 1, 'max': 3}
    fixed_assertions = [{**range_assertion, 'role': 'condition'}, fixed_assertion]
    contexts = []

    class Main:
        """只替代远端决策；拒绝反馈检查发生在真实第二次主模型调用之前。"""
        def complete(self, messages, **kwargs):
            context = json.loads(messages[-1]['content'])
            contexts.append(context)
            if len(contexts) == 1:
                return action_payload(create_call(bad_requirement, call_id='rejected-create'), assertions=bad_assertions)
            if len(contexts) == 2:
                assert_no_business_effects(store, gateway)
                rejected = next(item for item in context['turn_tool_evidence'] if item['name'] == 'create_game' and not item['called'])
                assert rejected['allowed'] is False and rejected['error']
                assert rejected['result']['validation_error']['code'] == 'explicit_task_fact_consistency_rejected'
                reference = rejected['result']['reference_requirement']
                assert reference['stake'] == '2'
                if conflict_in == 'reference':
                    assert reference['stake_range'] == {'min': 1, 'max': 3}
                    assert rejected['result']['reference_fact_sources'] == {
                        'stake_range': {'source_message_id': range_source, 'role': 'current'},
                        'stake': {'source_message_id': 'fixed-source', 'role': 'current'}}
                assert any(turn.role.value == 'tool' and 'validation_error' in turn.content for turn in store.recent_turns(current.conversation_id))
                # 可接受范围仍作为条件保留；新原话已选固定2块，不把两种表示同时声明为当前工具事实。
                return action_payload(create_call(requirement, call_id='corrected-create'), assertions=fixed_assertions,
                    known_facts={'accepted_stake_range': {'value': {'min': 1, 'max': 3}, 'source_message_id': old.message_id}})
            if len(contexts) == 3:
                game_id = next(item for item in context['turn_tool_evidence'] if item['name'] == 'create_game' and item['called'])['result']['game']['game_id']
                return action_payload(ToolCall(name='search_customers', call_id='candidate-read', arguments={'game_id': game_id,
                    'requirement': deepcopy(requirement)}, reason='继续真实招募闭环'), assertions=fixed_assertions)
            assert len(contexts) == 4
            return action_payload(reply='好，先给你记下。')

    review = StaticAgentClient([json.dumps({'approved': True, 'needs_human': False, 'reasoning_summary': '真实登记后的短收讫',
        'violations': [], 'item_reviews': [{'item_id': 'reply_to_user', 'approved': True, 'suggested_safe_text': '好，先给你记下。',
        'reasoning_summary': '未声称已外发或持续通知', 'violations': []}]}, ensure_ascii=False)])
    runtime = AgentRuntime(llm_client=Main(), store=store, tool_gateway=gateway, trace_recorder=trace,
        reply_self_review_enabled=True, reply_self_review_client=review)
    result = runtime.handle_user_message(current, trace_id='normalization-recovery')
    assert result.status == 'completed' and result.final_reply == '好，先给你记下。'
    assert len(contexts) == 4 and gateway.executed == ['create_game', 'search_customers']
    rejected = result.tool_results[0]
    assert rejected.called is False and rejected.allowed is False
    assert ('invalid explicit current-task fact reference' if conflict_in == 'reference' else 'invalid proposed') in rejected.error
    assert len(store.games) == 1 and not store.invite_drafts and not store.outbound_message_drafts
    game = next(iter(store.games.values()))
    assert game.requirement['stake'] == '2' and 'stake_range' not in game.requirement
    assert game.seat_summary()['claimed_seats'] == 1 and game.remaining_seats() == 3
    events = trace.get_trace(result.trace_id)
    feedback_index = next(i for i, event in enumerate(events) if event.step == 'tool_argument_consistency_feedback')
    assert not any(event.step in {'tool_called', 'state_transition', 'final_output'} for event in events[:feedback_index])
    assert sum(event.step == 'tool_explicit_fact_consistency_error' for event in events) == 1
    assert [turn.content for turn in store.recent_turns(current.conversation_id) if turn.role.value == 'user'] == [old.text, current.text]
    assert [turn.content for turn in store.recent_turns(current.conversation_id) if turn.role.value == 'assistant'] == [result.final_reply]
    (tmp_path/'runtime_trace.json').write_text(json.dumps({'store_kind': 'sqlite' if isinstance(store, SQLiteAgentStore) else 'memory',
        'conflict_in': conflict_in, 'same_source': same_source, 'result': result.to_dict(),
        'events': [event.to_dict() for event in events]}, ensure_ascii=False, indent=2)+'\n')
    if isinstance(store, SQLiteAgentStore):
        recovered = SQLiteAgentStore(store.path)
        try:
            assert recovered.require_game(game.game_id).requirement['stake'] == '2'
            feedback_turns = [turn for turn in recovered.recent_turns(current.conversation_id) if turn.role.value == 'tool' and 'validation_error' in turn.content]
            assert len(feedback_turns) == 1
        finally:
            recovered._connection.close()


@pytest.mark.parametrize('role', ['condition', 'quoted', 'uncertain'])
def test_noncurrent_range_does_not_poison_valid_fixed_stake_execution(store, role):
    """同源角色的区别保留；条件、引语或未确定范围不能被当成当前固定档位的冲突事实。"""
    from mahjong_agent_runtime.services import ToolExecutionService
    trace = InMemoryTraceRecorder()
    gateway = TrackingGateway(store, trace)
    message = UserMessage(conversation_id='role-chat', sender_id='customer', sender_name='客户',
        text='以前说过1到3，这次2块。', message_id='role-source')
    store.append_user_turn(message, 'role-trace')
    assertions = [{'field': 'stake_range', 'value': {'min': 1, 'max': 3}, 'role': role, 'source_message_id': message.message_id},
                  {'field': 'stake', 'value': '2', 'role': 'current', 'source_message_id': message.message_id}]
    service = ToolExecutionService(store=store, tool_gateway=gateway, trace_recorder=trace)
    result = service.execute_tool_calls(AgentAction(goal='验证语义角色', objective_status='needs_tool', reasoning_summary='',
        objective_state={'fact_assertions': assertions}, tool_calls=[ToolCall(name='search_current_games',
            call_id='role-read', arguments={'requirement': {'stake': '2'}}, reason='只读查询')]),
        message=message, trace_id='role-trace', previous_step_tool_results=[], step_index=1, run_id='role-run',
        run_version=store.conversation_version(message.conversation_id))
    assert result.tool_results[0].called and result.tool_results[0].allowed
    assert gateway.executed == ['search_current_games'] and not store.games


def test_domain_conflict_stays_strict_and_guard_returns_full_invalid_reference():
    """不降低领域合同：同一冲突仍不可规范化，但执行守卫将其变成完整拒绝反馈。"""
    reference = {'stake_range': {'min': 1, 'max': 3}, 'stake': '2'}
    with pytest.raises(ValueError, match='stake_range cannot be combined'):
        normalize_requirement(reference)
    error, returned = validate_explicit_task_fact_consistency(ToolCall(name='search_current_games', arguments={'requirement': {'stake': '2'}}),
        {'explicit_task_facts': {'facts': reference}})
    assert error and 'invalid explicit current-task fact reference' in error
    assert returned == reference
    returned['stake_range']['min'] = 0
    assert reference['stake_range']['min'] == 1


@pytest.mark.parametrize('side', ['proposal', 'reference'])
def test_read_to_write_consistency_also_rejects_invalid_normalization(side):
    """相邻跨工具守卫同样不因规范化冲突崩溃，不在错误后继续建局。"""
    invalid = {'stake_range': {'min': 1, 'max': 3}, 'stake': '2'}
    call = create_call(invalid if side == 'proposal' else {'stake': '2'}, call_id='invalid-create')
    previous = ToolResult(name='search_current_games', called=True, allowed=True,
        result={'requirement': invalid if side == 'reference' else {'stake': '2'}})
    error = validate_tool_call_consistency(call, [previous])
    assert error and 'invalid' in error


@pytest.mark.parametrize('source_kind', ['other_customer', 'invalidated', 'missing'])
def test_fact_binding_keeps_source_identity_and_invalidation_guards(store, source_kind):
    """新增反馈来源不扩权；其他客户、已失效或不存在的消息不能建立当前范围参考。"""
    from mahjong_agent_runtime.services import ToolExecutionService
    trace = InMemoryTraceRecorder()
    gateway = TrackingGateway(store, trace)
    current = UserMessage(conversation_id='source-chat', sender_id='customer', sender_name='客户',
        text='这次2块', message_id='own-fixed')
    foreign = UserMessage(conversation_id='source-chat', sender_id='other' if source_kind == 'other_customer' else 'customer',
        sender_name='另一来源', text='1到3', message_id='range-source',
        metadata={'context_invalidated': source_kind == 'invalidated'})
    if source_kind != 'missing':
        store.append_user_turn(foreign, 'source-trace')
    store.append_user_turn(current, 'source-trace')
    assertions = [{'field': 'stake_range', 'value': {'min': 1, 'max': 3}, 'role': 'current', 'source_message_id': foreign.message_id},
                  {'field': 'stake', 'value': '2', 'role': 'current', 'source_message_id': current.message_id}]
    service = ToolExecutionService(store=store, tool_gateway=gateway, trace_recorder=trace)
    result = service.execute_tool_calls(AgentAction(goal='验证来源', objective_status='needs_tool', reasoning_summary='',
        objective_state={'fact_assertions': assertions}, tool_calls=[ToolCall(name='search_current_games', call_id='source-read',
            arguments={'requirement': {'stake': '2'}}, reason='只读')]), message=current, trace_id='source-trace',
        previous_step_tool_results=[], step_index=1, run_id='source-run', run_version=store.conversation_version(current.conversation_id))
    assert result.tool_results[0].called and result.tool_results[0].allowed
    assert gateway.executed == ['search_current_games'] and not store.games
