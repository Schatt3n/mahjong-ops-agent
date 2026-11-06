"""执行前工具参数拒绝只有完整无副作用证据与真实合法重试，才可计为安全恢复。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from mahjong_agent_runtime import (
    AgentLLMConfig, AgentRuntime, CustomerProfile, InMemoryAgentStore, InMemoryTraceRecorder,
    OpenAICompatibleAgentClient, SQLiteAgentStore, TokenBudget, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_result_dict
from mahjong_agent_runtime.tracing import TraceEvent
from tests.test_action_contract_recovery_audit import Response, actual_model_prompt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_real_owner_chat_live_eval as owner


@pytest.fixture(params=['memory', 'sqlite'])
def store(request, tmp_path):
    """两种正式持久层均走真实写工具，SQLite临时文件不借历史运行数据。"""
    value = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path / 'parameter.sqlite3')
    value.upsert_customer(CustomerProfile(customer_id='self', display_name='客户',
        preferred_games=['national_standard_mahjong'], preferred_stakes=['1', '2'], smoke_preference='any'))
    yield value
    if request.param == 'sqlite':
        value._connection.close()


def requirement():
    return {'game_type': 'national_standard_mahjong', 'stake_range': {'min': 1, 'max': 3},
        'cap_score': 256, 'smoke_preference': 'any', 'start_time_kind': 'asap_when_full',
        'known_player_count': 1, 'needed_seats': 3}


def action(name=None, arguments=None, *, call_id=None, explicit=False, status=None):
    """动作固定只为验证审计机制，独立语义验收在此通过显式布尔分别测试。"""
    state = {'fact_assertions': [{'field': 'stake_range', 'value': {'min': 1, 'max': 3},
        'role': 'current', 'source_message_id': 'range-input'}]} if explicit else {}
    return {'goal': '承接本人1到3块256封顶的组局委托',
        'objective_status': status or ('needs_tool' if name else 'completed'),
        'reasoning_summary': '离线机制夹具，不代表真实模型语义结论', 'objective_state': state,
        'reply_to_user': '' if name else '好，给你记上了，人齐开。',
        'tool_calls': [{'name': name, 'arguments': arguments, 'call_id': call_id,
                        'reason': '推进同一用户目标'}] if name else [],
        'needs_human': False, 'stop_reason': {'can_stop': not name, 'why': '按实际工具状态推进',
            'pending_work': ['完成本次组局'] if name else [], 'depends_on_tool_results': True}}


def run_case(store, *, explicit=True, recover=True, deny_write=False):
    """真实Runtime/网关/反馈装配；只以本地HTTP响应替身替换远端推理。"""
    calls, snapshots = [], []
    trace = InMemoryTraceRecorder()
    create_args = {'organizer_id': 'self', 'organizer_name': '客户', 'requirement': requirement(),
        'known_players': [{'customer_id': 'self', 'display_name': '客户', 'seat_count': 1}]}

    def transport(request, **kwargs):
        body = json.loads(request.data)
        calls.append(body)
        payload = json.loads(body['messages'][-1]['content'])
        snapshots.append({'game_count': len(store.games), 'draft_count': len(store.invite_drafts),
                          'outbound_count': len(store.outbound_message_drafts)})
        step = len(calls)
        if step == 1:
            result = action('search_current_games', {'requirement': requirement()}, call_id='search')
        elif step == 2:
            bad = deepcopy(create_args); bad['requirement']['base_stake'] = 1
            result = action('create_game', bad, call_id='rejected-create', explicit=explicit)
        elif step == 3 and recover:
            result = action('create_game', deepcopy(create_args), call_id='corrected-create', explicit=explicit)
        elif step == 4 and store.games:
            game = next(iter(store.games.values()))
            result = action('search_customers', {'game_id': game.game_id, 'exclude_customer_ids': ['self']}, call_id='candidates')
        else:
            result = action(status='waiting_user' if not recover or deny_write else 'completed')
        return Response(json.dumps(result, ensure_ascii=False), step)

    client = OpenAICompatibleAgentClient(config=AgentLLMConfig(api_key='unit-only-key', model='unit-model',
        base_url='https://example.invalid', retry_attempts=1, max_estimated_tokens_per_day=0), urlopen=transport)
    gateway = ToolGateway(store)
    if deny_write:
        gateway.allowed_execution_modes = {'read_only', 'audit_write'}
    runtime = AgentRuntime(llm_client=client, store=store, tool_gateway=gateway, trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=64000, max_calls_per_turn=8))
    result = runtime.handle_user_message(UserMessage(conversation_id='range-chat', sender_id='self', sender_name='客户',
        text='这次底注1到3块都能打，256封顶，我一个人，烟不限，人齐开，帮我组', message_id='range-input'),
        trace_id='tool-parameter-recovery')
    # 序列化往返强制使用已落盘形态，不能借LLMCompletionText附带的内存诊断过关。
    events = [TraceEvent(trace_id=e.trace_id, step=e.step, content=json.loads(json.dumps(e.content)),
                        level=e.level, occurred_at=e.occurred_at) for e in trace.get_trace(result.trace_id)]
    return result, events, calls, snapshots


def audit(result, events, *, business=True, semantics=True):
    return owner.assess_action_contract_recovery(events, runtime_status=result.status, final_reply=result.final_reply,
        business_checks_passed=business, semantic_passed=semantics)


def rejection(events):
    return next(e for e in events if e.step == 'tool_result' and e.content.get('call_id') == 'rejected-create')


def mutate_prompt(events, step, change):
    """只改实际发送前最后一份请求，不能仅破坏未发送的压缩前草稿。"""
    event = actual_model_prompt(events, step)
    payload = json.loads(event.content['messages'][-1]['content'])
    change(payload)
    event.content['messages'][-1]['content'] = json.dumps(payload, ensure_ascii=False)


def assert_blocked(result, events, **kwargs):
    metrics = audit(result, events, **kwargs)
    assert metrics['quality_status'] == 'failed', metrics
    assert metrics['tool_parameter_unrecovered_count'] or metrics['unresolved_tool_error_count'], metrics
    return metrics


@pytest.mark.parametrize('explicit,code', [(True, 'explicit_task_fact_consistency_rejected'),
                                         (False, 'tool_requirement_consistency_rejected')])
def test_actual_rejection_feedback_legal_execution_and_final_store_are_all_required(store, explicit, code):
    result, events, calls, snapshots = run_case(store, explicit=explicit)
    assert result.status == 'completed'
    refused = rejection(events).content
    assert refused['result']['validation_error']['code'] == code
    assert refused['called'] is refused['allowed'] is False and refused['state_transitions'] == []
    # 第三次HTTP输入发生在失败提案之后、合法重试之前，此时业务状态必须仍为空。
    assert snapshots[2] == {'game_count': 0, 'draft_count': 0, 'outbound_count': 0}
    sent = json.loads(calls[2]['messages'][-1]['content'])
    latest = sent['previous_tool_results'][0]
    feedback = sent['turn_tool_evidence'][latest['result_reference']['index']] if 'result_reference' in latest else latest
    assert feedback == compact_tool_result_dict(refused)
    assert len(store.games) == 1
    game = next(iter(store.games.values()))
    assert game.requirement['stake_range'] == {'min': 1, 'max': 3} and game.requirement['cap_score'] == 256
    assert 'base_stake' not in game.requirement and 'stake' not in game.requirement
    assert game.remaining_seats() == 3
    assert not store.invite_drafts and not store.outbound_message_drafts
    metrics = audit(result, events)
    assert metrics['quality_status'] == 'passed', metrics
    assert metrics['action_contract_error_count'] == metrics['safe_recovered_count'] == 0
    assert metrics['tool_parameter_error_count'] == metrics['tool_parameter_safe_recovered_count'] == 1
    assert metrics['tool_parameter_unrecovered_count'] == metrics['tool_parameter_unsafe_rejected_step_count'] == 0
    assert metrics['unresolved_tool_error_count'] == 0
    item = metrics['tool_parameter_events'][0]
    assert item['validation_code'] == code and item['call_id'] == 'rejected-create'
    assert item['safely_rejected'] and item['feedback_returned_to_model'] and item['recovered']
    assert item['recovered_at_step'] == 3


@pytest.mark.parametrize('fault', ['called_true', 'state_transition', 'gateway_execution', 'outbound',
    'missing_feedback', 'wrong_feedback_pointer', 'missing_http', 'missing_retry_http', 'wrong_http_trace', 'missing_error_event',
    'forged_validation_code', 'different_call', 'different_tool_success', 'different_goal', 'different_customer',
    'missing_success_result', 'final_reply_not_delivered', 'foreign_success_trace', 'corrected_call_not_proposed',
    'corrected_proposal_not_from_http_response'])
def test_trace_gaps_side_effects_and_borrowed_success_cannot_be_recovered(store, fault):
    result, events, _, _ = run_case(store)
    refused = rejection(events)
    error = next(e for e in events if e.step == 'tool_explicit_fact_consistency_error')
    success = next(e for e in events if e.step == 'tool_result' and e.content.get('call_id') == 'corrected-create')
    if fault == 'called_true':
        refused.content['called'] = True
    elif fault == 'state_transition':
        refused.content['state_transitions'] = [{'entity_type': 'game', 'entity_id': 'bad', 'to_status': 'forming'}]
    elif fault in {'gateway_execution', 'outbound'}:
        step = 'tool_gateway_completed' if fault == 'gateway_execution' else 'outbound_message_sent'
        events.insert(events.index(refused), TraceEvent(trace_id=result.trace_id, step=step,
            content={'tool_name': 'create_game', 'called': True, 'allowed': True, 'step_index': 201,
                     'recipient_id': 'self', 'text': '错误步骤已发送'}))
    elif fault == 'missing_feedback':
        mutate_prompt(events, 3, lambda p: (p.update(previous_tool_results=[]), p.update(turn_tool_evidence=[])))
    elif fault == 'wrong_feedback_pointer':
        mutate_prompt(events, 3, lambda p: p['previous_tool_results'][0].update(result_reference={'source': 'turn_tool_evidence', 'index': 0, 'call_id': 'search'}))
    elif fault in {'missing_http', 'missing_retry_http', 'wrong_http_trace'}:
        step = 3 if fault == 'missing_retry_http' else 2
        response = next(e for e in events if e.step == 'llm_response' and e.content['step_index'] == step)
        if fault in {'missing_http', 'missing_retry_http'}:
            response.content.pop('completion_diagnostics')
        else:
            response.content['response_source']['trace_id'] = 'other-trace'
    elif fault == 'missing_error_event':
        events.remove(error)
    elif fault == 'forged_validation_code':
        refused.content['result']['validation_error']['code'] = 'permission_denied'
    elif fault == 'different_call':
        refused.content['call_id'] = 'other-proposal'
    elif fault == 'different_tool_success':
        success.content['name'] = 'search_current_games'
    elif fault == 'different_goal':
        mutate_prompt(events, 3, lambda p: p['task_context_window'].update(task_context_id='another-goal'))
    elif fault == 'different_customer':
        mutate_prompt(events, 3, lambda p: p['current_message'].update(sender_id='another-customer'))
    elif fault == 'missing_success_result':
        events.remove(success)
    elif fault == 'final_reply_not_delivered':
        next(e for e in events if e.step == 'final_output').content['reply'] = '另一条最终回复'
    elif fault == 'foreign_success_trace':
        begin = next(i for i, e in enumerate(events) if e.step == 'llm_response' and e.content['step_index'] == 3)
        end = next(i for i, e in enumerate(events) if e.step == 'llm_response' and e.content['step_index'] == 4)
        for event in events[begin:end]:
            if event.step in {'tool_called', 'tool_gateway_received', 'tool_gateway_completed', 'tool_result'}:
                event.trace_id = 'another-business-trace'
    elif fault == 'corrected_call_not_proposed':
        for event in events:
            call = event.content.get('call')
            if event.step in {'tool_called', 'tool_gateway_received'} and isinstance(call, dict) and call.get('call_id') == 'corrected-create':
                call['arguments']['organizer_id'] = 'another-customer'
    elif fault == 'corrected_proposal_not_from_http_response':
        proposal = next(e for e in events if e.step == 'action_proposed'
            and any(call.get('call_id') == 'corrected-create' for call in e.content.get('tool_calls', [])))
        proposal.content['tool_calls'][0]['arguments']['organizer_id'] = 'another-customer'
    assert_blocked(result, events)


@pytest.mark.parametrize('business,semantics', [(False, True), (True, False), (False, False)])
def test_final_business_or_independent_semantic_failure_still_blocks(store, business, semantics):
    result, events, _, _ = run_case(store)
    assert_blocked(result, events, business=business, semantics=semantics)


def test_unrecovered_parameter_rejection_cannot_hide_behind_completed_reply(store):
    result, events, _, _ = run_case(store, recover=False)
    assert not store.games
    assert_blocked(result, events)


def test_real_gateway_permission_denial_is_never_reclassified_as_parameter_recovery(store):
    result, events, _, _ = run_case(store, deny_write=True)
    assert not store.games
    denied = [e for e in events if e.step == 'tool_permission_checked' and e.content.get('allowed') is False]
    assert denied
    metrics = assert_blocked(result, events)
    assert metrics['unresolved_tool_error_count'] >= 1


def test_rejected_proposal_cannot_satisfy_required_successful_tool_execution(store):
    """required工具表示真实成功执行，不能把仅出现工具名的失败提案算作承接。"""
    result, events, _, _ = run_case(store, recover=False)
    scenario = owner.LiveEvalScenario(
        scenario_id='rejected-proposal-only', name='拒绝提案不算成功建局',
        message=UserMessage(conversation_id='range-chat', sender_id='self', sender_name='客户',
            text='这次底注1到3块都能打，256封顶，我一个人，烟不限，人齐开，帮我组', message_id='range-input'),
        setup=lambda _: None,
        required_tool_names=['create_game'], required_tool_name_any=['create_game'])
    report = owner.validate_result(result, scenario, events, store,
        semantic_assessment={'passed': True, 'error': None})
    checks = {item['name']: item for item in report['checks']}
    assert checks['should_call_create_game']['passed'] is False
    assert checks['should_call_any_state_tracking_tool']['passed'] is False
    assert not store.games
