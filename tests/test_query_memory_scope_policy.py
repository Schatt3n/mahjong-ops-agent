"""D01只询局允许有来源的本次条件记录，不能借此放宽原状态、权限或长期画像。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, CustomerProfile, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.task_context import TaskContextManager
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_conversation_topic_eval as runner


@pytest.fixture(params=['memory', 'sqlite'])
def actual_memory(request, tmp_path):
    store = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path / 'memory.sqlite3')
    store.upsert_customer(CustomerProfile(customer_id='topic_customer', display_name='客户'))
    message = UserMessage(conversation_id='query-memory', sender_id='topic_customer', sender_name='客户',
        text='只看现成的杭麻1块无烟局，我一个人，最多能打四小时，先别帮我招人。', message_id='current-input')
    TaskContextManager(store).prepare(message, trace_id='current-trace')
    store.append_user_turn(message, 'current-trace')
    gateway = ToolGateway(store)
    def write(field, value, evidence, trace_id):
        return gateway.execute(ToolCall(name='record_user_memory', arguments={'task_memories': [{
            'customer_id': 'topic_customer', 'memory_type': 'constraint', 'field': field, 'value': value,
            'evidence': evidence, 'confidence': 1.0, 'scope': 'current_task'}]}), trace_id=trace_id,
            conversation_id=message.conversation_id, sender_id=message.sender_id, sender_name=message.sender_name, step_index=1)
    previous = write('smoke_preference', 'no_smoke', '无烟局', 'previous-trace')
    assert previous.called and previous.allowed and not previous.error
    before = runner.state_snapshot(store, message.conversation_id)
    result = write('max_duration_hours', 4, '最多能打四小时', 'current-trace')
    assert result.called and result.allowed and not result.error
    after = runner.state_snapshot(store, message.conversation_id)
    yield before, after, [result.to_dict()], message.to_dict()
    if request.param == 'sqlite': store._connection.close()


def test_real_current_condition_memory_is_allowed_and_old_memory_is_preserved(actual_memory):
    before, after, tools, request = actual_memory
    assert runner.grounded_current_task_memory(before, after, tools, request=request, trace_id='current-trace')
    assert len(after['all_task_memories']) == len(before['all_task_memories']) + 1
    turn = {'game_count': 0, 'no_operational_change': True, 'allow_grounded_current_task_memory': True}
    checks = runner.checks_for(turn, before, after, tools, request=request, trace_id='current-trace')
    assert all(checks.values())
    # 不强迫记忆写入；直接按真实约束查询同样合法。
    assert runner.grounded_current_task_memory(before, before, [], request=request, trace_id='current-trace')


@pytest.mark.parametrize('fault', ['other_customer', 'other_conversation', 'other_task', 'source_trace', 'no_quote',
    'unrelated_quote', 'no_tool', 'tool_failed', 'missing_transition', 'old_memory_changed', 'old_memory_removed',
    'long_term', 'pending', 'arbitrary_new_memory'])
def test_memory_exception_rejects_unproven_or_out_of_scope_writes(actual_memory, fault):
    before, after, tools, request = deepcopy(actual_memory)
    new = after['all_task_memories'][-1]
    if fault == 'other_customer': new['customer_id'] = 'someone_else'
    elif fault == 'other_conversation': new['conversation_id'] = 'another_conversation'
    elif fault == 'other_task': new['metadata']['task_context_id'] = 'another_goal'
    elif fault == 'source_trace': new['source_trace_id'] = 'previous-trace'
    elif fault == 'no_quote': new['evidence'] = ''
    elif fault == 'unrelated_quote': new['evidence'] = '隔壁老板说六小时'
    elif fault == 'no_tool': tools = []
    elif fault == 'tool_failed': tools[0]['error'] = 'write failed'
    elif fault == 'missing_transition': tools[0]['state_transitions'] = []
    elif fault == 'old_memory_changed': after['all_task_memories'][0]['value'] = 'smoking'
    elif fault == 'old_memory_removed': after['all_task_memories'].pop(0)
    elif fault == 'long_term': new['scope'] = 'long_term'
    elif fault == 'pending': after['pending_memory_candidates'].append({'candidate_id': 'unrequested'})
    else: after['all_task_memories'].append({**new, 'memory_id': 'unproven-extra'})
    if tools: tools[0]['result']['task_memories'] = [deepcopy(new)]
    assert not runner.grounded_current_task_memory(before, after, tools, request=request, trace_id='current-trace')


def test_pure_casual_turn_still_rejects_even_grounded_memory_change(actual_memory):
    before, after, tools, request = actual_memory
    checks = runner.checks_for({'game_count': 0, 'no_business_change': True}, before, after, tools,
        request=request, trace_id='current-trace')
    assert checks['no_unrequested_business_change'] is False


def test_revision_preserves_original_source_and_all_duration_assertions():
    original = runner.DATASET.read_bytes()
    policy = json.loads(runner.QUERY_MEMORY_POLICY.read_text())
    assert hashlib.sha256(original).hexdigest() == policy['source_asset_sha256']
    old_cases = [json.loads(line) for line in original.decode().splitlines()]
    cases = runner.load_cases()
    target = next(case for case in cases if case['case_id'] == policy['source_case_id'])
    previous = next(case for case in old_cases if case['case_id'] == target['case_id'])
    assert previous['turns'][0]['no_business_change'] is True
    assert 'no_business_change' not in target['turns'][0]
    assert target['turns'][0]['no_operational_change'] is True
    for key in ('required_duration_filter', 'required_tools', 'game_count'):
        assert previous['turns'][0][key] == target['turns'][0][key]
    assert previous['inventory_duration_hours'] == target['inventory_duration_hours'] == [4, 5, None]
    assert [case for case in cases if case['case_id'] != target['case_id']] == [case for case in old_cases if case['case_id'] != target['case_id']]
    assert runner.DATASET.read_bytes() == original
