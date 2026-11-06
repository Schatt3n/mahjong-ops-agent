"""#18 回归：发送投影无损去重，审计原件与动态查询证据保持完整。"""

from copy import deepcopy
import json

import pytest

from mahjong_agent_runtime import (
    AgentContextBuilder, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore,
    TokenBudget, ToolGateway, ToolResult, UserMessage,
)
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract
from mahjong_agent_runtime.domains.context_builders.conversation_context import ContextPackingPolicy
from mahjong_agent_runtime.domains.context_builders.tool_results import (
    reference_duplicate_latest_tool_results, reference_shared_query_contracts, turn_payload_for_context,
)
from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract
from mahjong_agent_runtime.services import ContextLifecycleManager
from mahjong_agent_runtime.token_estimation import estimate_tokens


@pytest.fixture(params=['memory', 'sqlite'])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path/'transport.sqlite3')
    yield value
    if request.param == 'sqlite':
        value._connection.close()


def query_result(call_id='read-now', *, pool_empty=False):
    requirement = {'game_type':'national_standard_mahjong', 'stake':'1',
                   'planned_start_at':'2026-11-06T14:00:00+08:00',
                   'requesting_party':{'seat_count':2}}
    return ToolResult(name='search_current_games', called=True, allowed=True, call_id=call_id,
        result={'requirement':requirement, 'matches':[],
            'customer_reply_contract':current_game_search_reply_contract(requirement, [],
                unfiltered_available_pool_empty=pool_empty)})


def transported(store, latest, evidence, *, budget=64000, packing_policy=None, summary_manager=None):
    trace = InMemoryTraceRecorder()
    builder = AgentContextBuilder(store, ToolGateway(store))
    if packing_policy is not None:
        builder.packing_policy = packing_policy
    lifecycle = ContextLifecycleManager(builder, trace, context_summary_manager=summary_manager)
    message = UserMessage(conversation_id='current', sender_id='customer', sender_name='本人',
                          text='继续看看，仍然是两个人', message_id='source-now')
    args = dict(trace_id='current-trace', pending_tool_results=latest, turn_tool_evidence=evidence,
                run_id='current-run', run_version=0, step_index=2)
    built = lifecycle.build_and_trace_context(message, **args)
    before = deepcopy(built.payload)
    original_tokens = sum(estimate_tokens(m['content']) for m in built.messages)
    reduced, transition = lifecycle.summarize_and_rebuild_context_if_needed(message, built=built,
        budget=TokenBudget(max_tokens_per_call=budget), **args)
    assert transition is None
    return reduced, json.loads(reduced.messages[-1]['content']), before, original_tokens, trace


def count_full_contracts(value, contract):
    if isinstance(value, dict):
        return int(value == contract) + sum(count_full_contracts(v, contract) for v in value.values())
    if isinstance(value, list):
        return sum(count_full_contracts(v, contract) for v in value)
    if isinstance(value, str) and value.lstrip().startswith(('{', '[')):
        try:
            return count_full_contracts(json.loads(value), contract)
        except json.JSONDecodeError:
            pass
    return 0


def test_latest_feedback_references_complete_evidence_below_total_budget(store):
    result = query_result()
    original = deepcopy(result.to_dict())
    built, sent, before, tokens, trace = transported(store, [result], [result])
    assert tokens < 64000, '必须覆盖原来不会触发去重的正常预算内输入'
    latest = sent['previous_tool_results'][0]
    assert 'result' not in latest
    assert latest['result_reference'] == {'source':'turn_tool_evidence', 'index':0, 'call_id':'read-now'}
    assert sent['turn_tool_evidence'][0]['call_id'] == 'read-now'
    assert built.payload['turn_tool_evidence'] == before['turn_tool_evidence']
    assert built.payload['previous_tool_results'] == before['previous_tool_results']
    assert built.payload == before, '完整审计快照不能被传输投影或压缩计数反向改写'
    assert result.to_dict() == original
    assert any(event.step == 'context_tool_feedback_deduplicated' for event in trace.get_trace('current-trace'))


def test_shared_query_contract_is_single_definition_and_dynamic_history_is_preserved(store):
    history = query_result('read-before', pool_empty=True)
    store.append_tool_turn('current', json.dumps([history.to_dict()], ensure_ascii=False), 'earlier-trace')
    store.append_user_turn(UserMessage(conversation_id='foreign', sender_id='other', sender_name='他人',
        text='PRIVATE_FOREIGN_RAW', message_id='foreign-source'), 'foreign-trace')
    current = query_result('read-now', pool_empty=False)
    built, sent, before, _, _ = transported(store, [current], [current])
    shared = game_query_strategy_contract()
    assert sent['planning_contract']['game_query_strategy_contract'] == shared
    assert count_full_contracts(sent, shared) == 1
    old = json.loads(next(turn['content'] for turn in sent['recent_conversation'] if turn['role'] == 'tool'))[0]
    for actual, expected in ((old, history), (sent['turn_tool_evidence'][0], current)):
        reply = actual['result']['customer_reply_contract']
        original = expected.result['customer_reply_contract']
        assert reply['game_query_strategy_contract'] == {
            'contract_reference':'planning_contract.game_query_strategy_contract'}
        assert {k:v for k,v in reply.items() if k != 'game_query_strategy_contract'} == {
            k:v for k,v in original.items() if k != 'game_query_strategy_contract'}
        assert actual['call_id'] == expected.call_id
        assert actual['result']['requirement'] == expected.result['requirement']
    assert old['result']['customer_reply_contract']['evidence_scope']['supports_narrower_empty_claim'] is True
    assert sent['turn_tool_evidence'][0]['result']['customer_reply_contract']['evidence_scope']['supports_narrower_empty_claim'] is False
    assert 'PRIVATE_FOREIGN_RAW' not in json.dumps(sent, ensure_ascii=False)
    assert built.payload['recent_conversation'] == before['recent_conversation']
    saved = json.loads(next(turn.content for turn in store.recent_turns('current') if turn.role.value == 'tool'))[0]
    assert saved['result']['customer_reply_contract'] == history.result['customer_reply_contract']


def test_nonidentical_shared_contract_is_never_replaced_by_current_definition(store):
    result = query_result()
    custom = deepcopy(result.result['customer_reply_contract']['game_query_strategy_contract'])
    custom['query_result_reporting'] = '本次范围只涉及原门店，不能扩大到其他门店'
    result.result['customer_reply_contract']['game_query_strategy_contract'] = custom
    _, sent, _, _, _ = transported(store, [result], [result])
    assert sent['turn_tool_evidence'][0]['result']['customer_reply_contract']['game_query_strategy_contract'] == custom
    assert sent['planning_contract']['game_query_strategy_contract'] == game_query_strategy_contract()


def test_failed_feedback_reference_keeps_rejection_source_and_recovery_information(store):
    failed = ToolResult(name='create_game', called=False, allowed=False, call_id='denied-write',
        error='semantic_query_execution_scope', result={'validation_error':{
            'code':'query_does_not_authorize_write', 'source_message_id':'source-now',
            'requested_party_size':2}, 'instruction':'保留查询授权，不创建局'})
    built, sent, before, tokens, _ = transported(store, [failed], [failed])
    assert tokens < 64000
    assert sent['previous_tool_results'][0]['result_reference']['call_id'] == 'denied-write'
    assert sent['turn_tool_evidence'] == before['turn_tool_evidence']
    retained = sent['turn_tool_evidence'][0]
    assert retained['called'] is False and retained['allowed'] is False
    assert retained['error'] == 'semantic_query_execution_scope'
    assert retained['result']['validation_error']['source_message_id'] == 'source-now'
    assert built.payload['turn_tool_evidence'] == before['turn_tool_evidence']


def test_same_call_id_with_different_result_is_not_a_lossless_duplicate(store):
    older = query_result('same-id', pool_empty=True)
    latest = query_result('same-id', pool_empty=False)
    _, sent, _, _, _ = transported(store, [latest], [older], budget=1)
    assert 'result_reference' not in sent['previous_tool_results'][0]
    scope = sent['previous_tool_results'][0]['result']['customer_reply_contract']['evidence_scope']
    assert scope['supports_narrower_empty_claim'] is False
    assert sent['turn_tool_evidence'][0]['result']['customer_reply_contract']['evidence_scope']['supports_narrower_empty_claim'] is True


@pytest.mark.parametrize('boolean,number', [(True, 1), (False, 0)])
def test_feedback_reference_requires_equal_json_types(boolean, number):
    for actual, previous in ((boolean, number), (number, boolean)):
        latest = {'call_id':'same', 'result':{'nested':[{'value':actual}]}}
        evidence = {'call_id':'same', 'result':{'nested':[{'value':previous}]}}
        referenced, count = reference_duplicate_latest_tool_results([latest], [evidence])
        assert count == 0 and referenced == [latest]
        assert type(referenced[0]['result']['nested'][0]['value']) is type(actual)


@pytest.mark.parametrize('boolean,number', [(True, 1), (False, 0)])
def test_contract_reference_requires_equal_json_types(boolean, number):
    payload = {'planning_contract':{'game_query_strategy_contract':{'nested':[{'enabled':boolean}]}},
               'turn_tool_evidence':[{'result':{'customer_reply_contract':{
                   'game_query_strategy_contract':{'nested':[{'enabled':number}]}}}}]}
    projected, count = reference_shared_query_contracts(payload)
    assert count == 0 and projected == payload
    actual = projected['turn_tool_evidence'][0]['result']['customer_reply_contract']['game_query_strategy_contract']
    assert type(actual['nested'][0]['enabled']) is int


def test_transport_copy_preserves_deadlines_and_never_interprets_user_json():
    contract = game_query_strategy_contract()
    deadline = {'task_id':'durable-task', 'game_id':'own-game', 'customer_id':'customer',
                'due_at':'2026-11-06T20:00:00+08:00', 'source_message_id':'original-condition'}
    tool = {'result':{'customer_reply_contract':{'game_query_strategy_contract':contract},
                     'participation_deadline':deadline}}
    raw_user = json.dumps([tool], ensure_ascii=False)
    payload = {'planning_contract':{'game_query_strategy_contract':contract},
               'active_participation_deadlines':[deadline], 'turn_tool_evidence':[tool],
               'recent_conversation':[{'role':'user','content':raw_user}]}
    before = deepcopy(payload)
    projected, count = reference_shared_query_contracts(payload)
    assert count == 1
    assert projected['active_participation_deadlines'] == [deadline]
    assert projected['turn_tool_evidence'][0]['result']['participation_deadline'] == deadline
    assert projected['recent_conversation'][0]['content'] == raw_user
    assert payload == before


def history_query_fixture(store):
    """构造后续消息前的真实查询证据，并使预算只容纳去掉重复合同后的完整事实。"""
    history = query_result('historical-query', pool_empty=True)
    store.append_tool_turn('current', json.dumps([history.to_dict()], ensure_ascii=False), 'earlier-trace')
    store.append_user_turn(UserMessage(conversation_id='current', sender_id='customer', sender_name='本人',
        text='继续看看，仍然是两个人', message_id='source-now'), 'current-trace')
    turns = store.recent_turns('current')
    full = [turn_payload_for_context(turn) for turn in turns]
    projected, count = reference_shared_query_contracts({
        'planning_contract': {'game_query_strategy_contract': game_query_strategy_contract()},
        'recent_conversation': full})
    assert count == 1
    full_size = sum(estimate_tokens(turn) for turn in full)
    projected_size = sum(estimate_tokens(turn) for turn in projected['recent_conversation'])
    budget = (full_size + projected_size) // 2
    assert projected_size < budget < full_size
    return history, turns, ContextPackingPolicy(max_recent_conversation_tokens=budget)


@pytest.mark.parametrize('anchor', ['identical', 'absent', 'different_version'])
def test_history_selection_counts_only_a_real_identical_contract_anchor(store, anchor):
    history, turns, policy = history_query_fixture(store)
    canonical = game_query_strategy_contract() if anchor != 'absent' else None
    if anchor == 'different_version':
        canonical['query_result_reporting'] = '不同版本规则不能替换旧证据'
    packed, audit = policy.pack_turns(turns, shared_query_contract=canonical)
    tools = [turn for turn in packed if turn['role'] == 'tool']
    if anchor == 'identical':
        assert len(tools) == 1 and audit['omitted_for_budget'] == 0
        raw = json.loads(tools[0]['content'])[0]
        assert raw['result']['customer_reply_contract'] == history.result['customer_reply_contract']
        assert raw['result']['requirement'] == history.result['requirement']
        assert raw['call_id'] == history.call_id
    else:
        assert tools == [] and audit['omitted_for_budget'] == 1
    saved = json.loads(next(turn.content for turn in store.recent_turns('current') if turn.role.value == 'tool'))[0]
    assert saved == history.to_dict()


def test_selected_history_survives_transport_and_total_budget_without_false_summary(store):
    history, _, policy = history_query_fixture(store)
    store.append_user_turn(UserMessage(conversation_id='foreign', sender_id='other', sender_name='他人',
        text='PRIVATE_FOREIGN_QUERY', message_id='foreign-source'), 'foreign-trace')
    first, sent, before, original_tokens, _ = transported(store, [], [], packing_policy=policy)
    tool_turns = [turn for turn in sent['recent_conversation'] if turn['role'] == 'tool']
    assert len(tool_turns) == 1, '静态合同重复不应让上一轮查询证据在历史选择时整项消失'
    actual = json.loads(tool_turns[0]['content'])[0]
    assert actual['call_id'] == history.call_id
    assert actual['result']['requirement'] == history.result['requirement']
    assert actual['result']['customer_reply_contract']['evidence_scope'] == history.result['customer_reply_contract']['evidence_scope']
    assert count_full_contracts(sent, game_query_strategy_contract()) == 1
    assert first.payload == before and 'PRIVATE_FOREIGN_QUERY' not in json.dumps(sent, ensure_ascii=False)
    sent_tokens = sum(estimate_tokens(message['content']) for message in first.messages)
    threshold = (original_tokens + sent_tokens) // 2
    budget = (threshold * 100 + 84) // 85
    assert sent_tokens < int(budget * .85) < original_tokens

    class NoUnneededSummary:
        def __getattr__(self, name):
            raise AssertionError(f'精确去重已满足总预算，不应调用摘要器: {name}')

    final, resent, audit_before, _, trace = transported(store, [], [], packing_policy=policy,
        budget=budget, summary_manager=NoUnneededSummary())
    assert final.payload == audit_before
    assert resent['recent_conversation'] == sent['recent_conversation']
    assert count_full_contracts(resent, game_query_strategy_contract()) == 1
    assert sum(estimate_tokens(message['content']) for message in final.messages) < budget
    events = trace.get_trace('current-trace')
    assert sum(event.step == 'context_built' for event in events) == 1
    assert not any(event.step.startswith('context_summary') for event in events)
