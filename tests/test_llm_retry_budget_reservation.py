"""#18 回归：一次慢读不能耗尽所有尝试，传输重试不产生额外业务副作用。"""

import json
from datetime import timedelta

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder,
    OpenAICompatibleAgentClient, SQLiteAgentStore, TokenBudget, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.llm import completion_diagnostics
from mahjong_agent_runtime.models import now
from tests.test_llm_resilience import FakeClock, IncrementalResponse, config, encoded_response


def timed_out_body(clock, timeout, content='{"ignored_partial_action":true}'):
    response = IncrementalResponse(clock, [(0, encoded_response(content)[:-3]), (timeout, b'')])
    response.status = 200
    return response


@pytest.mark.parametrize('attempt_cap', [None, 60, 45, 10])
def test_attempt_upper_bound_leaves_time_for_read_timeout_recovery(attempt_cap):
    clock = FakeClock()
    requests, responses = [], []

    def urlopen(request, *, timeout):
        requests.append((timeout, request.data))
        response = timed_out_body(clock, timeout) if len(requests) == 1 else IncrementalResponse(
            clock, [(.01, encoded_response('{"recovered":true}'))])
        responses.append(response)
        return response

    client = OpenAICompatibleAgentClient(config=config(attempt_timeout_seconds=attempt_cap, retry_attempts=3),
        urlopen=urlopen, monotonic_fn=clock, sleep_fn=clock.advance)
    value = client.complete([{'role':'user','content':'bounded-request'}], trace_id='read-timeout-recovery', timeout_seconds=45)
    assert value == '{"recovered":true}'
    assert len(requests) == 2
    assert 0 < requests[0][0] < 45
    if attempt_cap is not None:
        assert requests[0][0] <= attempt_cap
    if attempt_cap == 10:
        assert requests[0][0] == 10, '较短的显式cap仍是有效上限，不能被自动预算放大'
    assert 0 < requests[1][0] <= 45 - requests[0][0]
    assert requests[0][1] == requests[1][1], '重试不能换提示词、模型或悄悄改授权'
    assert clock.elapsed <= 45 and all(response.closed for response in responses)
    attempts = completion_diagnostics(value)['attempts']
    assert attempts[0]['http_status'] == 200 and attempts[0]['body_complete'] is False
    assert attempts[0]['error_type'] == 'TimeoutError'
    assert attempts[1]['body_complete'] is True
    assert client.concurrency_metrics()['http_recoveries'] == 1


@pytest.mark.parametrize('attempt_cap', [None, 60, 10])
def test_repeated_timeouts_obey_attempt_and_absolute_deadline_limits(attempt_cap):
    clock = FakeClock()
    timeouts, responses, starts = [], [], []

    def urlopen(request, *, timeout):
        starts.append(clock.elapsed)
        timeouts.append(timeout)
        response = timed_out_body(clock, timeout)
        responses.append(response)
        return response

    client = OpenAICompatibleAgentClient(config=config(attempt_timeout_seconds=attempt_cap, retry_attempts=3,
        retry_base_delay_seconds=.5, retry_max_delay_seconds=.5), urlopen=urlopen,
        monotonic_fn=clock, sleep_fn=clock.advance, random_fn=lambda:0)
    with pytest.raises(RuntimeError, match='TimeoutError') as failure:
        client.complete([], trace_id='all-read-timeouts', timeout_seconds=45)
    assert 2 <= len(timeouts) <= 3, '已配置重试次数必须有实际剩余预算，而不只是名义上允许'
    assert all(0 < timeout <= 45-start for timeout,start in zip(timeouts, starts))
    assert clock.elapsed <= 45
    assert all(response.closed for response in responses)
    assert len(completion_diagnostics(failure.value)['attempts']) == len(timeouts)
    metrics = client.concurrency_metrics()
    assert metrics['http_attempts'] == metrics['http_timeouts'] == len(timeouts)
    assert metrics['active_provider_requests'] == metrics['waiting_provider_requests'] == 0


@pytest.mark.parametrize('attempt_cap', [None, 60])
def test_single_attempt_may_use_whole_budget_without_hidden_retry(attempt_cap):
    clock = FakeClock()
    requested = []
    def urlopen(request, *, timeout):
        requested.append(timeout)
        return IncrementalResponse(clock, [(44, encoded_response())])
    client = OpenAICompatibleAgentClient(config=config(retry_attempts=1, attempt_timeout_seconds=attempt_cap),
        urlopen=urlopen, monotonic_fn=clock)
    assert client.complete([], trace_id='single-attempt', timeout_seconds=45) == '{"ok":true}'
    assert requested == [45] and clock.elapsed == 44


@pytest.mark.parametrize('kind', ['memory', 'sqlite'])
@pytest.mark.parametrize('attempt_cap', [None, 10, 60])
def test_partial_model_write_response_never_executes_and_recovery_writes_once(kind, attempt_cap, tmp_path):
    store = InMemoryAgentStore() if kind == 'memory' else SQLiteAgentStore(tmp_path/'retry-effects.sqlite3')
    clock = FakeClock()
    planned = (now() + timedelta(days=1)).replace(hour=19, minute=30, second=0, microsecond=0)
    create = AgentAction(goal='登记本人的已授权新局', objective_status='needs_tool', reasoning_summary='条件明确后登记',
        tool_calls=[ToolCall(name='create_game', call_id='create-once', arguments={
            'organizer_id':'owner', 'organizer_name':'本人',
            'requirement':{'game_type':'hangzhou_mahjong','stake':'0.5','smoke_preference':'no_smoking',
                           'start_time_kind':'scheduled','start_time':'19:30','planned_start_at':planned.isoformat()},
            'requesting_party':{'contact_id':'owner','contact_name':'本人','seat_count':1}}, reason='登记本人需求')],
        stop_reason={'can_stop':False,'why':'等登记结果','pending_work':['登记'],'depends_on_tool_results':True})
    final = AgentAction(goal='登记本人的已授权新局', objective_status='completed', reasoning_summary='真实工具已登记',
        reply_to_user='登记好了。', stop_reason={'can_stop':True,'why':'登记完成','pending_work':[], 'depends_on_tool_results':True})
    calls, responses = [], []
    def urlopen(request, *, timeout):
        calls.append(request.data)
        body = json.dumps((create if len(calls) <= 2 else final).to_dict(), ensure_ascii=False)
        response = timed_out_body(clock, timeout, body) if len(calls) == 1 else IncrementalResponse(
            clock, [(0, encoded_response(body))])
        responses.append(response)
        return response
    client = OpenAICompatibleAgentClient(config=config(attempt_timeout_seconds=attempt_cap, retry_attempts=3),
        urlopen=urlopen, monotonic_fn=clock, sleep_fn=clock.advance)
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=client, store=store, tool_gateway=ToolGateway(store), trace_recorder=trace,
        llm_timeout_seconds=45, token_budget=TokenBudget(max_tokens_per_call=64000,max_calls_per_turn=4),
        reply_self_review_enabled=False, customer_visible_text_generation_enabled=False)
    try:
        result = runtime.handle_user_message(UserMessage(conversation_id='own-chat',sender_id='owner',sender_name='本人',
            message_id='create-source',text='帮我登记明晚七点半杭麻0.5无烟的新局，我一个人'), trace_id='partial-write')
        assert result.status == 'completed'
        executed = [tool for tool in result.tool_results if tool.name == 'create_game' and tool.called and tool.allowed]
        assert len(executed) == len(store.games) == 1
        assert len([item for item in store.transitions if item.entity_type == 'game' and item.from_status is None]) == 1
        assert calls[0] == calls[1] and len(calls) == 3
        assert client.concurrency_metrics()['http_timeouts'] == client.concurrency_metrics()['http_recoveries'] == 1
        assert all(response.closed for response in responses)
        assert len([event for event in trace.get_trace('partial-write') if event.step == 'tool_called']) == 1
    finally:
        if kind == 'sqlite':
            store._connection.close()
