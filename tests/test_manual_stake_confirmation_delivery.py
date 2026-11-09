"""手动模拟仅领取本轮登记的到期核档任务，桌主收件箱按实际私聊回执投影。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import timedelta
import json
from pathlib import Path
import sys
from threading import Event
from types import SimpleNamespace
import time

import pytest

from mahjong_agent_runtime import AgentAction, AgentRuntime, AgentRuntimeResult, SQLiteAgentStore, ToolResult
from mahjong_agent_runtime.group_chat import stake_confirmation
from mahjong_agent_runtime.models import MessageReference, ScheduledAgentTask, now
from mahjong_agent_runtime.scheduled_tasks import ScheduledTaskDeferred
from mahjong_agent_runtime.tracing import InMemoryTraceRecorder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from manual_simulation import handle_manual_simulation
from manual_group_simulation import deliver_manual_stake_confirmations
import manual_simulation
from simulation_observability import simulation_observability_payload, update_simulation_control


@pytest.mark.parametrize('task_type,handler_name', [
    (stake_confirmation.BOARD_STAKE_CONFIRMATION_TASK_TYPE, 'handle_board_stake_confirmation'),
    (stake_confirmation.BOARD_CANDIDATE_UPDATE_TASK_TYPE, 'handle_board_candidate_update'),
])
def test_retry_projects_persisted_sandbox_receipt_without_sending(tmp_path, monkeypatch, task_type, handler_name):
    store = SQLiteAgentStore(tmp_path / 'stake-delivery.db')
    task = ScheduledAgentTask('ask', task_type, 'game', 'game-a', 'private-a', 'a', '客户A', now(), 'ask')
    store.save_follow_up_task(task, trace_id='inquiry')
    store.register_message_reference(MessageReference('ask', 'private-a', task_type, 'game-a',
        text='七点那桌打多大的？', channel='console', recipient_id='a', recipient_name='客户A',
        metadata={'trace_id': 'original-send', 'delivery_id': 'sandbox-output:1:original-send'}))

    def should_not_send(*args):
        raise AssertionError('已有真实回执，不能再次生成或发送')

    monkeypatch.setattr(stake_confirmation, handler_name, should_not_send)
    with closing(store._connection):
        runtime = SimpleNamespace(store=store, trace_recorder=InMemoryTraceRecorder())
        delivery, = deliver_manual_stake_confirmations(runtime, [], {'ask': task_type})
        assert delivery['outputs'][0]['metadata']['recovered_from_reference'] is True
        assert delivery['runtime_result']['final_reply'] == '七点那桌打多大的？'
        assert delivery['runtime_result']['tool_results'] == []
        assert store.scheduled_tasks['ask'].status.value == 'completed'


def _wait_for_delivery_worker(report_path):
    """等待当前临时沙盒后台结束，避免卸载测试替身时仍有任务运行。"""
    deadline = time.monotonic() + 4
    while report_path in manual_simulation._MANUAL_STAKE_WORKERS and time.monotonic() < deadline:
        time.sleep(0.01)
    assert report_path not in manual_simulation._MANUAL_STAKE_WORKERS


def test_known_stake_candidate_receipt_binds_question_and_update_to_requester():
    question_type = stake_confirmation.BOARD_STAKE_CONFIRMATION_TASK_TYPE
    update_type = stake_confirmation.BOARD_CANDIDATE_UPDATE_TASK_TYPE
    tool = ToolResult('request_board_candidate_confirmation', True, True,
                      result={'candidate_confirmation': {'task_id': 'known-stake-question', 'status': 'pending'}})
    requester_turn = {'sequence': 1, 'manual_request_id': 'ask', 'channel': 'private', 'conversation_id': 'private:b',
                      'user': {'customer_id': 'b'}, 'agent': {'reply': '有一桌，我问一下能不能补。', 'trace_id': 'inquiry'},
                      'manual_deferred_stake_confirmations': True, 'tool_results': [tool.to_dict()]}
    report = {'run_id': 'test', 'transcript': [requester_turn], 'manual_requests': {'ask': 1}}
    sources, inquiries, recorded = manual_simulation._manual_stake_sources(report)
    assert sources == {'known-stake-question': (1, 'ask', question_type)}
    assert list(inquiries) == ['inquiry'] and not recorded
    update = ScheduledAgentTask('known-stake-update', update_type, 'game', 'game', 'private:b', 'b', '客户B', now(),
                               'known-stake-update', payload={'room_id': 'manual-room:test', 'inquiry_trace_id': 'inquiry'})
    assert manual_simulation._manual_candidate_source(report, update, inquiries) == (1, 'ask', update_type)

    requester_turn['tool_results'][0]['allowed'] = False
    sources, inquiries, _ = manual_simulation._manual_stake_sources(report)
    assert not sources and not inquiries
    assert manual_simulation._manual_candidate_source(report, update, inquiries) is None


def test_manual_private_inquiry_delivers_only_evidenced_due_task_to_host(tmp_path, monkeypatch):
    run_id = handle_manual_simulation({'action': 'create', 'request_id': 'create-stake-room'},
                                      runtime_dir=tmp_path, mode='mock')['run_id']
    folder = tmp_path / 'runs' / run_id
    host_id, requester_id = 'manual_customer', 'manual_customer_2'
    host_conversation = f'manual:{run_id}'
    requester_conversation = f'manual:{run_id}:{requester_id}'
    requester_text = '这是B的私聊原话，0.5到2块都行。'
    requester_reply = '这桌的档位还要确认。'
    host_reply = '你七点那桌财敲打多大的？'
    model_calls, delivered_tasks = [], []

    def requester_result(runtime, message, *, trace_id):
        """工具替身真实落下任务，验证模拟器只消费该次成功结果指向的到期任务。"""
        assert message.conversation_id == requester_conversation
        model_calls.append(trace_id)
        stamp = now()
        for task_id in ('current-due', 'unrelated-due', 'denied-due', 'current-future'):
            due_at = stamp + timedelta(hours=1) if task_id == 'current-future' else stamp
            task = ScheduledAgentTask(task_id=task_id, task_type=stake_confirmation.BOARD_STAKE_CONFIRMATION_TASK_TYPE,
                aggregate_type='game', aggregate_id=f'game:{task_id}', conversation_id=host_conversation,
                subject_id=host_id, subject_name='客户A', due_at=due_at, idempotency_key=task_id)
            runtime.store.ensure_board_stake_confirmation_task(task, trace_id=trace_id)
        runtime.store.append_assistant_turn(message.conversation_id, requester_reply, trace_id, metadata={
            'delivery_status': 'pending_operator_send', 'conversation_version': runtime.store.conversation_version(message.conversation_id)})
        evidence = [ToolResult('request_board_stake_confirmation', True, task_id != 'denied-due',
            result={'stake_confirmation': {'task_id': task_id, 'status': 'pending'}})
            for task_id in ('current-due', 'current-due', 'current-future', 'denied-due')]
        return AgentRuntimeResult(trace_id, message.conversation_id, requester_reply, tool_results=evidence)

    def deliver(runtime, messenger, task, trace_id):
        """替代主模型生成环节，实际调度与本地收件人、内容和交付回执仍走正式模拟接线。"""
        assert task.status.value == 'processing' and task.conversation_id == host_conversation
        delivered_tasks.append(task.task_id)
        messenger.send_private_message(host_id, host_reply, metadata={'trace_id': trace_id})
        # 缓存结果可有不同生成trace，回执仍来自本次真实调度。
        return AgentRuntimeResult('host-generation-trace', host_conversation, host_reply, status='waiting_user',
            actions=[AgentAction(goal='向桌主确认底注', objective_status='waiting_user', reasoning_summary='受控通知替身')])

    monkeypatch.setattr(AgentRuntime, 'handle_user_message', requester_result)
    monkeypatch.setattr(stake_confirmation, 'handle_board_stake_confirmation', deliver)
    payload = {'action': 'send', 'run_id': run_id, 'request_id': 'ask-board-stake',
               'actor_id': requester_id, 'channel': 'private', 'text': requester_text}
    result = handle_manual_simulation(payload, runtime_dir=tmp_path)
    assert result['ok'] and delivered_tasks == ['current-due']
    report_path = folder / 'report.json'
    saved = report_path.read_bytes()
    raw_turn, = json.loads(saved)['transcript']
    delivery, = raw_turn['scheduled_private_deliveries']
    output, = delivery['outputs']
    assert delivery['task_id'] == 'current-due'
    assert output['channel'] == 'private' and output['recipient_id'] == host_id
    assert output['conversation_id'] == host_conversation and output['text'] == host_reply
    assert output['metadata']['trace_id'] == delivery['delivery_trace_id']
    assert delivery['runtime_result']['trace_id'] == 'host-generation-trace'
    assert raw_turn['agent']['reply'] == requester_reply

    selected = simulation_observability_payload(runtime_dir=tmp_path, selected_run_id=run_id)['selected_run']
    views = {view['conversation_id']: view for view in selected['conversations']}
    host_turn, = views[host_conversation]['turns']
    requester_turn, = views[requester_conversation]['turns']
    assert host_turn['agent']['reply'] == host_reply and host_turn['agent']['trace_id'] == 'host-generation-trace'
    assert host_turn['private_delivery_outputs'] == [output]
    assert host_turn['user']['customer_id'] == host_id and requester_text not in json.dumps(host_turn, ensure_ascii=False)
    assert requester_turn['agent']['reply'] == requester_reply
    assert host_reply not in json.dumps(requester_turn, ensure_ascii=False)
    assert host_reply not in json.dumps(selected['participants'] + selected['topics'], ensure_ascii=False)
    assert report_path.read_bytes() == saved
    assert handle_manual_simulation(payload, runtime_dir=tmp_path)['deduplicated'] is True
    assert len(model_calls) == 1 and delivered_tasks == ['current-due'] and report_path.read_bytes() == saved
    store = SQLiteAgentStore(folder / 'test_sim.db')
    with closing(store._connection):
        assert {key: task.status.value for key, task in store.scheduled_tasks.items()} == {
            'current-due': 'completed', 'unrelated-due': 'pending', 'denied-due': 'pending', 'current-future': 'pending'}


def test_background_stake_confirmation_keeps_requester_responsive_and_retries(tmp_path, monkeypatch):
    run_id = handle_manual_simulation({'action': 'create', 'request_id': 'create-background-room'},
                                      runtime_dir=tmp_path, mode='mock')['run_id']
    folder = tmp_path / 'runs' / run_id
    report_path = folder / 'report.json'
    host_id, requester_id = 'manual_customer', 'manual_customer_2'
    host_conversation = f'manual:{run_id}'
    requester_conversation = f'manual:{run_id}:{requester_id}'
    host_reply = '七点那桌财敲是打多大的？'
    first_started, release = Event(), Event()
    attempts = []

    def requester_result(runtime, message, *, trace_id):
        """第一条请求登记一个真实任务，后续消息仅回复，用来检测清单合并。"""
        if message.text == '再看看别的。':
            reply, tools = '有新消息再告诉你。', []
        else:
            reply = '这桌档位还要核实。'
            task = ScheduledAgentTask(task_id='background-stake', task_type=stake_confirmation.BOARD_STAKE_CONFIRMATION_TASK_TYPE,
                aggregate_type='game', aggregate_id='game:background-stake', conversation_id=host_conversation,
                subject_id=host_id, subject_name='客户A', due_at=now(), idempotency_key='background-stake')
            runtime.store.ensure_board_stake_confirmation_task(task, trace_id=trace_id)
            tools = [ToolResult('request_board_stake_confirmation', True, True,
                result={'stake_confirmation': {'task_id': task.task_id, 'status': 'pending'}})]
        runtime.store.append_assistant_turn(message.conversation_id, reply, trace_id, metadata={
            'delivery_status': 'pending_operator_send', 'conversation_version': runtime.store.conversation_version(message.conversation_id)})
        return AgentRuntimeResult(trace_id, message.conversation_id, reply, tool_results=tools)

    def deliver(runtime, messenger, task, trace_id):
        """核档开始时询局人回复已提交，首轮临时延期后按真实到期时间再送达。"""
        attempts.append(trace_id)
        if len(attempts) == 1:
            report = json.loads(report_path.read_text())
            turn, = report['transcript']
            assert report['manual_operations']['ask-background-stake']['status'] == 'done'
            assert turn['manual_processing'] is False and turn['agent']['reply'] == '这桌档位还要核实。'
            first_started.set()
            assert release.wait(4)
            raise ScheduledTaskDeferred('测试暂缓核档', delay_seconds=1)
        messenger.send_private_message(host_id, host_reply, metadata={'trace_id': trace_id})
        return AgentRuntimeResult(trace_id, task.conversation_id, host_reply, status='waiting_user')

    monkeypatch.setattr(AgentRuntime, 'handle_user_message', requester_result)
    monkeypatch.setattr(stake_confirmation, 'handle_board_stake_confirmation', deliver)
    payload = {'action': 'send', 'run_id': run_id, 'request_id': 'ask-background-stake',
               'actor_id': requester_id, 'channel': 'private', 'text': '有今晚的局吗？'}
    options = {'runtime_dir': tmp_path, 'background_stake_confirmations': True}
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(handle_manual_simulation, payload, **options)
        try:
            assert first_started.wait(3)
            assert first.result(timeout=0.5)['sequence'] == 1
            assert not update_simulation_control('clear_history', confirm=True, runtime_dir=tmp_path)['ok']
            assert report_path.exists() and not (tmp_path / 'archives').exists()
            second = handle_manual_simulation({**payload, 'request_id': 'send-next', 'text': '再看看别的。'}, **options)
            assert second['sequence'] == 2
            assert handle_manual_simulation(payload, **options)['deduplicated'] is True
            assert len(attempts) == 1
        finally:
            release.set()

    _wait_for_delivery_worker(report_path)
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    report = json.loads(report_path.read_text())
    first_turn, second_turn = report['transcript']
    delivery, = first_turn['scheduled_private_deliveries']
    assert delivery['delivery_trace_id'] == attempts[1]
    assert second_turn['agent']['reply'] == '有新消息再告诉你。'
    assert report['manual_requests'] == {'ask-background-stake': 1, 'send-next': 2}

    selected = simulation_observability_payload(runtime_dir=tmp_path, selected_run_id=run_id)['selected_run']
    views = {view['conversation_id']: view for view in selected['conversations']}
    host_turn, = views[host_conversation]['turns']
    assert host_turn['agent']['reply'] == host_reply and host_turn['user']['customer_id'] == host_id
    assert '有今晚的局吗？' not in json.dumps(host_turn, ensure_ascii=False)
    assert host_reply not in json.dumps(views[requester_conversation], ensure_ascii=False)
    store = SQLiteAgentStore(folder / 'test_sim.db')
    with closing(store._connection):
        assert store.scheduled_tasks['background-stake'].status.value == 'completed'


@pytest.mark.parametrize('background_acceptance', [True, False])
def test_notifies_only_requester_after_stake_and_candidate_decision(tmp_path, monkeypatch, background_acceptance):
    run_id = handle_manual_simulation({'action': 'create', 'request_id': 'create-stake-update-room'},
                                      runtime_dir=tmp_path, mode='mock')['run_id']
    folder = tmp_path / 'runs' / run_id
    report_path = folder / 'report.json'
    host_id, requester_id = 'manual_customer', 'manual_customer_2'
    host_conversation = f'manual:{run_id}'
    requester_conversation = f'manual:{run_id}:{requester_id}'
    question, update = '七点那桌财敲是打多大的？', '七点那桌财敲，底注确定是两块。'
    candidate_update = '七点那桌可以补位，两块的，你来吗？'
    received = []
    inquiry_trace = None
    expires_at = now() + timedelta(hours=1)

    def model(runtime, message, *, trace_id):
        """模拟模型和持久化动作；后台仍由同一 SQLite 和正式调度器消费。"""
        nonlocal inquiry_trace
        tools = []
        if message.text == '有今晚的局吗？':
            inquiry_trace = trace_id
            question_task = ScheduledAgentTask(task_id='host-question', task_type=stake_confirmation.BOARD_STAKE_CONFIRMATION_TASK_TYPE,
                aggregate_type='game', aggregate_id='game:board', conversation_id=host_conversation, subject_id=host_id,
                subject_name='客户A', due_at=now(), idempotency_key='host-question')
            update_task = ScheduledAgentTask(task_id='requester-update', task_type=stake_confirmation.BOARD_STAKE_UPDATE_TASK_TYPE,
                aggregate_type='game', aggregate_id='game:board', conversation_id=requester_conversation, subject_id=requester_id,
                subject_name='客户B', due_at=expires_at, idempotency_key='requester-update',
                payload={'expires_at': expires_at.isoformat()})
            runtime.store.ensure_board_stake_confirmation_task(question_task, trace_id=trace_id)
            # 此处是调度投影夹具；领域层原子登记与激活有单独测试。
            runtime.store.save_follow_up_task(update_task, trace_id=trace_id)
            tools.append(ToolResult('request_board_stake_confirmation', True, True, result={
                'stake_confirmation': {'task_id': question_task.task_id}, 'stake_update': {'task_id': update_task.task_id}}))
            reply = '七点有桌财敲，档位还要核实。'
        elif message.text == '男生也行。':
            for task_id, room_id in (('candidate-update', f'manual-room:{run_id}'), ('foreign-room', 'other-room')):
                task = ScheduledAgentTask(task_id=task_id, task_type=stake_confirmation.BOARD_CANDIDATE_UPDATE_TASK_TYPE,
                    aggregate_type='game', aggregate_id='game:board', conversation_id=requester_conversation,
                    subject_id=requester_id, subject_name='客户B', due_at=now(), idempotency_key=task_id,
                    payload={'room_id': room_id, 'expires_at': expires_at.isoformat(),
                             'inquiry_trace_id': inquiry_trace, 'decision_task_id': 'decision'})
                runtime.store.save_follow_up_task(task, trace_id=trace_id)
            # 已提交的业务动作不能因为同一轮桌主可见回复过期而丢失。
            return AgentRuntimeResult(trace_id, message.conversation_id, '', status='superseded')
        elif message.conversation_id == host_conversation:
            task = runtime.store.scheduled_tasks['requester-update']
            task.due_at = now()
            runtime.store.save_follow_up_task(task, trace_id=trace_id)
            reply = '记下了，两块。'
        else:
            reply = '好。'
        runtime.store.append_assistant_turn(message.conversation_id, reply, trace_id, metadata={
            'delivery_status': 'pending_operator_send', 'conversation_version': runtime.store.conversation_version(message.conversation_id)})
        return AgentRuntimeResult(trace_id, message.conversation_id, reply, tool_results=tools)

    def deliver(runtime, messenger, task, trace_id):
        """发出仅给各任务目标的真实沙盒私信。"""
        text = {'host-question': question, 'requester-update': update, 'candidate-update': candidate_update}[task.task_id]
        received.append((task.task_id, task.subject_id))
        messenger.send_private_message(task.subject_id, text, metadata={'trace_id': trace_id})
        return AgentRuntimeResult(trace_id, task.conversation_id, text, status='completed')

    monkeypatch.setattr(AgentRuntime, 'handle_user_message', model)
    monkeypatch.setattr(stake_confirmation, 'handle_board_stake_confirmation', deliver)
    monkeypatch.setattr(stake_confirmation, 'handle_board_stake_update', deliver)
    monkeypatch.setattr(stake_confirmation, 'handle_board_candidate_update', deliver)

    def send(request_id, actor_id, text, *, background=True):
        """经真实手动发送入口唤醒当前沙盒。"""
        return handle_manual_simulation({'action': 'send', 'run_id': run_id, 'request_id': request_id,
                                         'actor_id': actor_id, 'channel': 'private', 'text': text},
                                        runtime_dir=tmp_path, background_stake_confirmations=background)

    wake = manual_simulation._wake_manual_stake_worker
    monkeypatch.setattr(manual_simulation, '_wake_manual_stake_worker', lambda path: None)
    assert send('ask', requester_id, '有今晚的局吗？')['sequence'] == 1
    assert not received
    monkeypatch.setattr(manual_simulation, '_wake_manual_stake_worker', wake)
    assert send('ask', requester_id, '有今晚的局吗？')['deduplicated'] is True
    _wait_for_delivery_worker(report_path)
    assert received == [('host-question', host_id)]
    assert send('more', requester_id, '好，知道了。')['sequence'] == 2
    _wait_for_delivery_worker(report_path)
    assert send('answer', host_id, '是我那一桌，两块。')['sequence'] == 3
    _wait_for_delivery_worker(report_path)
    assert received == [('host-question', host_id), ('requester-update', requester_id)]
    assert send('answer', host_id, '是我那一桌，两块。')['deduplicated'] is True
    _wait_for_delivery_worker(report_path)
    assert len(received) == 2
    assert send('accept', host_id, '男生也行。', background=background_acceptance)['sequence'] == 4
    _wait_for_delivery_worker(report_path)
    assert received == [('host-question', host_id), ('requester-update', requester_id), ('candidate-update', requester_id)]
    assert send('accept', host_id, '男生也行。')['deduplicated'] is True
    _wait_for_delivery_worker(report_path)
    assert len(received) == 3

    selected = simulation_observability_payload(runtime_dir=tmp_path, selected_run_id=run_id)['selected_run']
    views = {view['conversation_id']: view['turns'] for view in selected['conversations']}
    assert [turn['agent']['reply'] for turn in views[requester_conversation]] == ['七点有桌财敲，档位还要核实。', '好。', update, candidate_update]
    assert [turn['agent']['reply'] for turn in views[host_conversation]] == [question, '记下了，两块。', '']
    assert views[requester_conversation][-1]['user']['text'] == '（老板主动私聊更新补位进展）'
    assert question not in json.dumps(views[requester_conversation], ensure_ascii=False)
    assert '是我那一桌，两块。' not in json.dumps(views[requester_conversation], ensure_ascii=False)
    assert update not in json.dumps(views[host_conversation], ensure_ascii=False)
    assert '男生也行。' not in json.dumps(views[requester_conversation], ensure_ascii=False)
    assert candidate_update not in json.dumps(views[host_conversation], ensure_ascii=False)
    initial, _, answer, acceptance = selected['transcript']
    assert [item['task_id'] for item in initial['scheduled_private_deliveries']] == ['host-question', 'requester-update', 'candidate-update']
    assert views[requester_conversation][-2]['observed_at'] >= answer['observed_at']
    assert views[requester_conversation][-1]['observed_at'] >= acceptance['observed_at']
    store = SQLiteAgentStore(folder / 'test_sim.db')
    with closing(store._connection):
        assert store.scheduled_tasks['candidate-update'].status.value == 'completed'
        assert store.scheduled_tasks['foreign-room'].status.value == 'pending'
