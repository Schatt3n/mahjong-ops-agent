"""手动测试只能操作自身沙盒，撤回保留审计，并复用真实运行时合同。"""

from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from manual_simulation import handle_manual_simulation
from mahjong_agent_runtime import SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.models import now


def create(tmp_path):
    return handle_manual_simulation({'action':'create','request_id':'create-test'},runtime_dir=tmp_path,mode='mock')['run_id']


def operate(tmp_path, run_id, action, request_id, **kwargs):
    return handle_manual_simulation({'run_id':run_id,'action':action,'request_id':request_id,**kwargs},runtime_dir=tmp_path)


def test_manual_send_retry_and_recall_are_auditable_and_idempotent(tmp_path):
    run_id=create(tmp_path)
    assert create(tmp_path)==run_id
    assert len(list((tmp_path/'runs').iterdir()))==1
    result=operate(tmp_path,run_id,'send','send-once',text='三缺一的时候叫我')
    assert result['sequence']==1
    assert operate(tmp_path,run_id,'send','send-once',text='三缺一的时候叫我')['deduplicated']
    result=operate(tmp_path,run_id,'recall','recall-once',sequence=1)
    assert result['sequence']==2
    report=json.loads((tmp_path/'runs'/run_id/'report.json').read_text())
    assert len(report['transcript'])==2
    assert report['transcript'][0]['user']['text']=='三缺一的时候叫我'
    assert report['transcript'][0]['recalled'] is True
    event=report['transcript'][1]
    assert event['agent']['response_mode']=='silent'
    assert event['agent']['reply']==''
    assert event['simulation_observation']['model_call_observed'] is False
    assert event['recalled_message_id']==report['transcript'][0]['message_id']
    assert operate(tmp_path,run_id,'recall','recall-again',sequence=1)['already_recalled']
    assert len((tmp_path/'runs'/run_id/'live_events.jsonl').read_text().splitlines())==2


def test_automatic_runs_and_invalid_messages_are_not_modified(tmp_path):
    folder=tmp_path/'runs'/'automatic_run'
    folder.mkdir(parents=True)
    original='{"status":"completed","transcript":[]}'
    (folder/'report.json').write_text(original)
    with pytest.raises(ValueError,match='历史自动测试记录只读'):
        operate(tmp_path,'automatic_run','send','attempt',text='不能改')
    assert (folder/'report.json').read_text()==original
    run_id=create(tmp_path)
    with pytest.raises(ValueError,match='4000'):
        operate(tmp_path,run_id,'send','empty',text='')
    with pytest.raises(ValueError,match='自己发送'):
        operate(tmp_path,run_id,'recall','missing',sequence=99)
    with pytest.raises(ValueError):
        operate(tmp_path,'../../outside','send','path',text='不能越界')


def test_manual_clock_triggers_only_its_follow_up_and_preserves_other_due_tasks(tmp_path):
    run_id=create(tmp_path)
    operate(tmp_path,run_id,'send','first',text='先等等')
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    stamp=now(); due=stamp+timedelta(minutes=3)
    gateway=ToolGateway(store)
    result=gateway.execute(ToolCall('schedule_follow_up',{'goal':'再确认是否要打','waiting_for':'customer','due_at':due.isoformat(),
                                                        'expires_at':(stamp+timedelta(hours=1)).isoformat()},'手动登记'),
                           trace_id='schedule',step_index=1,conversation_id=f'manual:{run_id}',sender_id='manual_customer',sender_name='手动测试客户')
    assert result.allowed
    unrelated,_=store.ensure_waiting_demand_expiration_task(due_at=stamp+timedelta(minutes=1),trace_id='expiry')
    store._connection.close()
    result=operate(tmp_path,run_id,'advance','advance-once')
    report=json.loads((tmp_path/'runs'/run_id/'report.json').read_text())
    assert result['sequence']==2
    assert report['transcript'][-1]['event_type']=='follow_up_timeout'
    assert report['transcript'][-1]['user']['generation']['source']=='scheduler'
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    assert store.scheduled_tasks[unrelated.task_id].status.value=='pending'
    store._connection.close()
