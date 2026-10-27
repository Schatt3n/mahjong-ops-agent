"""手动测试只能操作自身沙盒，撤回保留审计，并复用真实运行时合同。"""

from datetime import timedelta
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from manual_simulation import handle_manual_simulation
from simulation_observability import simulation_observability_payload, simulation_transcript_export
from mahjong_agent_runtime import AgentRuntime, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.models import now


def create(tmp_path):
    return handle_manual_simulation({'action':'create','request_id':'create-test'},runtime_dir=tmp_path,mode='mock')['run_id']


def operate(tmp_path, run_id, action, request_id, **kwargs):
    return handle_manual_simulation({'run_id':run_id,'action':action,'request_id':request_id,**kwargs},runtime_dir=tmp_path)


def read_report(tmp_path,run_id):
    return json.loads((tmp_path/'runs'/run_id/'report.json').read_text())


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


def test_workspace_has_only_controlled_roles_and_profile_edits_are_audited(tmp_path):
    run_id=create(tmp_path)
    report=read_report(tmp_path,run_id)
    assert report['manual_workspace_version']==2 and len(report['manual_actors'])==3
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    assert set(store.customers)=={actor['customer_id'] for actor in report['manual_actors']}
    store._connection.close()
    profile={'display_name':'张三','preferred_games':['国标'],'preferred_stakes':['1'],'smoke_preference':'any',
             'profile_facts':['通常一个人来'],'notes':'老板私有测试备注'}
    operate(tmp_path,run_id,'update_profile','profile-a',actor_id='manual_customer',profile=profile)
    assert operate(tmp_path,run_id,'update_profile','profile-a',actor_id='manual_customer',profile=profile)['deduplicated']
    operate(tmp_path,run_id,'update_profile','profile-b',actor_id='manual_customer_2',profile={'display_name':'李四','preferred_games':['川麻'],'preferred_stakes':['0.5']})
    for actor,text in [('manual_customer','A的私聊消息'),('manual_customer_2','B的私聊消息')]:
        operate(tmp_path,run_id,'send',f'send-{actor}',actor_id=actor,text=text)
    report=read_report(tmp_path,run_id)
    assert len(report['manual_profile_events'])==2
    assert report['manual_profile_events'][0]['before']['display_name']=='客户A'
    assert report['manual_profile_events'][0]['before']['preferred_games']==[]
    assert report['manual_profile_events'][0]['after']['display_name']=='张三'
    first,second=report['transcript']
    assert first['manual_profile_at_input']['preferred_games']==['national_standard_mahjong']
    assert second['manual_profile_at_input']['preferred_games']==['sichuan_mahjong']
    assert first['conversation_id']!=second['conversation_id']
    assert '老板私有测试备注' not in json.dumps(first['simulation_observation'],ensure_ascii=False)
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    assert 'B的私聊消息' not in repr(store.recent_turns(first['conversation_id']))
    assert store.customers['manual_customer'].notes=='老板私有测试备注'
    store._connection.close()


@pytest.mark.parametrize('profile',[{'display_name':'客户B'},{'preferred_games':'国标'},{'smoke_preference':[]},{'unknown_field':'no'}])
def test_profile_schema_and_role_boundaries_do_not_mutate_on_failure(tmp_path,profile):
    run_id=create(tmp_path);before=read_report(tmp_path,run_id)
    with pytest.raises(ValueError):operate(tmp_path,run_id,'update_profile','bad',actor_id='manual_customer',profile=profile)
    assert read_report(tmp_path,run_id)==before
    with pytest.raises(ValueError,match='不属于'):operate(tmp_path,run_id,'update_profile','foreign',actor_id='outsider',profile={'display_name':'外人'})


def test_add_role_name_stable_identity_and_retry(tmp_path):
    run_id=create(tmp_path)
    result=operate(tmp_path,run_id,'add_actor','add-user',profile={'display_name':'新客','preferred_games':['其他地方麻将']})
    assert operate(tmp_path,run_id,'add_actor','add-user',profile={'display_name':'新客'})['deduplicated']
    report=read_report(tmp_path,run_id)
    assert len(report['manual_actors'])==4
    actor=next(item for item in report['manual_actors'] if item['customer_id']==result['actor_id'])
    assert actor['preferred_games']==['其他地方麻将']


def test_input_profile_snapshot_precedes_runtime_memory_updates(tmp_path,monkeypatch):
    run_id=create(tmp_path)
    original_handle=AgentRuntime.handle_user_message
    def handle(runtime,message,**kwargs):
        result=original_handle(runtime,message,**kwargs)
        customer=runtime.store.customers[message.sender_id]
        runtime.store.upsert_customer(replace(customer,profile_facts=[*customer.profile_facts,'本轮执行后新增事实']))
        return result
    monkeypatch.setattr(AgentRuntime,'handle_user_message',handle)
    operate(tmp_path,run_id,'send','learn',text='今天我想打国标')
    report=read_report(tmp_path,run_id);turn=report['transcript'][0]
    assert '本轮执行后新增事实' not in repr(turn['manual_profile_at_input'])
    assert '本轮执行后新增事实' in report['manual_actors'][0]['profile_facts']
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    assert '本轮执行后新增事实' in store.customers['manual_customer'].profile_facts
    store._connection.close()


def test_group_entry_preserves_room_timeline_but_isolates_main_contexts_and_topics(tmp_path):
    run_id=create(tmp_path)
    for index,(actor,text) in enumerate([('manual_customer','现在有人吗'),('manual_customer_2','川麻有人吗'),('manual_customer','国标呢')]):
        operate(tmp_path,run_id,'send',f'group-{index}',actor_id=actor,channel='group',text=text)
    report=read_report(tmp_path,run_id);turns=report['transcript']
    assert len({turn['conversation_id'] for turn in turns})==1
    assert all(turn['channel']=='group' for turn in turns)
    assert turns[0]['thread_id']==turns[2]['thread_id']
    assert turns[0]['thread_id']!=turns[1]['thread_id']
    assert turns[0]['group_processing']['entrypoint']=='GroupMessageHandler'
    assert turns[0]['group_processing']['runtime_results'][0]['conversation_id']!=turns[1]['group_processing']['runtime_results'][0]['conversation_id']
    assert turns[2]['group_processing']['routing']['action']=='ignore'
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    a=turns[0]['group_processing']['runtime_results'][0]['conversation_id']
    assert '川麻有人吗' not in repr(store.recent_turns(a))
    store._connection.close()


def test_group_quote_restores_topic_and_recall_clears_only_matching_source(tmp_path):
    run_id=create(tmp_path)
    operate(tmp_path,run_id,'send','a',actor_id='manual_customer',channel='group',text='国标有人吗')
    operate(tmp_path,run_id,'send','b',actor_id='manual_customer_2',channel='group',text='我也想打',quote_sequence=1)
    report=read_report(tmp_path,run_id)
    assert report['transcript'][0]['thread_id']==report['transcript'][1]['thread_id']
    with pytest.raises(ValueError,match='自己发送'):
        operate(tmp_path,run_id,'recall','wrong-actor',actor_id='manual_customer_2',sequence=1)
    operate(tmp_path,run_id,'recall','recalled-a',actor_id='manual_customer',sequence=1)
    report=read_report(tmp_path,run_id)
    assert report['transcript'][0]['recalled'] and not report['transcript'][1].get('recalled')
    assert report['transcript'][-1]['agent']['reply']==''
    assert report['transcript'][-1]['group_processing']['topic_analysis']['model_calls']==[]
    sessions=report['manual_group_session_state']['sessions']
    assert any(set(session['participants'])=={'manual_customer_2'} for session in sessions)


def test_group_private_switch_deliveries_and_recall_are_distinct(tmp_path):
    run_id=create(tmp_path)
    operate(tmp_path,run_id,'send','switch',channel='group',text='帮我组个国标1块人齐开的局')
    report=read_report(tmp_path,run_id);turn=report['transcript'][0]
    assert turn['group_processing']['routing']['action']=='private_switch'
    outputs=turn['group_processing']['outputs']
    assert {item['channel'] for item in outputs}=={'group','private'}
    assert all(item['recipient_id']=='manual_customer' for item in outputs if item['channel']=='private')
    assert turn['group_processing']['recall_targets']
    path=tmp_path/'runs'/run_id/'report.json';original=path.read_bytes()
    projected=simulation_observability_payload(runtime_dir=tmp_path,selected_run_id=run_id)['selected_run']
    private=next(view for view in projected['conversations'] if view['conversation_id']==f'manual:{run_id}')
    assert len(private['turns'])==1 and private['turns'][0]['event_type']=='group_private_delivery'
    assert private['turns'][0]['user']['generation']['source']=='group_private_continuation'
    assert len(projected['transcript'])==1 and path.read_bytes()==original
    exported,_,_=simulation_transcript_export(run_id,format='markdown',runtime_dir=tmp_path)
    assert 'private → 客户A' in exported.decode()
    operate(tmp_path,run_id,'recall','recall-switch',sequence=1)
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    assert any(t.metadata.get('context_invalidated') for t in store.recent_turns(f'manual:{run_id}'))
    store._connection.close()


def test_batch_order_retry_resume_and_cancel_preserve_inputs(tmp_path):
    run_id=create(tmp_path)
    items=[{'actor_id':'manual_customer','channel':'group','text':'现在有人吗'},
           {'actor_id':'manual_customer_2','channel':'group','text':'川麻有人吗'},
           {'actor_id':'manual_customer','channel':'private','text':'我打国标'}]
    operate(tmp_path,run_id,'queue_batch','queue',messages=items)
    operate(tmp_path,run_id,'batch_next','step-1')
    assert operate(tmp_path,run_id,'batch_next','step-1')['deduplicated']
    assert read_report(tmp_path,run_id)['manual_batch']['position']==1
    operate(tmp_path,run_id,'batch_next','step-2')
    operate(tmp_path,run_id,'cancel_batch','stop')
    operate(tmp_path,run_id,'batch_next','after-stop')
    report=read_report(tmp_path,run_id)
    assert [t['user']['text'] for t in report['transcript']]==[item['text'] for item in items[:2]]
    assert report['manual_batch']['items'][2]['status']=='pending'
    operate(tmp_path,run_id,'queue_batch','new-queue',messages=[items[2]])
    operate(tmp_path,run_id,'batch_next','new-step')
    report=read_report(tmp_path,run_id)
    assert report['manual_batch']['status']=='completed'
    assert report['manual_batch_history'][0]['status']=='cancelled'
    assert len(report['transcript'])==3


def test_group_post_publishes_real_board_and_number_claim_uses_same_aggregate(tmp_path):
    run_id=create(tmp_path)
    operate(tmp_path,run_id,'send','post',channel='group',text='cq371 人齐开 1块无烟')
    first=read_report(tmp_path,run_id)['transcript'][0]
    assert first['group_processing']['routing']['action']=='board_import'
    assert any('当前缺人局' in item['text'] for item in first['group_processing']['outputs'])
    game_id=first['group_processing']['routing']['game_id']
    operate(tmp_path,run_id,'send','claim',actor_id='manual_customer_2',channel='group',text='1来')
    report=read_report(tmp_path,run_id);last=report['transcript'][-1]
    assert last['group_processing']['routing']['action']=='claim'
    assert last['group_processing']['routing']['game_id']==game_id
    assert last['group_processing']['topic_analysis']['model_calls'][0]['input']['board_state']['items']
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    assert store.require_game(game_id).remaining_seats()==0
    store._connection.close()


def test_partial_workspace_creation_retries_same_folder_without_reset(tmp_path):
    import hashlib
    digest=hashlib.sha256(b'create-test').hexdigest()[:16]
    folder=tmp_path/'runs'/f'20261027T000000Z_manual_{digest}'
    folder.mkdir(parents=True)
    run_id=create(tmp_path)
    assert run_id==folder.name and (folder/'report.json').is_file()
    assert len(list((tmp_path/'runs').iterdir()))==1
