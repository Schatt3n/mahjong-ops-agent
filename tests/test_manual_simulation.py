"""手动测试只能操作自身沙盒，撤回保留审计，并复用真实运行时合同。"""

from datetime import timedelta
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

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


def test_pending_main_input_accepts_supplement_before_previous_model_finishes(tmp_path,monkeypatch):
    """真实主循环配可控模型桩：第二条先落库，旧输出失效，原文和固定时间线均保留。"""
    import manual_simulation as manual
    from mahjong_agent_runtime import AgentAction, InMemoryTraceRecorder
    first_started=threading.Event();release_first=threading.Event();second_received=threading.Event()
    contexts=[]
    class MainClient:
        config=SimpleNamespace(model='deterministic-main-ingress-test')
        def complete(self,messages,**kwargs):
            payload=json.loads(messages[-1]['content']);contexts.append(payload)
            text=payload['current_message']['text']
            if text=='先约晚上':
                first_started.set()
                assert release_first.wait(8),'测试未释放旧模型调用'
            return json.dumps(AgentAction(goal='回应当前安排意愿',objective_status='waiting_user',reasoning_summary='受控测试按当前原话确认，不执行业务写工具。',
                objective_state={'current_phase':'waiting_user','known_facts':{},'missing_facts':[],'blockers':[]},
                reply_to_user='下午的，知道了。' if text=='改成下午' else '晚上的，知道了。',
                stop_reason={'can_stop':True,'why':'已回应当前消息','pending_work':[],'depends_on_tool_results':False}).to_dict(),ensure_ascii=False)
    client=MainClient()
    def build(store,mode):
        return AgentRuntime(llm_client=client,store=store,tool_gateway=ToolGateway(store),trace_recorder=InMemoryTraceRecorder(),
            customer_visible_text_generation_enabled=False,reply_self_review_enabled=False,context_summary_manager=None),client
    monkeypatch.setattr(manual,'build_runtime',build)
    original_accept=SQLiteAgentStore.accept_user_input
    def accept(store,message,**kwargs):
        result=original_accept(store,message,**kwargs)
        if message.text=='改成下午':second_received.set()
        return result
    monkeypatch.setattr(SQLiteAgentStore,'accept_user_input',accept)
    run_id=handle_manual_simulation({'action':'create','request_id':'real-main-fixture'},runtime_dir=tmp_path,mode='real')['run_id']
    with ThreadPoolExecutor(max_workers=2) as executor:
        first=executor.submit(operate,tmp_path,run_id,'send','old-request',text='先约晚上')
        assert first_started.wait(5)
        second=executor.submit(operate,tmp_path,run_id,'send','new-request',text='改成下午')
        try:
            assert second_received.wait(5),'补充消息被观测锁挡在正式接收入口之外'
            pending=read_report(tmp_path,run_id)
            assert [(turn['sequence'],turn['user']['text']) for turn in pending['transcript']]==[(1,'先约晚上'),(2,'改成下午')]
            assert all(turn['manual_processing'] for turn in pending['transcript'])
            repeated=operate(tmp_path,run_id,'send','new-request',text='改成下午')
            assert repeated['deduplicated'] and repeated['processing'] and repeated['sequence']==2
            store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
            try:
                turns=[turn for turn in store.recent_turns(f'manual:{run_id}') if turn.role.value=='user']
                assert [turn.content for turn in turns]==['先约晚上','改成下午']
                assert [turn.metadata['conversation_version'] for turn in turns]==[1,2]
            finally:store._connection.close()
        finally:release_first.set()
        assert first.result(timeout=5)['sequence']==1
        assert second.result(timeout=5)['sequence']==2
    report=read_report(tmp_path,run_id);old,new=report['transcript']
    assert report['llm_mode']=='real' and report['main_agent_model']=='deterministic-main-ingress-test'
    assert old['agent']['status']=='superseded' and old['agent']['reply']==''
    assert new['agent']['reply']=='下午的，知道了。'
    assert not old['manual_processing'] and not new['manual_processing']
    assert report['quality_status']=='passed' and report['agent_internal_failure_count']==0
    assert len((tmp_path/'runs'/run_id/'live_events.jsonl').read_text().splitlines())==2
    latest=next(payload for payload in contexts if payload['current_message']['text']=='改成下午')
    assert '先约晚上' in repr(latest['recent_conversation'])
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    try:
        delivered=next(turn for turn in store.recent_turns(f'manual:{run_id}') if turn.trace_id==new['agent']['trace_id'] and turn.role.value=='assistant')
        assert delivered.metadata['delivery_status']=='delivered'
    finally:store._connection.close()


def test_other_actor_completion_does_not_wait_for_model_or_replace_pending_snapshot(tmp_path,monkeypatch):
    """不同客户可先完成，迟到的旧快照不得抹掉客户B画像或已完成对话。"""
    run_id=create(tmp_path);started=threading.Event();release=threading.Event()
    original=AgentRuntime.handle_user_message
    def handle(runtime,message,**kwargs):
        if message.text=='客户A正在处理':
            started.set();assert release.wait(8)
        return original(runtime,message,**kwargs)
    monkeypatch.setattr(AgentRuntime,'handle_user_message',handle)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first=executor.submit(operate,tmp_path,run_id,'send','actor-a-pending',text='客户A正在处理')
        assert started.wait(5)
        try:
            second=executor.submit(operate,tmp_path,run_id,'send','actor-b-done',actor_id='manual_customer_2',text='客户B自己的消息')
            assert second.result(timeout=5)['sequence']==2
            operate(tmp_path,run_id,'update_profile','change-b',actor_id='manual_customer_2',profile={'display_name':'更新后的客户B'})
            other_run=handle_manual_simulation({'action':'create','request_id':'independent-run'},runtime_dir=tmp_path,mode='mock')['run_id']
            assert other_run!=run_id
            assert operate(tmp_path,other_run,'send','other-run-message',text='另一次运行自己的消息')['sequence']==1
            other_report=read_report(tmp_path,other_run)
            report=read_report(tmp_path,run_id)
            assert report['transcript'][0]['manual_processing'] is True
            assert report['transcript'][1]['manual_processing'] is False
        finally:release.set()
        assert first.result(timeout=5)['sequence']==1
    report=read_report(tmp_path,run_id)
    assert [turn['user']['customer_id'] for turn in report['transcript']]==['manual_customer','manual_customer_2']
    assert report['transcript'][1]['user']['text']=='客户B自己的消息'
    assert next(actor for actor in report['manual_actors'] if actor['customer_id']=='manual_customer_2')['display_name']=='更新后的客户B'
    assert len(report['manual_profile_events'])==1
    assert read_report(tmp_path,other_run)==other_report


def test_batch_cancel_while_model_runs_preserves_sent_item_and_unsent_raw_messages(tmp_path,monkeypatch):
    """批次取消可立即落盘；本条结束后不得复活已取消批次或删除未发原文。"""
    run_id=create(tmp_path);started=threading.Event();release=threading.Event()
    operate(tmp_path,run_id,'queue_batch','batch-live',messages=[{'actor_id':'manual_customer','channel':'private','text':'第一条'},
                                                             {'actor_id':'manual_customer_2','channel':'private','text':'保留但不要发送'}])
    original=AgentRuntime.handle_user_message
    def handle(runtime,message,**kwargs):
        started.set();assert release.wait(8)
        return original(runtime,message,**kwargs)
    monkeypatch.setattr(AgentRuntime,'handle_user_message',handle)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first=executor.submit(operate,tmp_path,run_id,'batch_next','batch-first')
        assert started.wait(5)
        try:
            operate(tmp_path,run_id,'cancel_batch','cancel-live')
            assert read_report(tmp_path,run_id)['manual_batch']['status']=='cancelled'
        finally:release.set()
        assert first.result(timeout=5)['sequence']==1
    batch=read_report(tmp_path,run_id)['manual_batch']
    assert batch['status']=='cancelled' and batch['position']==1
    assert batch['items'][0]['status']=='done'
    assert batch['items'][1]['status']=='pending' and batch['items'][1]['text']=='保留但不要发送'


def test_recall_completed_before_old_report_merge_remains_monotonic(tmp_path,monkeypatch):
    """旧主循环已返回但还未展示时，撤回可先完成；迟到观测不能抹掉撤回或外发旧回复。"""
    run_id=create(tmp_path);returned=threading.Event();release=threading.Event()
    original=AgentRuntime.handle_user_message
    def handle(runtime,message,**kwargs):
        result=original(runtime,message,**kwargs)
        if message.text=='准备撤回的原文':
            returned.set();assert release.wait(8)
        return result
    monkeypatch.setattr(AgentRuntime,'handle_user_message',handle)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first=executor.submit(operate,tmp_path,run_id,'send','send-before-recall',text='准备撤回的原文')
        assert returned.wait(5)
        try:
            assert operate(tmp_path,run_id,'recall','recall-before-report',sequence=1)['sequence']==2
            assert read_report(tmp_path,run_id)['transcript'][0]['recalled'] is True
        finally:release.set()
        assert first.result(timeout=5)['sequence']==1
    old,recall=read_report(tmp_path,run_id)['transcript']
    assert old['recalled'] and old['user']['text']=='准备撤回的原文'
    assert old['agent']['reply']=='' and old['agent']['status']=='superseded'
    assert recall['recalled_message_id']==old['message_id']


def test_shown_reply_is_delivered_and_later_input_does_not_retract_it(tmp_path):
    """本地清单展示成功即标交付，新输入只能淘汰尚未显示的输出。"""
    run_id=create(tmp_path)
    operate(tmp_path,run_id,'send','shown-first',text='现在有人吗')
    first=read_report(tmp_path,run_id)['transcript'][0]
    assert first['agent']['reply']
    operate(tmp_path,run_id,'send','shown-second',text='改成下午')
    assert read_report(tmp_path,run_id)['transcript'][0]==first
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    try:
        reply=next(turn for turn in store.recent_turns(f'manual:{run_id}') if turn.role.value=='assistant' and turn.trace_id==first['agent']['trace_id'])
        assert reply.metadata['delivery_status']=='delivered'
    finally:store._connection.close()


def _observer_snapshot(message_ids,*,session_id='topic',fact=0,mapping='base'):
    return {'version':1,'sessions':[{'session_id':session_id,'messages':[
        {'message_id':message_id,'sender_external_id':'actor-a','sent_at':'2026-10-29T00:00:00+00:00'} for message_id in message_ids],
        'participants':['actor-a'] if message_ids else [],'extracted_state':{'value':fact},'last_activity_at':'2026-10-29T00:00:00+00:00'}],
        'message_sessions':[['room','m',mapping]],'classifications':[['room','m',{'meaning':mapping}]]}


def test_observer_three_way_merge_preserves_recall_and_latest_per_key():
    """各键独立版本防止完成顺序改写候选事实，删除及撤回保持单调。"""
    from copy import deepcopy
    from manual_simulation import _merge_group_snapshot, _exclude_recalled_observations
    before=_observer_snapshot(['m'])
    current=_merge_group_snapshot(before,before,_observer_snapshot([],mapping='newer'),incoming_sequence=3)
    merged=_merge_group_snapshot(current,before,_observer_snapshot(['m','n'],mapping='older'),incoming_sequence=2)
    assert len(merged['sessions'])==1
    assert [message['message_id'] for message in merged['sessions'][0]['messages']]==['n']
    assert merged['message_sessions']==[['room','m','newer']]
    assert merged['classifications']==[['room','m',{'meaning':'newer'}]]
    empty={'version':1,'sessions':[],'message_sessions':[],'classifications':[]}
    late=_merge_group_snapshot(empty,empty,_observer_snapshot(['m']),incoming_sequence=1)
    late=_exclude_recalled_observations(late,{'m'})
    assert all(not row['messages'] for row in late['sessions'])
    assert not late['message_sessions'] and not late['classifications']

    # 独立B话题的第4条不能阻止A话题第3条覆盖A自己的第2条。
    base=_observer_snapshot(['a'],session_id='a',fact=0)
    second=_observer_snapshot(['a'],session_id='a',fact=2)
    fourth=deepcopy(second);fourth['sessions'].extend(_observer_snapshot(['b'],session_id='b',fact=4)['sessions'])
    third=_observer_snapshot(['a'],session_id='a',fact=3)
    after_second=_merge_group_snapshot(base,base,second,incoming_sequence=2)
    order_a=_merge_group_snapshot(after_second,second,fourth,incoming_sequence=4)
    order_a=_merge_group_snapshot(order_a,base,third,incoming_sequence=3)
    order_b=_merge_group_snapshot(after_second,base,third,incoming_sequence=3)
    order_b=_merge_group_snapshot(order_b,second,fourth,incoming_sequence=4)
    assert {row['session_id']:row['extracted_state'] for row in order_a['sessions']}=={'a':{'value':3},'b':{'value':4}}
    assert order_a==order_b


def test_concurrent_group_observation_keeps_both_actor_sessions(tmp_path,monkeypatch):
    """候选群观察在锁外完成，迟到的A快照只能合并自己的增量，不能覆盖B的分类与消息。"""
    import manual_simulation as manual
    run_id=create(tmp_path);observed=threading.Event();release=threading.Event()
    original=manual.process_manual_group
    def process(runtime,report,actor,text,**kwargs):
        result=original(runtime,report,actor,text,**kwargs)
        if actor['customer_id']=='manual_customer':
            observed.set();assert release.wait(8)
        return result
    monkeypatch.setattr(manual,'process_manual_group',process)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first=executor.submit(operate,tmp_path,run_id,'send','group-observe-a',actor_id='manual_customer',channel='group',text='国标有人吗')
        assert observed.wait(5)
        try:
            second=executor.submit(operate,tmp_path,run_id,'send','group-observe-b',actor_id='manual_customer_2',channel='group',text='川麻有人吗')
            assert second.result(timeout=5)['sequence']==2
        finally:release.set()
        assert first.result(timeout=5)['sequence']==1
    report=read_report(tmp_path,run_id);snapshot=report['manual_group_session_state']
    assert {message['sender_external_id'] for session in snapshot['sessions'] for message in session['messages']}=={'manual_customer','manual_customer_2'}
    assert len(snapshot['message_sessions'])==len(snapshot['classifications'])==2
    assert {row[1] for row in snapshot['message_sessions']}=={turn['message_id'] for turn in report['transcript']}
    assert [turn['sequence'] for turn in report['transcript']]==[1,2]


def test_manual_pending_receipt_recovers_after_process_exit_without_duplicate_input(tmp_path,monkeypatch):
    """模型已完成但清单未合并时重启重试，沿用同一序号与核心幂等结果。"""
    import manual_simulation as manual
    run_id=create(tmp_path);original=manual._operate;original_build=manual.build_runtime;notices=[]
    def build_with_notices(store,mode):
        runtime,client=original_build(store,mode)
        def notice(event):notices.append((event.name,manual._MANUAL_LOCK.locked()))
        runtime.hook_manager.register('after_task_context_prepared',notice)
        runtime.hook_manager.register('after_user_turn_appended',notice)
        return runtime,client
    monkeypatch.setattr(manual,'build_runtime',build_with_notices)
    class SimulatedProcessExit(BaseException):pass
    def interrupted(*args,**kwargs):
        original(*args,**kwargs)
        raise SimulatedProcessExit()
    monkeypatch.setattr(manual,'_operate',interrupted)
    with pytest.raises(SimulatedProcessExit):operate(tmp_path,run_id,'send','recover-request',text='原始请求保留')
    pending=read_report(tmp_path,run_id)
    assert pending['manual_operations']['recover-request']['status']=='processing'
    assert pending['transcript'][0]['manual_processing'] is True
    monkeypatch.setattr(manual,'_operate',original)
    assert operate(tmp_path,run_id,'send','recover-request',text='原始请求保留')['sequence']==1
    report=read_report(tmp_path,run_id)
    assert len(report['transcript'])==1 and not report['transcript'][0]['manual_processing']
    assert len((tmp_path/'runs'/run_id/'live_events.jsonl').read_text().splitlines())==1
    assert notices==[('after_task_context_prepared',False),('after_user_turn_appended',False)]
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    try:
        turns=store.recent_turns(f'manual:{run_id}')
        assert [turn.content for turn in turns if turn.role.value=='user']==['原始请求保留']
        assert len([turn for turn in turns if turn.role.value=='assistant'])==1
    finally:store._connection.close()


def test_no_output_operation_does_not_reuse_reserved_timeline_sequence(tmp_path,monkeypatch):
    """已被并发领取的复查可无输出，但其幂等序号不能随后指向另一条用户消息。"""
    import manual_simulation as manual
    run_id=create(tmp_path)
    operate(tmp_path,run_id,'send','before-gap',text='先等等')
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db');stamp=now()
    try:
        result=ToolGateway(store).execute(ToolCall('schedule_follow_up',{'goal':'复查','waiting_for':'customer','due_at':(stamp+timedelta(minutes=3)).isoformat(),
            'expires_at':(stamp+timedelta(hours=1)).isoformat()},'测试'),trace_id='no-op-schedule',step_index=1,conversation_id=f'manual:{run_id}',sender_id='manual_customer',sender_name='客户A')
        assert result.allowed
    finally:store._connection.close()
    original=manual._operate
    def no_output(runtime,path,report,payload,request_id):
        if payload['action']=='advance':return {'ok':True,'run_id':run_id,'message':'复查已经被另一个执行者领取'}
        return original(runtime,path,report,payload,request_id)
    monkeypatch.setattr(manual,'_operate',no_output)
    operate(tmp_path,run_id,'advance','no-output-advance')
    assert [turn['sequence'] for turn in read_report(tmp_path,run_id)['transcript']]==[1]
    assert operate(tmp_path,run_id,'send','after-gap',text='补充一句')['sequence']==3
    assert operate(tmp_path,run_id,'advance','no-output-advance')['sequence']==2


def test_manual_accepted_order_reaches_core_before_first_execution_starts(tmp_path,monkeypatch):
    """第一条已被清单接受但执行尚未开始时，补充不能抢先成为核心第一条输入。"""
    import manual_simulation as manual
    run_id=create(tmp_path);paused=threading.Event();release=threading.Event()
    original=manual._operate
    def delay_before_execution(runtime,path,report,payload,request_id):
        if request_id=='earlier-private-input':
            paused.set();assert release.wait(8)
        return original(runtime,path,report,payload,request_id)
    monkeypatch.setattr(manual,'_operate',delay_before_execution)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first=executor.submit(operate,tmp_path,run_id,'send','earlier-private-input',text='先约晚上')
        assert paused.wait(5)
        try:
            second=executor.submit(operate,tmp_path,run_id,'send','later-private-input',text='改成下午')
            assert second.result(timeout=5)['sequence']==2
            paused_report=read_report(tmp_path,run_id)
            store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
            try:paused_core=[turn.content for turn in store.recent_turns(f'manual:{run_id}') if turn.role.value=='user']
            finally:store._connection.close()
        finally:release.set()
        assert first.result(timeout=5)['sequence']==1
    report=read_report(tmp_path,run_id)
    accepted_manifest=[turn['user']['text'] for turn in report['transcript']]
    assert accepted_manifest==['先约晚上','改成下午']
    store=SQLiteAgentStore(tmp_path/'runs'/run_id/'test_sim.db')
    try:
        core=[turn for turn in store.recent_turns(f'manual:{run_id}') if turn.role.value=='user']
        core_text=[turn.content for turn in core]
        assert core_text==accepted_manifest, f'清单接受顺序={accepted_manifest!r}，核心接受顺序={core_text!r}'
        assert [turn.metadata['conversation_version'] for turn in core]==[1,2]
    finally:store._connection.close()
    assert paused_core==accepted_manifest
    assert paused_report['transcript'][0]['manual_processing'] is True
    assert report['transcript'][0]['agent']['status']=='superseded'
    assert report['transcript'][0]['agent']['reply']==''
    assert report['transcript'][1]['agent']['reply']
    assert [turn['manual_input_receipt']['conversation_version'] for turn in report['transcript']]==[1,2]


def test_manual_private_pre_receipt_uses_short_entry_and_keeps_group_routing(tmp_path,monkeypatch):
    """只预接收私聊；群消息仍经群入口，所有模型调用都位于清单锁外。"""
    import manual_simulation as manual
    from sim_adapter import StaticAgentLLMClient
    received=[];handled=[];model_locks=[]
    original_receive=AgentRuntime.receive_user_message
    original_handle=AgentRuntime.handle_user_message
    original_complete=StaticAgentLLMClient.complete
    def receive(runtime,message,**kwargs):
        received.append({'conversation_id':message.conversation_id,'message_id':message.message_id,
                         'held':manual._MANUAL_LOCK.locked(),'sender_id':message.sender_id,'object_id':id(message)})
        return original_receive(runtime,message,**kwargs)
    def handle(runtime,message,**kwargs):
        handled.append((message.message_id,id(message)))
        return original_handle(runtime,message,**kwargs)
    def complete(client,*args,**kwargs):
        model_locks.append(manual._MANUAL_LOCK.locked())
        return original_complete(client,*args,**kwargs)
    monkeypatch.setattr(AgentRuntime,'receive_user_message',receive)
    monkeypatch.setattr(AgentRuntime,'handle_user_message',handle)
    monkeypatch.setattr(StaticAgentLLMClient,'complete',complete)
    run_id=create(tmp_path)
    operate(tmp_path,run_id,'send','private-pre-receipt',text='现在有人吗')
    operate(tmp_path,run_id,'send','group-no-bypass',actor_id='manual_customer_2',channel='group',text='川麻有人吗')
    assert model_locks and not any(model_locks)
    held=[item for item in received if item['held']]
    assert held and all(item['conversation_id']==f'manual:{run_id}' and item['sender_id']=='manual_customer' for item in held)
    assert not any(item['conversation_id']==f'manual-group:{run_id}' for item in received)
    assert all((item['message_id'],item['object_id']) in handled for item in held)
    report=read_report(tmp_path,run_id)
    assert report['transcript'][0]['manual_input_receipt']['accepted'] is True
    assert report['transcript'][0]['manual_input_receipt']['message_id']==report['transcript'][0]['message_id']
    assert 'manual_input_receipt' not in report['transcript'][1]
    assert report['transcript'][1]['group_processing']['entrypoint']=='GroupMessageHandler'


def test_manual_receipt_failure_keeps_raw_input_and_never_starts_model(tmp_path,monkeypatch):
    """正式接收失败不伪造 accepted，也不能继续旧模型；同请求可用原序号安全重试。"""
    run_id=create(tmp_path)
    original_receive=AgentRuntime.receive_user_message
    original_handle=AgentRuntime.handle_user_message
    def unavailable(*args,**kwargs):raise RuntimeError('deterministic receipt storage failure')
    def unexpected_model(*args,**kwargs):raise AssertionError('接收失败后不得进入执行入口')
    monkeypatch.setattr(AgentRuntime,'receive_user_message',unavailable)
    monkeypatch.setattr(AgentRuntime,'handle_user_message',unexpected_model)
    with pytest.raises(RuntimeError,match='deterministic receipt storage failure'):
        operate(tmp_path,run_id,'send','receipt-failure-retry',text='原始消息保留')
    report=read_report(tmp_path,run_id)
    assert report['transcript'][0]['user']['text']=='原始消息保留'
    assert report['transcript'][0]['agent']['status']=='failed'
    assert not report['transcript'][0]['manual_processing']
    assert 'manual_input_receipt' not in report['transcript'][0]
    assert report['manual_operations']['receipt-failure-retry']['status']=='failed'
    assert report['agent_internal_failure_count']==1 and report['quality_status']=='degraded'
    monkeypatch.setattr(AgentRuntime,'receive_user_message',original_receive)
    monkeypatch.setattr(AgentRuntime,'handle_user_message',original_handle)
    assert operate(tmp_path,run_id,'send','receipt-failure-retry',text='原始消息保留')['sequence']==1
    report=read_report(tmp_path,run_id)
    assert len(report['transcript'])==1 and report['transcript'][0]['manual_input_receipt']['accepted'] is True


def test_manual_batch_pre_receipt_failure_can_retry_the_same_original_input(tmp_path,monkeypatch):
    """批次正式接收失败后暂停，原输入和序号不丢，恢复不永久卡在 processing。"""
    run_id=create(tmp_path)
    operate(tmp_path,run_id,'queue_batch','receipt-batch',messages=[{'actor_id':'manual_customer','channel':'private','text':'批次原文'},
        {'actor_id':'manual_customer_2','channel':'private','text':'尚未发送的下一条'}])
    original_receive=AgentRuntime.receive_user_message
    def unavailable(*args,**kwargs):raise RuntimeError('deterministic batch receipt failure')
    monkeypatch.setattr(AgentRuntime,'receive_user_message',unavailable)
    with pytest.raises(RuntimeError,match='deterministic batch receipt failure'):
        operate(tmp_path,run_id,'batch_next','receipt-batch-first')
    report=read_report(tmp_path,run_id);batch=report['manual_batch']
    assert batch['status']=='paused_error' and batch['position']==0
    assert batch['items'][0]['status']=='failed' and batch['items'][0]['text']=='批次原文'
    assert batch['items'][1]['status']=='pending' and batch['items'][1]['text']=='尚未发送的下一条'
    monkeypatch.setattr(AgentRuntime,'receive_user_message',original_receive)
    assert operate(tmp_path,run_id,'batch_next','receipt-batch-first')['sequence']==1
    report=read_report(tmp_path,run_id)
    assert report['manual_batch']['position']==1 and report['manual_batch']['status']=='running'
    assert len(report['transcript'])==1
    assert report['transcript'][0]['manual_input_receipt']['conversation_version']==1
