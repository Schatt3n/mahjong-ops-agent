"""提供多角色、可编辑画像及群聊入口的隔离手动测试，不连接微信。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from simulation_observability import DEFAULT_RUNTIME_DIR, DEFAULT_ENV_FILE, ROOT, _atomic_write_json, _read_env_metadata, _safe_run_id

SIMULATION_PATH = ROOT / 'tests' / 'simulation'
if str(SIMULATION_PATH) not in sys.path:
    sys.path.insert(0, str(SIMULATION_PATH))
from behavior_policy import SimulationAction
from sim_adapter import RequestOutcome, build_runtime, simulation_execution_observation
from sim_orchestrator import outcome_to_transcript_entry
from mahjong_agent_runtime import CustomerProfile, SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.group_chat import ChannelIdentity, GroupRoomPolicy
from mahjong_agent_runtime.knowledge import default_terminology_repository
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.follow_up import handle_follow_up_task
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler
from manual_group_simulation import process_manual_group

_MANUAL_LOCK = threading.Lock()
MANUAL_CUSTOMER_ID = 'manual_customer'
WORKSPACE_VERSION = 2


def _private_conversation(run_id: str, actor_id: str) -> str:
    """每位角色有稳定且独立的私聊；默认角色保留约定的短会话编号。"""
    return f'manual:{run_id}' if actor_id == MANUAL_CUSTOMER_ID else f'manual:{run_id}:{actor_id}'


def _actor(report: dict, actor_id=None) -> dict:
    """仅接受当前沙盒名单中的角色，不允许借画像接口修改其他客户。"""
    selected = actor_id or report['default_actor_id']
    actor = next((item for item in report['manual_actors'] if item['customer_id'] == selected), None)
    if actor is None:
        raise ValueError('角色不属于当前测试场景')
    return actor


def _save_profile(store, report: dict, payload: dict, request_id: str) -> dict:
    """经白名单验证后保存操作员画像，并追加变更证据；不会把私有备注交给主模型。"""
    adding = payload['action'] == 'add_actor'
    if adding and len(report['manual_actors']) >= 20:
        raise ValueError('一个测试场景最多20位角色')
    actor_id = ('manual_' + hashlib.sha256(request_id.encode()).hexdigest()[:16]) if adding else _actor(report,payload.get('actor_id'))['customer_id']
    before = None if adding else store.customers[actor_id].to_dict()
    changes = payload.get('profile')
    allowed = {'display_name','gender','preferred_games','preferred_stakes','preferred_time_tags','profile_facts','smoke_preference','private_remark','notes','no_contact'}
    if not isinstance(changes,dict) or set(changes)-allowed:
        raise ValueError('画像字段无效')
    # 审计中的 before 必须保留原值，不能与待修改的字段字典共享引用。
    values = deepcopy(before) if before is not None else CustomerProfile(customer_id=actor_id,display_name='新客户').to_dict()
    for key,value in changes.items():
        if key in {'preferred_games','preferred_stakes','preferred_time_tags','profile_facts'}:
            if not isinstance(value,list) or len(value)>12 or any(not isinstance(item,str) or not item.strip() or len(item)>240 for item in value):
                raise ValueError('画像列表最多12项，每项须为1至240字')
            values[key] = list(dict.fromkeys(item.strip() for item in value))
        elif key=='no_contact':
            if not isinstance(value,bool):raise ValueError('触达开关须为布尔值')
            values[key]=value
        elif key=='smoke_preference':
            if value is not None and (not isinstance(value,str) or value not in {'any','no_smoke','smoking'}):raise ValueError('烟况须为未知、不限、无烟或有烟')
            values[key]=value
        elif key=='gender':
            if value is not None and (not isinstance(value,str) or value not in {'未知','男','女'}):raise ValueError('性别值无效')
            values[key]=value
        else:
            if not isinstance(value,str) or len(value)>1000:raise ValueError('画像文本须在1000字以内')
            values[key]=value.strip()
    name=values['display_name']
    if not name or len(name)>30:raise ValueError('角色名须为1至30字')
    if any(item['customer_id']!=actor_id and item['display_name']==name for item in report['manual_actors']):
        raise ValueError('角色名不能重复')
    normalized=[]
    for game in values['preferred_games']:
        term=next((term for term in default_terminology_repository().terms if term.category=='game_type'
                   and game.casefold() in {alias.casefold() for alias in term.aliases}),None)
        normalized.append(str(term.canonical['requested_game']) if term else game)
    values['preferred_games']=list(dict.fromkeys(normalized));values['public_name']=name
    profile=CustomerProfile(**values);store.upsert_customer(profile)
    store.upsert_channel_identity(ChannelIdentity(channel='console',external_user_id=actor_id,customer_id=actor_id,public_name=name,
        private_conversation_id=_private_conversation(report['run_id'],actor_id),can_private_message=True,is_friend=True))
    report['manual_private_conversations'][actor_id]=_private_conversation(report['run_id'],actor_id)
    report['manual_actors']=[profile.to_dict() if item['customer_id']==actor_id else item for item in report['manual_actors']]
    if adding:report['manual_actors'].append(profile.to_dict())
    event={'actor_id':actor_id,'request_id':request_id,'before':before,'after':profile.to_dict(),'saved_at':now().isoformat()}
    report.setdefault('manual_profile_events',[]).append(event)
    return {'ok':True,'run_id':report['run_id'],'actor_id':actor_id,'message':'画像已保存，下条消息会使用新画像'}


def handle_manual_simulation(payload: dict, *, runtime_dir: Path | None = None, mode: str | None = None) -> dict:
    """串行执行沙盒操作；分条批量发送沿用消息幂等、真实群入口和原始证据格式。"""
    action=str(payload.get('action') or '')
    if action not in {'create','send','recall','advance','add_actor','update_profile','queue_batch','batch_next','cancel_batch'}:
        raise ValueError('unsupported manual simulation action')
    request_id=str(payload.get('request_id') or '')
    if not request_id or len(request_id)>100:raise ValueError('缺少有效请求编号')
    base=(runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve()
    with _MANUAL_LOCK:
        if action=='create':return _create_session(base,payload,request_id,mode)
        run_id=_safe_run_id(str(payload.get('run_id') or ''))
        path=base/'runs'/run_id/'report.json'
        if not run_id or not path.is_file():raise ValueError('找不到手动测试记录')
        with (path.parent/'manual.lock').open('a+') as lock_file:
            fcntl.flock(lock_file,fcntl.LOCK_EX)
            report=json.loads(path.read_text())
            if not report.get('manual_session'):raise ValueError('只能操作手动测试会话，历史自动测试记录只读')
            if report.get('manual_workspace_version')!=WORKSPACE_VERSION:
                raise ValueError('历史单角色记录只读，请新建多角色测试场景')
            previous=report['manual_requests'].get(request_id)
            if previous is not None:return {'ok':True,'run_id':run_id,'sequence':previous,'deduplicated':True}
            store=SQLiteAgentStore(path.parent/'test_sim.db')
            try:
                if action in {'add_actor','update_profile'}:
                    result=_save_profile(store,report,payload,request_id)
                    with (path.parent/'profile_events.jsonl').open('a') as output:
                        output.write(json.dumps(report['manual_profile_events'][-1],ensure_ascii=False)+'\n')
                elif action=='cancel_batch':
                    if report.get('manual_batch'):report['manual_batch']['status']='cancelled'
                    result={'ok':True,'run_id':run_id,'message':'未发消息已停止；已发消息及批次原文保留'}
                elif action=='queue_batch':
                    items=payload.get('messages')
                    if not isinstance(items,list) or not 1<=len(items)<=30:raise ValueError('批量输入须为1至30条消息')
                    normalized=[]
                    for index,item in enumerate(items):
                        if not isinstance(item,dict):raise ValueError('批量消息格式无效')
                        actor=_actor(report,item.get('actor_id'));channel=item.get('channel','group');text=item.get('text')
                        if channel not in {'group','private'} or not isinstance(text,str) or not 1<=len(text.strip())<=4000:
                            raise ValueError('批量消息须指定群聊或私聊，正文为1至4000字')
                        normalized.append({'actor_id':actor['customer_id'],'actor_name':actor['display_name'],'channel':channel,'text':text.strip(),'status':'pending',
                            'message_id':f"batch:{hashlib.sha256(request_id.encode()).hexdigest()[:16]}:{index}"})
                    if report.get('manual_batch',{}).get('status')=='running':raise ValueError('已有未完成批次，请先完成或更换新测试场景')
                    if report.get('manual_batch'):report.setdefault('manual_batch_history',[]).append(report['manual_batch'])
                    report['manual_batch']={'items':normalized,'position':0,'status':'running'}
                    result={'ok':True,'run_id':run_id,'message':f'已载入{len(normalized)}条，按顺序执行'}
                else:
                    runtime,_=build_runtime(store,report['llm_mode'])
                    if action=='batch_next':
                        batch=report.get('manual_batch')
                        if not batch or batch['status']=='cancelled' or batch['position']>=len(batch['items']):
                            return {'ok':True,'run_id':run_id,'message':'批次已完成'}
                        item=batch['items'][batch['position']]
                        result=_operate(runtime,path,report,{**item,'action':'send'},request_id)
                        item.update(status='done',sequence=result['sequence']);batch['position']+=1
                        batch['status']='completed' if batch['position']==len(batch['items']) else 'running'
                        if report['transcript'][-1]['agent']['status'] in {'needs_help','needs_human'}:
                            batch['status']='paused_error'
                            result['message']='本条 Agent 执行异常，批量发送已暂停，可查看证据后继续'
                    else:result=_operate(runtime,path,report,payload,request_id)
                # 名单是当前画像的展示投影；模型合法记忆更新后，编辑器也应从 Store 读取最新事实。
                report['manual_actors']=[store.customers[item['customer_id']].to_dict() for item in report['manual_actors']]
                report['manual_requests'][request_id]=len(report['transcript'])
                report['updated_at']=now().isoformat()
                _atomic_write_json(path,report)
                return result
            finally:
                store._connection.close()


def _create_session(base: Path, payload: dict, request_id: str, mode: str | None) -> dict:
    """创建只包含受控角色的当前格式沙盒，不再掺入百名随机客户。"""
    digest=hashlib.sha256(request_id.encode()).hexdigest()[:16];runs=base/'runs';runs.mkdir(parents=True,exist_ok=True)
    existing=list(runs.glob(f'*_manual_{digest}'))
    if existing and (existing[0]/'report.json').is_file():return {'ok':True,'run_id':existing[0].name,'manual_session':True}
    initial_name=str(payload.get('display_name') or '客户A').strip()
    if not initial_name or len(initial_name)>30:raise ValueError('角色名须为1至30字')
    # 创建未完成时只恢复同一请求的初始化，不重置数据库或另造一次运行。
    run_id=existing[0].name if existing else datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_manual_'+digest
    folder=runs/run_id;folder.mkdir(exist_ok=True);store=SQLiteAgentStore(folder/'test_sim.db')
    try:
        store.configure_rooms(['manual_room_1','manual_room_2','manual_room_3'])
        selected_mode=mode or _read_env_metadata(DEFAULT_ENV_FILE).get('MAHJONG_PERIODIC_SIM_LLM_MODE','real')
        runtime,client=build_runtime(store,selected_mode)
        report={'run_id':run_id,'manual_session':True,'manual_workspace_version':WORKSPACE_VERSION,
            'manual_actors':[],'default_actor_id':MANUAL_CUSTOMER_ID,'manual_profile_events':[],
            'manual_private_conversations':{},'manual_group_conversation_id':f'manual-group:{run_id}',
            'status':'interactive','quality_status':'unknown','llm_mode':selected_mode,
            'main_agent_model':getattr(getattr(client,'config',None),'model',None),
            'main_agent_runtime_config':{'max_tokens_per_call':runtime.token_budget.max_tokens_per_call},
            'message_generation_mode':'manual','started_at':now().isoformat(),
            'configuration':{'channels':['private','group'],'external_channel':False,'group_entrypoint':'GroupMessageHandler',
                             'topic_analysis':'GroupSessionPipeline (candidate observation)','rooms':list(store.room_ids)},
            'manual_game_catalog':[{'id':term.canonical['requested_game'],'label':term.canonical['game_type']}
                                   for term in default_terminology_repository().terms if term.category=='game_type'],
            'total_messages':0,'transcript':[],'manual_requests':{}}
        names=[initial_name,*[name if name!=initial_name else name+'2' for name in ('客户B','客户C')]]
        for index,name in enumerate(names):
            actor_id=MANUAL_CUSTOMER_ID if index==0 else f'manual_customer_{index+1}'
            profile=CustomerProfile(customer_id=actor_id,display_name=name,public_name=name,gender='未知')
            store.upsert_customer(profile);report['manual_actors'].append(profile.to_dict())
            report['manual_private_conversations'][actor_id]=_private_conversation(run_id,actor_id)
            store.upsert_channel_identity(ChannelIdentity(channel='console',external_user_id=actor_id,customer_id=actor_id,public_name=name,
                private_conversation_id=_private_conversation(run_id,actor_id),can_private_message=True,is_friend=True))
        store.upsert_group_room_policy(GroupRoomPolicy(room_id=f'manual-room:{run_id}',channel='console'))
        _atomic_write_json(folder/'report.json',report)
        return {'ok':True,'run_id':run_id,'manual_session':True,'actor_id':MANUAL_CUSTOMER_ID}
    finally:store._connection.close()


def _operate(runtime,path: Path,report: dict,payload: dict,request_id: str) -> dict:
    """按受控角色与通道处理一条消息；群转私聊、静默和撤回都保存真实执行证据。"""
    run_id=report['run_id'];actor=_actor(report,payload.get('actor_id'));command=payload['action']
    channel=payload.get('channel','private')
    if channel not in {'group','private'}:raise ValueError('未知输入通道')
    conversation_id=f'manual-group:{run_id}' if channel=='group' else _private_conversation(run_id,actor['customer_id'])
    profile_at_input=deepcopy(runtime.store.customers[actor['customer_id']].to_model_context())
    text=str(payload.get('text') or '').strip();target=None;processing=None
    if command=='send' and not 1<=len(text)<=4000:raise ValueError('消息须为1至4000个字符')
    if command=='recall':
        target=next((turn for turn in report['transcript'] if turn['sequence']==payload.get('sequence') and turn['event_type']=='text'
                     and turn['user']['generation']['source']=='manual' and turn['user']['customer_id']==actor['customer_id']),None)
        if target is None:raise ValueError('只能撤回自己发送的消息，请切换到对应角色')
        if target.get('recalled'):return {'ok':True,'run_id':run_id,'already_recalled':True}
        channel=target['channel'];conversation_id=target['conversation_id'];text='撤回了一条消息'
    stamp=max(now(),datetime.fromisoformat(report.get('manual_virtual_time') or now().isoformat()))
    message_id=payload.get('message_id') or 'manual:'+hashlib.sha256(f'{run_id}:{request_id}'.encode()).hexdigest()[:24]

    def record(result,*,content,event_type,source,latency_ms,group_processing=None):
        """沿用统一审阅格式，保留当前画像快照、实际输出通道和候选话题来源。"""
        topic=next((item.get('session_id') for item in (group_processing or {}).get('topic_analysis',{}).get('outcomes',[]) if item.get('session_id')),None)
        action=SimulationAction(due_simulated_seconds=0,sequence=len(report['transcript'])+1,channel=channel,conversation_id=conversation_id,
            sender_id=actor['customer_id'],sender_name=actor['display_name'],text=content,thread_id=topic or conversation_id,
            event_type=event_type,generation_source=source)
        response=result.to_dict();response['simulation_observation']=simulation_execution_observation(runtime,result.trace_id)
        response['operational_state']={'follow_ups':[task.to_dict() for task in runtime.store.scheduled_tasks.values() if task.subject_id==actor['customer_id']]}
        entry=outcome_to_transcript_entry(RequestOutcome(action=action,sent=True,status_code=200,response=response,latency_ms=latency_ms))
        entry['message_id']=message_id;entry['manual_profile_at_input']=profile_at_input
        if group_processing:
            entry['group_processing']=group_processing;entry['topic_label_source']='candidate_group_analysis'
        else:entry['topic_label_source']='manual_private_conversation'
        if event_type=='recall' and target:entry['recalled_message_id']=target['message_id']
        report['transcript'].append(entry)
        with (path.parent/'live_events.jsonl').open('a') as output:output.write(json.dumps(entry,ensure_ascii=False)+'\n')

    if command=='advance':
        pending=[task for task in runtime.store.scheduled_tasks.values() if task.task_type=='conversation_follow_up' and task.subject_id==actor['customer_id'] and task.status.value=='pending']
        if not pending:return {'ok':True,'run_id':run_id,'message':'这位角色当前没有待复查任务'}
        task=min(pending,key=lambda item:item.due_at);due=task.due_at;conversation_id=task.conversation_id;channel='private'
        def trigger(task,trace_id):
            """仅领取当前角色的指定复查，保持其他角色和领域任务不动。"""
            started=time.monotonic();result=handle_follow_up_task(runtime,task,trace_id,at=due)
            if result:record(result,content='（测试触发：到达复查时间）',event_type='follow_up_timeout',source='scheduler',latency_ms=(time.monotonic()-started)*1000)
        ScheduledAgentTaskScheduler(runtime.store,trigger,runtime.trace_recorder).run_task_once(task.task_id,at=due)
        report['manual_virtual_time']=due.isoformat()
    else:
        started=time.monotonic()
        if channel=='group':
            quoted=None
            if payload.get('quote_sequence'):
                quoted=next((turn for turn in report['transcript'] if turn['sequence']==payload['quote_sequence'] and turn['channel']=='group' and not turn.get('recalled')),None)
                if quoted is None:raise ValueError('引用必须是本测试群尚未撤回的消息')
            result,processing,_=process_manual_group(runtime,report,actor,'' if target else text,message_id=message_id,stamp=stamp,
                trace_id=f'manual_group_{hashlib.sha256(message_id.encode()).hexdigest()[:16]}',quote_id=quoted['message_id'] if quoted else None,recall_target=target)
        else:
            metadata={'channel':'console','simulation':True}
            if target:metadata.update(event_type='message_recalled',target_message_id=target['message_id'])
            result=runtime.handle_user_message(UserMessage(conversation_id=conversation_id,sender_id=actor['customer_id'],sender_name=actor['display_name'],
                text='' if target else text,message_id=message_id,sent_at=stamp,metadata=metadata))
        record(result,content=text,event_type='recall' if target else 'text',source='manual',latency_ms=(time.monotonic()-started)*1000,group_processing=processing)
        if target:target['recalled']=True
    report['total_messages']=len(report['transcript'])
    failures=[turn for turn in report['transcript'] if turn['agent']['status'] in {'needs_help','needs_human'} or turn['agent']['objective_status']=='needs_tool']
    topic_failures=sum(any(outcome['action'] in {'classification_failed','topic_analysis_failed'} for outcome in turn.get('group_processing',{}).get('topic_analysis',{}).get('outcomes',[])) for turn in report['transcript'])
    report.update(status='interactive',quality_status='degraded' if failures or topic_failures else 'passed',agent_internal_failure_count=len(failures),topic_analysis_failure_count=topic_failures,
        pending_follow_up_count=sum(task.task_type=='conversation_follow_up' and task.status.value=='pending' for task in runtime.store.scheduled_tasks.values()))
    return {'ok':True,'run_id':run_id,'sequence':report['transcript'][-1]['sequence']}
