"""提供多角色、可编辑画像及群聊入口的隔离手动测试，不连接微信。"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

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
_MANUAL_PROCESS_TOKEN = uuid.uuid4().hex
_MANUAL_ACTIVE_REQUESTS: set[tuple[str,str]] = set()
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
    """短锁接收并持久化输入，模型在锁外执行，完成时仅合并本次请求的证据。

    Runtime 负责会话版本和业务顺序；私聊在清单接受序号的同一短锁内完成正式接收，
    群消息仍由独立群入口分流。任何模型调用都在锁外，迟到结果不能覆盖新消息。
    """
    action=str(payload.get('action') or '')
    if action not in {'create','send','recall','advance','add_actor','update_profile','queue_batch','batch_next','cancel_batch'}:
        raise ValueError('unsupported manual simulation action')
    request_id=str(payload.get('request_id') or '')
    if not request_id or len(request_id)>100:raise ValueError('缺少有效请求编号')
    base=(runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve()
    if action=='create':
        # 创建只初始化本地库和客户端，不执行模型；同请求仍恢复原目录。
        with _MANUAL_LOCK:return _create_session(base,payload,request_id,mode)
    run_id=_safe_run_id(str(payload.get('run_id') or ''))
    path=base/'runs'/run_id/'report.json'
    if not run_id or not path.is_file():raise ValueError('找不到手动测试记录')
    store=None;owns_operation=False;active_key=(str(path),request_id)
    try:
        with _locked_manual_report(path) as report:
            existing=_existing_manual_request_result(report,request_id,active_key)
            if existing is not None:return existing
            selected_mode=report['llm_mode']
        # 客户端和连接准备不调用模型，但也不占清单锁；序号仅在准备完成后正式分配。
        store=SQLiteAgentStore(path.parent/'test_sim.db')
        runtime=None
        if action not in {'add_actor','update_profile','queue_batch','cancel_batch'}:
            runtime,_=build_runtime(store,selected_mode)
        with _locked_manual_report(path) as report:
            # 准备期间可能已有同请求完成，必须重读后再决定是否接收，不能凭旧快照写回。
            existing=_existing_manual_request_result(report,request_id,active_key)
            if existing is not None:return existing
            if action in {'add_actor','update_profile','queue_batch','cancel_batch'}:
                result=_manual_control(store,path,report,payload,request_id)
                report['manual_requests'][request_id]=len(report['transcript'])
                report['updated_at']=now().isoformat()
                _atomic_write_json(path,report)
                return result
            prepared=_prepare_manual_operation(store,report,payload,request_id)
            if prepared.get('finished'):
                return prepared['result']
            working,effective_payload=prepared['working'],prepared['payload']
            _MANUAL_ACTIVE_REQUESTS.add(active_key);owns_operation=True
            _atomic_write_json(path,report)
            if effective_payload['action']=='send' and effective_payload['channel']=='private':
                try:
                    _receive_manual_private_input(runtime,report,working,effective_payload,request_id)
                except Exception as exc:
                    # 原文已保存但尚未确认接收，失败不能伪装 accepted 或让请求永久 processing。
                    report['manual_operations'][request_id]['status']='failed'
                    entry=next(turn for turn in report['transcript'] if turn['sequence']==effective_payload['_manual_sequence'])
                    entry['agent'].update(status='failed',objective_status='needs_help')
                    entry.update(manual_processing=False,error=type(exc).__name__,status_code=503)
                    report['quality_status']='degraded';report['updated_at']=now().isoformat()
                    report['agent_internal_failure_count']=sum(turn['agent']['status'] in {'failed','needs_help','needs_human'} or turn['agent']['objective_status']=='needs_tool' for turn in report['transcript'])
                    if effective_payload.get('_manual_batch_message_id'):
                        for batch in [report.get('manual_batch'),*report.get('manual_batch_history',[])]:
                            if not batch:continue
                            item=next((item for item in batch['items'] if item['message_id']==effective_payload['_manual_batch_message_id']),None)
                            if item is not None:
                                item.update(status='failed',sequence=effective_payload['_manual_sequence'])
                                if batch['status']!='cancelled':batch['status']='paused_error'
                    _atomic_write_json(path,report)
                    raise
                _atomic_write_json(path,report)
        # 原话及版本已在短临界区按序接收；模型执行仍完全不占清单锁或文件锁。
        try:
            result=_operate(runtime,path,working,effective_payload,request_id)
        except Exception as exc:
            _finish_manual_operation(path,working,effective_payload,request_id,store,error=exc)
            raise
        return _finish_manual_operation(path,working,effective_payload,request_id,store,result=result)
    finally:
        if owns_operation:
            with _MANUAL_LOCK:_MANUAL_ACTIVE_REQUESTS.discard(active_key)
        if store is not None:store._connection.close()


def _existing_manual_request_result(report: dict,request_id: str,active_key: tuple[str,str]) -> dict | None:
    """准备客户端前后复用幂等检查，接收只承认当前清单和仍存活的执行者。"""
    previous=report['manual_requests'].get(request_id)
    if previous is not None:return {'ok':True,'run_id':report['run_id'],'sequence':previous,'deduplicated':True}
    pending=report.get('manual_operations',{}).get(request_id)
    if pending and pending['status']=='processing' and _manual_operation_is_live(pending,active_key):
        return {'ok':True,'run_id':report['run_id'],'sequence':pending['sequence'],'deduplicated':True,'processing':True}
    return None


def _manual_private_user_message(report: dict,payload: dict) -> UserMessage:
    """接收和执行共用原始私聊消息，固定身份、正文、消息编号和服务端接受时刻。"""
    return UserMessage(conversation_id=_private_conversation(report['run_id'],payload['actor_id']),
        sender_id=payload['actor_id'],sender_name=payload['_manual_actor_name'],text=str(payload.get('text') or '').strip(),
        message_id=payload['_manual_message_id'],sent_at=datetime.fromisoformat(payload['_manual_stamp']),
        metadata={'channel':'console','simulation':True})


def _receive_manual_private_input(runtime,report: dict,working: dict,payload: dict,request_id: str) -> None:
    """在清单接受序号的同一短锁内交给正式接收入口；不传票据控制后续核心版本。"""
    message=_manual_private_user_message(working,payload)
    trace_id='manual_private_'+hashlib.sha256(message.message_id.encode()).hexdigest()[:16]
    receipt=runtime.receive_user_message(message,trace_id=trace_id)
    # 仅保存后端生成的最小审计投影；handle 仍重新查 Store，不信任调用方给版本。
    receipt={key:receipt[key] for key in ('conversation_id','message_id','trace_id','conversation_version','accepted')}
    payload['_manual_trace_id']=receipt['trace_id'];payload['_manual_input_receipt']=receipt
    payload['_manual_received_user_message']=message
    report['manual_operations'][request_id]['input_receipt']=receipt
    next(turn for turn in report['transcript'] if turn['sequence']==payload['_manual_sequence'])['manual_input_receipt']=deepcopy(receipt)


def _manual_operation_is_live(operation: dict,active_key: tuple[str,str]) -> bool:
    """只把仍有执行者的请求判为处理中；进程退出后重试复用原消息和序号由核心幂等恢复。"""
    owner=operation.get('owner_process_id')
    if owner==os.getpid():
        return operation.get('owner_process_token')==_MANUAL_PROCESS_TOKEN and active_key in _MANUAL_ACTIVE_REQUESTS
    if not isinstance(owner,int) or owner<=0:return False
    try:os.kill(owner,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


@contextmanager
def _locked_manual_report(path: Path):
    """保护清单及其对应的短输入接收；调用者不得在此作用域请求模型或网络。"""
    with _MANUAL_LOCK:
        with (path.parent/'manual.lock').open('a+') as lock_file:
            fcntl.flock(lock_file,fcntl.LOCK_EX)
            report=json.loads(path.read_text())
            if not report.get('manual_session'):raise ValueError('只能操作手动测试会话，历史自动测试记录只读')
            if report.get('manual_workspace_version')!=WORKSPACE_VERSION:
                raise ValueError('历史单角色记录只读，请新建多角色测试场景')
            yield report


def _manual_control(store,path: Path,report: dict,payload: dict,request_id: str) -> dict:
    """仅处理无模型调用的画像和批次控制，使用当前清单避免覆盖正在处理的输入。"""
    action=payload['action'];run_id=report['run_id']
    if action in {'add_actor','update_profile'}:
        result=_save_profile(store,report,payload,request_id)
        with (path.parent/'profile_events.jsonl').open('a') as output:
            output.write(json.dumps(report['manual_profile_events'][-1],ensure_ascii=False)+'\n')
        return result
    if action=='cancel_batch':
        if report.get('manual_batch'):report['manual_batch']['status']='cancelled'
        return {'ok':True,'run_id':run_id,'message':'未发消息已停止；已发消息及批次原文保留'}
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
    return {'ok':True,'run_id':run_id,'message':f'已载入{len(normalized)}条，按顺序执行'}


def _prepare_manual_operation(store,report: dict,payload: dict,request_id: str) -> dict:
    """在短临界区固定输入身份、原文和序号，先保存处理中占位，执行由 Runtime 接管。"""
    payload=dict(payload);command=payload['action'];run_id=report['run_id']
    operation=report.setdefault('manual_operations',{}).get(request_id)
    if command=='batch_next':
        batch=report.get('manual_batch')
        if operation:
            item=next((item for candidate in [batch,*report.get('manual_batch_history',[])] if candidate
                       for item in candidate['items'] if item['message_id']==operation['input']['message_id']),None)
            if item is None:raise ValueError('找不到已经接收的批次消息，原始记录不能替换')
        elif not batch or batch['status']=='cancelled' or batch['position']>=len(batch['items']):
            return {'finished':True,'result':{'ok':True,'run_id':run_id,'message':'批次已完成'}}
        else:item=batch['items'][batch['position']]
        recovering=operation and operation['input'].get('message_id')==item['message_id']
        if item['status']=='processing' and not recovering:raise ValueError('当前批次消息仍在处理，可直接补充输入或稍后继续')
        # 批次完成序号和状态是观察进度，恢复时不能把它们变成新的用户输入证据。
        payload={key:item[key] for key in ('actor_id','channel','text','message_id')}
        payload.update(action='send',_manual_batch_message_id=item['message_id'])
    actor=_actor(report,payload.get('actor_id'));channel=payload.get('channel','private')
    if channel not in {'group','private'}:raise ValueError('未知输入通道')
    text=str(payload.get('text') or '').strip();target=None
    if payload['action']=='send' and not 1<=len(text)<=4000:raise ValueError('消息须为1至4000个字符')
    if payload['action']=='recall':
        target=next((turn for turn in report['transcript'] if turn['sequence']==payload.get('sequence') and turn['event_type']=='text'
                     and turn['user']['generation']['source']=='manual' and turn['user']['customer_id']==actor['customer_id']),None)
        if target is None:raise ValueError('只能撤回自己发送的消息，请切换到对应角色')
        if target.get('recalled') and not operation:return {'finished':True,'result':{'ok':True,'run_id':run_id,'already_recalled':True}}
        channel=target['channel'];text='撤回了一条消息'
    if payload['action']=='advance' and not operation and not any(task.task_type=='conversation_follow_up' and task.subject_id==actor['customer_id'] and task.status.value=='pending'
                                               for task in store.scheduled_tasks.values()):
        return {'finished':True,'result':{'ok':True,'run_id':run_id,'message':'这位角色当前没有待复查任务'}}
    if payload.get('quote_sequence') and not any(turn['sequence']==payload['quote_sequence'] and turn['channel']=='group' and not turn.get('recalled')
                                                for turn in report['transcript']):
        raise ValueError('引用必须是本测试群尚未撤回的消息')
    used_sequences=[turn['sequence'] for turn in report['transcript']]+[item['sequence'] for item in report['manual_operations'].values()]
    sequence=operation['sequence'] if operation else max(used_sequences,default=0)+1
    message_id=payload.get('message_id') or 'manual:'+hashlib.sha256(f'{run_id}:{request_id}'.encode()).hexdigest()[:24]
    input_evidence={key:payload[key] for key in ('action','actor_id','channel','text','quote_sequence','sequence','message_id') if key in payload}
    input_evidence.update(actor_id=actor['customer_id'],channel=channel,text=text,message_id=message_id)
    if operation and operation['input']!=input_evidence:raise ValueError('同一请求编号不能替换已经接收的原始输入')
    stamp=max(now(),datetime.fromisoformat(report.get('manual_virtual_time') or now().isoformat()))
    conversation_id=target['conversation_id'] if target else f'manual-group:{run_id}' if channel=='group' else _private_conversation(run_id,actor['customer_id'])
    previous_entry=next((turn for turn in report['transcript'] if turn['sequence']==sequence),None) if operation else None
    profile=deepcopy(previous_entry['manual_profile_at_input'] if previous_entry else store.customers[actor['customer_id']].to_model_context())
    actor_name=previous_entry['user']['display_name'] if previous_entry else actor['display_name']
    payload.update(actor_id=actor['customer_id'],channel=channel,_manual_sequence=sequence,_manual_stamp=stamp.isoformat(),
                   _manual_message_id=message_id,_manual_profile_at_input=profile,_manual_input_evidence=input_evidence,_manual_actor_name=actor_name)
    working=deepcopy(report)
    working['_manual_group_state_before']=deepcopy(report.get('manual_group_session_state'))
    action=SimulationAction(due_simulated_seconds=0,sequence=sequence,channel=channel,conversation_id=conversation_id,
        sender_id=actor['customer_id'],sender_name=actor_name,text=text if payload['action']!='advance' else '（测试触发：到达复查时间）',
        thread_id=conversation_id,event_type='recall' if target else 'follow_up_timeout' if payload['action']=='advance' else 'text',
        generation_source='scheduler' if payload['action']=='advance' else 'manual')
    entry=outcome_to_transcript_entry(RequestOutcome(action=action,sent=True,status_code=202,response={'status':'processing'},latency_ms=0))
    entry.update(message_id=message_id,manual_request_id=request_id,manual_profile_at_input=profile,manual_processing=True,
                 manual_input=input_evidence,topic_label_source='manual_private_conversation' if channel=='private' else 'candidate_group_analysis')
    report['transcript']=[turn for turn in report['transcript'] if turn['sequence']!=sequence]+[entry]
    report['transcript'].sort(key=lambda turn:turn['sequence'])
    report['total_messages']=len(report['transcript']);report['updated_at']=now().isoformat()
    report['manual_operations'][request_id]={'sequence':sequence,'status':'processing','actor_id':actor['customer_id'],'action':command,'input':input_evidence,
        'owner_process_id':os.getpid(),'owner_process_token':_MANUAL_PROCESS_TOKEN}
    if command=='batch_next':item['status']='processing'
    return {'working':working,'payload':payload}


def _merge_group_snapshot(current: dict | None,before: dict | None,incoming: dict | None,*,incoming_sequence: int) -> dict | None:
    """逐字段三方合并候选观察增量；各键记录输入版本，独立话题不能压住彼此更新。"""
    if not incoming or incoming==before:return current
    current=deepcopy(current or {'version':1,'sessions':[],'message_sessions':[],'classifications':[]})
    before=before or {'sessions':[],'message_sessions':[],'classifications':[]}
    revisions=current.setdefault('_manual_revisions',{})
    missing=object()

    def apply_delta(target,key,value,baseline,revision_parts):
        """仅提交相对起点变化的字段，删除也保留版本墓碑，旧结果不能复活它。"""
        if value==baseline:return
        revision_key=json.dumps(revision_parts,ensure_ascii=False,separators=(',',':'))
        if incoming_sequence<revisions.get(revision_key,0):return
        if value is missing:target.pop(key,None)
        else:target[key]=deepcopy(value)
        revisions[revision_key]=incoming_sequence

    sessions={row['session_id']:row for row in current['sessions']}
    original={row['session_id']:row for row in before['sessions']}
    for row in incoming['sessions']:
        session_key=row['session_id'];base=original.get(session_key,{})
        merged=sessions.setdefault(session_key,{'session_id':session_key,'messages':[],'participants':[],'extracted_state':{}})
        for field in set(base)|set(row):
            if field not in {'session_id','messages','participants','extracted_state'}:
                apply_delta(merged,field,row.get(field,missing),base.get(field,missing),['session',session_key,'field',field])
        messages={message['message_id']:message for message in merged['messages']}
        baseline={message['message_id']:message for message in base.get('messages',[])}
        updates={message['message_id']:message for message in row['messages']}
        for key in set(baseline)|set(updates):
            apply_delta(messages,key,updates.get(key,missing),baseline.get(key,missing),['session',session_key,'message',key])
        merged['messages']=sorted(messages.values(),key=lambda message:(message['sent_at'],message['message_id']))
        merged['participants']=sorted({message['sender_external_id'] for message in merged['messages']})
        facts=merged['extracted_state'];baseline_facts=base.get('extracted_state',{})
        for key in set(baseline_facts)|set(row['extracted_state']):
            apply_delta(facts,key,row['extracted_state'].get(key,missing),baseline_facts.get(key,missing),['session',session_key,'fact',key])
    current['sessions']=list(sessions.values())
    for field in ('message_sessions','classifications'):
        values={(row[0],row[1]):row for row in current[field]}
        baseline={(row[0],row[1]):row for row in before[field]}
        updates={(row[0],row[1]):row for row in incoming[field]}
        for key in set(baseline)|set(updates):
            apply_delta(values,key,updates.get(key,missing),baseline.get(key,missing),[field,*key])
        current[field]=list(values.values())
    return current


def _claim_manual_delivery(store,entry: dict) -> list[tuple[str,str]]:
    """在可见清单提交前原子领取精确回复；过期回复只保留审计，不进入本地收件箱。"""
    claimed=[];processing=entry.get('group_processing')
    results=processing.get('runtime_results',[]) if processing else [{
        'conversation_id':entry['conversation_id'],'trace_id':entry['agent']['trace_id'],
        'final_reply':entry['agent']['reply'],'status':entry['agent']['status'],
    }]
    for result in results:
        trace_id=result['trace_id'];conversation_id=result['conversation_id']
        if not result.get('final_reply') or result.get('status')=='superseded':continue
        if processing and not any(output.get('metadata',{}).get('trace_id')==trace_id for output in processing['outputs']):continue
        if store.claim_assistant_reply_delivery(conversation_id,trace_id=trace_id):
            claimed.append((conversation_id,trace_id));continue
        if entry['agent']['trace_id']==trace_id:
            status=entry['agent']['status']
            entry['agent'].update(reply='',status=status if status in {'failed','needs_help','needs_human','needs_tool'} else 'superseded',response_mode='silent')
        if processing:
            blocked=[output for output in processing['outputs'] if output.get('metadata',{}).get('trace_id')==trace_id]
            processing.setdefault('blocked_outputs',[]).extend(blocked)
            processing['outputs']=[output for output in processing['outputs'] if output not in blocked]
    return claimed


def _exclude_recalled_observations(snapshot: dict,recalled_ids: set[str]) -> dict:
    """撤回标记单调约束候选观察投影；原文继续保存在不可丢失的 transcript 与 trace。"""
    removed=set(recalled_ids)
    for session in snapshot['sessions']:
        kept=[]
        for message in session['messages']:
            sources=set(message.get('metadata',{}).get('accumulated_source_message_ids') or [message['message_id']])
            if sources & recalled_ids or message['message_id'] in recalled_ids:
                removed.update(sources);removed.add(message['message_id'])
            else:kept.append(message)
        session['messages']=kept
        session['participants']=sorted({message['sender_external_id'] for message in kept})
    for field in ('message_sessions','classifications'):
        snapshot[field]=[row for row in snapshot[field] if row[1] not in removed]
    return snapshot


def _finish_manual_operation(path: Path,working: dict,payload: dict,request_id: str,store,*,result: dict | None=None,error: Exception | None=None) -> dict:
    """重读清单后只替换本请求占位，保留新输入、画像、批次取消和其他客户的结果。"""
    sequence=payload['_manual_sequence']
    with _locked_manual_report(path) as report:
        if request_id in report['manual_requests']:
            return {'ok':True,'run_id':report['run_id'],'sequence':report['manual_requests'][request_id],'deduplicated':True}
        deliveries=[]
        entry=working.get('_manual_result_entry')
        if error is not None:
            entry=deepcopy(next(turn for turn in report['transcript'] if turn['sequence']==sequence))
            entry['agent'].update(status='failed',objective_status='needs_help')
            entry.update(error=type(error).__name__,status_code=503)
        if entry is None:
            report['transcript']=[turn for turn in report['transcript'] if turn['sequence']!=sequence]
        else:
            existing=next(turn for turn in report['transcript'] if turn['sequence']==sequence)
            if existing.get('recalled') and existing['message_id']==entry['message_id']:entry['recalled']=True
            entry['manual_processing']=False
            deliveries=_claim_manual_delivery(store,entry) if error is None else []
            report['transcript']=[entry if turn['sequence']==sequence else turn for turn in report['transcript']]
            with (path.parent/'live_events.jsonl').open('a') as output:output.write(json.dumps(entry,ensure_ascii=False)+'\n')
            if entry.get('recalled_message_id'):
                for turn in report['transcript']:
                    if turn['message_id']==entry['recalled_message_id']:turn['recalled']=True
        snapshot=_merge_group_snapshot(report.get('manual_group_session_state'),working.get('_manual_group_state_before'),working.get('manual_group_session_state'),
            incoming_sequence=sequence)
        if snapshot:
            report['manual_group_session_state']=_exclude_recalled_observations(snapshot,{turn['message_id'] for turn in report['transcript'] if turn.get('recalled')})
        if working.get('manual_virtual_time'):
            report['manual_virtual_time']=max(working['manual_virtual_time'],report.get('manual_virtual_time',''))
        batch_message=payload.get('_manual_batch_message_id')
        if batch_message:
            # 取消只停止未发部分；已在执行的这一条仍保存，不把 cancelled 覆盖回 running。
            for batch in [report.get('manual_batch'),*report.get('manual_batch_history',[])]:
                if not batch:continue
                item=next((item for item in batch['items'] if item['message_id']==batch_message),None)
                if item is None:continue
                item.update(status='done' if error is None else 'failed',sequence=sequence)
                while batch['position']<len(batch['items']) and batch['items'][batch['position']]['status']=='done':batch['position']+=1
                if batch['status']!='cancelled':
                    failed=error is not None or entry and entry['agent']['status'] in {'failed','needs_help','needs_human'}
                    batch['status']='paused_error' if failed else 'completed' if batch['position']==len(batch['items']) else 'running'
                    if failed and result is not None:result['message']='本条 Agent 执行异常，批量发送已暂停，可查看证据后继续'
        report['manual_operations'][request_id]['status']='failed' if error is not None else 'done'
        if error is None:report['manual_requests'][request_id]=sequence
        report['manual_actors']=[store.customers[item['customer_id']].to_dict() for item in report['manual_actors']]
        report['total_messages']=len(report['transcript']);report['updated_at']=now().isoformat()
        failures=[turn for turn in report['transcript'] if turn['agent']['status'] in {'failed','needs_help','needs_human'} or turn['agent']['objective_status']=='needs_tool']
        topic_failures=sum(any(outcome['action'] in {'classification_failed','topic_analysis_failed'} for outcome in turn.get('group_processing',{}).get('topic_analysis',{}).get('outcomes',[])) for turn in report['transcript'])
        report.update(status='interactive',quality_status='degraded' if failures or topic_failures else 'passed',agent_internal_failure_count=len(failures),topic_analysis_failure_count=topic_failures,
            pending_follow_up_count=sum(task.task_type=='conversation_follow_up' and task.status.value=='pending' for task in store.scheduled_tasks.values()))
        _atomic_write_json(path,report)
        # report 已是可读取的本地交付证据；精确 trace 确认，不复活 superseded 回复。
        for conversation_id,trace_id in deliveries:store.mark_assistant_reply_delivered(conversation_id,trace_id=trace_id)
    return result or {'ok':False,'run_id':working['run_id'],'sequence':sequence}


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
    report['_manual_result_entry']=None
    run_id=report['run_id'];actor=_actor(report,payload.get('actor_id'));command=payload['action']
    actor={**actor,'display_name':payload['_manual_actor_name']}
    channel=payload.get('channel','private')
    if channel not in {'group','private'}:raise ValueError('未知输入通道')
    conversation_id=f'manual-group:{run_id}' if channel=='group' else _private_conversation(run_id,actor['customer_id'])
    profile_at_input=deepcopy(payload['_manual_profile_at_input'])
    text=str(payload.get('text') or '').strip();target=None;processing=None;recorded=[]
    if command=='send' and not 1<=len(text)<=4000:raise ValueError('消息须为1至4000个字符')
    if command=='recall':
        target=next((turn for turn in report['transcript'] if turn['sequence']==payload.get('sequence') and turn['event_type']=='text'
                     and turn['user']['generation']['source']=='manual' and turn['user']['customer_id']==actor['customer_id']),None)
        if target is None:raise ValueError('只能撤回自己发送的消息，请切换到对应角色')
        if target.get('recalled'):return {'ok':True,'run_id':run_id,'already_recalled':True}
        channel=target['channel'];conversation_id=target['conversation_id'];text='撤回了一条消息'
    stamp=datetime.fromisoformat(payload['_manual_stamp'])
    message_id=payload['_manual_message_id']

    def record(result,*,content,event_type,source,latency_ms,group_processing=None):
        """沿用统一审阅格式，保留当前画像快照、实际输出通道和候选话题来源。"""
        topic=next((item.get('session_id') for item in (group_processing or {}).get('topic_analysis',{}).get('outcomes',[]) if item.get('session_id')),None)
        action=SimulationAction(due_simulated_seconds=0,sequence=payload['_manual_sequence'],channel=channel,conversation_id=conversation_id,
            sender_id=actor['customer_id'],sender_name=actor['display_name'],text=content,thread_id=topic or conversation_id,
            event_type=event_type,generation_source=source)
        response=result.to_dict();response['simulation_observation']=simulation_execution_observation(runtime,result.trace_id)
        response['operational_state']={'follow_ups':[task.to_dict() for task in runtime.store.scheduled_tasks.values() if task.subject_id==actor['customer_id']]}
        entry=outcome_to_transcript_entry(RequestOutcome(action=action,sent=True,status_code=200,response=response,latency_ms=latency_ms))
        entry['message_id']=message_id;entry['manual_profile_at_input']=profile_at_input;entry['manual_request_id']=request_id
        entry['manual_input']=deepcopy(payload['_manual_input_evidence'])
        if '_manual_input_receipt' in payload:entry['manual_input_receipt']=deepcopy(payload['_manual_input_receipt'])
        if group_processing:
            entry['group_processing']=group_processing;entry['topic_label_source']='candidate_group_analysis'
        else:entry['topic_label_source']='manual_private_conversation'
        if event_type=='recall' and target:entry['recalled_message_id']=target['message_id']
        report['transcript'].append(entry)
        recorded.append(entry)
        report['_manual_result_entry']=entry

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
            if command=='send':
                result=runtime.handle_user_message(payload['_manual_received_user_message'],trace_id=payload['_manual_trace_id'])
            else:
                # 撤回继续走正式事件路由；群消息也不从这里整段灌入私聊接收入口。
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
    return {'ok':True,'run_id':run_id,'sequence':recorded[-1]['sequence'] if recorded else None}
