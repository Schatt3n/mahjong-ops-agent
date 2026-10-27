"""让手动沙盒走正式群消息入口，并独立记录候选话题分析；所有外发只进入本地收件箱。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import json
import time

from mahjong_agent_runtime import AgentAction, AgentRuntimeResult, UserMessage
from mahjong_agent_runtime.group_chat import (
    BoardEngine, ClaimHandler, GroupMessage, GroupMessageHandler, GroupSessionClassifier,
    GroupSessionPipeline, L1RuleEngine, L2IntentRouter, MessageAccumulator, NotifyDispatcher,
    OwnerMessageParser, QuickFilter, SessionRouter, GroupSessionOutcome,
    BoardState, BoardItem, BOARD_TASK_TYPE,
)
from mahjong_agent_runtime.group_chat.projections import public_group_game_summary, public_game_start_display
from mahjong_agent_runtime.knowledge import default_terminology_repository
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler


class SandboxMessenger:
    """只记录群内与私聊收件人，不持有微信客户端或任何外部发送能力。"""
    def __init__(self, actors):
        self.actors=actors
        self.outputs=[]

    def send_group_message(self, room_id, text, *, metadata=None):
        """把群回复加入本地收件箱，返回可引用的测试消息编号。"""
        return self._record('group',room_id,'测试群',text,metadata)

    def send_private_message(self, external_user_id, text, *, metadata=None):
        """私聊输出只对一个受控角色登记，绝不伪装成群回复。"""
        actor=next((item for item in self.actors if item['customer_id']==external_user_id),None)
        return self._record('private',external_user_id,actor['display_name'] if actor else external_user_id,text,metadata)

    def _record(self,channel,recipient,name,text,metadata):
        """保留真实路由与发送元数据，便于审阅群转私聊及通知。"""
        output={'channel':channel,'recipient_id':recipient,'recipient_name':name,'text':text,'metadata':dict(metadata or {})}
        self.outputs.append(output)
        return f"sandbox-output:{len(self.outputs)}:{(metadata or {}).get('trace_id','')}"


class RecordingRuntime:
    """旁录正式入口实际产生的 Runtime 结果，不替换规划或工具执行。"""
    def __init__(self,runtime):
        self.runtime=runtime
        self.results=[]

    def __getattr__(self,name):
        return getattr(self.runtime,name)

    def handle_user_message(self,message,**kwargs):
        """委托真实主 Agent 并保留结果。"""
        result=self.runtime.handle_user_message(message,**kwargs);self.results.append(result);return result

    def handle_system_trigger(self,message,**kwargs):
        """群转私聊和通知仍通过相同的系统触发入口。"""
        result=self.runtime.handle_system_trigger(message,**kwargs);self.results.append(result);return result


class TopicClient:
    """候选话题模型只接触公开 Session；离线模式明确使用固定结构桩。"""
    def __init__(self,runtime,mode):
        self.runtime=runtime
        self.mode=mode
        self.calls=[]

    def complete(self,messages,**kwargs):
        """保留分类输入与输出，真实模式复用供应商，绝不共享主 Agent 对话历史。"""
        entry={'mode':self.mode,'model':getattr(getattr(self.runtime.llm_client,'config',None),'model','offline_stub'),
               'trace_id':kwargs.get('trace_id'),'input':json.loads(messages[-1]['content'])}
        self.calls.append(entry);started=time.monotonic()
        try:
            if self.mode=='real':
                raw=self.runtime.llm_client.complete(messages,**kwargs)
            else:
                raw=json.dumps({'intent':'query','extracted_features':{'game_type':None,'stakes':None,'time':None,'participant_code':None},
                                'matched_board_no':None,'confidence':0.5,'reasoning':'[T1] 离线分类桩，仅验证入口与持久化',
                                'response':None,'channel_action':'group_reply'},ensure_ascii=False)
            entry['output']=raw
        except Exception as exc:
            entry['error_type']=type(exc).__name__
            raise
        finally:entry['latency_ms']=round((time.monotonic()-started)*1000,2)
        return raw


def _sync_published_board(store,room_id,stamp):
    """候选分类只接收已经在沙盒群发布的看板投影，不复制联系人或私聊资料。"""
    snapshot=store.get_latest_board_snapshot(room_id)
    if snapshot is None:return
    names={term.canonical['requested_game']:term.canonical['game_type'] for term in default_terminology_repository().terms if term.category=='game_type'}
    items=[]
    for entry in snapshot.items:
        game=store.require_game(entry.game_id);public=public_group_game_summary(game);need=public['public_requirement']
        items.append(BoardItem(id=game.game_id,display_no=entry.item_no,game_type=names.get(need.get('game_type'),str(need.get('game_type') or '')),
            participant_code=public['seat_code'],time=public_game_start_display(game),smoking=str(need.get('smoke_preference') or ''),
            stakes=str(need.get('stake_label') or need.get('stake') or ''),special_rules=None,slots_filled=public['claimed_seats']))
    store.upsert_group_board_state(BoardState(room_id=room_id,items=items,source_message_id=snapshot.external_message_id,last_published_at=stamp))


def process_manual_group(runtime,report,actor,text,*,message_id,stamp,trace_id,quote_id=None,recall_target=None):
    """处理一个受控角色的群发言，分别保存正式路由、实际输出和候选话题快照。"""
    room_id=f"manual-room:{report['run_id']}"
    message=GroupMessage(room_id=room_id,conversation_id=f"manual-group:{report['run_id']}",sender_external_id=actor['customer_id'],
        sender_name=actor['display_name'],text=text,message_id=message_id,sent_at=stamp,channel='console',quoted_message_id=quote_id,
        metadata={'content_type':'revoke_event','target_message_id':recall_target['message_id']} if recall_target else {})
    router=SessionRouter(clock=lambda:stamp)
    if report.get('manual_group_session_state'):
        router.restore_snapshot(report['manual_group_session_state'])
    _sync_published_board(runtime.store,room_id,stamp)
    topic_client=TopicClient(runtime,report['llm_mode'])
    pipeline=GroupSessionPipeline(store=runtime.store,owner_parser=OwnerMessageParser(),quick_filter=QuickFilter(),
        accumulator=MessageAccumulator(quiet_seconds=1),session_router=router,classifier=GroupSessionClassifier(topic_client,timeout_seconds=45))
    try:
        accepted=pipeline.accept(message,trace_id=trace_id)
        outcomes=pipeline.flush_due(at=stamp+timedelta(seconds=2)) if accepted.action=='buffered' else [accepted]
    except Exception as exc:
        # 观察流程故障不能阻断真实群入口；失败类型仍作为独立证据呈现。
        outcomes=[GroupSessionOutcome(action='topic_analysis_failed',detail={'error_type':type(exc).__name__})]
    report['manual_group_session_state']=router.snapshot()
    messenger=SandboxMessenger(report['manual_actors']);captured=RecordingRuntime(runtime)
    before_transitions=len(runtime.store.transitions)
    if recall_target:
        targets=recall_target.get('group_processing',{}).get('recall_targets',[])
        for index,target in enumerate(targets):
            captured.handle_user_message(UserMessage(conversation_id=target['conversation_id'],sender_id=actor['customer_id'],sender_name=actor['display_name'],
                text='',message_id=f'{message_id}:{index}',sent_at=stamp,metadata={'event_type':'message_recalled','target_message_id':target['message_id']}),trace_id=f'{trace_id}:{index}')
        processing={'action':'message_recalled'}
    else:
        dispatcher=NotifyDispatcher(store=runtime.store,messenger=messenger,runtime=captured)
        board=BoardEngine(store=runtime.store,messenger=messenger,clock=lambda:stamp)
        handler=GroupMessageHandler(rule_engine=L1RuleEngine(bot_id='manual-agent'),intent_router=L2IntentRouter(runtime.store,ack_picker=lambda options:options[0]),
            board_engine=board,claim_handler=ClaimHandler(board_engine=board,store=runtime.store,notify_dispatcher=dispatcher,clock=lambda:stamp),
            notify_dispatcher=dispatcher,messenger=messenger,runtime=captured,clock=lambda:stamp)
        processing=asdict(handler.handle(message,trace_id=trace_id))
        # 手动输入以一次显式刷新为边界，仅推进本测试群的看板发布任务；不运行其他客户的任务。
        board_tasks=[task for task in runtime.store.scheduled_tasks.values() if task.task_type==BOARD_TASK_TYPE and task.aggregate_id==room_id and task.status.value=='pending']
        scheduler=ScheduledAgentTaskScheduler(runtime.store,lambda task,trace:board.publish(room_id,trace_id=trace),runtime.trace_recorder)
        for task in board_tasks:scheduler.run_task_once(task.task_id,at=max(stamp,task.due_at))
        processing['board_flush_count']=len(board_tasks)
    targets=[]
    for result in captured.results:
        for turn in runtime.store.recent_turns(result.conversation_id,limit=60):
            source=str(turn.metadata.get('source_message_id') or '')
            if turn.trace_id==result.trace_id and turn.role.value=='user' and turn.sender_id==actor['customer_id'] and source:
                target={'conversation_id':result.conversation_id,'message_id':source}
                if target not in targets:targets.append(target)
    identity=runtime.store.get_channel_identity('console',actor['customer_id'])
    own_conversations={f"group:{room_id}:customer:{actor['customer_id']}",identity.private_conversation_id if identity else ''}
    primary=next((result for result in captured.results if result.conversation_id in own_conversations),None)
    if primary is None:
        reply='\n'.join(output['text'] for output in messenger.outputs if output['channel']=='group')
        primary=AgentRuntimeResult(trace_id=trace_id,conversation_id=f"group:{room_id}:customer:{actor['customer_id']}",final_reply=reply,
            actions=[AgentAction(goal='处理群通道事件',objective_status='completed',reasoning_summary='正式群入口的处理结果见路由证据',
                reply_to_user=reply,response_mode='reply' if reply else 'silent',silence_reason='' if reply else 'group_route_no_reply')])
    return primary,{'entrypoint':'GroupMessageHandler','routing':processing,'outputs':messenger.outputs,'recall_targets':targets,
                    'state_transitions':[item.to_dict() for item in runtime.store.transitions[before_transitions:]],
                    'runtime_results':[result.to_dict() for result in captured.results],
                    'topic_analysis':{'entrypoint':'GroupSessionPipeline','scope':'candidate_observation_only','outcomes':[asdict(outcome) for outcome in outcomes],
                                      'sessions':[session.to_context() for session in router.list_sessions(room_id)],'model_calls':topic_client.calls}},message
