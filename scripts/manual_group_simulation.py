"""手动沙盒复用正式群语义入口与权威话题收据；所有外发只进入本地收件箱。"""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy

from mahjong_agent_runtime import AgentAction, AgentRuntimeResult, UserMessage
from mahjong_agent_runtime.group_chat import (
    BoardEngine, ClaimHandler, GroupMessage, GroupMessageHandler,
    L1RuleEngine, L2IntentRouter, NotifyDispatcher, BOARD_TASK_TYPE,
)
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
        """成局等领域通知复用正式触发入口；群转私聊原话仍走用户输入。"""
        result=self.runtime.handle_system_trigger(message,**kwargs);self.results.append(result);return result


def process_manual_group(runtime,report,actor,text,*,message_id,stamp,trace_id,quote_id=None,recall_target=None):
    """处理一个受控角色的群发言，保存正式路由、实际输出及同一次分类的权威收据。"""
    room_id=f"manual-room:{report['run_id']}"
    message=GroupMessage(room_id=room_id,conversation_id=f"manual-group:{report['run_id']}",sender_external_id=actor['customer_id'],
        sender_name=actor['display_name'],text=text,message_id=message_id,sent_at=stamp,channel='console',quoted_message_id=quote_id,
        metadata={'content_type':'revoke_event','target_message_id':recall_target['message_id']} if recall_target else {})
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
        handler=GroupMessageHandler(rule_engine=L1RuleEngine(bot_id='manual-agent'),intent_router=L2IntentRouter(runtime.store,llm=runtime.llm_client,ack_picker=lambda options:options[0]),
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
                    'topic_analysis':_topic_analysis(processing)},message


def _topic_analysis(processing):
    """面板只投影正式入口已经保存的分类证据，不再次调用模型或维护第二套话题状态。"""
    receipt = deepcopy((processing.get('detail') or {}).get('topic_routing') or {})
    classification = (receipt.get('message') or {}).get('classification') or {}
    audit = classification.get('audit') or {}
    topic_id = (receipt.get('topic') or {}).get('topic_id')
    calls = [{**attempt, 'input':deepcopy(audit.get('input') or {})}
             for attempt in audit.get('attempts', [])]
    outcome = {'action':processing.get('action'), 'session_id':topic_id, 'classification':classification}
    return {'entrypoint':'SemanticTopicClassifier', 'scope':'authoritative_routing_receipt',
            'receipt':receipt, 'outcomes':[outcome] if receipt else [], 'model_calls':calls}
