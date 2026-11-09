"""模拟通道只负责装配正式群入口及记录真实去向，不执行独立的群聊业务策略。"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from mahjong_agent_runtime import UserMessage
from mahjong_agent_runtime.group_chat import (
    BoardEngine, ClaimHandler, GroupMessage, GroupMessageHandler, L1RuleEngine,
    L2IntentRouter, NotifyDispatcher,
)


class SimulationMessenger:
    """只向本次模拟的受控成员记录外发；不含微信或网络发送能力。"""

    def __init__(self, store, room_id: str) -> None:
        """绑定可信沙盒房间与权威Store，不接受模型选择收件范围。"""
        self.store, self.room_id, self.outputs = store, room_id, []

    def send_group_message(self, room_id, text, *, metadata=None):
        """群发送必须属于已授权的模拟房间。"""
        policy = self.store.get_group_room_policy(room_id)
        if room_id != self.room_id or not policy or not policy.managed or not policy.outbound_enabled:
            raise ValueError('simulation room output is not authorized')
        return self._record('group', room_id, '', text, metadata, f'sim:group:{room_id}')

    def send_private_message(self, external_user_id, text, *, metadata=None):
        """私聊只给有通道权限的对应成员，绝不按原输入的群标签广播。"""
        identity = self.store.get_channel_identity('console', external_user_id)
        if identity is None or not identity.can_private_message:
            raise ValueError('simulation private output is not authorized')
        return self._record('private', external_user_id, identity.public_name, text, metadata,
                            identity.private_conversation_id)

    def _record(self, channel, recipient, name, text, metadata, conversation_id):
        """真实发送收据与私聊会话随输出保存，观测台和客户收件箱共用这份证据。"""
        output = {'channel': channel, 'recipient_id': recipient, 'recipient_name': name, 'text': text,
                  'conversation_id': conversation_id, 'metadata': dict(metadata or {})}
        self.outputs.append(output)
        return f"simulation-output:{len(self.outputs)}:{(metadata or {}).get('trace_id', '')}"


class CapturedRuntime:
    """旁录正式主 Agent 的输入和结果，以便保留引用撤回及执行证据。"""

    def __init__(self, runtime):
        """每条群消息独立旁录，避免并发请求共享输出列表。"""
        self.runtime, self.results, self.targets = runtime, [], []
        self.claimed, self.blocked_outputs = [], []

    @property
    def tool_gateway(self):
        """静默报局复用运行时的可信工具网关，注册时不调用主模型。"""
        return self.runtime.tool_gateway

    def handle_user_message(self, message, **kwargs):
        """正常消息继续使用正式会话隔离、幂等和工具边界。"""
        self.targets.append({'conversation_id': message.conversation_id, 'message_id': message.message_id})
        result = self.runtime.handle_user_message(message, **kwargs)
        self.results.append(result)
        return result

    def handle_system_trigger(self, message, **kwargs):
        """旁录成局等领域通知；群转私聊的真实输入走 handle_user_message。"""
        result = self._claim_result(self.runtime.handle_system_trigger(message, **kwargs))
        self.results.append(result)
        return result

    def prepare_reply_delivery(self, result):
        """业务入口选择实际发送后才声明交付，静默承接不会被标成正在投递。"""
        checked = self._claim_result(result)
        self.results[-1] = checked
        return bool(checked.final_reply)


    def _claim_result(self, result):
        """生成回复先走正式交付栅栏；已过期结果不能进入沙盒发送器。"""
        if result.final_reply:
            if self.runtime.store.claim_assistant_reply_delivery(result.conversation_id, trace_id=result.trace_id):
                self.claimed.append(result)
            else:
                self.blocked_outputs.append({'conversation_id': result.conversation_id,
                    'trace_id': result.trace_id, 'text': result.final_reply, 'reason':'reply_no_longer_deliverable'})
                result = replace(result, final_reply='', status='superseded')
        return result

    def finish_deliveries(self, outputs):
        """实际记入沙盒收据后才完成交付，不把静态群确认当作生成回复的证据。"""
        for result in self.claimed:
            if any(item['text'] == result.final_reply and item['metadata'].get('trace_id') == result.trace_id
                   for item in outputs):
                self.runtime.store.mark_assistant_reply_delivered(result.conversation_id, trace_id=result.trace_id)


def process_group_payload(runtime, payload: dict, receipts: dict) -> dict[str, Any]:
    """将模拟群消息交给生产 Handler；没有群路由输出就没有客户可见回复。"""
    meta = dict(payload.get('metadata') or {})
    raw = meta.get('raw_wechat_payload') or {}
    room_id = str((raw.get('room') or {}).get('id') or meta.get('group_id') or '')
    sender = str(payload['sender_id'])
    identity = runtime.store.get_channel_identity('console', sender)
    policy = runtime.store.get_group_room_policy(room_id)
    if not room_id or identity is None or not policy or not policy.managed:
        raise ValueError('unknown simulation room or member')
    messenger = SimulationMessenger(runtime.store, room_id)
    captured = CapturedRuntime(runtime)
    trace_id = str(payload.get('trace_id') or payload['message_id'])
    message = GroupMessage(room_id=room_id, conversation_id=str(payload['conversation_id']),
        sender_external_id=sender, sender_name=str(payload['sender_name']), text=str(payload.get('text') or ''),
        message_id=str(payload['message_id']), channel='console',
        quoted_message_id=raw.get('quoted_message_id') or meta.get('quoted_message_id'))
    if meta.get('event_type') == 'message_recalled':
        target = receipts.get((room_id, sender, str(meta.get('target_message_id') or '')), {})
        for index, item in enumerate(target.get('group_processing', {}).get('recall_targets', [])):
            captured.handle_user_message(UserMessage(conversation_id=item['conversation_id'], sender_id=sender,
                sender_name=message.sender_name, text='', message_id=f'{message.message_id}:{index}',
                metadata={'event_type': 'message_recalled', 'target_message_id': item['message_id']}),
                trace_id=f'{trace_id}:{index}')
        routing = {'action': 'message_recalled', 'matched_original': bool(target)}
    else:
        dispatcher = NotifyDispatcher(store=runtime.store, messenger=messenger, runtime=captured)
        board = BoardEngine(store=runtime.store, messenger=messenger)
        handler = GroupMessageHandler(rule_engine=L1RuleEngine(bot_id='simulation-agent'),
            intent_router=L2IntentRouter(runtime.store, llm=runtime.llm_client),
            board_engine=board, claim_handler=ClaimHandler(board_engine=board, store=runtime.store,
                notify_dispatcher=dispatcher), notify_dispatcher=dispatcher, messenger=messenger, runtime=captured,
            prepare_reply_delivery=captured.prepare_reply_delivery)
        routing = asdict(handler.handle(message, trace_id=trace_id))
    captured.finish_deliveries(messenger.outputs)
    for result in captured.results:
        for turn in runtime.store.recent_turns(result.conversation_id, limit=60):
            source = str(turn.metadata.get('source_message_id') or '')
            if turn.trace_id == result.trace_id and turn.role.value == 'user' and turn.sender_id == sender and source:
                target = {'conversation_id':result.conversation_id, 'message_id':source}
                if target not in captured.targets:
                    captured.targets.append(target)
    # 当前客户只能看到群公开输出和发给本人的私聊；其他成员的私聊只进内部执行证据。
    private = [item for item in messenger.outputs if item['channel'] == 'private' and item['recipient_id'] == sender]
    public = [item for item in messenger.outputs if item['channel'] == 'group']
    visible = private or public
    own_ids = {identity.private_conversation_id, f'group:{room_id}:customer:{identity.customer_id}'}
    primary = next((result for result in captured.results if result.conversation_id in own_ids), None)
    response = primary.to_dict() if primary else {'trace_id': trace_id, 'status': 'completed', 'actions': [],
                                                'tool_results': [], 'state_transitions': []}
    response['final_reply'] = '\n'.join(item['text'] for item in visible if item['text'])
    response['tool_results'] = [tool.to_dict() for result in captured.results for tool in result.tool_results]
    response['state_transitions'] = [item.to_dict() for result in captured.results for item in result.state_transitions]
    if any(result.status in {'needs_help', 'needs_human', 'failed'} for result in captured.results):
        response['status'] = 'needs_help'
    response['group_processing'] = {'entrypoint': 'GroupMessageHandler', 'routing': routing,
        'outputs': messenger.outputs, 'runtime_results': [item.to_dict() for item in captured.results],
        'recall_targets': captured.targets, 'blocked_outputs': captured.blocked_outputs, 'external_wechat_sent': False}
    if private:
        response['reply_delivery'] = {'channel': 'private', 'conversation_id': identity.private_conversation_id,
                                      'recipient_id': sender}
    elif public:
        response['reply_delivery'] = {'channel': 'group', 'conversation_id': message.conversation_id}
    response['group_route_silent'] = (not messenger.outputs and not captured.results
        and routing['action'] in {'ignore', 'register_self_recruitment', 'board_import', 'claim', 'message_recalled'})
    return response
