"""确有可选局时只私聊本人，桥接交付键与重试 trace 相互独立。"""

from dataclasses import replace
from datetime import timedelta

import pytest

from mahjong_agent_runtime import AgentRuntimeResult, InMemoryAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.group_chat import NotifyDispatcher
from tests.test_group_chat.test_claim_and_switch import TEST_NOW, _group_message, _identity, _stack


class DeliveryMessenger:
    """按桥接的 draft_id 优先规则去重，记录尝试和实际发送以区分两者。"""

    def __init__(self):
        self.attempts = []
        self.delivered = {}

    def send_group_message(self, recipient, text, *, metadata=None):
        return self._send('group', recipient, text, metadata)

    def send_private_message(self, recipient, text, *, metadata=None):
        return self._send('private', recipient, text, metadata)

    def _send(self, channel, recipient, text, metadata):
        row = {'channel':channel, 'recipient':recipient, 'text':text, 'metadata':dict(metadata or {})}
        self.attempts.append(row)
        key = row['metadata'].get('draft_id') or row['metadata'].get('source_message_id')
        assert key, '私聊交付必须有稳定来源键，不能依赖本次trace'
        self.delivered.setdefault(key, row)
        return key


def test_private_switch_same_source_different_trace_delivers_reply_once():
    store = InMemoryAgentStore()
    _, runtime, _, _, handler = _stack(store)
    _identity(store, 'friend', friend=True, public_name='群友')
    store.create_game(conversation_id='wechaty:contact:host', organizer_id='host', organizer_name='发局人',
                      requirement={'game_type': 'hangzhou_mahjong', 'party_size': 3,
                                   'planned_start_at': (TEST_NOW + timedelta(hours=2)).isoformat()},
                      known_players=[], trace_id='seed')
    gateway = ToolGateway(store)
    handle = runtime.handle_user_message

    def handle_with_option(message, *, trace_id):
        result = handle(message, trace_id=trace_id)
        option = gateway.execute(ToolCall('search_current_games', {'requirement': {}}, '查询实际可选局'),
            trace_id=trace_id, conversation_id=message.conversation_id, sender_id=message.sender_id,
            sender_name=message.sender_name, step_index=1, source_message_id=message.message_id)
        return replace(result, tool_results=[option])

    runtime.handle_user_message = handle_with_option
    messenger = DeliveryMessenger()
    handler.messenger = handler.notify_dispatcher.messenger = messenger
    message = _group_message('帮我找个合适的局', 'friend', message_id='original-source')

    handler.handle(message, trace_id='first-attempt')
    handler.handle(message, trace_id='retry-with-new-trace')

    assert len(messenger.delivered) == 1
    assert {item['channel'] for item in messenger.delivered.values()} == {'private'}
    private = [item for item in messenger.attempts if item['channel'] == 'private']
    assert len(private) == 2
    assert private[0]['metadata']['draft_id'] == private[1]['metadata']['draft_id']
    assert private[0]['metadata']['trace_id'] != private[1]['metadata']['trace_id']
    assert private[0]['metadata']['draft_id'] != message.message_id
    assert all(item['metadata']['source_message_id'] == message.message_id for item in private)


@pytest.mark.parametrize('changed', ['channel', 'room', 'customer', 'message'])
def test_private_delivery_key_isolates_complete_source_scope(changed):
    store = InMemoryAgentStore()
    identity = _identity(store, 'friend', friend=True, public_name='群友')
    source = _group_message('帮我找个合适的局', 'friend', message_id='original-source')
    other_source, other_identity = source, identity
    if changed == 'channel':
        other_source = replace(source, channel='console')
    elif changed == 'room':
        other_source = replace(source, room_id='another-room')
    elif changed == 'customer':
        other_identity = replace(identity, customer_id='another-customer', external_user_id='another-contact')
    else:
        other_source = replace(source, message_id='another-source')
    messenger = DeliveryMessenger()
    dispatcher = NotifyDispatcher(store=store, messenger=messenger, runtime=object())
    result = AgentRuntimeResult(trace_id='trace', conversation_id=identity.private_conversation_id, final_reply='我帮你找找。')
    for current_identity, current_source in ((identity, source), (other_identity, other_source)):
        dispatcher.send_private_switch_reply(current_identity, result, trace_id='trace', source_message=current_source)
    assert len(messenger.delivered) == 2


def test_empty_private_reply_does_not_create_delivery_attempt():
    store = InMemoryAgentStore()
    identity = _identity(store, 'friend', friend=True, public_name='群友')
    messenger = DeliveryMessenger()
    dispatcher = NotifyDispatcher(store=store, messenger=messenger, runtime=object())
    result = AgentRuntimeResult(trace_id='trace', conversation_id=identity.private_conversation_id, final_reply='')
    assert dispatcher.send_private_switch_reply(identity, result, trace_id='trace',
        source_message=_group_message('好', 'friend', message_id='source')) == ''
    assert messenger.attempts == []
