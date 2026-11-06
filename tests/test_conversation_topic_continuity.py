"""真实运行内核验证话题插入不丢业务事实；语义选择另由目标模型评测覆盖。"""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from mahjong_agent_runtime import (
    AgentRuntime, CustomerProfile, InMemoryAgentStore, InMemoryTraceRecorder, QuotedMessageRef,
    SQLiteAgentStore, StaticAgentClient, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.task_context import TaskContextManager


@pytest.fixture(params=['memory', 'sqlite'])
def continuity_store(request, tmp_path):
    """每例独立真实持久层；不复制历史库或把摘要充当业务事实。"""
    store = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path / 'topics.sqlite3')
    store.upsert_customer(CustomerProfile(customer_id='self', display_name='本人',
        preferred_games=['hangzhou_mahjong'], preferred_stakes=['0.5'], smoke_preference='no_smoke'))
    yield store
    if request.param == 'sqlite':
        store._connection.close()


def response(reply, *, calls=()):
    """固定模型提案用于核验装配和执行，不把桩输出宣称为真实语义识别成绩。"""
    return json.dumps({'goal': '回应当前输入并保留此前业务目标',
        'objective_status': 'needs_tool' if calls else 'completed', 'reasoning_summary': '离线执行夹具',
        'objective_state': {}, 'tool_calls': list(calls), 'reply_to_user': '' if calls else reply,
        'needs_human': False, 'stop_reason': {'can_stop': not calls, 'why': '核验当前动作',
            'pending_work': ['执行当前变更'] if calls else [], 'depends_on_tool_results': bool(calls)}}, ensure_ascii=False)


def reviewed(reply):
    return json.dumps({'approved': True, 'needs_human': False, 'reasoning_summary': '离线审查夹具',
        'violations': [], 'item_reviews': [{'item_id': 'reply_to_user', 'approved': True,
            'suggested_safe_text': reply, 'reasoning_summary': '普通话题', 'violations': []}]}, ensure_ascii=False)


def seed_goal(store, *, create=True):
    """先由网关留下委托事实、待办和可选真实局，随后测试仅观察原记录是否保留。"""
    source = UserMessage(conversation_id='topic-continuity', sender_id='self', sender_name='本人',
        text='帮我找杭麻0.5无烟的，我一个人，人齐开，最多四小时。', message_id='delegation')
    task = TaskContextManager(store).prepare(source, trace_id='seed-goal').context
    store.append_user_turn(source, 'seed-goal')
    gateway = ToolGateway(store)
    operations = [('record_user_memory', {'task_memories': [{'customer_id': 'self', 'memory_type': 'constraint',
        'field': 'max_duration_hours', 'value': 4, 'evidence': '最多四小时', 'confidence': 0.99, 'scope': 'current_task'}]}),
        ('update_context_checkpoint', {'summary': '用户已委托寻找杭麻0.5无烟，人齐开，最多四小时。',
            'facts': {'delegated_matching': True, 'authorization_source_message_id': source.message_id,
                'game_type': 'hangzhou_mahjong', 'stake': '0.5', 'smoke_preference': 'no_smoke'},
            'open_questions': ['还需协调牌友']})]
    if create:
        operations.append(('create_game', {'organizer_id': 'self', 'organizer_name': '本人',
            'requirement': {'game_type': 'hangzhou_mahjong', 'stake': '0.5', 'smoke_preference': 'no_smoke',
                'start_time_kind': 'asap_when_full'}, 'known_players': []}))
    for index, (name, arguments) in enumerate(operations, 1):
        result = gateway.execute(ToolCall(name=name, arguments=arguments), trace_id='seed-goal',
            conversation_id=source.conversation_id, sender_id=source.sender_id, sender_name=source.sender_name,
            source_message_id=source.message_id, step_index=index)
        assert result.called and result.allowed and not result.error, result
    return source, task


def runtime_for(store, replies):
    client = StaticAgentClient([response(reply) for reply in replies])
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder(),
        reply_self_review_enabled=True, reply_self_review_client=StaticAgentClient([reviewed(reply) for reply in replies]))
    return runtime, client


def message(text, index, *, quoted=None):
    return UserMessage(conversation_id='topic-continuity', sender_id='self', sender_name='本人',
        text=text, message_id=f'topic-{index}', quoted_message=quoted)


def payload(client):
    return json.loads(client.calls[-1]['messages'][-1]['content'])


def test_life_work_technology_quote_then_status_keep_same_goal_and_authoritative_game(continuity_store):
    """T01/T02/T05/T07：当前回复完成不关闭原任务，技术词正常通过最终审查过滤。"""
    store = continuity_store
    source, task = seed_goal(store)
    initial_game = deepcopy(next(iter(store.games.values())).to_dict())
    initial_memories = deepcopy(store.task_memory_context(source.conversation_id, source.sender_id))
    replies = ['下雨去看个展也挺好。', '预算审批慢确实耽误事。',
               'AI 的效果得看具体任务，日志和 prompt 都能帮助排查。', '嗯，懂了。', '现在还差三个人。']
    runtime, client = runtime_for(store, replies)
    inputs = [message('周末又下雨了，出去玩不方便。', 1), message('今天工作一直等预算审批，草稿改了好几遍。', 2),
        message('说起来，AI 的效果怎么评估，日志和 prompt 有用吗？', 3),
        message('一致', 4, quoted=QuotedMessageRef(message_id='quote', sender_id='self', sender_name='本人',
            text='自然聊天的语气要和人一直')),
        message('刚才那个局现在几个人了？', 5)]
    for index, (incoming, reply) in enumerate(zip(inputs, replies), 1):
        result = runtime.handle_user_message(incoming, trace_id=f'topic-turn-{index}')
        assert result.final_reply == reply and result.status == 'completed'
        assert all(item.name == 'customer_visible_content_review' for item in result.tool_results)
        assert next(iter(store.games.values())).to_dict() == initial_game
        assert store.current_task_context(source.conversation_id, source.sender_id).task_context_id == task.task_context_id
        assert store.task_memory_context(source.conversation_id, source.sender_id) == initial_memories
        assert not store.invite_drafts and not store.outbound_message_drafts
        context = payload(client)
        assert context['current_message']['text'] == incoming.text
        assert context['task_context_window']['task_context_id'] == task.task_context_id
        assert context['conversation_checkpoint']['facts']['delegated_matching'] is True
        assert context['conversation_checkpoint']['open_questions'] == ['还需协调牌友']
        # checkpoint覆盖的早期轮次可不重复传输，但授权来源与已知条件必须留在摘要，原话仍可审计。
        assert context['conversation_checkpoint']['facts']['authorization_source_message_id'] == source.message_id
        assert any(turn.content == source.text for turn in store.recent_turns(source.conversation_id, 60))
    assert context['active_game_visible_summaries'][0]['seat_summary']['remaining_seats'] == 3


def test_pending_delegation_survives_unrelated_topic_before_missing_fact_is_answered(continuity_store):
    """T03：尚无Game也不因闲聊清空授权来源、已知条件和未完成待办。"""
    store = continuity_store
    source, task = seed_goal(store, create=False)
    runtime, client = runtime_for(store, ['听起来挺有意思。', '好。'])
    runtime.handle_user_message(message('今天看了一个天文展，挺震撼。', 1), trace_id='new-topic')
    runtime.handle_user_message(message('补一下，我今天七点以后有空。', 2), trace_id='supply-fact')
    context = payload(client)
    assert context['task_context_window']['task_context_id'] == task.task_context_id
    assert context['conversation_checkpoint']['facts']['authorization_source_message_id'] == source.message_id
    assert context['conversation_checkpoint']['open_questions'] == ['还需协调牌友']
    assert '已委托' in context['conversation_checkpoint']['summary']
    assert any(turn.content == source.text for turn in store.recent_turns(source.conversation_id, 60))
    assert any(item['field'] == 'max_duration_hours' and item['value'] == 4 for item in context['task_memories'])
    assert not store.games


def test_mixed_topic_does_not_prevent_explicit_cancel_tool_execution(continuity_store):
    """T04：后端不以闲聊标签封锁真实业务工具；语义是否选择退出由目标模型回放验证。"""
    store = continuity_store
    source, _ = seed_goal(store)
    game = next(iter(store.games.values()))
    reply = '好，这次先取消。'
    client = StaticAgentClient([response('', calls=[{'name': 'update_game_status',
        'arguments': {'game_id': game.game_id, 'status': 'cancelled', 'reason': '本人本次明确取消'},
        'reason': '处理混合话题中的真实取消'}]), response(reply)])
    runtime = AgentRuntime(llm_client=client, store=store, reply_self_review_enabled=True,
        reply_self_review_client=StaticAgentClient([reviewed(reply)]))
    result = runtime.handle_user_message(message('刚聊的那个电影我也想看，不过我今晚加班，刚才的局先取消。', 1),
        trace_id='mixed-cancel')
    assert result.status == 'completed' and result.final_reply == reply
    assert store.games[game.game_id].status.value == 'cancelled'
    assert len(store.games) == 1 and not store.invite_drafts and not store.outbound_message_drafts
    assert any(item.name == 'update_game_status' and item.called and item.allowed and not item.error
               for item in result.tool_results)
