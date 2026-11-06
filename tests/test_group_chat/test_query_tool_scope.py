from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolGateway
from mahjong_agent_runtime.models import ToolCall
import pytest
from datetime import datetime

from mahjong_agent_runtime.models import UserMessage
from mahjong_agent_runtime.group_chat.models import ChannelIdentity
from mahjong_agent_runtime.group_chat.tool_scope import tool_execution_scope
from mahjong_agent_runtime.task_context import TaskContextManager


@pytest.mark.parametrize('backend', ['memory','sqlite'])
def test_query_scope_blocks_create_before_claim_or_state_write(tmp_path,backend):
    store=InMemoryAgentStore() if backend=='memory' else SQLiteAgentStore(tmp_path/'scope.db')
    gateway=ToolGateway(store)
    call=ToolCall(name='create_game',arguments={'organizer_id':'a','organizer_name':'A','requirement':{'requested_game':'hangzhou_mahjong','stake':'0.5','known_player_count':2},'known_players':[]})
    result=gateway.execute(call,trace_id='query',conversation_id='c',sender_id='a',sender_name='A',step_index=1,execution_scope='query_only')
    assert not result.called and not result.allowed
    assert '只读询局' in result.error
    assert not store.games
    assert not store.invite_drafts
    assert store.idempotent_result(result.idempotency_key) is None
    # 后续用户明确委托时新一轮按原网关权限执行；不污染全局permission或缓存拒绝。
    allowed=gateway.execute(call,trace_id='delegate',conversation_id='c',sender_id='a',sender_name='A',step_index=1)
    assert allowed.called and allowed.allowed
    assert len(store.games)==1


@pytest.mark.parametrize('backend', ['memory','sqlite'])
def test_query_scope_preserves_read_tools(tmp_path,backend):
    store=InMemoryAgentStore() if backend=='memory' else SQLiteAgentStore(tmp_path/'scope.db')
    gateway=ToolGateway(store)
    result=gateway.execute(ToolCall(name='search_current_games',arguments={'requirement':{}}),trace_id='query',conversation_id='c',
        sender_id='a',sender_name='A',step_index=1,execution_scope='query_only')
    assert result.called and result.allowed
    assert not store.games


@pytest.mark.parametrize('backend',['memory','sqlite'])
@pytest.mark.parametrize('intent,topic_action,goal_relation',[
    ('update','uncertain','uncertain'),('cancel','uncertain','uncertain'),
    ('update','none','new'),('delegate','new','uncertain'),
])
def test_unresolved_target_is_read_only_and_never_falls_into_previous_task(tmp_path,backend,intent,topic_action,goal_relation):
    store=InMemoryAgentStore() if backend=='memory' else SQLiteAgentStore(tmp_path/'unresolved.db')
    store.upsert_channel_identity(ChannelIdentity(channel='console',external_user_id='ea',customer_id='a',
        public_name='A',private_conversation_id='private-a'))
    stamp=datetime.fromisoformat('2026-11-05T11:00:00+08:00')
    conversation='group:room:customer:a'
    previous,_=store.activate_task_context(conversation_id=conversation,customer_id='a',trace_id='before',
        activity_at=stamp,started_at=stamp,reason='first',force_new=True,archive_previous=False)
    store.record_group_topic_message(channel='console',room_id='room',customer_id='a',message_id='m',
        text='改成八点',sent_at=stamp,intent=intent,public_summary={},create_topic=False,
        classification={'intent':intent,'topic_action':topic_action,'goal_relation':goal_relation,'task_context_id':None})
    message=UserMessage(conversation_id=conversation,sender_id='a',sender_name='A',text='改成八点',message_id='m',
        sent_at=stamp,metadata={'group_topic_source':{'channel':'console','room_id':'room','external_user_id':'ea'}})
    assert tool_execution_scope(store,message)=='query_only'
    prepared=TaskContextManager(store).prepare(message,trace_id='clarify')
    assert prepared.context.task_context_id!=previous.task_context_id
    assert prepared.reason=='unresolved_group_topic_target'
    call=ToolCall(name='create_game',arguments={'organizer_id':'a','organizer_name':'A',
        'requirement':{'requested_game':'hangzhou_mahjong','known_player_count':2},'known_players':[]})
    blocked=ToolGateway(store).execute(call,trace_id='blocked',conversation_id=conversation,sender_id='a',sender_name='A',
        step_index=1,execution_scope=tool_execution_scope(store,message))
    assert not blocked.called and not store.games


@pytest.mark.parametrize('backend',['memory','sqlite'])
def test_verified_own_continuation_is_not_overblocked(tmp_path,backend):
    store=InMemoryAgentStore() if backend=='memory' else SQLiteAgentStore(tmp_path/'continue.db')
    store.upsert_channel_identity(ChannelIdentity(channel='console',external_user_id='ea',customer_id='a',
        public_name='A',private_conversation_id='private-a'))
    stamp=datetime.fromisoformat('2026-11-05T11:00:00+08:00')
    conversation='group:room:customer:a'
    task,_=store.activate_task_context(conversation_id=conversation,customer_id='a',trace_id='before',
        activity_at=stamp,started_at=stamp,reason='first',force_new=True,archive_previous=False)
    first=store.record_group_topic_message(channel='console',room_id='room',customer_id='a',message_id='original',
        text='帮我找今晚的',sent_at=stamp,intent='delegate',public_summary={})
    topic=first['topic']['topic_id']
    store.bind_group_topic_customer(topic_id=topic,customer_id='a',conversation_id=conversation,
        task_context_id=task.task_context_id,source_message_id='original')
    store.record_group_topic_message(channel='console',room_id='room',customer_id='a',message_id='m',text='改成八点',
        sent_at=stamp,intent='update',topic_id=topic,expected_version=1,public_summary={},
        classification={'intent':'update','topic_action':'existing','goal_relation':'continue','task_context_id':task.task_context_id})
    message=UserMessage(conversation_id=conversation,sender_id='a',sender_name='A',text='改成八点',message_id='m',
        sent_at=stamp,metadata={'group_topic_source':{'channel':'console','room_id':'room','external_user_id':'ea'}})
    assert tool_execution_scope(store,message) is None
    assert TaskContextManager(store).prepare(message,trace_id='continue').context.task_context_id==task.task_context_id


@pytest.mark.parametrize('backend',['memory','sqlite'])
def test_unresolved_no_topic_receive_execute_and_replay_keep_one_task(tmp_path,backend,monkeypatch):
    import json
    from mahjong_agent_runtime import AgentAction, AgentRuntime, StaticAgentClient
    store=InMemoryAgentStore() if backend=='memory' else SQLiteAgentStore(tmp_path/'receipt.db')
    store.upsert_channel_identity(ChannelIdentity(channel='console',external_user_id='ea',customer_id='a',
        public_name='A',private_conversation_id='private-a'))
    stamp=datetime.fromisoformat('2026-11-05T11:00:00+08:00')
    conversation='group:room:customer:a'
    store.record_group_topic_message(channel='console',room_id='room',customer_id='a',message_id='m',
        text='改成八点',sent_at=stamp,intent='update',public_summary={},create_topic=False,
        classification={'intent':'update','topic_action':'uncertain','goal_relation':'uncertain','task_context_id':None})
    message=UserMessage(conversation_id=conversation,sender_id='a',sender_name='A',text='改成八点',message_id='m',
        sent_at=stamp,metadata={'group_topic_source':{'channel':'console','room_id':'room','external_user_id':'ea'}})
    final=AgentAction(goal='澄清修改目标',objective_status='waiting_user',reasoning_summary='无法确定是哪一个本人需求',
        reply_to_user='你是要改哪一场？',stop_reason={'can_stop':True,'why':'需要客户说明目标',
        'pending_work':['客户说明目标'],'depends_on_tool_results':False})
    client=StaticAgentClient([json.dumps(final.to_dict(),ensure_ascii=False)])
    prepared_ids=[]
    original_prepare=TaskContextManager.prepare
    def counted_prepare(self,*args,**kwargs):
        prepared=original_prepare(self,*args,**kwargs)
        prepared_ids.append(prepared.context.task_context_id)
        return prepared
    monkeypatch.setattr(TaskContextManager,'prepare',counted_prepare)
    runtime=AgentRuntime(llm_client=client,store=store)
    receipt=runtime.receive_user_message(message,trace_id='receive')
    task_id=receipt['task_preparation']['task_context']['task_context_id']
    handled=runtime.handle_user_message(message,trace_id='execute')
    replay=runtime.handle_user_message(message,trace_id='replay')
    assert handled.status==replay.status=='waiting_user'
    assert prepared_ids==[task_id] and len(store.task_contexts)==1
    assert store.current_task_context(conversation,'a').task_context_id==task_id
    assert len(client.calls)==1


def test_query_scope_records_standard_permission_rejection_before_idempotency_claim():
    from mahjong_agent_runtime.tracing import InMemoryTraceRecorder
    store=InMemoryAgentStore();recorder=InMemoryTraceRecorder();gateway=ToolGateway(store,trace_recorder=recorder)
    call=ToolCall(name='create_game',arguments={'organizer_id':'a','organizer_name':'A','requirement':{}})
    result=gateway.execute(call,trace_id='scope-audit',conversation_id='c',sender_id='a',sender_name='A',step_index=1,execution_scope='query_only')
    events=recorder.get_trace('scope-audit')
    assert not result.called and not result.allowed
    assert any(e.step=='tool_permission_checked' and e.content['allowed'] is False for e in events)
    assert not any(e.step=='tool_idempotency_claimed' for e in events)
    assert not store.games
