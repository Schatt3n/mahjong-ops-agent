from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.group_chat.models import ChannelIdentity
from mahjong_agent_runtime.group_chat.topic_context import topic_context_projection, verified_topic_message
from mahjong_agent_runtime.models import UserMessage
from mahjong_agent_runtime.runtime import AgentRuntime
from mahjong_agent_runtime.task_context import TaskContextManager

STAMP = datetime.fromisoformat('2026-11-05T11:00:00+08:00')
GROUP = 'group:room:customer:a'


@pytest.fixture(params=['memory','sqlite'])
def store(request,tmp_path):
    value=InMemoryAgentStore() if request.param=='memory' else SQLiteAgentStore(tmp_path/'context.sqlite3')
    value.upsert_channel_identity(ChannelIdentity(channel='wechaty',external_user_id='external-a',customer_id='a',
        public_name='甲',private_conversation_id='private:a',can_private_message=True,is_friend=True))
    yield value
    if hasattr(value,'_connection'):value._connection.close()


def source(store, number, *, topic_id=None, selected=None, new=True, intent='query', constraints=None, conversation=GROUP, stamp=None):
    when=stamp or STAMP+timedelta(seconds=number)
    message=UserMessage(conversation_id=conversation,sender_id='a',sender_name='甲',text=f'本人消息{number}',
        message_id=f'm{number}',sent_at=when,metadata={'group_topic_source':{
            'channel':'wechaty','room_id':'room','external_user_id':'external-a'}})
    prior=store.get_group_topic(topic_id) if topic_id else None
    saved=store.record_group_topic_message(channel='wechaty',room_id='room',customer_id='a',message_id=message.message_id,
        text=message.text,sent_at=when,intent=intent,topic_id=topic_id,expected_version=prior['version'] if prior else None,
        public_summary={},classification={'intent':intent,'goal_relation':'new' if new else 'continue',
        'task_context_id':selected,'constraints':constraints or {}},trace_id=f't{number}',idle_seconds=10)
    return message,saved['topic']['topic_id']


def test_repeat_prepare_and_replaying_old_input_keep_original_task(store):
    manager=TaskContextManager(store)
    original,topic=source(store,1)
    first=manager.prepare(original,trace_id='t1').context
    assert manager.prepare(original,trace_id='replay').context.task_context_id==first.task_context_id
    following,_=source(store,2,topic_id=topic,selected=first.task_context_id,new=False)
    assert manager.prepare(following,trace_id='t2').context.task_context_id==first.task_context_id
    assert manager.prepare(original,trace_id='replay-old').context.task_context_id==first.task_context_id
    assert len(store.task_contexts)==1


def test_different_topics_resume_original_id_and_keep_checkpoint_memory(store):
    manager=TaskContextManager(store)
    original,topic=source(store,1,constraints={'game_type':'hangzhou_mahjong'})
    first=manager.prepare(original,trace_id='t1').context
    store.upsert_conversation_checkpoint(conversation_id=GROUP,summary='本人杭麻需求',facts={'stake':'0.5'},
        open_questions=[],trace_id='c1',task_context_id=first.task_context_id)
    memory,_=store.record_task_memory(conversation_id=GROUP,customer_id='a',memory_type='requirement',field='stake',value='0.5',trace_id='mem1')
    different,_=source(store,2,constraints={'game_type':'sichuan_mahjong'})
    second=manager.prepare(different,trace_id='t2').context
    later,_=source(store,3,topic_id=topic,selected=first.task_context_id,new=False)
    restored=manager.prepare(later,trace_id='t3').context
    assert restored.task_context_id==first.task_context_id!=second.task_context_id
    assert store.get_task_context_checkpoint(restored.task_context_id).summary=='本人杭麻需求'
    assert store.task_memories[memory.memory_id].status=='active'
    assert store.get_task_context(second.task_context_id).status=='closed'


def test_same_topic_multiple_personal_goals_do_not_mix_conditions(store):
    manager=TaskContextManager(store)
    original,topic=source(store,1,constraints={'time_window':{'start':'19:30'}})
    first=manager.prepare(original,trace_id='t1').context
    separate,_=source(store,2,topic_id=topic,constraints={'time_window':{'start':'22:00'}})
    second=manager.prepare(separate,trace_id='t2').context
    later,_=source(store,3,topic_id=topic,new=False,selected=first.task_context_id,constraints={'party_size':2})
    manager.prepare(later,trace_id='t3')
    projected=topic_context_projection(store,later)
    assert [row['source_message_id'] for row in projected['own_condition_sources']]==['m1','m3']
    assert first.task_context_id!=second.task_context_id


def test_forged_identity_conversation_text_time_metadata_cannot_restore_topic(store):
    original,topic=source(store,1)
    task=TaskContextManager(store).prepare(original,trace_id='t1').context
    for field,value in [('sender_id','b'),('conversation_id','private:foreign'),('text','篡改原文'),
                        ('sent_at',STAMP+timedelta(days=1))]:
        forged=deepcopy(original);setattr(forged,field,value)
        assert verified_topic_message(store,forged) is None
    forged=deepcopy(original);forged.metadata['group_topic_source']['room_id']='another'
    assert verified_topic_message(store,forged) is None
    forged=deepcopy(original);forged.metadata['task_context_id']='another-task'
    assert TaskContextManager(store).prepare(forged,trace_id='safe').context.task_context_id==task.task_context_id
    assert store.get_group_topic(topic)['version']==1


def test_initial_query_cannot_forge_private_route_but_delegation_preserves_own_conditions(store):
    manager=TaskContextManager(store)
    original,topic=source(store,1,constraints={'game_type':'hangzhou_mahjong','party_size':2})
    group_task=manager.prepare(original,trace_id='t1').context
    forged=deepcopy(original);forged.conversation_id='private:a'
    assert verified_topic_message(store,forged) is None
    delegated,_=source(store,2,topic_id=topic,new=False,selected=group_task.task_context_id,
        intent='delegate',conversation='private:a')
    private_task=manager.prepare(delegated,trace_id='t2').context
    assert private_task.task_context_id!=group_task.task_context_id
    assert private_task.conversation_id=='private:a'
    assert topic_context_projection(store,delegated)['own_condition_sources'][0]['constraints']['party_size']==2
    later,_=source(store,3,topic_id=topic,new=False,selected=private_task.task_context_id,
        intent='update',conversation='private:a',constraints={'latest_wait_at':'2026-11-05T20:00:00+08:00'})
    assert manager.prepare(later,trace_id='t3').context.task_context_id==private_task.task_context_id
    assert [row['source_message_id'] for row in topic_context_projection(store,later)['own_condition_sources']]==['m1','m2','m3']


def test_timezone_equivalent_original_is_verified_and_receipt_is_once(store):
    message,_=source(store,1,stamp=STAMP.astimezone(timezone.utc))
    assert verified_topic_message(store,message) is not None
    runtime=AgentRuntime(llm_client=object(),store=store)
    first=runtime.receive_user_message(message,trace_id='receipt-first')
    second=runtime.receive_user_message(message,trace_id='receipt-replay')
    assert first['accepted'] is True and second['accepted'] is False
    assert first['conversation_version']==second['conversation_version']==1
    assert len(store.task_contexts)==1
    assert store.recent_turns(GROUP)[0].content==message.text


def test_cold_topic_recovery_never_cancels_game_or_archives_memory(store):
    manager=TaskContextManager(store,idle_reset_seconds=5)
    message,topic=source(store,1)
    context=manager.prepare(message,trace_id='first').context
    memory,_=store.record_task_memory(conversation_id=GROUP,customer_id='a',memory_type='requirement',field='stake',value='0.5',trace_id='m')
    game,_=store.create_game(conversation_id=GROUP,organizer_id='a',organizer_name='甲',
        requirement={'game_type':'hangzhou_mahjong','stake':'0.5','start_time':'19:30'},known_players=[],trace_id='g')
    assert not store.list_group_topics(room_id='room',at=STAMP+timedelta(hours=6))
    continuation,_=source(store,2,topic_id=topic,selected=context.task_context_id,new=False,stamp=STAMP+timedelta(hours=6))
    resumed=manager.prepare(continuation,trace_id='cold')
    assert resumed.context.task_context_id==context.task_context_id
    assert store.games[game.game_id].status==game.status
    assert store.task_memories[memory.memory_id].status=='active'


def test_private_context_projection_filters_unknown_and_foreign_conditions(store):
    original,topic=source(store,1,constraints={'party_size':2,'private_remark':'不可暴露的原文','participants':['他人']})
    TaskContextManager(store).prepare(original,trace_id='t1')
    store.record_group_topic_message(channel='wechaty',room_id='room',customer_id='b',message_id='other',
        text='其他人原文',sent_at=STAMP+timedelta(seconds=2),intent='query',topic_id=topic,expected_version=1,
        classification={'intent':'query','constraints':{'party_size':4}},public_summary={})
    projected=topic_context_projection(store,original)
    assert projected['own_condition_sources'][0]['constraints']=={'party_size':2}
    assert '其他人原文' not in str(projected) and '不可暴露' not in str(projected)
