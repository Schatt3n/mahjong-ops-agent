from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import sqlite3

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore

STAMP = datetime.fromisoformat('2026-11-05T11:00:00+08:00')


@pytest.fixture(params=['memory', 'sqlite'])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path / 'topic.sqlite3')
    yield value
    if hasattr(value, '_connection'):
        value._connection.close()


def record(store, **overrides):
    args = dict(channel='console', room_id='room', customer_id='a', message_id='m1',
                text='明晚七点半杭麻有人吗', sent_at=STAMP, intent='query',
                public_summary={'game_type':'hangzhou_mahjong','stakes':['0.5'], 'date':'2026-11-06',
                                'time_window':{'start':'19:30','end':None},'smoking':'no_smoking'},
                classification={'intent':'query'}, trace_id='trace-1', idle_seconds=60)
    args.update(overrides)
    return store.record_group_topic_message(**args)


def test_topic_replay_keeps_original_decision_and_activity(store):
    first = record(store)
    replay = record(store, intent='delegate', classification={'intent':'delegate'})
    assert replay['deduplicated'] is True
    assert replay['message'] == first['message']
    assert replay['topic']['version'] == 1
    assert replay['topic']['active_until'] == (STAMP+timedelta(seconds=60)).isoformat()
    assert not store.games


def test_same_message_id_different_original_is_rejected_without_mutation(store):
    first=record(store)
    with pytest.raises(ValueError, match='message.*conflict'):
        record(store, text='改成明天下午')
    assert store.get_group_topic(first['topic']['topic_id'])['version']==1


def test_cold_topic_is_recoverable_and_does_not_delete_source_or_binding(store):
    first=record(store);topic=first['topic']['topic_id']
    store.bind_group_topic_customer(topic_id=topic, customer_id='a',conversation_id='private:a')
    assert not store.list_group_topics(room_id='room',at=STAMP+timedelta(minutes=2))
    recovered=store.list_group_topics(room_id='room',customer_id='a',at=STAMP+timedelta(minutes=2),include_cold=True)
    assert recovered[0]['topic_id']==topic
    assert store.get_group_topic_message('console','room','a','m1')['text']=='明晚七点半杭麻有人吗'
    assert store.get_group_topic_customer(topic,'a')['conversation_id']=='private:a'
    assert not store.list_group_topics(room_id='room',customer_id='b',include_cold=True,at=STAMP)


def test_foreign_room_and_stale_version_cannot_update_topic(store):
    first=record(store);topic=first['topic']['topic_id']
    with pytest.raises(ValueError,match='room'):
        record(store,room_id='other',message_id='m2',topic_id=topic,expected_version=1)
    with pytest.raises(ValueError,match='version'):
        record(store,message_id='m2',topic_id=topic,expected_version=0)
    assert store.get_group_topic_message('console','room','a','m2') is None
    assert store.get_group_topic(topic)['version']==1


def test_each_customer_retains_own_source_without_overwriting_public_anchor(store):
    first=record(store);topic=first['topic']['topic_id']
    second=record(store,customer_id='b',message_id='m2',text='我只能八点',sent_at=STAMP+timedelta(seconds=5),
                  topic_id=topic,expected_version=1,public_summary={'time_window':{'start':'20:00','end':None}})
    assert second['topic']['public_summary']['time_window']['start']=='19:30'
    assert store.list_group_topic_customer_messages(topic,'b')[0]['text']=='我只能八点'
    assert [m['text'] for m in store.list_group_topic_customer_messages(topic,'a')]==['明晚七点半杭麻有人吗']
    assert second['topic']['version']==2


def test_out_of_order_source_cannot_move_activity_backwards(store):
    first=record(store);topic=first['topic']['topic_id']
    second=record(store,message_id='m2',topic_id=topic,expected_version=1,sent_at=STAMP-timedelta(minutes=5))
    assert second['topic']['last_activity_at']==STAMP.isoformat()
    assert second['topic']['active_until']==first['topic']['active_until']


def test_public_summary_rejects_identity_or_arbitrary_text(store):
    for summary in ({'name':'秘密名字'},{'game_type':'秘密名字'},{'date':'明晚'},
                    {'time_window':{'start':'客户A今天请假'}},{'stakes':['手机号123']}, {'smoking':'私人备注'}):
        with pytest.raises(ValueError):record(store,public_summary=summary)
    assert store.list_group_topics(room_id='room',at=STAMP)==[]


def test_irrelevant_message_has_durable_decision_without_topic(store):
    result=record(store,intent='chitchat',public_summary={},create_topic=False)
    assert result['topic'] is None
    assert result['message']['intent']=='chitchat'
    assert not store.list_group_topics(room_id='room',at=STAMP)


def test_same_source_concurrent_creation_is_idempotent(store):
    with ThreadPoolExecutor(max_workers=4) as executor:results=list(executor.map(lambda _: record(store),range(4)))
    assert len({x['topic']['topic_id'] for x in results})==1
    assert sum(not x['deduplicated'] for x in results)==1


def test_task_binding_is_validated_and_preserves_history(store):
    topic=record(store)['topic']['topic_id']
    context,_=store.activate_task_context(conversation_id='private:a',customer_id='a',trace_id='t',activity_at=STAMP,reason='new',force_new=True,started_at=STAMP,archive_previous=False)
    store.bind_group_topic_customer(topic_id=topic,customer_id='a',conversation_id='private:a',task_context_id=context.task_context_id)
    other,_=store.activate_task_context(conversation_id='private:a',customer_id='a',trace_id='t2',activity_at=STAMP+timedelta(seconds=1),reason='new',force_new=True,started_at=STAMP,archive_previous=False)
    store.bind_group_topic_customer(topic_id=topic,customer_id='a',conversation_id='private:a',task_context_id=other.task_context_id)
    assert len(store.list_group_topic_customer_bindings(topic,'a'))==2
    with pytest.raises(ValueError,match='owner'):
        store.bind_group_topic_customer(topic_id=topic,customer_id='b',conversation_id='private:a',task_context_id=other.task_context_id)
    with pytest.raises(ValueError,match='conversation'):
        store.bind_group_topic_customer(topic_id=topic,customer_id='a',conversation_id='other')


def test_sqlite_topic_reopens_and_conflicting_writers_keep_one_version(tmp_path):
    path=tmp_path/'reopen.sqlite3';left=SQLiteAgentStore(path);first=record(left);right=SQLiteAgentStore(path)
    topic=first['topic']['topic_id']
    def write(pair):
        target,msg=pair
        try:return record(target,message_id=msg,topic_id=topic,expected_version=1)
        except ValueError as exc:return str(exc)
    with ThreadPoolExecutor(max_workers=2) as executor:results=list(executor.map(write,[(left,'m2'),(right,'m3')]))
    assert sum(isinstance(item,dict) for item in results)==1
    left._connection.close();right._connection.close();reopened=SQLiteAgentStore(path)
    assert reopened.get_group_topic(topic)['version']==2
    reopened._connection.close()


def test_schema3_rejected_without_automatic_topic_migration(tmp_path):
    path=tmp_path/'old.sqlite3'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE original_evidence(value TEXT)');connection.execute('PRAGMA user_version=3')
    with pytest.raises(RuntimeError,match='schema version 3'):
        SQLiteAgentStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='runtime_group_topics'").fetchone() is None


def test_resume_personal_task_keeps_identity_checkpoint_and_memory(store):
    args=dict(conversation_id='private:a',customer_id='a',trace_id='t1',activity_at=STAMP,started_at=STAMP,
              reason='new',force_new=True,archive_previous=False)
    first,_=store.activate_task_context(**args)
    store.upsert_conversation_checkpoint(conversation_id='private:a',summary='只查杭麻',facts={'requested_game':'hangzhou_mahjong'},
                                        open_questions=['有没有局'],trace_id='t1',task_context_id=first.task_context_id)
    memory,_=store.record_task_memory(conversation_id='private:a',customer_id='a',memory_type='requirement',field='stake',value='0.5',trace_id='mem')
    second,_=store.activate_task_context(**{**args,'trace_id':'t2','activity_at':STAMP+timedelta(seconds=1)})
    old_checkpoint=store.get_task_context_checkpoint(first.task_context_id).to_dict()
    restored,transitions=store.resume_task_context(task_context_id=first.task_context_id,conversation_id='private:a',customer_id='a',
                                                 trace_id='t3',activity_at=STAMP+timedelta(seconds=2))
    assert restored.task_context_id==first.task_context_id and restored.closed_at is None
    assert store.current_task_context('private:a','a').task_context_id==first.task_context_id
    assert store.get_task_context(second.task_context_id).status=='closed'
    assert store.get_task_context_checkpoint(first.task_context_id).to_dict()==old_checkpoint
    assert store.task_memories[memory.memory_id].status=='active'
    assert len(transitions)==2
    again,replayed=store.resume_task_context(task_context_id=first.task_context_id,conversation_id='private:a',customer_id='a',
                                           trace_id='t3',activity_at=STAMP+timedelta(seconds=2))
    assert again.task_context_id==first.task_context_id and replayed==[]


def test_resume_task_rejects_cross_owner_and_conversation_without_closing_active(store):
    args=dict(conversation_id='private:a',customer_id='a',trace_id='t1',activity_at=STAMP,started_at=STAMP,
              reason='new',force_new=True,archive_previous=False)
    first,_=store.activate_task_context(**args)
    for changes in ({'customer_id':'b'},{'conversation_id':'private:other'},{'task_context_id':'missing'}):
        with pytest.raises(ValueError):
            store.resume_task_context(**{k:v for k,v in {**args,'task_context_id':first.task_context_id,**changes}.items()
                                         if k in {'conversation_id','customer_id','trace_id','activity_at','task_context_id'}})
    assert store.current_task_context('private:a','a').task_context_id==first.task_context_id


def test_public_reference_resolves_cold_topic_but_never_ambiguous_message_id(store):
    first=record(store)['topic']
    assert store.get_group_topic_for_message_reference('console','room','m1') == first
    assert store.get_group_topic_for_message_reference('console','other','m1') is None
    record(store,customer_id='b',text='另一个川麻讨论',public_summary={'game_type':'sichuan_mahjong'})
    assert store.get_group_topic_for_message_reference('console','room','m1') is None


def test_integer_version_contract_rejects_bool_and_float_without_source_write(store):
    first=record(store)['topic']
    for invalid in (True, 1.0):
        with pytest.raises(ValueError,match='version'):
            record(store,message_id='m2',topic_id=first['topic_id'],expected_version=invalid)
    assert store.get_group_topic_message('console','room','a','m2') is None
