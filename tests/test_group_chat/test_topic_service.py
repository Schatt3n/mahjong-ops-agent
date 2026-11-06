"""话题编排回归：事实回源、本人上下文隔离、幂等和版本竞争。"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.group_chat.models import BoardSnapshot, BoardSnapshotItem, ChannelIdentity, GroupMessage
from mahjong_agent_runtime.group_chat.topic_service import TopicRoutingService

STAMP = datetime.fromisoformat('2026-11-05T11:00:00+08:00')


class Classifier:
    max_candidates = 5

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def classify(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        result = self.results.pop(0)
        return result(kwargs) if callable(result) else deepcopy(result)


@pytest.fixture(params=['memory','sqlite'])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path/'service.sqlite3')
    value.upsert_channel_identity(ChannelIdentity(channel='console',external_user_id='external-a',customer_id='a',
        public_name='甲',private_conversation_id='private-a',can_private_message=True))
    yield value
    if hasattr(value,'_connection'):
        value._connection.close()


def msg(text='帮我找明晚杭麻', message_id='current', **kwargs):
    return GroupMessage(room_id='room',conversation_id='group-room',sender_external_id='external-a',
        sender_name='甲',text=text,message_id=message_id,channel='console',sent_at=kwargs.pop('sent_at',STAMP),**kwargs)


def result(intent='delegate', **kwargs):
    return {'intent':intent,'topic_action':'new','topic_id':None,'goal_relation':'new','task_context_id':None,
        'public_summary':{'game_type':'hangzhou_mahjong'},'constraints':{},'source_message_id':'current',
        'confidence':0.9,'reasoning':'按本人授权判断。','audit':{},**kwargs}


def seed(store, *, customer_id='a', message_id='past', at=STAMP, **kwargs):
    return store.record_group_topic_message(channel='console',room_id='room',customer_id=customer_id,
        message_id=message_id,text=f'{customer_id}自己的消息',sent_at=at,intent='delegate',
        public_summary={'game_type':'hangzhou_mahjong'},classification=result(),idle_seconds=60,**kwargs)


def bind_task(store, topic, conversation, source='past'):
    task,_ = store.activate_task_context(conversation_id=conversation,customer_id='a',trace_id='seed',
        activity_at=STAMP,started_at=STAMP,reason='first_message',force_new=True,archive_previous=False)
    store.bind_group_topic_customer(topic_id=topic,customer_id='a',conversation_id=conversation,
        task_context_id=task.task_context_id,source_message_id=source,allow_conversation_change=True,at=STAMP)
    return task


def test_new_demand_is_only_topic_record_not_game_and_replay_does_not_reclassify(store):
    classifier=Classifier(result())
    service=TopicRoutingService(store,classifier)
    first=service.classify(msg(),trace_id='trace')
    replay=service.classify(msg(),trace_id='different-trace')
    assert first['message']['customer_id']=='a'
    assert first['message']['classification']['intent']=='delegate'
    assert replay['deduplicated'] and replay['message']==first['message']
    assert len(classifier.calls)==1 and not store.games
    with pytest.raises(ValueError,match='conflict'):
        service.classify(msg('换了原文'),trace_id='trace')
    assert len(classifier.calls)==1


@pytest.mark.parametrize('intent',['chitchat','uncertain'])
def test_non_business_or_unresolved_message_keeps_evidence_without_creating_topic(store,intent):
    classifier=Classifier(result(intent,topic_action='none' if intent=='chitchat' else 'uncertain',
                                 goal_relation='none' if intent=='chitchat' else 'uncertain',public_summary={}))
    receipt=TopicRoutingService(store,classifier).classify(msg('这是什么意思'),trace_id='trace')
    assert receipt['topic'] is None and receipt['message']['intent']==intent
    assert store.list_group_topics(room_id='room',at=STAMP)==[]


def test_own_cold_topic_survives_other_peoples_active_topics_without_their_raw_messages(store):
    own=seed(store,at=STAMP-timedelta(hours=2))['topic']['topic_id']
    task=bind_task(store,own,'private-a')
    for i in range(7):seed(store,customer_id='other',message_id=f'other-{i}',at=STAMP+timedelta(seconds=i))
    classifier=Classifier(result('update',topic_action='existing',topic_id=own,goal_relation='continue',task_context_id=task.task_context_id))
    TopicRoutingService(store,classifier).classify(msg('改成八点'),trace_id='trace')
    call=classifier.calls[0]
    assert len(call['candidates'])<=5
    assert own in [row['topic_id'] for row in call['candidates']]
    assert all(row['customer_id']=='a' for row in call['own_context']['recent_messages'])
    assert 'other自己的消息' not in str(call)
    assert call['candidates'][0]['own_task_context_id']==task.task_context_id


def test_same_topic_multiple_personal_goals_are_present_without_implicitly_picking_latest(store):
    topic=seed(store)['topic']['topic_id']
    first=bind_task(store,topic,'private-a-first')
    second=bind_task(store,topic,'private-a-second')
    classifier=Classifier(result('update',topic_action='existing',topic_id=topic,goal_relation='continue',task_context_id=first.task_context_id))
    TopicRoutingService(store,classifier).classify(msg('前一个改成八点'),trace_id='trace')
    call=classifier.calls[0]
    assert {row['task_context_id'] for row in call['own_context']['tasks']}=={first.task_context_id,second.task_context_id}
    assert 'own_task_context_id' not in call['candidates'][0]


def test_cas_conflict_reload_and_reclassification_happens_once(store):
    topic=seed(store)['topic']['topic_id']
    def race(kwargs):
        seed(store,customer_id='other',message_id='racing',topic_id=topic,expected_version=1)
        return result(topic_action='existing',topic_id=topic)
    classifier=Classifier(race,result(topic_action='existing',topic_id=topic))
    receipt=TopicRoutingService(store,classifier).classify(msg(),trace_id='trace')
    assert len(classifier.calls)==2
    assert classifier.calls[0]['candidates'][0]['version']==1
    assert classifier.calls[1]['candidates'][0]['version']==2
    assert receipt['topic']['version']==3
    assert len(receipt['message']['classification']['routing_audit']['cas_conflicts'])==1


def test_cache_failure_falls_back_to_sqlite_fact_and_does_not_block_record(store):
    class BrokenCache:
        def list_candidates(self,*args,**kwargs):raise ConnectionError('redis down')
        def put(self,*args,**kwargs):raise ConnectionError('redis down')
    topic=seed(store)['topic']['topic_id']
    classifier=Classifier(result(topic_action='existing',topic_id=topic))
    receipt=TopicRoutingService(store,classifier,BrokenCache()).classify(msg(),trace_id='trace')
    assert receipt['topic']['version']==2
    assert classifier.calls[0]['candidates'][0]['topic_id']==topic


def test_quote_keeps_cold_other_topic_but_never_other_raw_text(store):
    topic=seed(store,customer_id='other',message_id='old-public',at=STAMP-timedelta(hours=2))['topic']['topic_id']
    classifier=Classifier(result(topic_action='existing',topic_id=topic))
    TopicRoutingService(store,classifier).classify(msg(quoted_message_id='old-public'),trace_id='trace')
    call=classifier.calls[0]
    assert call['candidates'][0]['topic_id']==topic and call['candidates'][0]['quoted_reference']
    assert 'other自己的消息' not in str(call)


def test_two_consecutive_cas_conflicts_are_visible_without_infinite_retry(store):
    topic=seed(store)['topic']['topic_id']
    def race(kwargs):
        version=kwargs['candidates'][0]['version']
        seed(store,customer_id='other',message_id=f'racing-{version}',topic_id=topic,expected_version=version)
        return result(topic_action='existing',topic_id=topic)
    classifier=Classifier(race,race)
    with pytest.raises(ValueError,match='topic version conflict'):
        TopicRoutingService(store,classifier).classify(msg(),trace_id='trace')
    assert len(classifier.calls)==2
    assert store.get_group_topic_message('console','room','a','current') is None


def test_same_task_many_source_bindings_remains_one_personal_goal(store):
    topic=seed(store)['topic']['topic_id']
    task=bind_task(store,topic,'private-a')
    seed(store,message_id='followup',topic_id=topic,expected_version=1)
    store.bind_group_topic_customer(topic_id=topic,customer_id='a',conversation_id='private-a',
        task_context_id=task.task_context_id,source_message_id='followup',at=STAMP+timedelta(seconds=1))
    classifier=Classifier(result('update',topic_action='existing',topic_id=topic,goal_relation='continue',task_context_id=task.task_context_id))
    TopicRoutingService(store,classifier).classify(msg('改成八点'),trace_id='trace')
    assert len(classifier.calls[0]['own_context']['tasks'])==1
    assert classifier.calls[0]['candidates'][0]['own_task_context_id']==task.task_context_id


def test_service_failure_does_not_save_a_fake_ignore_receipt(store):
    def fail(kwargs):raise TimeoutError('provider timed out')
    classifier=Classifier(fail)
    with pytest.raises(TimeoutError):TopicRoutingService(store,classifier).classify(msg(),trace_id='trace')
    assert store.get_group_topic_message('console','room','a','current') is None
    assert not store.list_group_topics(room_id='room',at=STAMP)


def test_replay_compares_timestamp_instant_after_store_timezone_normalization(store):
    classifier=Classifier(result())
    service=TopicRoutingService(store,classifier)
    first=service.classify(msg(sent_at=STAMP.astimezone(timezone.utc)),trace_id='trace')
    replay=service.classify(msg(sent_at=STAMP),trace_id='trace-replay')
    assert replay['deduplicated']
    assert replay['message']==first['message']
    assert len(classifier.calls)==1


def board_fixture(store):
    games=[]
    for index,start in enumerate(('19:30','20:00')):
        game,_=store.create_game(conversation_id=f'private-board-{index}',organizer_id=f'other-{index}',
            organizer_name='PRIVATE_PARTICIPANT',requirement={'requested_game':'hangzhou_mahjong','stake':'0.5',
            'smoke_preference':'no_smoking','start_time':f'2026-11-06T{start}:00+08:00',
            'known_player_count':3,'internal_note':'PRIVATE_NOTE'},known_players=[],trace_id='seed')
        games.append(game)
    def save(snapshot_id,external,ordered):
        return store.save_board_snapshot(BoardSnapshot(snapshot_id=snapshot_id,room_id='room',
            conversation_id='wechaty:room:room',external_message_id=external,rendered_text='RAW_BOARD_PRIVATE',
            items=[BoardSnapshotItem(item_no=i+1,game_id=game.game_id,rendered_text='RAW_ITEM_PRIVATE')
                   for i,game in enumerate(ordered)],created_at=STAMP))
    first=save('snapshot-old','board-old',games)
    latest=save('snapshot-new','board-new',list(reversed(games)))
    return games,first,latest


def test_classifier_uses_snapshot_numbers_with_current_public_game_projection(store):
    games,_,latest=board_fixture(store)
    classifier=Classifier(result('query'))
    receipt=TopicRoutingService(store,classifier).classify(msg('看板第一桌什么情况'),trace_id='trace')
    rows=classifier.calls[0]['board']
    assert store.get_group_board_state('room') is None
    assert rows[0]['game_id']==games[1].game_id and rows[0]['display_no']==1
    assert rows[0]['snapshot_id']==latest.snapshot_id and rows[0]['reference_status']=='latest'
    assert rows[0]['game_type']=='hangzhou_mahjong' and rows[0]['stakes']==['0.5']
    assert rows[0]['slots_filled']==3 and rows[0]['slots_total']==4
    assert 'PRIVATE' not in str(rows)
    assert receipt['message']['classification']['routing_audit']['board_source']['snapshot_id']==latest.snapshot_id


def test_explicit_quote_uses_exact_snapshot_not_latest_renumbering(store):
    games,first,_=board_fixture(store)
    classifier=Classifier(result('query'))
    TopicRoutingService(store,classifier).classify(msg('第一桌还能来吗',quoted_message_id='board-old'),trace_id='trace')
    rows=classifier.calls[0]['board']
    assert rows[0]['game_id']==games[0].game_id and rows[0]['snapshot_id']==first.snapshot_id
    assert rows[0]['reference_status']=='quoted'


def test_unknown_quote_does_not_guess_latest_number_mapping(store):
    board_fixture(store)
    classifier=Classifier(result('uncertain',topic_action='uncertain',goal_relation='uncertain',public_summary={}))
    receipt=TopicRoutingService(store,classifier).classify(msg('第一桌还缺吗',quoted_message_id='unknown-board'),trace_id='trace')
    assert classifier.calls[0]['board']==[]
    assert receipt['message']['classification']['routing_audit']['board_source']['reference_status']=='unresolved_quote'


def test_board_source_anchor_reference_resolves_same_immutable_snapshot(store):
    from mahjong_agent_runtime.models import MessageReference
    games,first,_=board_fixture(store)
    store.register_message_reference(MessageReference(message_id='quoted-anchor',conversation_id='wechaty:room:room',
        business_ref_type='group_board_snapshot',business_ref_id=first.snapshot_id,text='PRIVATE_REFERENCE_BODY',
        channel='wechaty',recipient_id='room'))
    classifier=Classifier(result('query'))
    TopicRoutingService(store,classifier).classify(msg('第一桌还缺吗',quoted_message_id='quoted-anchor'),trace_id='trace')
    assert classifier.calls[0]['board'][0]['game_id']==games[0].game_id
    assert 'PRIVATE_REFERENCE_BODY' not in str(classifier.calls[0])


def test_old_snapshot_keeps_number_but_reads_current_cancelled_game_state(store):
    games,_,_=board_fixture(store)
    store.update_game_status(game_id=games[0].game_id,status='cancelled',reason='cancelled by organizer',trace_id='cancel')
    classifier=Classifier(result('query'))
    TopicRoutingService(store,classifier).classify(msg('第一桌还有吗',quoted_message_id='board-old'),trace_id='trace')
    row=classifier.calls[0]['board'][0]
    assert row['game_id']==games[0].game_id and row['display_no']==1
    assert row['status']=='cancelled'


def test_reference_to_another_room_cannot_supply_number_mapping(store):
    from mahjong_agent_runtime.models import MessageReference
    games,_,_=board_fixture(store)
    store.save_board_snapshot(BoardSnapshot(snapshot_id='foreign-snapshot',room_id='foreign-room',
        conversation_id='wechaty:room:foreign-room',external_message_id='foreign-board',rendered_text='FOREIGN_RAW',
        items=[BoardSnapshotItem(item_no=1,game_id=games[0].game_id,rendered_text='FOREIGN_ITEM')],created_at=STAMP))
    store.register_message_reference(MessageReference(message_id='cross-room-anchor',conversation_id='wechaty:room:room',
        business_ref_type='group_board_snapshot',business_ref_id='foreign-snapshot',text='PRIVATE_REFERENCE_BODY',
        channel='wechaty',recipient_id='room'))
    classifier=Classifier(result('query'))
    TopicRoutingService(store,classifier).classify(msg('第一桌还有吗',quoted_message_id='cross-room-anchor'),trace_id='trace')
    assert classifier.calls[0]['board']==[]


def test_task_facts_keep_original_null_task_source_after_many_bound_updates(store):
    original=store.record_group_topic_message(channel='console',room_id='room',customer_id='a',message_id='initial',
        text='明晚杭麻0.5无烟我们两人，先看看有没有局',sent_at=STAMP,intent='query',
        public_summary={'game_type':'hangzhou_mahjong'},classification=result('query',constraints={
            'game_type':'hangzhou_mahjong','stakes':['0.5'],'smoking':'no_smoking','party_size':2}))
    topic=original['topic']['topic_id']
    task=bind_task(store,topic,'private-a',source='initial')
    for i in range(8):
        source=f'change-{i}'
        when=STAMP+timedelta(seconds=i+1)
        store.record_group_topic_message(channel='console',room_id='room',customer_id='a',message_id=source,
            text=f'时间改成{i+12}点',sent_at=when,intent='update',topic_id=topic,expected_version=i+1,
            public_summary={},classification=result('update',topic_action='existing',topic_id=topic,
                goal_relation='continue',task_context_id=task.task_context_id,
                constraints={'planned_start_at':f'2026-11-06T{i+12:02}:00:00+08:00'}))
        store.bind_group_topic_customer(topic_id=topic,customer_id='a',conversation_id='private-a',
            task_context_id=task.task_context_id,source_message_id=source,at=when)
    classifier=Classifier(result('update',topic_action='existing',topic_id=topic,goal_relation='continue',task_context_id=task.task_context_id))
    TopicRoutingService(store,classifier).classify(msg('还是原来两人',sent_at=STAMP+timedelta(minutes=1)),trace_id='trace')
    row=classifier.calls[0]['own_context']['tasks'][0]
    assert row['goal_summary']=='明晚杭麻0.5无烟我们两人，先看看有没有局'
    assert row['facts']['party_size']==2 and row['facts']['game_type']=='hangzhou_mahjong'
    assert row['facts']['smoking']=='no_smoking' and row['facts']['stakes']==['0.5']
    assert row['facts']['planned_start_at']=='2026-11-06T19:00:00+08:00'
