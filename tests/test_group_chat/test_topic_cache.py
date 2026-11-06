from datetime import timedelta
import json
import shutil
import socket
import subprocess
import time
import uuid

import pytest

from mahjong_agent_runtime import InMemoryAgentStore
from mahjong_agent_runtime.models import now


@pytest.fixture
def redis_client():
    redis = pytest.importorskip('redis')
    binary = shutil.which('redis-server')
    if not binary:
        pytest.skip('isolated Redis executable unavailable')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen([binary, '--bind', '127.0.0.1', '--port', str(port),
                                '--save', '', '--appendonly', 'no'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = redis.Redis(host='127.0.0.1', port=port, decode_responses=True, socket_timeout=0.2)
    try:
        for _ in range(50):
            try:
                if client.ping():
                    break
            except redis.RedisError:
                time.sleep(0.02)
        else:
            pytest.fail('isolated Redis did not start')
        yield client
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=3)


def record(store, stamp, **kwargs):
    args = dict(channel='console', room_id='room', customer_id='a', message_id='m1',
                text='杭麻有人吗', sent_at=stamp, intent='query', idle_seconds=60,
                public_summary={'game_type':'hangzhou_mahjong'}, classification={'reasoning':'本人私密解释'})
    args.update(kwargs)
    return store.record_group_topic_message(**args)


def test_redis_cache_stale_write_replay_ttl_and_public_projection(redis_client):
    from mahjong_agent_runtime.group_chat.topic_cache import RedisGroupTopicCache
    cache = RedisGroupTopicCache(redis_client, namespace='test:'+uuid.uuid4().hex)
    store = InMemoryAgentStore(); stamp = now(); first = record(store, stamp)['topic']
    assert cache.put(first)
    key = cache._topic_key(first['topic_id'])
    ttl = redis_client.pttl(key)
    assert not cache.put(first)
    assert redis_client.pttl(key) <= ttl
    updated = record(store, stamp+timedelta(seconds=1), message_id='m2', topic_id=first['topic_id'], expected_version=1)['topic']
    assert cache.put(updated)
    assert not cache.put(first)
    assert cache.get(first['topic_id'])['version'] == 2
    assert cache.list_candidates('room', at=stamp, limit=5)[0]['version'] == 2
    assert not cache.list_candidates('other', at=stamp, limit=5)
    payload = json.loads(redis_client.get(key))
    assert not {'text','classification','customer_id','task_context_id','game_ids'} & payload.keys()
    assert cache.last_error is None


def test_cache_expiry_keeps_sqlite_equivalent_source_and_personal_binding(redis_client):
    from mahjong_agent_runtime.group_chat.topic_cache import RedisGroupTopicCache, load_group_topic_candidates
    cache = RedisGroupTopicCache(redis_client, namespace='test:'+uuid.uuid4().hex)
    store = InMemoryAgentStore(); stamp = now(); first = record(store, stamp)['topic']
    cache.put(first)
    future = stamp+timedelta(seconds=61)
    assert cache.list_candidates('room', at=future, limit=5) == []
    assert store.get_group_topic_message('console','room','a','m1')['text'] == '杭麻有人吗'
    assert not load_group_topic_candidates(store,cache,room_id='room',at=future)
    own = load_group_topic_candidates(store,cache,room_id='room',customer_id='a',at=future,include_cold=True)
    assert own[0]['topic_id'] == first['topic_id']
    assert not load_group_topic_candidates(store,cache,room_id='room',customer_id='b',at=future,include_cold=True)


def test_missing_partial_or_stale_cache_cannot_hide_committed_topic(redis_client):
    from mahjong_agent_runtime.group_chat.topic_cache import RedisGroupTopicCache, load_group_topic_candidates
    cache = RedisGroupTopicCache(redis_client, namespace='test:'+uuid.uuid4().hex)
    store = InMemoryAgentStore(); stamp = now(); first = record(store,stamp)['topic'];cache.put(first)
    second = record(store,stamp+timedelta(seconds=1),message_id='m2')['topic']
    results = load_group_topic_candidates(store,cache,room_id='room',at=stamp)
    assert {r['topic_id'] for r in results} == {first['topic_id'],second['topic_id']}
    assert cache.get(second['topic_id'])['version'] == 1


def test_unavailable_redis_falls_back_to_store():
    from mahjong_agent_runtime.group_chat.topic_cache import RedisGroupTopicCache, load_group_topic_candidates
    class Down:
        def get(self,*args,**kwargs):raise ConnectionError('down')
        def eval(self,*args,**kwargs):raise ConnectionError('down')
        def zrevrangebyscore(self,*args,**kwargs):raise ConnectionError('down')
    cache=RedisGroupTopicCache(Down());store=InMemoryAgentStore();stamp=now();first=record(store,stamp)['topic']
    result=load_group_topic_candidates(store,cache,room_id='room',at=stamp)
    assert result[0]['topic_id']==first['topic_id']
    assert cache.last_error == 'ConnectionError'
