"""整包场景先校验再发布；画像、好友通道和每次只发一条的事实必须可核对。"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import manual_simulation as manual
from mahjong_agent_runtime import SQLiteAgentStore


def scenario():
    return {'version': 1, 'actors': [
        {'id': 'a', 'is_friend': True, 'profile': {'display_name': '客户甲', 'preferred_games': ['国标'],
                                                 'preferred_stakes': ['0.5'], 'smoke_preference': 'no_smoke'}},
        {'id': 'b', 'is_friend': False, 'profile': {'display_name': '客户乙', 'preferred_games': ['川麻']}},
    ], 'messages': [
        {'actor_id': 'a', 'channel': 'group', 'text': '  帮我组个国标1块人齐开的局\n  '},
        {'actor_id': 'b', 'channel': 'group', 'text': '帮我组个国标1块人齐开的局'},
    ]}


def import_scene(tmp_path, value=None, request_id='import-once'):
    return manual.handle_manual_simulation({'action': 'import_scenario', 'request_id': request_id,
                                           'scenario': scenario() if value is None else value},
                                          runtime_dir=tmp_path, mode='mock')


def report(tmp_path, run_id):
    return json.loads((tmp_path / 'runs' / run_id / 'report.json').read_text())


def operation(tmp_path, run_id, action, request_id, **kwargs):
    return manual.handle_manual_simulation({'run_id': run_id, 'action': action, 'request_id': request_id, **kwargs},
                                          runtime_dir=tmp_path)


def test_import_never_builds_model_or_executes_pending_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(manual, 'build_runtime', lambda *args, **kwargs: pytest.fail('导入不能构造主Agent或调用模型'))
    source = scenario()
    result = import_scene(tmp_path, source)
    saved = report(tmp_path, result['run_id'])
    assert saved['manual_import']['source'] == source
    assert saved['transcript'] == [] and saved['total_messages'] == 0
    assert saved['manual_batch']['position'] == 0
    assert all(item['status'] == 'pending' for item in saved['manual_batch']['items'])
    assert saved['manual_batch']['items'][0]['text'] == source['messages'][0]['text']
    assert len(saved['manual_actors']) == 2
    store = SQLiteAgentStore(tmp_path / 'runs' / result['run_id'] / 'test_sim.db')
    try:
        assert set(store.customers) == set(result['actor_ids'].values())
        assert store.customers[result['actor_ids']['a']].preferred_games == ['national_standard_mahjong']
        for alias, friend in [('a', True), ('b', False)]:
            identity = store.get_channel_identity('console', result['actor_ids'][alias])
            assert identity.is_friend is friend
            assert identity.can_private_message is friend
        assert not store.games and not store.invite_drafts and not store.outbound_message_drafts
    finally:
        store._connection.close()


@pytest.mark.parametrize('mutate', [
    lambda value: value.update(unknown=True),
    lambda value: value.update(version=True),
    lambda value: value.update(version=2),
    lambda value: value.update(actors=[]),
    lambda value: value.update(messages=[]),
    lambda value: value['actors'][0].update(extra='bad'),
    lambda value: value['actors'][0].update(is_friend='false'),
    lambda value: value['actors'][0].update(id='../outside'),
    lambda value: value['actors'][1].update(id='a'),
    lambda value: value['actors'][1]['profile'].update(display_name='客户甲'),
    lambda value: value['actors'][1]['profile'].update(preferred_games='川麻'),
    lambda value: value['actors'][1]['profile'].update(smoke_preference='no_smoking'),
    lambda value: value['actors'][1]['profile'].update(customer_id='other'),
    lambda value: value['actors'][1]['profile'].pop('display_name'),
    lambda value: value['messages'][1].update(actor_id='missing'),
    lambda value: value['messages'][1].update(channel='wechat'),
    lambda value: value['messages'][1].update(channel=[]),
    lambda value: value['messages'][1].update(text=' ' * 3),
    lambda value: value['messages'][1].update(text='字' * 4001),
    lambda value: value['messages'][1].update(text=123),
    lambda value: value['messages'][1].update(sender_id='other'),
    lambda value: value.update(messages=value['messages'] * 16),
])
def test_invalid_complete_scene_creates_no_partial_run_or_database(tmp_path, mutate):
    value = scenario()
    mutate(value)
    with pytest.raises(ValueError):
        import_scene(tmp_path, value)
    assert list(tmp_path.iterdir()) == []


def test_import_deduplicates_same_request_and_rejects_changed_source_without_overwrite(tmp_path):
    result = import_scene(tmp_path)
    path = tmp_path / 'runs' / result['run_id'] / 'report.json'
    original = path.read_bytes()
    retry = import_scene(tmp_path)
    assert retry['deduplicated'] and retry['actor_ids'] == result['actor_ids']
    changed = scenario()
    changed['messages'][0]['text'] += '别改原件'
    with pytest.raises(ValueError, match='不能替换'):
        import_scene(tmp_path, changed)
    assert path.read_bytes() == original
    assert len(list((tmp_path / 'runs').iterdir())) == 1


def test_concurrent_imports_publish_one_complete_workspace(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: import_scene(tmp_path), range(4)))
    assert len({item['run_id'] for item in results}) == 1
    assert sum(bool(item.get('deduplicated')) for item in results) == 3
    assert len(list((tmp_path / 'runs').iterdir())) == 1
    assert len(report(tmp_path, results[0]['run_id'])['manual_actors']) == 2


def test_failed_initialization_keeps_evidence_hidden_and_can_retry(tmp_path, monkeypatch):
    original = SQLiteAgentStore.upsert_customer
    calls = []
    def fail_second(store, profile):
        calls.append(profile.customer_id)
        if len(calls) == 2:
            raise OSError('受控存储失败')
        return original(store, profile)
    monkeypatch.setattr(SQLiteAgentStore, 'upsert_customer', fail_second)
    with pytest.raises(OSError):
        import_scene(tmp_path)
    assert list((tmp_path / 'runs').iterdir()) == []
    evidence = list((tmp_path / '.scenario_imports').glob('*/import_failure.json'))
    assert len(evidence) == 1
    assert json.loads(evidence[0].with_name('source.json').read_text()) == scenario()
    monkeypatch.setattr(SQLiteAgentStore, 'upsert_customer', original)
    result = import_scene(tmp_path)
    assert len(report(tmp_path, result['run_id'])['manual_actors']) == 2
    assert evidence[0].is_file(), '重试不能删除失败证据'


def test_imported_group_steps_use_real_route_and_friend_identity(tmp_path, monkeypatch):
    from sim_adapter import StaticAgentLLMClient
    original = StaticAgentLLMClient._semantic_topic_result
    def classified(self, payload):
        self.semantic_intent = 'delegate'
        return original(self, payload)
    monkeypatch.setattr(StaticAgentLLMClient, '_semantic_topic_result', classified)
    source = scenario()
    result = import_scene(tmp_path, source)
    run_id = result['run_id']
    first = operation(tmp_path, run_id, 'batch_next', 'step-a')
    saved = report(tmp_path, run_id)
    assert saved['manual_batch']['position'] == 1
    assert len(saved['transcript']) == 1 and first['sequence'] == 1
    a = saved['transcript'][0]
    assert a['user']['text'] == source['messages'][0]['text']
    assert a['manual_input']['text'] == source['messages'][0]['text']
    assert a['group_processing']['entrypoint'] == 'GroupMessageHandler'
    assert a['group_processing']['routing']['action'] == 'private_switch'
    assert a['group_processing']['outputs'] == []
    assert a['group_processing']['routing']['detail']['private_reply_suppressed'] is True
    assert a['group_processing']['routing']['detail']['private_conversation_id'] == saved['manual_private_conversations'][result['actor_ids']['a']]
    assert operation(tmp_path, run_id, 'batch_next', 'step-a')['deduplicated']
    assert report(tmp_path, run_id)['manual_batch']['position'] == 1
    recovered = import_scene(tmp_path, source)
    assert recovered['deduplicated'] and recovered['run_id'] == run_id
    assert report(tmp_path, run_id)['manual_batch']['position'] == 1
    assert '尚未执行' not in recovered['message']
    operation(tmp_path, run_id, 'batch_next', 'step-b')
    saved = report(tmp_path, run_id)
    assert len(saved['transcript']) == 2 and saved['manual_batch']['position'] == 2
    b = saved['transcript'][1]
    assert b['group_processing']['routing']['action'] == 'agent_loop'
    assert not any(item['channel'] == 'private' for item in b['group_processing']['outputs'])
    assert next(actor for actor in saved['manual_actors'] if actor['customer_id'] == result['actor_ids']['b'])['is_friend'] is False


def test_profile_edit_preserves_nonfriend_until_explicitly_changed(tmp_path):
    result = import_scene(tmp_path)
    run_id, actor_id = result['run_id'], result['actor_ids']['b']
    operation(tmp_path, run_id, 'update_profile', 'rename', actor_id=actor_id, profile={'display_name': '客户乙新名字'})
    saved = report(tmp_path, run_id)
    assert next(actor for actor in saved['manual_actors'] if actor['customer_id'] == actor_id)['is_friend'] is False
    with pytest.raises(ValueError, match='布尔'):
        operation(tmp_path, run_id, 'update_profile', 'invalid-friend', actor_id=actor_id, profile={}, is_friend='true')
    operation(tmp_path, run_id, 'update_profile', 'friend', actor_id=actor_id, profile={}, is_friend=True)
    saved = report(tmp_path, run_id)
    assert next(actor for actor in saved['manual_actors'] if actor['customer_id'] == actor_id)['is_friend'] is True
    assert saved['manual_profile_events'][-1]['before_is_friend'] is False
    assert saved['manual_profile_events'][-1]['after_is_friend'] is True


@pytest.mark.parametrize('example_name', ['group_scenario.json', 'seven_customer_group_scenario.json'])
def test_documented_json_example_is_accepted_without_sending(tmp_path, example_name):
    path = Path(__file__).resolve().parents[1] / 'docs' / 'examples' / example_name
    source = json.loads(path.read_text())
    result = import_scene(tmp_path, source)
    saved = report(tmp_path, result['run_id'])
    assert len(saved['manual_actors']) == len(source['actors'])
    assert len(saved['manual_batch']['items']) == len(source['messages'])
    assert saved['transcript'] == [] and saved['manual_batch']['position'] == 0
