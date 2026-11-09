"""手动群聊的新成员可直接选用，独立通道身份与已有场景记录都能保持。"""

from contextlib import closing
import json
from pathlib import Path
import sys

from mahjong_agent_runtime import SQLiteAgentStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from manual_simulation import handle_manual_simulation


def test_new_manual_group_has_seven_selectable_members(tmp_path):
    result = handle_manual_simulation({'action': 'create', 'request_id': 'new-roster'},
                                      runtime_dir=tmp_path, mode='mock')
    run_id = result['run_id']
    folder = tmp_path / 'runs' / run_id
    report = json.loads((folder / 'report.json').read_text())
    actors = report['manual_actors']
    expected_ids = ['manual_customer', *(f'manual_customer_{index}' for index in range(2, 8))]
    assert [actor['customer_id'] for actor in actors] == expected_ids
    assert [actor['display_name'] for actor in actors] == [f'客户{letter}' for letter in 'ABCDEFG']
    assert report['default_actor_id'] == expected_ids[0]
    assert all(actor['preferred_games'] == [] for actor in actors[:3])
    assert len({tuple(actor['preferred_games']) for actor in actors[3:]}) == 4
    assert len({actor['smoke_preference'] for actor in actors[3:]}) == 3
    conversations = report['manual_private_conversations']
    assert set(conversations) == set(expected_ids)
    assert len(set(conversations.values())) == 7
    assert conversations[expected_ids[0]] == f'manual:{run_id}'
    assert {actor['is_friend'] for actor in actors[3:]} == {True, False}
    store = SQLiteAgentStore(folder / 'test_sim.db')
    with closing(store._connection):
        assert set(store.customers) == set(expected_ids)
        for actor in actors:
            actor_id = actor['customer_id']
            assert store.customers[actor_id].to_dict() == {key: value for key, value in actor.items()
                                                         if key != 'is_friend'}
            identity = store.get_channel_identity('console', actor_id)
            assert identity.private_conversation_id == conversations[actor_id]
            assert identity.is_friend is identity.can_private_message is actor['is_friend']
        assert not store.games and not store.invite_drafts and not store.outbound_message_drafts
    handle_manual_simulation({'action': 'queue_batch', 'request_id': 'select-added-member', 'run_id': run_id,
                              'messages': [{'actor_id': 'manual_customer_7', 'channel': 'group', 'text': '大家好'}]},
                             runtime_dir=tmp_path)
    queued = json.loads((folder / 'report.json').read_text())
    item, = queued['manual_batch']['items']
    assert item['actor_id'] == 'manual_customer_7' and item['channel'] == 'group' and item['status'] == 'pending'
    assert queued['transcript'] == []


def test_existing_manual_workspace_keeps_its_saved_roster_on_create_retry(tmp_path):
    payload = {'action': 'create', 'request_id': 'saved-roster', 'display_name': '客户D'}
    result = handle_manual_simulation(payload, runtime_dir=tmp_path, mode='mock')
    run_id = result['run_id']
    path = tmp_path / 'runs' / run_id / 'report.json'
    original = json.loads(path.read_text())
    names = [actor['display_name'] for actor in original['manual_actors']]
    assert names[0] == '客户D' and names[3] == '客户D2' and len(set(names)) == 7
    handle_manual_simulation({'action': 'update_profile', 'request_id': 'edit-roster', 'run_id': run_id,
                              'actor_id': 'manual_customer_4', 'profile': {'display_name': '已修改的角色'}},
                             runtime_dir=tmp_path)
    saved = path.read_bytes()
    retry = handle_manual_simulation(payload, runtime_dir=tmp_path, mode='mock')
    assert retry['run_id'] == run_id
    assert path.read_bytes() == saved
    assert json.loads(saved)['manual_actors'][3]['display_name'] == '已修改的角色'
