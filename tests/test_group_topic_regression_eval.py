"""真实群入口回放的验收器不能把好看的回复或预存计划当成业务完成。"""

import importlib.util
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner():
    """载入独立回放入口，保持测试不依赖远端模型和密钥。"""
    sys.path.insert(0, str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('topic_regression_runner', ROOT / 'scripts/run_group_topic_regression_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation():
    """验收器只接真实运行层提供的规范化事实，不读模型计划自述。"""
    return {'intent': 'delegate', 'runtime_channel': 'private', 'status': 'completed',
            'tools': [], 'own_games': [], 'board_arrangements': [], 'deadline_tasks': [],
            'requester_joined_seeded': False, 'new_game_count': 0}


def test_delegation_requires_both_domain_and_board_evidence():
    module = runner()
    rules = {'require_persisted_recruitment': True, 'require_board_responsibility': True}
    actual = observation()
    assert not all(module.check_assertions(rules, actual).values())
    actual['own_games'] = [{'game_id': 'g', 'status': 'forming'}]
    assert module.check_assertions(rules, actual)['require_persisted_recruitment']
    assert not module.check_assertions(rules, actual)['require_board_responsibility']
    actual['board_arrangements'] = [{'game_id': 'g', 'status': 'scheduled', 'pending_task_count': 1,
                                   'linked_room_count': 1, 'outbound_authorized_room_count': 1}]
    assert all(module.check_assertions(rules, actual).values())
    actual['own_games'][0]['status'] = 'inviting'
    assert all(module.check_assertions(rules, actual).values())
    actual['board_arrangements'][0]['outbound_authorized_room_count'] = 0
    assert not module.check_assertions(rules, actual)['require_board_responsibility']


def test_query_rejects_business_writes_but_accepts_read_and_review():
    module = runner()
    actual = observation()
    actual['tools'] = [{'name': 'search_current_games', 'called': True, 'allowed': True, 'error': None},
                       {'name': 'customer_visible_content_review', 'called': True, 'allowed': True, 'error': None}]
    assert module.check_assertions({'no_business_writes': True}, actual)['no_business_writes']
    actual['tools'].append({'name': 'create_game', 'called': True, 'allowed': True, 'error': None})
    assert not module.check_assertions({'no_business_writes': True}, actual)['no_business_writes']


def test_join_confirmation_requires_membership_in_actual_seeded_game():
    module = runner()
    actual = observation()
    actual['own_games'] = [{'game_id': 'a-different-game', 'status': 'forming'}]
    assert not module.check_assertions({'requester_joined_seeded': True}, actual)['requester_joined_seeded']
    actual['requester_joined_seeded'] = True
    assert module.check_assertions({'requester_joined_seeded': True}, actual)['requester_joined_seeded']
    actual['own_games'] = []
    assert not module.check_assertions({'requester_joined_seeded': True}, actual)['requester_joined_seeded']


def test_failed_deadline_cannot_be_counted_as_arranged():
    module = runner()
    actual = observation()
    actual['tools'] = [{'name': 'set_participation_deadline', 'called': False, 'allowed': False, 'error': 'rejected'}]
    assert not module.check_assertions({'require_participation_deadline': True}, actual)['require_participation_deadline']
    actual['deadline_tasks'] = [{'task_type': 'activate_game_recruitment', 'status': 'pending'}]
    assert not module.check_assertions({'require_participation_deadline': True}, actual)['require_participation_deadline']


def test_condition_assertion_accepts_smoke_alias_only_and_rejects_wrong_or_missing_fact():
    module = runner()
    actual = observation()
    actual['own_games'] = [{'game_id': 'g', 'requirement': {'smoke_preference': 'no_smoking', 'stake': '1'}}]
    assert module.check_assertions({'expected_conditions': {'smoke_preference': 'no_smoke'}}, actual)['expected_conditions']
    assert not module.check_assertions({'expected_conditions': {'smoke_preference': 'smoking'}}, actual)['expected_conditions']
    assert not module.check_assertions({'expected_conditions': {'stake': 'no_smoke'}}, actual)['expected_conditions']
    del actual['own_games'][0]['requirement']['smoke_preference']
    assert not module.check_assertions({'expected_conditions': {'smoke_preference': 'no_smoke'}}, actual)['expected_conditions']


def deadline_observation():
    """显式保存本人当前目标/局/来源及工具回执，避免夹具只塞任意 pending 任务。"""
    task = {'task_id':'deadline-1', 'task_type':'participation_deadline', 'status':'pending',
            'subject_id':'self', 'conversation_id':'own-chat', 'due_at':'2026-11-06T22:00:00+08:00',
            'payload':{'customer_id':'self', 'task_context_id':'goal-1', 'source_message_id':'input-2',
                       'game_id':'g', 'deadline_at':'2026-11-06T22:00:00+08:00',
                       'condition':'game_not_ready', 'effect':'withdraw_self'}}
    actual = observation()
    actual.update(actor_id='self', conversation_id='own-chat', task_context_id='goal-1', message_id='input-2',
                  target_game_ids=['g'], own_games=[{'game_id':'g','status':'forming'}],
                  expected_deadline_at='2026-11-06T22:00:00+08:00', deadline_tasks=[task],
                  tools=[{'name':'set_participation_deadline','called':True,'allowed':True,'error':None,
                          'result':{'participation_deadline':deepcopy(task)}}])
    return actual


def test_deadline_requires_current_bound_task_and_accepts_equivalent_timezone():
    module, actual = runner(), deadline_observation()
    rules = {'require_participation_deadline':True, 'deadline_clock':'22:00'}
    assert all(module.check_assertions(rules,actual).values())
    for task in [actual['deadline_tasks'][0],actual['tools'][0]['result']['participation_deadline']]:
        task['due_at'] = task['payload']['deadline_at'] = '2026-11-06T14:00:00+00:00'
    assert all(module.check_assertions(rules,actual).values())


@pytest.mark.parametrize('path,value', [
    (('task_type',),'activate_game_recruitment'), (('status',),'cancelled'),
    (('subject_id',),'other'), (('conversation_id',),'foreign-chat'),
    (('payload','customer_id'),'other'), (('payload','task_context_id'),'old-goal'),
    (('payload','source_message_id'),'old-input'), (('payload','game_id'),'other-game'),
    (('payload','condition'),'always'), (('payload','effect'),'cancel_game'),
    (('payload','execution_outcome'),'withdraw_self'),
    (('due_at',),'2026-11-07T22:00:00+08:00'),
    (('payload','deadline_at'),'2026-11-07T22:00:00+08:00'),
    (('due_at',),'2026-11-06T22:00:00'), (('task_id',),'another-task'),
])
def test_deadline_cannot_borrow_unrelated_or_stale_pending_task(path,value):
    actual = deadline_observation()
    target = actual['deadline_tasks'][0]
    for key in path[:-1]:target = target[key]
    target[path[-1]] = value
    checks=runner().check_assertions({'require_participation_deadline':True,'deadline_clock':'22:00'},actual)
    assert checks['require_participation_deadline'] is False
    assert checks['deadline_clock'] is False


def test_deadline_requires_successful_receipt_for_the_persisted_task():
    actual = deadline_observation()
    actual['tools'][0].update(called=False,allowed=False,error='persistence rejected')
    assert not runner().check_assertions({'require_participation_deadline':True},actual)['require_participation_deadline']


def test_same_goal_requires_bound_history_not_a_classifier_claim():
    actual = observation()
    actual.update(goal_relation='continue',task_context_id='new-private-goal',classified_task_context_id='old-group-goal',
                  previous_task_context_ids=['old-group-goal'],previous_source_message_ids=['input-1'],
                  inherited_source_message_ids=['input-1','input-2'],source_bound_to_task=True)
    rules={'require_same_goal_continuation':True}
    assert runner().check_assertions(rules,actual)['require_same_goal_continuation']
    actual['inherited_source_message_ids']=['input-2']
    assert not runner().check_assertions(rules,actual)['require_same_goal_continuation']


def test_party_size_checks_requester_seats_not_whole_table():
    actual=observation()
    actual.update(actor_id='self',own_games=[{'game_id':'g','seat_summary':{'claimed_seats':4},
        'participants':[{'customer_id':'self','seat_count':1,'status':'confirmed'},
                        {'customer_id':'other','seat_count':3,'status':'confirmed'}]}])
    assert runner().check_assertions({'expected_conditions':{'party_size':1}},actual)['expected_conditions']
    assert not runner().check_assertions({'expected_conditions':{'party_size':4}},actual)['expected_conditions']


def test_unknown_assertion_is_rejected_instead_of_silently_skipped():
    import pytest
    with pytest.raises(ValueError, match='unknown assertion'):
        runner().check_assertions({'require_board_responsibilty': True}, observation())


def test_regression_assets_are_valid_and_keep_original_failure_inputs():
    module = runner()
    cases = module.load_cases(module.DEFAULT_DATASET)
    recovery = module.load_cases(ROOT / 'eval/golden/group_topic_recovery_20261105.jsonl')
    assert len(cases) == 8 and len(recovery) == 3
    assert recovery[0]['scenario']['messages'][-1]['text'] == '我明天下午的国标改成三点以后才方便，还是1块无烟。'
    assert recovery[1]['scenario']['messages'][-1]['text'] == '我明晚川麻九点半不变，最多等到十点，还没成的话我就不等了。'


def test_query_successor_view_changes_only_three_inputs_and_keeps_original_bytes():
    """明确店内查询的正向对照追加来源版本，业务/语义断言和原文件全部保留。"""
    module = runner()
    policy = json.loads(module.QUERY_POLICY_PATH.read_text())
    changed = []
    for source in policy['datasets']:
        path = ROOT / source['path']
        before = path.read_bytes()
        original = [json.loads(line) for line in before.decode().splitlines() if line.strip()]
        current = module.load_cases(path)
        assert len(current) == len(original)
        for prior, successor in zip(original, current):
            if 'input_revision' not in successor:
                assert successor == prior
                continue
            revision = successor['input_revision']
            assert revision['original_dataset_sha256'] == hashlib.sha256(before).hexdigest()
            assert revision['policy_sha256'] == module.QUERY_POLICY_SHA256
            assert successor['assertions'] == prior['assertions']
            assert successor['expectation'] == prior['expectation']
            assert '老板，店里' in successor['scenario']['messages'][revision['message_index']]['text']
            restored = deepcopy(successor)
            restored.pop('input_revision')
            restored['scenario']['messages'][revision['message_index']]['text'] = revision['previous_text']
            assert restored == prior
            changed.append(successor['case_id'])
        assert path.read_bytes() == before
    assert set(changed) == {'delegation_query_contrast', 'delegation_group_to_private_continuation',
                            'query_two_people_no_creation'}
    continued = next(case for case in module.load_cases(module.DEFAULT_DATASET)
                     if case['case_id'] == 'delegation_group_to_private_continuation')
    assert continued['assertions']['require_same_goal_continuation'] is True
    assert continued['scenario']['messages'][1]['text'] == '老板，帮我找明天下午两点国标1块无烟的，我一个人。'


def test_query_successor_policy_rejects_changed_source_case_or_policy(tmp_path, monkeypatch):
    module = runner()
    raw = module.DEFAULT_DATASET.read_bytes()
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    with pytest.raises(ValueError, match='original dataset changed'):
        module.apply_query_positive_control_policy(module.DEFAULT_DATASET, raw + b' ', cases)
    altered = deepcopy(cases)
    next(case for case in altered if case['case_id'] == 'delegation_query_contrast')['assertions']['no_business_writes'] = False
    with pytest.raises(ValueError, match='original case changed'):
        module.apply_query_positive_control_policy(module.DEFAULT_DATASET, raw, altered)
    changed = tmp_path / 'changed_policy.json'
    changed.write_bytes(module.QUERY_POLICY_PATH.read_bytes() + b' ')
    monkeypatch.setattr(module, 'QUERY_POLICY_PATH', changed)
    with pytest.raises(ValueError, match='policy changed without review'):
        module.load_cases(module.DEFAULT_DATASET)


def test_existing_option_fixture_really_has_three_host_seats_and_tomorrow_anchor(tmp_path):
    from mahjong_agent_runtime import SQLiteAgentStore
    module=runner()
    case=next(c for c in module.load_cases(module.DEFAULT_DATASET) if c['case_id']=='delegation_existing_option_needs_choice')
    profiles={a['id']:a['profile'] for a in module.validate_scenario(case['scenario'])['actors']}
    anchor=datetime.fromisoformat('2026-11-05T12:00:00+08:00')
    store=SQLiteAgentStore(tmp_path/'fixture.sqlite3')
    try:
        fixtures=module.seed_games(store,profiles,module.resolve_times(case['setup'],anchor),run_id='fixture')
        assert len(fixtures)==1
        game=fixtures[0]['persisted_game']
        assert game['planned_start_at']=='2026-11-06T14:00:00+08:00'
        assert game['seat_summary']['claimed_seats']==3 and game['seat_summary']['remaining_seats']==1
        assert [p['customer_id'] for p in game['participants']]==['host']
        assert game['participants'][0]['seat_count']==3
        assert module.successful(fixtures[0]['tool_result'])
    finally:store._connection.close()


def test_runner_uses_real_group_entry_and_keeps_expectations_out_of_actor_context(tmp_path, monkeypatch):
    """合同桩只验证回放装配，不将其结果称为真实模型的语义通过。"""
    from types import SimpleNamespace
    module = runner()
    original_build = module.build_runtime
    captured = []

    def build(store, mode):
        runtime, client = original_build(store, 'mock')
        client.config = SimpleNamespace(model='offline-mechanism', provider='stub', thinking_mode='disabled',
                                       reasoning_effort=None, max_tokens=1000, retry_attempts=1, attempt_timeout_seconds=None)
        original_complete = client.complete

        def complete(messages, **kwargs):
            captured.extend(m['content'] for m in messages)
            return original_complete(messages, **kwargs)

        client.complete = complete
        return runtime, client

    monkeypatch.setattr(module, 'build_runtime', build)
    monkeypatch.setattr(module, 'assess_reply_semantics', lambda *a, **kw: {'passed': True, 'scope': 'stub-mechanism-only'})
    case = next(c for c in module.load_cases(module.DEFAULT_DATASET) if c['case_id'] == 'delegation_query_contrast')
    result = module.run_case(case, tmp_path / 'query')
    assert result['status'] == 'passed', result
    assert result['turns'][0]['observation']['intent'] == 'query'
    assert result['turns'][0]['observation']['runtime_channel'] == 'group'
    assert all('required_meaning' not in text and 'target_message_index' not in text for text in captured)
    assert all('input_revision' not in text and 'source_bound_successor_input_view' not in text for text in captured)
