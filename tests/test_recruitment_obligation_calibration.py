"""评测须区分牌局招募、个人复查与截止责任，不能由惯用词推导新的承诺。"""

import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def task(kind='activate_game_recruitment'):
    """同一业务目标中的真实任务字段形状，不携带模型结论或预期标签。"""
    return {'task_id': 'work-1', 'task_type': kind, 'aggregate_type': 'game', 'aggregate_id': 'game-1',
        'conversation_id': 'chat-1', 'subject_id': 'self', 'status': 'pending',
        'due_at': '2026-11-06T19:30:00+08:00', 'lease_until': None,
        'payload': {'game_id': 'game-1', 'task_context_id': 'goal-1',
                    'event_type': 'game_recruitment_window_opened', 'private_note': '不得进入评测'}}


def project(tasks):
    """使用回放入口真正给独立判定器的投影，不仅验证局部自建数据。"""
    runner = importlib.import_module('run_group_topic_regression_eval')
    return runner.project_scheduled_responsibilities(tasks, customer_id='self', conversation_id='chat-1',
        task_context_id='goal-1', target_game_ids=['game-1'])


def test_activation_is_not_a_participation_deadline_and_keeps_actual_scope():
    original = task()
    projected = project([original])
    assert projected['participation_deadlines'] == []
    activation = projected['recruitment_activations'][0]
    assert activation['task_type'] == 'activate_game_recruitment'
    assert activation['due_at'] == original['due_at']
    assert activation['payload']['task_context_id'] == 'goal-1'
    assert 'private_note' not in str(projected)
    assert original['payload']['private_note'] == '不得进入评测'


@pytest.mark.parametrize('field,value', [('subject_id', 'other'), ('conversation_id', 'other-chat'),
    ('task_context_id', 'other-goal'), ('game_id', 'other-game'), ('aggregate_id', 'other-game'),
    ('aggregate_type', 'conversation_goal')])
def test_projection_never_borrows_other_person_goal_or_game(field, value):
    candidate = task()
    target = candidate['payload'] if field in {'task_context_id', 'game_id'} else candidate
    target[field] = value
    assert project([candidate]) == {'participation_deadlines': [], 'recruitment_activations': []}


def test_cancelled_or_executed_task_is_not_hidden_or_upgraded_to_active():
    cancelled = task(); cancelled['status'] = 'cancelled'
    executed = task('participation_deadline')
    executed.update(aggregate_type='game_participant', aggregate_id='game-1:self', status='processing')
    executed['payload'].update(customer_id='self', condition='game_not_ready', effect='withdraw_self',
        deadline_at='2026-11-06T22:00:00+08:00', execution_outcome='withdraw_self')
    result = project([cancelled, executed])
    assert result['recruitment_activations'][0]['status'] == 'cancelled'
    assert result['participation_deadlines'][0]['payload']['execution_outcome'] == 'withdraw_self'
    assert not project([task('conversation_follow_up')])['recruitment_activations']


def test_communication_contract_identifies_responsibility_before_mechanism():
    contract = importlib.import_module('run_inquiry_grounding_eval').communication_obligation_contract()
    assert contract['assessment_order'] == ['identify_object_and_trigger', 'identify_actual_scope', 'match_authoritative_mechanism']
    assert contract['recruitment_implies_personal_notification'] is False
    assert contract['activation_implies_unlimited_recheck'] is False
    assert '招募' in contract['recruitment_activation']
    assert '通知' in contract['future_notification']


def test_calibration_has_controlled_positive_and_negative_obligations():
    module = importlib.import_module('run_recruitment_obligation_calibration')
    cases = module.load_cases()
    assert {c['fixture'] for c in cases} == {'empty', 'board_activation', 'board_follow_up'}
    original = next(c for c in cases if c['case_id'] == 'recruitment_short_ack')
    missing = next(c for c in cases if c['case_id'] == 'recruitment_without_record')
    assert original['reply'] == missing['reply']
    assert original['expected_passed'] is True and missing['expected_passed'] is False
    assert original['reply'] == '好，明晚21:30川麻1块无烟这局我给你记上了，缺人我盯着。'
    assert len(cases) >= 5
    assert all(c['expected_passed'] is False for c in cases if c['case_id'] in {'board_not_personal_notice', 'activation_not_infinite_polling'})


@pytest.mark.parametrize('index', range(7))
def test_calibration_uses_gateway_and_sqlite_without_expected_label_leak(index, tmp_path):
    """独立模型前的事实必须真实落库，不能以正负标签伪造机制。"""
    import json
    module = importlib.import_module('run_recruitment_obligation_calibration')
    case = module.load_cases()[index]
    store, evidence, events = module.prepare_fixture(case, tmp_path)
    try:
        state = module.read_semantic_calibration_state(store.path)
        assert len(state['games']) == (0 if case['fixture'] == 'empty' else 1)
        assert any(e['step'] == 'tool_gateway_completed' for e in events)
        tasks = evidence['persisted_state']
        assert tasks['participation_deadlines'] == []
        assert len(tasks['recruitment_activations']) == (0 if case['fixture'] == 'empty' else 1)
        assert len(tasks['follow_up_state']['tasks']) == (1 if case['fixture'] == 'board_follow_up' else 0)
        if case['fixture'] != 'empty':
            board = evidence['execution_lifecycle']['activities'][0]['board_arrangement']
            assert board['linked_room_count'] == board['pending_task_count'] == board['outbound_authorized_room_count'] == 1
            assert board['publication_sent'] is False
        encoded = json.dumps(evidence, ensure_ascii=False)
        assert all(key not in encoded for key in ('expected_passed', case['case_id'], case['source'], 'fixture'))
        assert evidence['reply'] == case['reply']
    finally:
        store._connection.close()


def test_complete_frozen_reassessment_uses_every_source_once_without_main_agent(tmp_path, monkeypatch):
    """复评只能沿原运行清单，失败也不能偷偷替换回复或从其他目录补成绩。"""
    module = importlib.import_module('run_recruitment_obligation_calibration')
    source = tmp_path / 'frozen'; source.mkdir()
    module.write_json(source / 'summary.json', {'selected_cases': ['b', 'a'], 'cases': [{'case_id': 'b'}, {'case_id': 'a'}]})
    for name in ['a', 'b']:
        (source / name).mkdir(); module.write_json(source / name / 'report.json', {'case_id': name})
    calls = []

    def frozen_only(client, report, folder, *, judge):
        calls.append(report)
        return {'case_id': report.parent.name, 'status': 'failed' if report.parent.name == 'b' else 'passed'}

    monkeypatch.setattr(module, 'rejudge_frozen', frozen_only)
    result = module.rejudge_complete_run(None, source, tmp_path / 'new')
    assert calls == [source / 'b/report.json', source / 'a/report.json']
    assert result['case_count'] == 2 and result['passed_count'] == 1 and result['status'] == 'failed'
    assert result['main_agent_executed'] is False and result['source_summary_unchanged'] is True


def test_incomplete_frozen_run_is_rejected_before_any_judge(tmp_path):
    module = importlib.import_module('run_recruitment_obligation_calibration')
    module.write_json(tmp_path / 'summary.json', {'selected_cases': ['missing'], 'cases': []})
    with pytest.raises(ValueError, match='incomplete'):
        module.rejudge_complete_run(None, tmp_path, tmp_path / 'new')
