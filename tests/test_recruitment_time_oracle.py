"""以错误日期、时区及承诺窗口变异验证临时招募判定器，不改原始运行。"""
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_temporary_recruitment_probe as probe


def sample():
    game = {'game_id': 'g', 'status': 'forming', 'planned_start_at': '2026-11-03T19:30:00+08:00',
            'planned_end_at': '2026-11-03T23:30:00+08:00',
            'requirement': {'planned_start_at': '2026-11-03T19:30:00+08:00',
                            'planned_end_at': '2026-11-03T23:30:00+08:00'},
            'participants': [{'customer_id': probe.cid(a), 'status': 'joined', 'seat_count': 1} for a in 'ABC'],
            'remaining_seats': 1}
    expected = {'plans': [{'clock': '19:30', 'planned_start_at': game['planned_start_at'],
                          'planned_end_at': game['planned_end_at'], 'members': list('ABC'),
                          'allowed_statuses': ['forming', 'inviting'], 'remaining': 1}]}
    return {'games': [game]}, expected


@pytest.mark.parametrize('fault', ['next_day', 'wrong_offset', 'naive', 'missing_start', 'wrong_end', 'inconsistent_requirement'])
def test_oracle_rejects_changed_absolute_window_even_if_clock_and_members_match(fault):
    state, expected = sample();game = state['games'][0]
    if fault == 'next_day':
        for mapping in (game, game['requirement']):
            for key in ('planned_start_at', 'planned_end_at'):
                mapping[key] = (datetime.fromisoformat(mapping[key]) + timedelta(days=1)).isoformat()
    elif fault == 'wrong_offset':
        for mapping in (game, game['requirement']):
            for key in ('planned_start_at', 'planned_end_at'):
                mapping[key] = mapping[key].replace('+08:00', '+00:00')
    elif fault == 'naive':
        for mapping in (game, game['requirement']):
            for key in ('planned_start_at', 'planned_end_at'):
                mapping[key] = mapping[key][:-6]
    elif fault == 'missing_start':game['planned_start_at'] = None
    elif fault == 'wrong_end':game['planned_end_at'] = '2026-11-04T00:30:00+08:00'
    else:game['requirement']['planned_start_at'] = '2026-11-04T19:30:00+08:00'
    assert not all(row['passed'] for row in probe.evaluate_state(state, expected))


def test_equivalent_timezone_representation_and_cross_midnight_end_are_valid():
    state, expected = sample()
    for mapping in (state['games'][0], state['games'][0]['requirement']):
        mapping['planned_start_at'] = '2026-11-03T11:30:00+00:00'
        mapping['planned_end_at'] = '2026-11-03T15:30:00+00:00'
    assert all(row['passed'] for row in probe.evaluate_state(state, expected))
    for mapping in (state['games'][0], state['games'][0]['requirement'], expected['plans'][0]):
        mapping['planned_start_at'] = '2026-11-03T22:30:00+08:00'
        mapping['planned_end_at'] = '2026-11-04T02:30:00+08:00'
    expected['plans'][0]['clock'] = '22:30'
    assert all(row['passed'] for row in probe.evaluate_state(state, expected))


def test_two_non_overlapping_ready_games_are_not_a_double_commitment():
    state, _ = sample();first = state['games'][0];first['status'] = 'ready'
    second = deepcopy(first);second['game_id'] = 'later'
    for mapping in (second, second['requirement']):
        mapping['planned_start_at'] = '2026-11-03T23:30:00+08:00'
        mapping['planned_end_at'] = '2026-11-04T03:30:00+08:00'
    state['games'].append(second)
    assert all(row['passed'] for row in probe.evaluate_state(state, {}))
    second['planned_start_at'] = '2026-11-03T23:29:00+08:00'
    assert not all(row['passed'] for row in probe.evaluate_state(state, {}))


def test_v5_only_adds_complete_windows_without_rewriting_original_inputs_or_business_labels():
    old=json.loads((probe.ROOT/'eval/golden/temporary_recruitment_parallel_v4.json').read_text())
    current=json.loads((probe.ROOT/'eval/golden/temporary_recruitment_parallel_v5.json').read_text())
    for before,after in zip(old['cases'],current['cases'],strict=True):
        projected=deepcopy(after)
        for plan in projected['expected_final'].get('plans',[]):
            plan.pop('planned_start_at');plan.pop('planned_end_at')
        assert projected==before


def test_consensus_turn_must_persist_time_without_counting_unconfirmed_candidate():
    source=json.loads((probe.ROOT/'eval/golden/temporary_recruitment_parallel_v6.json').read_text())
    case=next(c for c in source['cases'] if c['case_id']=='e_1930_consensus')
    expected=probe.expected_turn_state(case,1)
    state,_=sample()
    assert all(r['passed'] for r in probe.evaluate_state(state,expected))
    for mapping in (state['games'][0],state['games'][0]['requirement']):
        mapping['planned_start_at']='2026-11-03T19:00:00+08:00'
        mapping['planned_end_at']='2026-11-03T23:00:00+08:00'
    assert not all(r['passed'] for r in probe.evaluate_state(state,expected))
    assert probe.expected_turn_state(case,0)=={'not_counted':['E'],'no_ready':True}
    assert probe.expected_turn_state(case,2)==case['expected_final']


def test_v6_preserves_all_inputs_and_final_labels_and_only_adds_consensus_checkpoint():
    old=json.loads((probe.ROOT/'eval/golden/temporary_recruitment_parallel_v5.json').read_text())
    current=json.loads((probe.ROOT/'eval/golden/temporary_recruitment_parallel_v6.json').read_text())
    current['schema_version']=old['schema_version']
    case=next(c for c in current['cases'] if c['case_id']=='e_1930_consensus')
    assert case.pop('expected_after_turns')['2']['not_counted']==['E']
    assert current==old
