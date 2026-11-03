"""纯查询口径追加修订仅撤销必选征询，原始证据、事实和授权仍精确锁定。"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import eval_product_contracts as policy
import run_real_owner_chat_live_eval as owner


def declaration():
    """读取固定产品资产，不根据被测回复生成标准。"""
    return json.loads(policy.QUERY_OPTION_POLICY_PATH.read_text())


def install_declaration(monkeypatch, tmp_path, record):
    """仅测试深层拒绝规则时重绑测试资产摘要，原仓库资产不写入。"""
    path = tmp_path / 'policy.jsonl'
    path.write_text(json.dumps(record, ensure_ascii=False) + '\n')
    monkeypatch.setattr(policy, 'QUERY_OPTION_POLICY_PATH', path)
    monkeypatch.setattr(policy, 'QUERY_OPTION_POLICY_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())


def original_catalog(record):
    """从实际场景恢复该声明精确绑定的前态，供应用边界测试。"""
    scenes = {scene.scenario_id: scene for scene in owner.live_eval_scenarios()}
    return [replace(scenes[item['live_scenario_id']], **deepcopy(item['previous_scenario_contract']),
                    requires_query_reply_state_check=False) for item in record['revisions']]


def test_two_queries_apply_only_optional_confirmation_and_blind_state_guard():
    record = declaration()
    assert hashlib.sha256(policy.QUERY_OPTION_POLICY_PATH.read_bytes()).hexdigest() == policy.QUERY_OPTION_POLICY_SHA256
    raw = policy.DATASET_PATH.read_bytes()
    assert hashlib.sha256(raw[:record['original_dataset_prefix_bytes']]).hexdigest() == record['original_dataset_prefix_sha256']
    assert record['source']['not_original_human_transcript'] is True
    assert record['source']['user_quotes'] == ['两种都允许，状态与实际回复一致']
    assert record['source_catalog_baseline']['kind'] == 'reviewed_code_catalog'
    scenes = {scene.scenario_id: scene for scene in owner.live_eval_scenarios()}
    cases = owner.golden_reply_cases()
    assert {key for key, scene in scenes.items() if scene.requires_query_reply_state_check} == {
        item['live_scenario_id'] for item in record['revisions']}
    for item in record['revisions']:
        scene = scenes[item['live_scenario_id']]
        case = next(case for case in cases if case['case_id'] == item['target_case_id'])
        assert case['expected'] == item['current_expectation']
        assert case['product_contract_revision_ids'][-1] == item['revision_id']
        assert all(getattr(scene, name) == value for name, value in item['current_scenario_contract'].items())
        assert scene.forbidden_tool_names == item['previous_scenario_contract']['forbidden_tool_names']
        assert {'create_game', 'search_customers', 'create_invite_drafts'} <= set(scene.forbidden_tool_names)
        assert item['current_expectation']['forbidden_meaning'] == item['previous_expectation']['forbidden_meaning']
        assert item['current_expectation'].get('required_tools') == item['previous_expectation'].get('required_tools')
        assert scene.required_tool_names == ['search_current_games']
        changed = next(entry for entry in case['expected']['required_meaning'] if entry['id'] == item['changed_criterion_ids'][0])
        assert '允许纯告知结果' in changed['description'] and 'waiting_user' in changed['description']
    flexible = record['revisions'][0]
    assert 'expected_runtime_status' not in flexible['current_expectation']
    assert flexible['current_expectation']['expected_runtime_status_any'] == ['completed', 'waiting_user']
    assert scenes['flexible_time_followup_matches_evening_game'].expected_status_any == ['completed', 'waiting_user']
    assert scenes['profile_range_matches_middle_stake'].expected_status_any == []
    assert scenes['flexible_time_followup_matches_evening_game'].expected_tool_result_paths['search_current_games']['result.requirement.stake'] == '1'
    assert scenes['profile_range_matches_middle_stake'].expected_tool_result_paths['search_current_games']['result.requirement.stake_range'] == {'min': 1, 'max': 3}
    assert scenes['flexible_time_empty_pool_offers_help'].require_help_offer is True
    assert scenes['profile_default_matched_game'].requires_query_reply_state_check is False
    assert scenes['profile_range_mixed_inventory_personal_query'].requires_query_reply_state_check is False


def test_asset_hash_is_required_before_applying_any_revision(monkeypatch, tmp_path):
    path = tmp_path / 'policy.jsonl'
    path.write_bytes(policy.QUERY_OPTION_POLICY_PATH.read_bytes() + b' ')
    monkeypatch.setattr(policy, 'QUERY_OPTION_POLICY_PATH', path)
    with pytest.raises(ValueError, match='source changed'):
        policy._query_option_revisions(policy.read_product_records())


@pytest.mark.parametrize('fault', ['source', 'prefix', 'case_hash', 'old_expected', 'missing', 'duplicate', 'extra',
    'criterion', 'facts', 'privacy', 'tools', 'money', 'time', 'status_scalar', 'status_missing', 'range_status',
    'guard_missing', 'guard_false', 'guard_numeric'])
def test_revision_cannot_hide_source_drift_or_change_other_contracts(monkeypatch, tmp_path, fault):
    record = declaration()
    item = record['revisions'][0]
    if fault == 'source':
        record['source']['user_quotes'] = ['并非这次用户原话']
    elif fault == 'prefix':
        record['original_dataset_prefix_sha256'] = '0' * 64
    elif fault == 'case_hash':
        item['source_case_sha256'] = '0' * 64
    elif fault == 'old_expected':
        item['previous_expectation']['required_tools'] = []
    elif fault == 'missing':
        record['revisions'].pop()
    elif fault == 'duplicate':
        record['revisions'][1] = deepcopy(item)
    elif fault == 'extra':
        record['revisions'].append(deepcopy(item))
    elif fault == 'criterion':
        item['changed_criterion_ids'] = ['evening_remove_stale_afternoon']
    elif fault == 'facts':
        item['current_expectation']['required_meaning'][0]['description'] = '不必检查时间'
    elif fault == 'privacy':
        item['current_expectation']['forbidden_meaning'] = []
    elif fault == 'tools':
        item['current_expectation']['required_tools'] = []
    elif fault == 'money':
        item['current_scenario_contract']['expected_tool_result_paths']['search_current_games']['result.requirement.stake'] = '2'
    elif fault == 'time':
        item['current_scenario_contract']['excluded_query_requirement_values'] = {}
    elif fault == 'status_scalar':
        item['current_expectation']['expected_runtime_status'] = 'completed'
    elif fault == 'status_missing':
        item['current_expectation'].pop('expected_runtime_status_any')
    elif fault == 'range_status':
        record['revisions'][1]['current_scenario_contract']['expected_status_any'] = ['completed']
    elif fault == 'guard_missing':
        item['evaluation_guard'] = {}
    else:
        item['evaluation_guard']['requires_query_reply_state_check'] = False if fault == 'guard_false' else 1
    install_declaration(monkeypatch, tmp_path, record)
    with pytest.raises(ValueError):
        policy._query_option_revisions(policy.read_product_records())


@pytest.mark.parametrize('fault', ['catalog', 'missing_field', 'duplicate', 'missing', 'already_applied'])
def test_exact_complete_catalog_and_single_application_are_required(monkeypatch, tmp_path, fault):
    record = declaration()
    scenes = original_catalog(record)
    if fault == 'catalog':
        scenes[0] = replace(scenes[0], expected_active_game_count=1)
    elif fault == 'missing_field':
        for field in ('previous_scenario_contract', 'current_scenario_contract'):
            record['revisions'][0][field].pop('forbidden_tool_names')
        install_declaration(monkeypatch, tmp_path, record)
    elif fault == 'duplicate':
        scenes.append(scenes[0])
    elif fault == 'missing':
        scenes.pop()
    else:
        scenes = policy.apply_query_option_policy_to_scenarios(scenes, policy.read_product_records())
    with pytest.raises(ValueError):
        policy.apply_query_option_policy_to_scenarios(scenes, policy.read_product_records())


@pytest.mark.parametrize('fault', ['hash', 'absent', 'duplicate', 'already_applied'])
def test_original_case_identity_and_exact_old_expectation_are_required(fault):
    records = policy.read_product_records()
    record = declaration()
    cases = owner.golden_reply_cases()
    for item in record['revisions']:
        case = next(case for case in cases if case['case_id'] == item['target_case_id'])
        if fault != 'already_applied':
            case['expected'] = deepcopy(item['previous_expectation'])
    target = next(case for case in cases if case['case_id'] == record['revisions'][0]['target_case_id'])
    if fault == 'hash':
        source = next(item for item in records if item['id'] == record['revisions'][0]['source_record_id'])
        source['eval_cases'][0]['name'] = 'unexpected source change'
    elif fault == 'absent':
        cases.remove(target)
    elif fault == 'duplicate':
        cases.append(deepcopy(target))
    with pytest.raises(ValueError):
        policy.apply_query_option_policy_to_cases(cases, records)
