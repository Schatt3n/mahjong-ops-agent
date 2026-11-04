"""校准允许集的局部修订保留原四行、必要错档义务与判定器输入隔离。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from mahjong_agent_runtime.llm import completion_diagnostics
from tests.semantic_http_stub import semantic_http_completion
from tests.test_reply_scope_calibration import build_scope_stub
import run_reply_scope_calibration as calibration
import run_inquiry_grounding_eval as inquiry

TARGET = 'known_option_wrong_stake_20261103'
EXTRA = 'flexible_time_followup_matches_evening_game.evening_no_unnecessary_time_or_stake_change'
ORIGINAL_SHA = '4368147698e28ac8ee6ae11cc4aacab9b066d1cb332053b7cc4192b5b704d24d'


def declaration():
    """只读单一追加修订，测试不写仓库中的原始校准记录。"""
    return json.loads(calibration.CALIBRATION_REVISION_PATH.read_text())


def install_revision(monkeypatch, tmp_path, record):
    """故障注入仅重绑临时资产摘要，验证摘要之后的类型、来源与范围门槛。"""
    path = tmp_path / 'revision.jsonl'
    path.write_text(json.dumps(record, ensure_ascii=False) + '\n')
    monkeypatch.setattr(calibration, 'CALIBRATION_REVISION_PATH', path)
    monkeypatch.setattr(calibration, 'CALIBRATION_REVISION_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())


def test_original_four_rows_remain_exact_and_only_overlap_is_allowed():
    raw = calibration.DATASET_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA
    original = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    current = calibration.load_cases()
    assert len(original) == len(current[:4]) == 4
    for old, new in zip(original, current):
        expected = deepcopy(old)
        if old['case_id'] == TARGET:
            expected['allowed_violation_ids'].append(EXTRA)
            assert new['expected_passed'] is False
            assert new['required_violation_ids'] == old['required_violation_ids']
        assert new == expected


def test_revision_source_is_pinned_before_loading(monkeypatch, tmp_path):
    path = tmp_path / 'revision.jsonl'
    path.write_bytes(calibration.CALIBRATION_REVISION_PATH.read_bytes() + b' ')
    monkeypatch.setattr(calibration, 'CALIBRATION_REVISION_PATH', path)
    with pytest.raises(ValueError, match='revision source changed'):
        calibration.load_cases()


@pytest.mark.parametrize('fault', ['dataset_hash', 'case_hash', 'cross_case', 'cross_scene', 'old_allowed',
    'unknown_id', 'missing_extra', 'missing_original_allowed', 'required', 'expected', 'rule_id', 'rule_text',
    'rule_case_hash', 'human_source', 'historical_score', 'extra_field', 'missing_field', 'multiple_revisions'])
def test_revision_cannot_change_labels_case_identity_or_other_rules(monkeypatch, tmp_path, fault):
    record = declaration()
    if fault == 'dataset_hash':record['source_dataset_sha256'] = '0' * 64
    elif fault == 'case_hash':record['source_case_sha256'] = '0' * 64
    elif fault == 'cross_case':record['target_case_id'] = 'unknown_time_unsupported_guarantee_20261103'
    elif fault == 'cross_scene':record['target_scenario_id'] = 'single_profile_game_used_for_query'
    elif fault == 'old_allowed':record['previous_allowed_violation_ids'].pop()
    elif fault == 'unknown_id':record['current_allowed_violation_ids'].append('privacy')
    elif fault == 'missing_extra':record['current_allowed_violation_ids'].remove(EXTRA)
    elif fault == 'missing_original_allowed':record['current_allowed_violation_ids'].pop(0)
    elif fault == 'required':record['preserved_required_violation_ids'] = []
    elif fault == 'expected':record['preserved_expected_passed'] = True
    elif fault == 'rule_id':record['rule_source']['criterion_id'] = 'privacy'
    elif fault == 'rule_text':record['rule_source']['description'] = '任意错误均允许'
    elif fault == 'rule_case_hash':record['rule_source']['case_sha256'] = '0' * 64
    elif fault == 'human_source':record['source']['not_original_human_transcript'] = False
    elif fault == 'historical_score':record['historical_scores_unchanged'] = False
    elif fault == 'extra_field':record['current_expected_passed'] = True
    elif fault == 'missing_field':record.pop('source_case_sha256')
    else:
        path = tmp_path / 'revision.jsonl'
        path.write_text((json.dumps(record) + '\n') * 2)
        monkeypatch.setattr(calibration, 'CALIBRATION_REVISION_PATH', path)
        monkeypatch.setattr(calibration, 'CALIBRATION_REVISION_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
        with pytest.raises(ValueError):calibration.load_cases()
        return
    install_revision(monkeypatch, tmp_path, record)
    with pytest.raises(ValueError):calibration.load_cases()


def test_original_case_edits_or_cross_case_changes_are_rejected(monkeypatch, tmp_path):
    original = calibration.DATASET_PATH.read_bytes()
    for index in (1, 2):
        records = [json.loads(line) for line in original.decode().splitlines() if line.strip()]
        records[index]['reply'] += '篡改原始文本'
        path = tmp_path / f'changed_{index}.jsonl'
        path.write_text('\n'.join(json.dumps(record) for record in records) + '\n')
        monkeypatch.setattr(calibration, 'DATASET_PATH', path)
        with pytest.raises(ValueError):calibration.load_cases()


def test_original_rule_overlap_passes_calibration_without_leaking_revision_labels(tmp_path):
    """离线替身按原三项规则拒绝；新允许集认可该重叠但不影响主原判及金额事实。"""
    real_assess = inquiry.assess_reply_semantics
    def assess(client, **kwargs):
        """仅注入固定受控判决，保留真实协议生成的独立诊断，不访问网络。"""
        result = real_assess(client, **kwargs)
        if kwargs['evidence']['reply'] == '晚上七点2块三缺一，你打吗？':
            next(item for item in result['criteria'] if item['id'] == EXTRA)['passed'] = False
            raw = json.dumps({'reason':result['reason'],'criteria':[
                {'id':item['id'],"verdict":("violated" if not item['passed'] else "clear"),'reason':item['reason']}
                for item in result['criteria']]},ensure_ascii=False)
            result['raw_response'] = raw
            actual = semantic_http_completion(raw, config=inquiry.make_semantic_judge_config(client.config, judge_transport=result["judge_transport"]),
                                              trace_id=result['trace_id'])
            result['attempts'][-1].update(raw_response=raw, completion_diagnostics=completion_diagnostics(actual))
        return result
    with patch.object(inquiry, 'assess_reply_semantics', assess):
        payload, calls = build_scope_stub(tmp_path / 'scope')
    assert payload['status'] == 'passed', [(item['case_id'], item.get('error'), item.get('semantic_assessment', {}).get('error')) for item in payload['reports']]
    target = next(item for item in payload['reports'] if item['case_id'] == TARGET)
    assert target['semantic_assessment']['passed'] is False
    assert EXTRA in target['combined_assessment']['general_violation_ids']
    assert target['money_fact_check']['status'] == 'conflict'
    assert calibration.validate_report(payload, tmp_path / 'report.json')['passed'] == 10
    for request in calls:
        raw = json.dumps(request,ensure_ascii=False)
        for forbidden in ('allowed_violation_ids','required_violation_ids','expected_passed','source_case_sha256',
                          'rule_source','overlapping_violation_20261103',TARGET):
            assert forbidden not in raw
