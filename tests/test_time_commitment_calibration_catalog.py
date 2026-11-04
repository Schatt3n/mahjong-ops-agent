"""时间负例与正例共享真实空房证据；不能只查错误保证而误杀条件或不确定答复。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_reply_scope_calibration as scope


def test_original_labels_and_inputs_remain_intact_and_time_boundaries_extend_coverage():
    assert hashlib.sha256(scope.DATASET_PATH.read_bytes()).hexdigest()=='4368147698e28ac8ee6ae11cc4aacab9b066d1cb332053b7cc4192b5b704d24d'
    cases=scope.load_cases()
    assert len(cases)==10
    extra=cases[4:]
    assert len(extra)==6 and sum(c['expected_passed'] for c in extra)==3
    assert all(c['source']['not_original_human_transcript'] is True for c in extra)
    assert len({c['case_id'] for c in cases})==len(cases)
    assert all(c['scenario_id']=='single_profile_game_used_for_query' for c in extra)
