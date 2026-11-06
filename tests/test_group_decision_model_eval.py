"""独立决策模型旁路实验机制；替身只验证输入边界、统计和失败，不证明语义能力。"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location('group_decision_model_eval_test', ROOT/'scripts/run_group_decision_model_eval.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_33_fixed_cases_keep_original_23_and_do_not_feed_labels_or_other_peoples_words():
    module = runner()
    cases = module.load_cases()
    assert len(cases) == 33
    original = json.loads((ROOT/'docs/examples/seven_customer_group_scenario.json').read_text())['messages']
    assert [case['input']['message']['text'] for case in cases[:23]] == [row['text'] for row in original]
    for case in cases:
        request = module.build_request(case)
        text = json.dumps(request, ensure_ascii=False)
        assert 'expected' not in text and 'annotation' not in text and case['case_id'] not in text
        assert len(request['questions']) == 3
        assert request['state']['message']['text'] == case['input']['message']['text']
        assert all(row['customer_id'] == case['input']['message']['customer_id'] for row in request['state']['own_history'])
        assert set(request['questions']['intent']['criteria']) == module.INTENTS


def test_foreign_history_or_freeform_candidate_fields_are_rejected_before_model_call():
    module = runner()
    case = module.load_cases()[0]
    bad = deepcopy(case)
    bad['input']['own_history'] = [{'customer_id':'other', 'text':'其他人的私聊'}]
    with pytest.raises(ValueError, match='own_history'):
        module.build_request(bad)
    bad = deepcopy(case)
    bad['input']['public_topics'] = [{'topic_id':'foreign','game_type':'杭麻','stakes':['1'],'summary':'公开卡片','notes':'其他私密备注'}]
    with pytest.raises(ValueError, match='public_topics'):
        module.build_request(bad)


def test_high_confidence_wrong_link_is_counted_and_never_authorizes_side_effects():
    module = runner()
    case = next(row for row in module.load_cases() if row['case_id']=='explicit_own_target')
    response = {'answers': {key:{'choice':value,'confidence':0.96} for key,value in
        [('intent','update'),('topic','topic_sichuan'),('own_goal','goal_evening')]},
        'usage':{'truncated':False,'state_tokens_dropped':0,'truncated_questions':[]}}
    result = module.score_response(case, response, latency_ms=12)
    assert result['intent_correct'] and not result['topic_correct'] and not result['own_goal_correct']
    assert result['misassociation'] and result['high_confidence_error']
    assert result['side_effects_authorized'] is False


@pytest.mark.parametrize('damage', ['truncation','missing_answer','unknown_choice'])
def test_invalid_or_truncated_outputs_cannot_count_as_pass(damage):
    module = runner()
    case = module.load_cases()[0]
    response = {'answers':{key:{'choice':value,'confidence':0.99} for key,value in
        [('intent','query'),('topic','new'),('own_goal','new')]},
        'usage':{'truncated':False,'state_tokens_dropped':0,'truncated_questions':[]}}
    if damage == 'truncation': response['usage']['state_tokens_dropped'] = 1
    elif damage == 'missing_answer': response['answers'].pop('own_goal')
    else: response['answers']['topic']['choice'] = 'invented-topic'
    result = module.score_response(case, response, latency_ms=10)
    assert not result['valid'] and not result['passed']


def test_localjev_adapter_uses_exact_same_input_and_refuses_remote_endpoint():
    module = runner()
    request = module.build_request(module.load_cases()[0])
    adapted = module.localjev_payload(request)
    assert adapted['state'] == request['state'] and adapted['questions'] == request['questions']
    assert module.validate_local_endpoint('http://127.0.0.1:8080')
    with pytest.raises(ValueError, match='loopback'):
        module.validate_local_endpoint('https://example.com')
