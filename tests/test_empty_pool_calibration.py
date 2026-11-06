"""空池推论校准使用真实网关和独立 HTTP 替身，固定判决仅检验验真接线。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from mahjong_agent_runtime import AgentLLMConfig
from tests.semantic_http_stub import semantic_http_completion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_empty_pool_calibration as runner
import run_inquiry_grounding_eval as inquiry


@pytest.fixture
def calibration(tmp_path, monkeypatch):
    cases=runner.load_cases();calls=[]
    class Judge:
        def __init__(self,*,config):self.config=config
        def complete(self,messages,**kwargs):
            request=json.loads(messages[-1]['content']);case=cases[len(calls)];calls.append(request)
            assert 'expected_passed' not in json.dumps(request)
            assert request['evidence']['reply']==case['reply']
            raw=json.dumps({'reason':'离线固定判决仅验接线','criteria':[{'id':item['id'],'reason':'原始范围证据',
                'verdict':'clear' if case['expected_passed'] else 'violated'} for item in request['expectation']['checks']]})
            return semantic_http_completion(raw,config=self.config,trace_id=kwargs['trace_id'])
    monkeypatch.setattr(inquiry,'OpenAICompatibleAgentClient',Judge)
    client=SimpleNamespace(config=AgentLLMConfig(api_key='offline-only',model='unit-model',provider='deepseek',base_url='https://example.invalid'))
    reports=[runner.run_case(case,client,tmp_path/case['case_id']) for case in cases]
    payload={'status':'passed','execution_scope':'controlled_empty_pool_semantic_calibration',
        'dataset_sha256':runner.hashlib.sha256(runner.DATASET.read_bytes()).hexdigest(),'reports':reports}
    runner.validate_scope(payload,tmp_path/'report.json')
    return payload,tmp_path/'report.json',calls


def test_real_gateway_proves_whole_empty_and_keeps_nonempty_false(calibration):
    payload,_,calls=calibration
    assert len(calls)==6 and len(payload['reports'])==6
    reports=payload['reports']; scope=lambda report:report['query']['result']['customer_reply_contract']['evidence_scope']
    assert scope(reports[0])['supports_narrower_empty_claim'] is True
    assert 'max_duration_hours' not in reports[0]['source']['requirement']
    for report in reports[1:4]:assert scope(report)['supports_narrower_empty_claim'] is False
    assert reports[4]['query']['called'] is False and reports[4]['query']['error']
    assert all(report['status']=='passed' for report in reports)
    assert all('empty_inventory_implication' in call['expectation']['criterion_evidence_contract'] for call in calls)


@pytest.mark.parametrize('fault',['input','reply','query','state','source','label','missing_call'])
def test_calibration_rejects_mismatched_actual_evidence(calibration,fault):
    payload,path,_=calibration; payload=deepcopy(payload);report=payload['reports'][0]
    if fault=='input':report['semantic_assessment']['evidence']['request']='other'
    elif fault=='reply':report['semantic_assessment']['evidence']['reply']='other'
    elif fault=='query':report['query']['result']['requirement']['max_duration_hours']=1
    elif fault=='state':report['state']['games']=[{}]
    elif fault=='source':report['source']['request']='other'
    elif fault=='label':report['semantic_assessment']['passed']=False
    else:report['events']=[event for event in report['events'] if event['step']!='tool_gateway_completed']
    with pytest.raises(ValueError):runner.validate_scope(payload,path)
