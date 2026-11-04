"""复杂语义判定固定新合同，实际HTTP配置不能被自报配置或主模型配置替代。"""
from copy import deepcopy
from email.message import Message
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_inquiry_grounding_eval as inquiry

EXPECTED={'required_meaning':[{'id':'grounded','description':'只说明有证据支持的事实。'}]}


class Response:
    """仅替代传输；真实客户端负责序列化、HTTP诊断与原文字段提取。"""
    def __init__(self,content):
        self.status=200;self.headers=Message()
        self.body=json.dumps({'model':'returned-model','choices':[{'finish_reason':'stop',
            'message':{'content':content,'reasoning_content':'不得进入判定报告的原始思考'}}]}).encode()
    def __enter__(self):return self
    def __exit__(self,*args):return None
    def read(self):return self.body


def run_http(monkeypatch,*,provider='deepseek',content=None,passed=True,judge_timeout_seconds=1):
    """固定业务判断输出只验证合同，绝不请求真实模型或声称真实语义通过。"""
    calls=[]
    raw=content if content is not None else json.dumps({'reason':'固定单测输出','criteria':[
        {'id':'grounded',"verdict":("violated" if not passed else "clear"),'reason':'仅验证判定合同'}]},ensure_ascii=False)
    def transport(request,**kwargs):
        calls.append(json.loads(request.data));return Response(raw)
    monkeypatch.setattr(inquiry,'OpenAICompatibleAgentClient',lambda *,config:OpenAICompatibleAgentClient(config=config,urlopen=transport))
    config=AgentLLMConfig(api_key='unit-only',provider=provider,model='requested-model',base_url='https://example.invalid',
        temperature=.9,max_tokens=64,thinking_mode='disabled',reasoning_effort=None,retry_attempts=1)
    before=deepcopy(config)
    assessment=inquiry.assess_reply_semantics(SimpleNamespace(config=config),evidence={'request':'看看','reply':'已查到。'},
        expectation=EXPECTED,trace_id='reasoned-config-test',judge_timeout_seconds=judge_timeout_seconds)
    assert config==before
    return assessment,calls


def test_reasoned_v2_is_fixed_in_real_http_payload_and_safe_diagnostics(monkeypatch):
    assessment,calls=run_http(monkeypatch)
    assert len(calls)==1 and assessment['passed'] is True and assessment['error'] is None
    assert assessment['judge_contract_version']=='compact_low_v6_20261104'
    assert calls[0]['thinking']=={'type':'enabled'} and calls[0]['reasoning_effort']=='low'
    assert calls[0]['max_tokens']==16384 and calls[0]['response_format']=={'type':'json_object'}
    assert 'temperature' not in calls[0]  # DeepSeek思考模式按提供方协议不发送采样温度。
    assert inquiry.semantic_judge_config_matches(assessment)
    diagnostics=assessment['attempts'][0]['completion_diagnostics']
    assert diagnostics['request']=={key:inquiry.SEMANTIC_JUDGE_FIXED_CONFIG[key] for key in
        ('max_tokens','response_format','thinking_mode','reasoning_effort')}
    serialized=json.dumps(assessment,ensure_ascii=False)
    assert 'unit-only' not in serialized and 'example.invalid' not in serialized
    assert '不得进入判定报告的原始思考' not in serialized


@pytest.mark.parametrize('fault',['version','temperature','max_tokens','thinking_mode','reasoning_effort','response_format',
    'http_max_tokens','http_thinking_mode','http_reasoning_effort','http_response_format','http_provider','http_model','http_length'])
def test_reasoned_validator_rejects_old_or_mismatched_actual_configuration(monkeypatch,fault):
    assessment,_=run_http(monkeypatch)
    if fault=='version':assessment['judge_contract_version']='nonreasoned_v1'
    elif fault in inquiry.SEMANTIC_JUDGE_FIXED_CONFIG:assessment['judge_config'][fault]='changed'
    else:
        diag=assessment['attempts'][0]['completion_diagnostics']
        if fault=='http_provider':diag['provider']='different-provider'
        elif fault=='http_model':diag['requested_model']='different-model'
        elif fault=='http_length':diag['attempts'][-1]['content']['length']+=1
        else:diag['request'][fault.removeprefix('http_')]='changed'
    assert not inquiry.semantic_judge_config_matches(assessment)


def test_provider_without_actual_reasoning_support_cannot_claim_reasoned_contract(monkeypatch):
    assessment,calls=run_http(monkeypatch,provider='openai')
    assert len(calls)==1 and assessment['passed'] is False and assessment['error']
    assert assessment['contract_repair_count']==0 and not inquiry.semantic_judge_config_matches(assessment)
    assert assessment['attempts'][0]['completion_diagnostics']['request']['thinking_mode'] is None


def test_reasoned_empty_body_keeps_evidence_and_blocks_after_one_contract_repair(monkeypatch):
    assessment,calls=run_http(monkeypatch,content='')
    assert len(calls)==2 and assessment['passed'] is False and assessment['error']
    assert assessment['raw_response']=='' and assessment['contract_repair_count']==1
    assert all(attempt['completion_diagnostics']['attempts'][-1]['content']['empty'] is True for attempt in assessment['attempts'])
    assert inquiry.semantic_judge_config_matches(assessment)


def test_valid_semantic_failure_is_not_retried_under_reasoned_contract(monkeypatch):
    assessment,calls=run_http(monkeypatch,passed=False)
    assert len(calls)==1 and assessment['passed'] is False and assessment['error'] is None
    assert assessment['contract_repair_count']==0 and inquiry.semantic_judge_config_matches(assessment)


def test_live_contract_cannot_silently_fall_back_to_offline_diagnostics(monkeypatch):
    assessment,_=run_http(monkeypatch)
    assert inquiry.semantic_judge_config_matches(assessment,require_http=True)
    assessment['attempts'][0]['completion_diagnostics']={}
    assert inquiry.semantic_judge_config_matches(assessment)
    assert not inquiry.semantic_judge_config_matches(assessment,require_http=True)
    assessment['attempts']=[]
    assert not inquiry.semantic_judge_config_matches(assessment,require_http=True)


@pytest.mark.parametrize('thinking,effort,tokens,temperature',[('disabled',None,64,.9),('enabled','low',8000,.4)])
def test_semantic_factory_does_not_inherit_main_sampling_or_budget(thinking,effort,tokens,temperature):
    original=AgentLLMConfig(api_key='unit-only',model='unit-model',base_url='https://example.invalid',
        thinking_mode=thinking,reasoning_effort=effort,max_tokens=tokens,temperature=temperature)
    result=inquiry.make_semantic_judge_config(original)
    assert result is not original and original.max_tokens==tokens
    assert all(getattr(result,key)==value for key,value in inquiry.SEMANTIC_JUDGE_FIXED_CONFIG.items())


@pytest.mark.parametrize('env,explicit,expected,source',[
    (None,None,180,'default'),('210',None,210,'environment'),('invalid',12,12,'explicit')])
def test_independent_transport_priority_matches_actual_client_and_complete(monkeypatch,env,explicit,expected,source):
    """真实协议替身证明实际单次上限与complete总预算相同，不借主模型60秒配置。"""
    if env is None:monkeypatch.delenv(inquiry.SEMANTIC_JUDGE_TIMEOUT_ENV,raising=False)
    else:monkeypatch.setenv(inquiry.SEMANTIC_JUDGE_TIMEOUT_ENV,env)
    actual=[];original=OpenAICompatibleAgentClient.complete
    def spy(client,messages,**kwargs):
        actual.append((client.config.attempt_timeout_seconds,kwargs['timeout_seconds']))
        return original(client,messages,**kwargs)
    monkeypatch.setattr(OpenAICompatibleAgentClient,'complete',spy)
    assessment,_=run_http(monkeypatch,judge_timeout_seconds=explicit)
    assert assessment['error'] is None and actual==[(expected,expected)]
    assert assessment['judge_transport']=={'total_timeout_seconds':expected,'attempt_timeout_seconds':expected,
        'retry_attempts':1,'budget_source':source}
    assert assessment['attempts'][0]['timeout_seconds']==expected
    assert inquiry.semantic_judge_config_matches(assessment,require_http=True)


@pytest.mark.parametrize('bad',[False,True,0,-1,float('nan'),float('inf'),'invalid'])
def test_invalid_explicit_judge_budget_blocks_before_http(monkeypatch,bad):
    assessment,calls=run_http(monkeypatch,judge_timeout_seconds=bad)
    assert calls==[] and assessment['passed'] is False and assessment['error']
    assert assessment['attempts']==[]


@pytest.mark.parametrize('bad',['','invalid','0','-1','nan','inf'])
def test_invalid_environment_budget_never_silently_falls_back(monkeypatch,bad):
    monkeypatch.setenv(inquiry.SEMANTIC_JUDGE_TIMEOUT_ENV,bad)
    assessment,calls=run_http(monkeypatch,judge_timeout_seconds=None)
    assert calls==[] and assessment['error'] and not assessment['passed']


@pytest.mark.parametrize('fault',['missing','source','total','cap','retries','actual_call','config_cap_bool'])
def test_original_transport_budget_is_bound_and_cannot_be_self_certified(monkeypatch,fault):
    assessment,_=run_http(monkeypatch)
    if fault=='missing':assessment.pop('judge_transport')
    elif fault=='source':assessment['judge_transport']['budget_source']='unknown'
    elif fault=='total':assessment['judge_transport']['total_timeout_seconds']=2
    elif fault=='cap':assessment['judge_transport']['attempt_timeout_seconds']=2
    elif fault=='retries':assessment['judge_transport']['retry_attempts']=True
    elif fault=='actual_call':assessment['attempts'][0]['timeout_seconds']=2
    else:assessment['judge_config']['attempt_timeout_seconds']=True
    assert not inquiry.semantic_judge_config_matches(assessment,require_http=True)


@pytest.mark.parametrize('fault',['missing_projection','wrong_projection_version','changed_projected_reply','changed_original_reply'])
def test_gate_binds_compact_view_to_complete_original_evidence(monkeypatch,fault):
    assessment,calls=run_http(monkeypatch)
    assert json.loads(calls[0]['messages'][1]['content'])['evidence']==assessment['judge_evidence']
    assert assessment['evidence']=={'request':'看看','reply':'已查到。'}
    if fault=='missing_projection':assessment.pop('judge_evidence')
    elif fault=='wrong_projection_version':assessment['judge_input_projection_version']='unreviewed'
    elif fault=='changed_projected_reply':assessment['judge_evidence']['reply']='改写回复'
    else:assessment['evidence']['reply']='另一个回复'
    assert not inquiry.semantic_judge_config_matches(assessment,require_http=True)
