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
    assert assessment['judge_contract_version']=='responsibility_scope_v9_20261124'
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


class JudgeRetryBudget:
    """为判定补试测试提供有限的无等待共用预算。"""

    def __init__(self, available, *, capped=0.5, active=False):
        self.available = available
        self.capped = capped
        self.active = active
        self.retries = []

    def limit_timeout(self, seconds):
        return min(seconds, self.capped) if self.active else seconds

    def begin_retry(self, failure, *, stage):
        if self.available <= 0:
            return False
        self.available -= 1
        self.active = True
        self.retries.append({**failure, 'stage': stage})
        return True


def install_judge_transport(monkeypatch, outcomes):
    """替换 HTTP 响应，保留生产诊断、截止时间和内容校验。"""
    calls = []
    outcomes = iter(outcomes)
    def transport(request, **kwargs):
        calls.append(json.loads(request.data))
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)
    monkeypatch.setattr(inquiry, 'OpenAICompatibleAgentClient',
                        lambda *, config: OpenAICompatibleAgentClient(config=config, urlopen=transport))
    config = AgentLLMConfig(api_key='unit-only', provider='deepseek', model='requested-model',
                            base_url='https://example.invalid', retry_attempts=1)
    return SimpleNamespace(config=config), calls


def run_budgeted_judge(client):
    return inquiry.assess_reply_semantics(client, evidence={'request': '看看', 'reply': '已查到。'},
                                         expectation=EXPECTED, trace_id='judge-timeout', judge_timeout_seconds=1)


def fixed_verdict(verdict):
    return json.dumps({'reason': '固定测试结果', 'criteria': [
        {'id': 'grounded', 'verdict': verdict, 'reason': '原证据与条目对照。'}]}, ensure_ascii=False)


def test_transport_retries_on_same_input_but_stops_at_valid_negative_verdict(monkeypatch):
    client, calls = install_judge_transport(monkeypatch, [TimeoutError('等待模型超时 unit-only'), fixed_verdict('violated')])
    budget = JudgeRetryBudget(3)
    monkeypatch.setattr(inquiry.eval_retry, 'current_budget', lambda: budget)
    result = run_budgeted_judge(client)

    assert result['passed'] is False and result['error'] is None and len(budget.retries) == 1
    assert len(calls) == 2 and calls[0] == calls[1]
    assert 'eval_inconclusive' not in result
    first, = result['transport_failures']
    assert first['failure']['retryable'] is True and first['trace_id'] == 'judge-timeout'
    assert first['completion_diagnostics']['attempts'][0]['failure_reason'] == 'timeout_waiting_for_response_headers'
    assert 'unit-only' not in json.dumps(result, ensure_ascii=False)
    verdict, = result['attempts']
    assert verdict['trace_id'] == 'judge-timeout:transport_retry_1'
    assert verdict['timeout_seconds'] == 1 and verdict['effective_timeout_seconds'] == 0.5
    assert inquiry.semantic_judge_config_matches(result, require_http=True)
    verdict['effective_timeout_seconds'] = 0.75
    assert not inquiry.semantic_judge_config_matches(result, require_http=True)


@pytest.mark.parametrize('repair_required', [False, True])
def test_resume_after_budget_exhaustion_keeps_frozen_request_even_in_sorted_report(monkeypatch, repair_required):
    outcomes = ['非法结构'] if repair_required else []
    client, calls = install_judge_transport(monkeypatch, [*outcomes, TimeoutError('暂时不可用'), fixed_verdict('clear')])
    budget = JudgeRetryBudget(0)
    monkeypatch.setattr(inquiry.eval_retry, 'current_budget', lambda: budget)
    original = run_budgeted_judge(client)
    frozen = json.loads(json.dumps(original, ensure_ascii=False, sort_keys=True))
    assert frozen['eval_inconclusive']['stage'] == 'judge'
    assert frozen['contract_repair_count'] == int(repair_required)

    budget.available = 1
    changed = deepcopy(frozen)
    changed['evidence']['reply'] = '新回复'
    with pytest.raises(ValueError, match='contract|input changed'):
        inquiry.resume_reply_semantics(client, changed)
    result = inquiry.resume_reply_semantics(client, frozen)

    assert calls[-2] == calls[-1] and len(calls) == len(outcomes) + 2
    assert result['passed'] is True and result['error'] is None
    assert result['transport_failures'] == original['transport_failures'] and not frozen['passed']
    assert len(result['attempts']) == int(repair_required) + 1
    assert inquiry.semantic_judge_config_matches(result, require_http=True)
    with pytest.raises(ValueError, match='no eligible inconclusive result'):
        inquiry.resume_reply_semantics(client, result)


def test_expired_agent_retry_window_does_not_invent_judge_transport_failure(monkeypatch):
    client, calls = install_judge_transport(monkeypatch, [])
    budget = JudgeRetryBudget(3, capped=0, active=True)
    monkeypatch.setattr(inquiry.eval_retry, 'current_budget', lambda: budget)
    result = run_budgeted_judge(client)
    assert result['eval_retry_budget_exhausted'] is True and result['passed'] is False
    assert 'eval_inconclusive' not in result and not calls and not budget.retries


def test_deferred_judge_reuses_completed_agent_evidence_in_new_window(monkeypatch):
    client, calls = install_judge_transport(monkeypatch, [fixed_verdict('clear')])
    budget = JudgeRetryBudget(0, capped=0, active=True)
    budget.last_failure = {'stage': 'agent', 'retryable': True,
                           'category': 'unattributed_timeout', 'reason': 'timeout_waiting_for_response_headers'}
    monkeypatch.setattr(inquiry.eval_retry, 'current_budget', lambda: budget)
    original = run_budgeted_judge(client)
    assert original['eval_inconclusive'] == inquiry.DEFERRED_JUDGE and not calls
    assert original['deferred_judge']['retry_trigger'] == budget.last_failure
    assert not original.get('transport_failures')

    fresh = JudgeRetryBudget(3)
    monkeypatch.setattr(inquiry.eval_retry, 'current_budget', lambda: fresh)
    result = inquiry.resume_reply_semantics(client, original)
    assert result['passed'] is True and len(calls) == len(fresh.retries) == 1
    assert result['deferred_judge'] == original['deferred_judge']
    assert inquiry.semantic_judge_config_matches(result, require_http=True)


def test_inquiry_gate_resumes_blocked_judge_after_an_isolated_agent_recovery(tmp_path, monkeypatch):
    """跨补试窗口保留旧报告、SQLite 和追踪，续跑只发受阻判定。"""
    from scripts.run_evals import validate_inquiry_scope_coverage

    agent_calls = []
    model_calls = []
    service_restored = False
    config = AgentLLMConfig(api_key='unit-only', provider='deepseek', model='requested-model',
                            base_url='https://example.invalid', retry_attempts=1)
    expected = {**EXPECTED, 'product_contract_revision_ids': []}
    monkeypatch.setattr(inquiry, 'CASES', inquiry.CASES[:2])
    monkeypatch.setattr(inquiry, 'inquiry_reply_expectation', lambda *args, **kwargs: expected)
    monkeypatch.setattr(inquiry, 'ROOT', tmp_path)
    monkeypatch.setattr(inquiry, '_inquiry_probe_messages', lambda store, message: [
        {'role': 'system', 'content': '审查固定负例'}, {'role': 'user', 'content': '已冻结的审查输入'}])
    monkeypatch.setattr(inquiry.eval_retry, 'sleep', lambda seconds: None)

    def transport(request, **kwargs):
        trace_id = next(value for key, value in request.header_items() if key.lower() == 'x-trace-id')
        model_calls.append(trace_id)
        if trace_id.startswith('judge:unknown_pool') and not service_restored:
            raise TimeoutError('等待模型响应头')
        if trace_id.startswith('probe:'):
            return Response('{"approved": false, "needs_human": false}')
        return Response(fixed_verdict('clear'))

    class Client:
        def __new__(cls, *, config):
            return OpenAICompatibleAgentClient(config=config, urlopen=transport)

        @staticmethod
        def from_env():
            return Client(config=config)

    def build_runtime(store, mode):
        assert mode == 'real'
        recorder = inquiry.JsonlTraceRecorder(store.path.parent / 'agent_trace.log')

        def handle(message):
            trace_id = f'runtime-{len(agent_calls)}'
            agent_calls.append((message.message_id, store.path))
            recorder.record(trace_id, 'user_input', {'message': message.to_dict()})
            if len(agent_calls) == 1:
                recorder.record(trace_id, 'llm_error', {'completion_diagnostics': {
                    'schema_version': 1, 'route': 'primary', 'error_type': 'RuntimeError',
                    'attempts': [{'failure_reason': 'timeout_waiting_for_response_headers'}]}})
                status, reply = 'failed', ''
            else:
                status, reply = 'completed', '按真实情况答复'
            recorder.record(trace_id, 'final_output', {'reply': reply, 'objective_status': status})
            return SimpleNamespace(trace_id=trace_id, status=status, final_reply=reply, actions=[],
                                   tool_results=[], state_transitions=[])

        return SimpleNamespace(trace_recorder=recorder, handle_user_message=handle,
                               token_budget=SimpleNamespace(max_tokens_per_call=64000)), Client.from_env()

    monkeypatch.setattr(inquiry, 'OpenAICompatibleAgentClient', Client)
    monkeypatch.setattr(inquiry, 'build_runtime', build_runtime)
    monkeypatch.setattr(sys, 'argv', ['run_inquiry_grounding_eval.py', '--run-id', 'inquiry_test'])
    scope = tmp_path / 'retry'
    scope.mkdir()
    monkeypatch.setenv(inquiry.eval_retry.SCOPE_ENV, str(scope))
    monkeypatch.setenv(inquiry.eval_retry.WINDOW_ENV, '0')
    assert inquiry.main() == inquiry.eval_retry.BLOCKED_EXIT_CODE
    root = tmp_path / 'runtime_data/periodic_chat_simulation/runs/inquiry_test'
    old_report = root / 'window_000/report.json'
    archived = old_report.read_bytes()
    blocked = json.loads((root / 'report.json').read_text())
    assert blocked['status'] == 'blocked'
    assert blocked['checks'] == {'unknown_empty': True, 'annotated_original_reply_rejected': True, 'unknown_pool': False}
    assert [case_id for case_id, _ in agent_calls] == ['unknown_empty', 'unknown_empty', 'unknown_pool']
    assert len({path for _, path in agent_calls}) == 3 and all(path.is_file() for _, path in agent_calls)
    failed_judge = blocked['transcript'][-1]['inquiry_assessment']
    assert len(failed_judge['transport_failures']) == 4
    old_source = agent_calls[-1][1].read_bytes()
    calls_before = len(model_calls)

    service_restored = True
    monkeypatch.setenv(inquiry.eval_retry.WINDOW_ENV, '1')
    assert inquiry.main() == 0
    assert len(agent_calls) == 3 and len(model_calls) == calls_before + 1
    assert model_calls[-1] == 'judge:unknown_pool:transport_retry_4'
    assert old_report.read_bytes() == archived and agent_calls[-1][1].read_bytes() == old_source
    final = json.loads((root / 'report.json').read_text())
    assert final['quality_status'] == 'passed' and final['reviewer_probe']['approved'] is False
    phase = {}
    validate_inquiry_scope_coverage(final, phase)
    assert phase['case_coverage']['executed_case_ids'] == ['unknown_empty', 'unknown_pool']
    assert final['transcript'][-1]['inquiry_assessment']['transport_failures'] == failed_judge['transport_failures']


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
