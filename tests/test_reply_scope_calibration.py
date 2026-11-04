"""四个原场景正反校准只替代远端判定，真实SQLite、Gateway及严格证据核验不可替代。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import patch
from tests.semantic_http_stub import semantic_http_completion

import pytest
from mahjong_agent_runtime import AgentLLMConfig
from mahjong_agent_runtime.llm import completion_diagnostics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_reply_scope_calibration as calibration
import run_inquiry_grounding_eval as inquiry
import run_real_owner_chat_live_eval as owner
import reply_money_eval as money


def build_scope_stub(work_dir, *, general_misses_money=False, money_scope='existing_option_fact', money_role='stake'):
    """受控标签只产生模型替身响应，不把离线结果声明为真实语义通过。"""
    cases=calibration.load_cases();calls=[]
    class Judge:
        """只替换模型输出，实际证据与解析仍由生产评测组件构建。"""
        def __init__(self,config):
            self.config=config
            assert config.temperature==0 and config.thinking_mode=='enabled'
            assert config.reasoning_effort=='low' and config.max_tokens==16384
        def complete(self,messages,**kwargs):
            request=json.loads(messages[-1]['content']);case=cases[len(calls)];calls.append(request)
            raw=json.dumps({'reason':'固定输出仅验收连线','criteria':[
                {'id':item['id'],"verdict":("violated" if item['id'] in case['required_violation_ids'] and not (
                    general_misses_money and item['id'] in calibration.MONEY_CRITERION_ROLES) else "clear"),'reason':'受控标签非真实语义结论'}
                for item in request['expectation']['checks']]},ensure_ascii=False)
            return semantic_http_completion(raw,config=self.config,trace_id=kwargs['trace_id'])
    class MoneyJudge:
        """只提供作用域，真实2块原字面及候选1块均由后端保留。"""
        def __init__(self,config):
            assert config.temperature==0 and config.thinking_mode=='disabled'
        def complete(self,messages,**kwargs):
            request=json.loads(messages[-1]['content'])
            assert set(request)=={'reply','spans'}
            return json.dumps({'items':[{'span_id':item['span_id'],'scope':money_scope,'role':money_role,
                'form':'single','reason':'固定作用域非真实语义通过'} for item in request['spans']]},ensure_ascii=False)
    def forbidden(*args,**kwargs):
        raise AssertionError('scope must not call the main Agent')
    client=SimpleNamespace(config=AgentLLMConfig(api_key='unit-only',provider='deepseek',model='unit-model',base_url='https://example.invalid'),complete=forbidden)
    with patch.object(inquiry,'OpenAICompatibleAgentClient',Judge),patch.object(money,'OpenAICompatibleAgentClient',MoneyJudge):
        result=calibration.run_calibration(client,SimpleNamespace(timeout_seconds=1),work_dir)
    return result,calls


@pytest.fixture
def scope(tmp_path):
    root=tmp_path/'current';root.mkdir()
    payload,calls=build_scope_stub(root/'scope')
    return payload,calls,root/'report.json'


def test_scope_runs_four_original_scenario_boundaries_without_label_or_plan_leak(scope):
    payload,calls,report_path=scope
    assert payload['status']=='passed',[(item['case_id'],item.get('error')) for item in payload['reports']]
    assert len(calls)==10 and sum(item['semantic_assessment']['passed'] for item in payload['reports'])==5
    sources=calibration.load_cases();scenarios=calibration.original_scenarios()
    for source,item,request in zip(sources,payload['reports'],calls):
        assert item['expectation']==owner.reply_semantic_expectation(scenarios[source['scenario_id']])
        serialized=json.dumps(request,ensure_ascii=False)
        for marker in (source['case_id'],'expected_passed','required_violation_ids','allowed_violation_ids','objective_plan','reasoning_summary'):
            assert marker not in serialized
        assert item['read_only'] is True
        raw=json.loads(Path(item['raw_trace_path']).read_text())
        assert raw['before_judge']==raw['after_judge']==calibration.sqlite_snapshot(Path(item['db_path']))
        assert raw['tool_result']['called'] and raw['tool_result']['allowed'] and not raw['tool_result']['error']
    coverage=calibration.validate_report(payload,report_path)
    assert coverage['total']==coverage['passed']==10


@pytest.mark.parametrize('fault',[
    'missing_case','duplicate_case','changed_source','changed_hash','claimed_main','numeric_count','missing_trace','outside_trace',
    'missing_gateway','missing_result','missing_assessment_event','changed_call','hidden_sqlite_write','changed_history',
    'changed_actual_reply','changed_known_fact','changed_expectation','missing_criterion','duplicate_criterion','numeric_criterion',
    'false_positive','unrelated_negative','extra_privacy_failure','changed_rubric','changed_configuration','missing_attempt',
    'changed_attempt_trace','changed_attempt_raw','valid_business_retried','missing_http','old_judge_version','changed_http_config',
])
def test_scope_gate_rejects_incomplete_or_self_certified_sources(scope,tmp_path,fault):
    """改动仅针对临时证据；来源、SQL、工具、原判定及正反边界必须逐项重验。"""
    payload,_,report_path=scope
    item=payload['reports'][1];path=Path(item['raw_trace_path']);raw=json.loads(path.read_text())
    if fault=='false_positive':
        item=payload['reports'][3];path=Path(item['raw_trace_path']);raw=json.loads(path.read_text())
    assessment=item['semantic_assessment']
    if fault=='missing_case':
        payload['reports'].pop()
    elif fault=='duplicate_case':
        payload['reports'][-1]=deepcopy(item)
    elif fault=='changed_source':
        item['source']['expected_passed']=True
    elif fault=='changed_hash':
        payload['dataset_sha256']='0'*64
    elif fault=='claimed_main':
        payload['main_agent_executed']=True
    elif fault=='numeric_count':
        payload['passed_count']=True
    elif fault=='missing_trace':
        item.pop('raw_trace_path')
    elif fault=='outside_trace':
        outside=tmp_path/'old_trace.json';outside.write_text(path.read_text());item['raw_trace_path']=str(outside)
    elif fault in {'missing_gateway','missing_result','missing_assessment_event'}:
        step={'missing_gateway':'tool_gateway_completed','missing_result':'tool_result','missing_assessment_event':'reply_scope_assessment'}[fault]
        raw['events']=[event for event in raw['events'] if event['step']!=step]
    elif fault=='changed_call':
        raw['tool_call']['arguments']['requirement']['stake']='2';item['tool_call']=deepcopy(raw['tool_call'])
    elif fault=='hidden_sqlite_write':
        with sqlite3.connect(item['db_path']) as connection:
            connection.execute('INSERT INTO runtime_rooms(room_id,updated_at) VALUES(?,?)',('hidden','2026-11-03T00:00:00+08:00'))
    elif fault in {'changed_history','changed_actual_reply','changed_known_fact'}:
        evidence=assessment['evidence']
        if fault=='changed_history':evidence['recent_conversation'].pop(0)
        elif fault=='changed_actual_reply':evidence['reply']='更容易通过的替代回复'
        else:evidence['known_profile']['preferred_stakes']=['2']
        raw['evidence']=deepcopy(evidence)
    elif fault=='changed_expectation':
        item['expectation']['required_meaning'].pop();raw['expectation']=deepcopy(item['expectation'])
        assessment['expectation']=deepcopy(item['expectation'])
    elif fault=='changed_rubric':assessment['rubric']='another judge rubric'
    elif fault=='changed_configuration':assessment['judge_config']['thinking_mode']='disabled'
    elif fault=='missing_attempt':assessment['attempts']=[]
    elif fault=='missing_http':assessment['attempts'][0]['completion_diagnostics']={}
    elif fault=='old_judge_version':assessment['judge_contract_version']='nonreasoned_v1'
    elif fault=='changed_http_config':assessment['attempts'][0]['completion_diagnostics']['request']['max_tokens']=4000
    elif fault=='changed_attempt_trace':assessment['attempts'][0]['trace_id']='other-call'
    elif fault=='changed_attempt_raw':assessment['attempts'][0]['raw_response']='{}'
    elif fault=='valid_business_retried':
        first=deepcopy(assessment['attempts'][0]);first['error']='claimed invalid contract'
        last=deepcopy(assessment['attempts'][0]);last['trace_id']+=':contract_repair_1'
        assessment['attempts']=[first,last];assessment['contract_repair_count']=1
    else:
        criteria=assessment['criteria']
        if fault=='missing_criterion':criteria.pop()
        elif fault=='duplicate_criterion':criteria.append(deepcopy(criteria[0]))
        elif fault=='numeric_criterion':criteria[0]['passed']=1
        elif fault=='false_positive':
            for criterion in criteria:criterion['passed']=True
        elif fault=='unrelated_negative':
            for criterion in criteria:criterion['passed']=criterion['id']!='privacy'
        elif fault=='extra_privacy_failure':
            next(criterion for criterion in criteria if criterion['id']=='privacy')['passed']=False
        assessment['passed']=all(criterion['passed'] for criterion in criteria)
        assessment['raw_response']=json.dumps({'reason':assessment['reason'],'criteria':[
            {'id':criterion['id'],"verdict":("violated" if not criterion['passed'] else "clear"),'reason':criterion['reason']} for criterion in criteria]})
        assessment['attempts'][-1].update(raw_response=assessment['raw_response'],passed=assessment['passed'])
    raw['semantic_assessment']=deepcopy(assessment)
    for event in raw['events']:
        if event['step']=='reply_scope_assessment':event['content']=deepcopy(assessment)
    item['trace_steps']=[event['step'] for event in raw['events']]
    path.write_text(json.dumps(raw))
    with pytest.raises((ValueError,TypeError)):
        calibration.validate_report(payload,report_path)


def test_explicit_wrong_stake_is_rejected_even_when_original_judge_ignores_reply_literal(tmp_path):
    """复现真实general误判，独立硬证据拒绝2块，不篡改原judge的passed=True。"""
    result,_=build_scope_stub(tmp_path/'scope',general_misses_money=True)
    item=result['reports'][1]
    assert result['status']=='passed' and item['semantic_assessment']['passed'] is True
    assert all(row['passed'] for row in item['semantic_assessment']['criteria'])
    check=item['money_fact_check'];assert check['status']=='conflict' and check['blocks'] is True
    row=check['comparisons'][0]
    assert row['literal']=='2块' and row['values']==['2'] and row['expected']['values']==['1']
    assert row['scope']=='existing_option_fact' and row['role']=='stake'
    assert item['combined_assessment']['general_violation_ids']==[]
    assert item['combined_assessment']['money_fact_violation_ids']==calibration.load_cases()[1]['required_violation_ids']
    assert calibration.validate_report(result,tmp_path/'report.json')['passed']==10


@pytest.mark.parametrize('scope,role',[('proposal','stake'),('uncertain','stake'),('existing_option_fact','fee')])
def test_wrong_stake_calibration_requires_actual_scoped_numeric_conflict(tmp_path,scope,role):
    """错误作用域、不确定或房费不能冒充识别了现成局底注错报。"""
    result,_=build_scope_stub(tmp_path/'scope',general_misses_money=True,money_scope=scope,money_role=role)
    assert result['reports'][1]['status']=='failed' and result['status']=='failed'
    with pytest.raises(ValueError):calibration.validate_report(result,tmp_path/'report.json')


@pytest.mark.parametrize('fault',['missing_event','replace_literal','replace_truth','remove_conflict','hide_original_general'])
def test_scope_gate_recomputes_independent_money_proof_instead_of_self_report(scope,fault):
    payload,_,report_path=scope;item=payload['reports'][1];path=Path(item['raw_trace_path']);raw=json.loads(path.read_text())
    check=item['money_fact_check']
    if fault=='missing_event':raw['events']=[event for event in raw['events'] if event['step']!='reply_money_fact_check']
    elif fault=='replace_literal':check['spans'][0]['literal']='1块'
    elif fault=='replace_truth':check['candidate']['money_facts']['stake']['values']=['2']
    elif fault=='remove_conflict':check.update(status='verified',blocks=False)
    else:item['combined_assessment']['general_semantic_passed']=True
    raw['money_fact_check']=deepcopy(check);raw['combined_assessment']=deepcopy(item['combined_assessment'])
    for event in raw['events']:
        if event['step']=='reply_money_fact_check':event['content']=deepcopy(check)
    item['trace_steps']=[event['step'] for event in raw['events']];path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):calibration.validate_report(payload,report_path)


def test_scope_allows_only_one_original_invalid_contract_repair(scope):
    """无效JSON可有限修复，合法语义失败不会被当作结构错误重判。"""
    payload,_,report_path=scope
    for item in payload['reports']:
        assessment=item['semantic_assessment'];final=deepcopy(assessment['attempts'][0]);final['trace_id']+=':contract_repair_1'
        config=AgentLLMConfig(api_key='unit-only',model='unit-model',provider='deepseek',base_url='https://example.invalid',
            **inquiry.SEMANTIC_JUDGE_FIXED_CONFIG,
            attempt_timeout_seconds=assessment['judge_transport']['attempt_timeout_seconds'],
            retry_attempts=assessment['judge_transport']['retry_attempts'])
        invalid=semantic_http_completion('{',config=config,trace_id=assessment['trace_id'])
        assessment['attempts']=[{'trace_id':assessment['trace_id'],'raw_response':'{','error':'invalid JSON',
            'completion_diagnostics':completion_diagnostics(invalid),
            'timeout_seconds':assessment['judge_transport']['total_timeout_seconds']},final]
        assessment['contract_repair_count']=1
        path=Path(item['raw_trace_path']);raw=json.loads(path.read_text());raw['semantic_assessment']=deepcopy(assessment)
        next(event for event in raw['events'] if event['step']=='reply_scope_assessment')['content']=deepcopy(assessment)
        path.write_text(json.dumps(raw))
    assert calibration.validate_report(payload,report_path)['passed']==10


def test_judge_rules_keep_real_mismatches_and_allow_only_originally_permitted_clarification():
    """通用规则只澄清证据归属，不取消原case的错档、未知精确时间或重问已知禁令。"""
    rule=inquiry.CRITERION_EVIDENCE_CONTRACT
    assert '真实参数、返回选项' in rule['interpretation_and_adoption']
    assert '事实不足且原case允许澄清' in rule['customer_communication']
    assert '具体询问真正缺项就是合理回应' in rule['customer_communication']
    assert '错档、错时间、重问已知条件、捏造结论及无关敷衍' in rule['customer_communication']
    scenarios=calibration.original_scenarios()
    original=owner.reply_semantic_expectation(scenarios['single_profile_game_used_for_query'])
    forbidden=' '.join(item['description'] for item in original['forbidden_meaning'])
    assert '精确最快时间' in forbidden and '不能再问客户想打什么麻将' in forbidden
    assert '未知底注可直接询问' in original['required_meaning'][0]['description']


def test_new_scope_is_explicit_in_full_plan_and_revalidated_by_gate(scope,tmp_path):
    """完整门禁独立新增四例校准，旧八例scope仍在且不被替代。"""
    spec=importlib.util.spec_from_file_location('reply_scope_gate',ROOT/'scripts/run_evals.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    args=SimpleNamespace(full=True,live_real_owner=False,live_real_group=False,live_concurrency=False)
    plan=gate.build_eval_plan(args,tmp_path/'gate-run')
    phases=[item for item in plan if item['scope']=='reply_scope_calibration']
    assert len(phases)==1 and 'scripts/run_reply_scope_calibration.py' in phases[0]['command'] and '--strict' in phases[0]['command']
    assert any(item['scope']=='reply_semantic_calibration' for item in plan)
    payload,_,report_path=scope;report_path.write_text(json.dumps(payload))
    phase={'scope':'reply_scope_calibration','report_path':str(report_path)}
    gate.validate_scope_report(phase)
    assert phase['case_coverage']['total']==phase['case_coverage']['passed']==10
