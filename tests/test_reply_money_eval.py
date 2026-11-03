"""金额值冻结于原文，模型只盲判作用域；正确后台事实不能盖过错误客户表述。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from mahjong_agent_runtime import AgentLLMConfig

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))
import reply_money_eval as money


def candidate(*,stake='1',cap=256,extra=False,stake_range=None):
    """来自真实查询结果同一候选的最小事实形状，不给模型真实金额或候选身份。"""
    requirement={'stake':stake,'cap_score':cap}
    if stake_range is not None:requirement={'stake_range':stake_range,'cap_score':cap}
    games=[{'game':{'game_id':'private-candidate-one','requirement':requirement}}]
    if extra:games.append({'game':{'game_id':'private-candidate-two','requirement':{'stake':'2'}}})
    return [{'name':'search_current_games','called':True,'allowed':True,'error':None,'call_id':'actual-query-call',
             'result':{'matches':games}}]


def run_check(monkeypatch,reply,*,scope='existing_option_fact',role='stake',form=None,tools=None,outputs=None):
    """固定作用域输出只验证合同及硬比较，不冒充真实语言分类能力通过。"""
    calls=[];queue=list(outputs) if outputs is not None else None
    class Judge:
        def __init__(self,*,config):
            assert config.temperature==0 and config.thinking_mode=='disabled'
        def complete(self,messages,**kwargs):
            calls.append(deepcopy(messages))
            if queue is not None:
                item=queue.pop(0)
                if isinstance(item,Exception):raise item
                return item
            payload=json.loads(messages[-1]['content'])
            return json.dumps({'items':[{'span_id':span['span_id'],'scope':scope,'role':role,
                'form':form or ('single' if len(money.money_spans(span['literal'])[0]['values'])==1 else 'range'),
                'reason':'固定语义标签只验证接线'} for span in payload['spans']]},ensure_ascii=False)
    monkeypatch.setattr(money,'OpenAICompatibleAgentClient',Judge)
    client=SimpleNamespace(config=AgentLLMConfig(api_key='unit-only',model='unit-model',base_url='https://example.invalid'))
    source=candidate() if tools is None else tools
    result=money.check_reply_money_facts(client,reply=reply,tool_results=source,trace_id='money-test',timeout_seconds=1)
    return result,calls,source


def test_literal_two_is_not_replaced_by_truth_one_and_hard_mismatch_rejects(monkeypatch):
    reply='晚上七点2块三缺一，你打吗？'
    result,calls,tools=run_check(monkeypatch,reply)
    assert len(calls)==1 and result['status']=='conflict' and result['blocks'] is True
    span=result['spans'][0]
    assert reply[span['start']:span['end']]==span['literal']=='2块' and span['values']==['2']
    payload=json.loads(calls[0][-1]['content'])
    assert set(payload)=={'reply','spans'}
    serialized=json.dumps(payload,ensure_ascii=False)
    for hidden in ('candidate','expected','money_facts','actual-query-call','private-candidate-one'):
        assert hidden not in serialized
    row=result['comparisons'][0]
    assert row['scope']=='existing_option_fact' and row['role']=='stake'
    assert row['values']==['2'] and row['expected']['values']==['1'] and row['game_id']=='private-candidate-one'
    assert money.validate_money_fact_check(result,reply=reply,tool_results=tools,trace_id='money-test')


@pytest.mark.parametrize('reply',['晚上七点三缺一，你打吗？','房间有空，你几个人，打多大？'])
def test_omitted_money_adds_no_model_call_or_repeat_obligation(reply):
    result=money.check_reply_money_facts(None,reply=reply,tool_results=candidate(),trace_id='no-money')
    assert result['status']=='not_applicable' and result['blocks'] is False and result['scope_assessment'] is None
    assert money.validate_money_fact_check(result,reply=reply,tool_results=candidate(),trace_id='no-money')


def test_multiple_candidates_are_uncovered_without_adding_single_option_requirement(monkeypatch):
    tools=candidate(extra=True)
    result=money.check_reply_money_facts(None,reply='有1块和2块的。',tool_results=tools,trace_id='multiple')
    assert result['status']=='uncovered' and result['blocks'] is False and result['candidate'] is None
    assert result['scope_assessment'] is None and result['coverage']=='no_unique_verified_candidate'
    assert money.validate_money_fact_check(result,reply='有1块和2块的。',tool_results=tools,trace_id='multiple')


def test_exact_match_and_alternative_are_not_misrepresented_as_one_candidate():
    tools=candidate();tools[0]['result']['alternatives']=candidate(stake='2')[0]['result']['matches']
    result=money.check_reply_money_facts(None,reply='有2块的。',tool_results=tools,trace_id='alternatives')
    assert result['status']=='uncovered' and result['blocks'] is False and result['candidate'] is None
    assert money.validate_money_fact_check(result,reply='有2块的。',tool_results=tools,trace_id='alternatives')


def test_single_actual_alternative_can_be_compared_without_using_query_wishes(monkeypatch):
    tools=candidate(stake='2');tools[0]['result']['alternatives']=tools[0]['result'].pop('matches');tools[0]['result']['matches']=[]
    result,calls,_=run_check(monkeypatch,'有2块的。',tools=tools)
    assert len(calls)==1 and result['status']=='verified' and result['candidate']['source_collection']=='alternatives'
    assert money.validate_money_fact_check(result,reply='有2块的。',tool_results=tools,trace_id='money-test')


@pytest.mark.parametrize('name',['create_game','update_game_requirement','new_unregistered_effect'])
def test_successful_write_after_query_cannot_bind_new_price_to_stale_option(name):
    tools=candidate();tools.append({'name':name,'called':True,'allowed':True,'error':None,
        'result':{'game':{'game_id':'new-or-updated','requirement':{'stake':'2'}}}})
    result=money.check_reply_money_facts(None,reply='这局2块。',tool_results=tools,trace_id='new-price')
    assert result['status']=='uncovered' and result['blocks'] is False and result['scope_assessment'] is None
    assert money.validate_money_fact_check(result,reply='这局2块。',tool_results=tools,trace_id='new-price')


def test_fresh_query_after_write_restores_binding_and_readonly_or_rejected_step_keeps_it(monkeypatch):
    tools=candidate()+[{'name':'update_game_requirement','called':True,'allowed':True,'error':None,'result':{}}]+candidate(stake='2')
    tools.extend([{'name':'check_room_availability','called':True,'allowed':True,'error':None,'result':{}},
                  {'name':'create_game','called':False,'allowed':False,'error':'rejected','result':{}}])
    result,_,_=run_check(monkeypatch,'这局2块。',tools=tools)
    assert result['status']=='verified' and result['candidate']['source_result_index']==2
    assert money.validate_money_fact_check(result,reply='这局2块。',tool_results=tools,trace_id='money-test')


@pytest.mark.parametrize('name',[money.CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,money.CUSTOMER_VISIBLE_TEXT_GENERATION_NAME])
def test_internal_model_evidence_is_readonly_and_does_not_disable_actual_price_check(monkeypatch,name):
    tools=candidate()+[{'name':name,'called':True,'allowed':True,'error':None,'result':{'approved':True}}]
    result,calls,_=run_check(monkeypatch,'这局2块。',tools=tools)
    assert len(calls)==1 and result['status']=='conflict' and result['candidate']['source_result_index']==0
    assert money.validate_money_fact_check(result,reply='这局2块。',tool_results=tools,trace_id='money-test')


def test_opaque_stake_label_remains_uncovered_without_crashing_or_guessing(monkeypatch):
    tools=candidate(stake='小档/某档位')
    result,_,_=run_check(monkeypatch,'有2块的。',tools=tools)
    assert result['status']=='uncovered' and result['blocks'] is False
    assert 'stake' not in result['candidate']['money_facts']
    assert money.validate_money_fact_check(result,reply='有2块的。',tool_results=tools,trace_id='money-test')


@pytest.mark.parametrize('requirement',[{'stake':'NaN'},{'stake':'Infinity'},{'stake':True},
    {'stake_range':{'min':1}},{'stake_range':{'min':1,'max':'not-numeric'}},
    {'stake_range':{'min':3,'max':1}}])
def test_malformed_candidate_money_is_structured_blocking_evidence(requirement):
    tools=candidate();tools[0]['result']['matches'][0]['game']['requirement']=requirement
    result=money.check_reply_money_facts(None,reply='有2块的。',tool_results=tools,trace_id='bad-source')
    assert result['status']=='source_error' and result['blocks'] is True and result['scope_assessment'] is None
    assert result['candidate']['source_errors']
    assert money.validate_money_fact_check(result,reply='有2块的。',tool_results=tools,trace_id='bad-source')
    result.update(status='uncovered',blocks=False)
    assert not money.validate_money_fact_check(result,reply='有2块的。',tool_results=tools,trace_id='bad-source')


@pytest.mark.parametrize('scope',['proposal','condition','negated','quoted','other'])
def test_nonasserted_prices_keep_literals_without_becoming_existing_stake(scope,monkeypatch):
    result,_,tools=run_check(monkeypatch,'2块可以再商量。',scope=scope)
    assert result['status']=='not_applicable' and result['blocks'] is False
    assert result['spans'][0]['values']==['2'] and result['comparisons'][0]['status']=='excluded'
    assert money.validate_money_fact_check(result,reply='2块可以再商量。',tool_results=tools,trace_id='money-test')


@pytest.mark.parametrize('role',['fee','other'])
def test_room_fee_or_other_money_is_not_compared_as_stake(monkeypatch,role):
    result,_,_=run_check(monkeypatch,'房费2元。',role=role)
    assert result['status']=='not_applicable' and not result['blocks']


@pytest.mark.parametrize('reply,role,form,tools',[
    ('有0.50元的。','stake','single',candidate(stake='.5')),
    ('封顶256元。','cap','single',candidate()),
    ('接受1到3块。','stake','range',candidate(stake_range={'min':1,'max':3})),
    ('1-256块这桌。','stake','stake_cap',candidate()),
])
def test_decimal_range_and_base_cap_roles_compare_exact_same_candidate(monkeypatch,reply,role,form,tools):
    result,_,_=run_check(monkeypatch,reply,role=role,form=form,tools=tools)
    assert result['status']=='verified' and not result['blocks']
    assert money.validate_money_fact_check(result,reply=reply,tool_results=tools,trace_id='money-test')


@pytest.mark.parametrize('fault',['drop','duplicate','new_id','extra_value'])
def test_missing_or_changed_span_contract_repairs_once_without_changing_literal(monkeypatch,fault):
    valid={'span_id':'money_0','scope':'existing_option_fact','role':'stake','form':'single','reason':'现成局事实'}
    bad={'items':[deepcopy(valid)]}
    if fault=='drop':bad['items']=[]
    elif fault=='duplicate':bad['items']*=2
    elif fault=='new_id':bad['items'][0]['span_id']='money_other'
    else:bad['items'][0]['amount']='1'
    result,calls,tools=run_check(monkeypatch,'有2块的。',outputs=[json.dumps(bad),json.dumps({'items':[valid]})])
    assert len(calls)==2 and result['scope_assessment']['contract_repair_count']==1
    assert result['status']=='conflict' and result['spans'][0]['values']==['2']
    assert money.validate_money_fact_check(result,reply='有2块的。',tool_results=tools,trace_id='money-test')


@pytest.mark.parametrize('fault',['schema_boolean','literal','span_value','candidate','scope_input','raw','comparison','status','valid_retry'])
def test_validation_recomputes_literal_truth_and_original_scope_response(monkeypatch,fault):
    reply='有2块的。';result,_,tools=run_check(monkeypatch,reply)
    if fault=='schema_boolean':result['schema_version']=True
    elif fault=='literal':result['spans'][0]['literal']='1块'
    elif fault=='span_value':result['spans'][0]['values']=['1']
    elif fault=='candidate':result['candidate']['money_facts']['stake']['values']=['2']
    elif fault=='scope_input':result['scope_assessment']['input']['reply']='有1块的。'
    elif fault=='raw':result['scope_assessment']['items'][0]['scope']='proposal'
    elif fault=='comparison':result['comparisons'][0]['status']='match'
    elif fault=='status':result.update(status='verified',blocks=False)
    else:
        assessment=result['scope_assessment'];first=deepcopy(assessment['attempts'][0]);first['error']='claimed invalid contract'
        last=deepcopy(assessment['attempts'][0]);last['trace_id']+=':contract_repair_1'
        assessment['attempts']=[first,last];assessment['contract_repair_count']=1
    assert not money.validate_money_fact_check(result,reply=reply,tool_results=tools,trace_id='money-test')


def test_valid_scope_is_not_retried_to_get_a_desired_result(monkeypatch):
    result,calls,_=run_check(monkeypatch,'2块可以吗？',scope='uncertain')
    assert len(calls)==1 and result['status']=='unproven' and result['blocks'] is True
