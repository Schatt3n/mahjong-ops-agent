"""动作内引用错误须先整体拒绝，后续安全恢复不得豁免业务或权限错误。"""
from copy import deepcopy
import json

import pytest

from mahjong_agent_runtime.action_contract import validate_action_contract
from tests.test_action_contract_recovery_audit import action_text, run_recovery_case, audit


def scoped_action():
    action = json.loads(action_text(query=True))
    action['tool_calls'][0].update(call_id='read', depends_on=[])
    action['tool_calls'].append({'name':'create_game', 'arguments':{'requirement':{'stake':'1'}},
                                 'reason':'引用一致后才可提出建局', 'call_id':'create_game_call', 'depends_on':['read']})
    action['objective_state'] = {'fact_assertions':[{'field':'stake','value':'1','role':'current',
        'source_message_id':'source-input','scope':{'call_id':'create_call'}}]}
    return action


def test_cross_field_reference_error_is_rejected_as_action_contract_not_silently_renamed():
    action=scoped_action();before=deepcopy(action)
    errors=validate_action_contract(action)
    assert any('scope.call_id' in error and 'create_call' in error and 'create_game_call' in error for error in errors)
    assert action==before
    action['tool_calls'][1]['call_id']='create_call'
    assert not validate_action_contract(action)


@pytest.mark.parametrize('scope',[None,'create_call',{'tool':'create_call'},{'call_id':''},{'game_id':3}])
def test_malformed_scope_does_not_disable_fact_binding(scope):
    action=scoped_action();action['objective_state']['fact_assertions'][0]['scope']=scope
    assert validate_action_contract(action)


def test_rejected_batch_executes_no_read_or_write_and_only_full_trace_can_prove_recovery():
    result,events,_=run_recovery_case(rejected=(json.dumps(scoped_action(),ensure_ascii=False),))
    error_index=next(i for i,e in enumerate(events) if e.step=='action_contract_error')
    next_response=next(i for i,e in enumerate(events) if i>error_index and e.step=='llm_response')
    assert not any(e.step in {'tool_called','tool_gateway_received','state_transition','final_output'}
                   for e in events[error_index:next_response])
    metrics=audit(result,events)
    assert metrics['action_contract_error_count']==metrics['safe_recovered_count']==1
    assert metrics['quality_status']=='passed'
    missing=[e for e in events if e.step!='contract_error_feedback']
    assert audit(result,missing)['quality_status']=='failed'
