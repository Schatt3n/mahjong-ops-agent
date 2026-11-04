"""旧工具白名单不能替代拒绝/约束证据；记忆等价路径必须由当次工具和Store读回证明。"""
from datetime import timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, InMemoryTraceRecorder, ToolGateway, ToolCall
from mahjong_agent_runtime.models import now
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_real_owner_chat_live_eval as owner


@pytest.fixture(params=['memory','sqlite'])
def case(request,tmp_path):
    store=InMemoryAgentStore() if request.param=='memory' else SQLiteAgentStore(tmp_path/'decline.sqlite3')
    scenario=next(s for s in owner.live_eval_scenarios() if s.scenario_id=='reject_smoking_game_updates_preference')
    scenario.setup(store);message=scenario.message
    store.activate_task_context(conversation_id=message.conversation_id,customer_id=message.sender_id,
        trace_id='current',activity_at=now(),started_at=now(),reason='本次拒绝',force_new=True,archive_previous=False)
    store.append_user_turn(message,'current')
    yield store,scenario
    if request.param=='sqlite':store._connection.close()


def execute(case,*,source='current',value='no_smoking'):
    store,scenario=case;message=scenario.message;trace=InMemoryTraceRecorder()
    result=ToolGateway(store,trace_recorder=trace).execute(ToolCall('record_user_memory',{'task_memories':[
        {'customer_id':message.sender_id,'memory_type':'preference','field':'smoke_preference','value':value,
         'evidence':message.text,'confidence':0.95,'risk_level':'low','scope':'current_task'}]}),
        trace_id=source,conversation_id=message.conversation_id,sender_id=message.sender_id,
        sender_name=message.sender_name,step_index=1,source_message_id=message.message_id)
    assert result.called and result.allowed and not result.error
    trace.record(source,'tool_result',result.to_dict())
    return result,trace.get_trace(source)


def tracking_passes(case,results,events):
    store,scenario=case
    result=SimpleNamespace(trace_id='current',final_reply='好，这桌先不排你。',status='completed',tool_results=results)
    checks=owner.validate_result(result,scenario,events,store,semantic_assessment={'passed':True,'error':None})['checks']
    return next(c for c in checks if c['name']=='should_call_any_state_tracking_tool')['passed']


@pytest.mark.parametrize('value',['no_smoking','no_smoke','无烟'])
def test_real_current_task_memory_is_a_valid_state_tracking_alternative(case,value):
    result,events=execute(case,value=value)
    assert tracking_passes(case,[result],events)


@pytest.mark.parametrize('fault',['no_write','stale_trace','wrong_value','missing_gateway','missing_tool_event','new_task','already_seated','pending_invite'])
def test_memory_alternative_cannot_hide_missing_or_wrong_business_effect(case,fault):
    store,scenario=case;message=scenario.message
    if fault=='no_write':
        assert not tracking_passes(case,[],[]);return
    result,events=execute(case,source='earlier' if fault=='stale_trace' else 'current',value='smoking' if fault=='wrong_value' else 'no_smoking')
    if fault=='missing_gateway':events=[e for e in events if e.step!='tool_gateway_completed']
    elif fault=='missing_tool_event':events=[e for e in events if e.step!='tool_result']
    elif fault=='new_task':
        store.activate_task_context(conversation_id=message.conversation_id,customer_id=message.sender_id,
            trace_id='new',activity_at=now(),started_at=now(),reason='另一个目标',force_new=True,archive_previous=True)
    elif fault=='already_seated':
        game=next(iter(store.games.values()))
        store.join_game(game_id=game.game_id,customer_id=message.sender_id,display_name=message.sender_name,trace_id='joined')
    elif fault=='pending_invite':
        game=next(iter(store.games.values()))
        store.update_game_requirement(game_id=game.game_id,requirement_patch={'planned_start_at':(now()+timedelta(hours=1)).isoformat()},reason='测试真实待处理邀约',trace_id='reschedule')
        store.create_invite_drafts(game_id=game.game_id,invitations=[{'customer_id':message.sender_id,
            'display_name':message.sender_name,'message_text':'这桌打吗？'}],trace_id='invited')
    assert not tracking_passes(case,[result],events)


def test_pure_query_policy_cannot_enable_memory_write_alternative():
    from copy import deepcopy
    import eval_product_contracts as policy
    records=policy.read_product_records();scenarios=deepcopy(owner.live_eval_scenarios())
    target=policy._query_option_revisions(records)[0]['live_scenario_id']
    next(s for s in scenarios if s.scenario_id==target).equivalent_state_tracking_memory={'field':'smoke_preference'}
    with pytest.raises(ValueError,match='cannot enable a constraint-memory write alternative'):
        policy.apply_query_option_policy_to_scenarios(scenarios,records)


def test_smoking_memory_review_preserves_all_previous_golden_bytes():
    """追加验收审查不得改写原始会话、标签或此前已审阅的合同。"""
    import hashlib
    path=Path(__file__).resolve().parents[1]/'eval/golden/real_owner_chat_golden.jsonl'
    prefix=b''.join(path.read_bytes().splitlines(keepends=True)[:17])
    assert hashlib.sha256(prefix).hexdigest()=='d7f33523e2dfe7fe914589f8b8662735291c418eb323ed409e2485b2a72c8106'
