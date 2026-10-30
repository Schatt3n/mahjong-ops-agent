"""咨询评测的真实工具和领域前后态，必须让暗中写状态也可被确定性门禁识别。"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import pytest
from mahjong_agent_runtime import CustomerProfile,InMemoryAgentStore,SQLiteAgentStore,ToolCall,ToolGateway
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_inquiry_grounding_eval import inquiry_business_state,inquiry_business_side_effect_checks,project_inquiry_tool_results
from mahjong_agent_runtime.domains.tools.audit_tools import update_context_checkpoint
from mahjong_agent_runtime.domains.tools.search_tools import search_current_games


@pytest.fixture(params=['memory','sqlite'])
def store(request,tmp_path):
    value=InMemoryAgentStore() if request.param=='memory' else SQLiteAgentStore(tmp_path/'evidence.sqlite3')
    yield value
    if request.param=='sqlite':value._connection.close()


def _state(store):
    return inquiry_business_state(store,customer_id='customer',conversation_id='current')


def _create(store,customer='customer',conversation='current'):
    return store.create_game(conversation_id=conversation,organizer_id=customer,organizer_name='PRIVATE_REMARK',
        requirement={'game_type':'national_standard_mahjong','stake_range':{'min':1,'max':3},'start_time_kind':'asap_when_full',
                     'internal_note':'PRIVATE_NOTE','user_visible_summary':'国标1至3块人齐开'},
        known_players=[{'customer_id':customer,'display_name':'PRIVATE_REMARK','seat_count':1}],trace_id='fixture')[0]


def test_read_query_projects_real_evidence_without_other_customer_private_fields(store):
    _create(store,customer='other',conversation='another-private')
    before=_state(store)
    result=search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':{}}),
                                'query','current','customer','客户')
    raw=[result.to_dict()];projected=project_inquiry_tool_results(raw)
    assert projected[0]['name']=='search_current_games' and projected[0]['called'] is True
    assert projected[0]['result']['matches'][0]['public_requirement']['stake_range']=={'min':1.0,'max':3.0}
    serialized=json.dumps(projected,ensure_ascii=False)
    assert 'PRIVATE_' not in serialized and 'participants' not in serialized and 'organizer_id' not in serialized
    assert before['customer_related_games']==[]
    checks=inquiry_business_side_effect_checks(raw,{'before':before,'after':_state(store)})
    assert all(row['passed'] for row in checks)


def test_silent_domain_write_fails_even_without_claimed_tool_or_changed_reply(store):
    before=_state(store);_create(store)
    checks=inquiry_business_side_effect_checks([],{'before':before,'after':_state(store)})
    by_name={row['name']:row for row in checks}
    assert by_name['no_unauthorized_business_tool_execution']['passed']
    assert not by_name['inquiry_business_state_unchanged']['passed']


def test_actual_write_tool_fails_even_if_state_projection_is_falsely_unchanged(store):
    before=_state(store)
    result=ToolGateway(store).execute(ToolCall(name='create_game',arguments={
        'organizer_id':'customer','organizer_name':'客户','requirement':{'game_type':'national_standard_mahjong','stake':'1','start_time_kind':'asap_when_full'},
        'known_players':[{'customer_id':'customer','display_name':'客户','seat_count':1}]}),
        trace_id='write-negative',conversation_id='current',sender_id='customer',sender_name='客户',step_index=1)
    assert result.called and result.allowed
    checks=inquiry_business_side_effect_checks([result.to_dict()],{'before':before,'after':before})
    assert not next(row for row in checks if row['name']=='no_unauthorized_business_tool_execution')['passed']
    assert project_inquiry_tool_results([result.to_dict()])[0]['name']=='create_game'


def test_legal_profile_and_checkpoint_records_are_not_business_side_effects(store):
    before=_state(store)
    store.upsert_customer(CustomerProfile(customer_id='customer',display_name='客户',preferred_games=['national_standard_mahjong']))
    result=update_context_checkpoint(store,ToolCall(name='update_context_checkpoint',arguments={
        'summary':'已知玩法待客户选择','facts':{'game_type':'national_standard_mahjong'},'open_questions':['档位']}),
        'checkpoint','current','customer','客户')
    assert store.get_conversation_checkpoint('current') is not None
    checks=inquiry_business_side_effect_checks([result.to_dict()],{'before':before,'after':_state(store)})
    assert all(row['passed'] for row in checks)


def test_domain_snapshot_is_immutable_and_contains_no_peer_identity(store):
    game=_create(store);before=_state(store)
    store.update_game_requirement(game_id=game.game_id,requirement_patch={'stake_range':{'min':2,'max':4}},reason='负例状态变更',trace_id='change')
    assert before['customer_related_games'][0]['public_requirement']['stake_range']=={'min':1.0,'max':3.0}
    assert _state(store)['customer_related_games'][0]['public_requirement']['stake_range']=={'min':2.0,'max':4.0}
    assert 'PRIVATE_' not in json.dumps(before)
