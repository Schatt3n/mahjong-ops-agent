"""精简证据只合并严格相同的副本，不代替原始事实或审计。"""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from semantic_judge_context import project,compact_game,compact_requirement
from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract


def test_compaction_retains_source_identity_permissions_and_follow_up_boundaries():
    request={'text':'几个人了','sender_id':'c','message_id':'m','metadata':{'channel':'group','outbound_enabled':False},'quoted_message':{'text':'七点'}}
    follow={'customer_id':'c','current_task_context_id':'t','observed_at':'2026-11-03T12:00:00+08:00',
        'tasks':[{'task_id':'f','status':'cancelled','expires_at':'2026-11-03T13:00:00+08:00','review_budget':1}]}
    source={'request':request,'reply':'两人','persisted_state':{'follow_up_state':follow},
        'tool_results':[{'name':'create_game','called':False,'allowed':False,'error':'denied','idempotency_key':'audit',
                         'board_arrangement':{'status':'scheduled','publication_sent':False}}]}
    before=deepcopy(source);result=project(source)
    assert source==before and result['request']==request
    assert result['persisted_state']['follow_up_state']==follow
    assert result['tool_results'][0]=={k:v for k,v in source['tool_results'][0].items() if k!='idempotency_key'}


def test_identical_related_games_are_referenced_without_losing_their_facts():
    party={'party_id':'p','seat_count':3,'known_member_ids':['c']}
    game={'game_id':'g','requirement':{'stake':'1','cap_score':256,'requesting_party':party,'seat_claims':[party]},
          'parties':[party],'seat_claims':[party],'status':'forming','remaining_seats':1}
    source={'persisted_state':{'games':[game],'related_active_games':[deepcopy(game)]}}
    result=project(source)['persisted_state'];one=result['games'][0]
    assert result['related_active_game_ids']==['g'] and 'related_active_games' not in result
    assert one['parties']==[party] and 'seat_claims' not in one
    assert one['requirement']=={'stake':'1','cap_score':256}


@pytest.mark.parametrize('different',[{'seat_count':2},{'seat_count':True},{'known_member_ids':['another']}])
def test_conflicting_occupancy_is_never_erased(different):
    party={'party_id':'p','seat_count':1,'known_member_ids':['c']};other={**party,**different}
    game={'parties':[party],'seat_claims':[other],'requirement':{'requesting_party':party,'seat_claims':[other]}}
    result=compact_game(game)
    assert result['seat_claims']==[other] and result['requirement']['seat_claims']==[other]


def test_same_game_id_with_different_amount_is_not_a_duplicate():
    source={'persisted_state':{'games':[{'game_id':'g','stake':'1'}],'related_active_games':[{'game_id':'g','stake':'2'}]}}
    assert project(source)==source
    assert project({'persisted_state':{}})=={'persisted_state':{}}


def test_only_recognized_backend_explanations_are_omitted():
    contract=current_game_search_reply_contract({'stake':'1'},[])
    source={'tool_results':[{'customer_reply_contract':contract}]};view=project(source)['tool_results'][0]['customer_reply_contract']
    assert 'claim_boundary' not in view['evidence_scope']
    assert view['evidence_scope']['supports_narrower_empty_claim']==contract['evidence_scope']['supports_narrower_empty_claim']
    custom=deepcopy(source);custom['tool_results'][0]['customer_reply_contract']['evidence_scope']['claim_boundary']='仅限这一家门店，禁止扩大范围'
    assert project(custom)['tool_results'][0]['customer_reply_contract']['evidence_scope']['claim_boundary']=='仅限这一家门店，禁止扩大范围'
