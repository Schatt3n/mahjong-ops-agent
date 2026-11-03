"""人数查询只强制准确人数，不强制复读已知时刻或新增参加征询。"""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_real_owner_chat_live_eval as owner
from eval_product_contracts import read_product_records,validate_reply_presentation_policy


def test_count_only_query_preserves_business_contract_without_redundant_anchors():
    scene=next(s for s in owner.live_eval_scenarios() if s.scenario_id=='later_people_count_query')
    assert scene.required_reply_any==[['两个人','2个人','两个']]
    assert scene.expected_active_game_count==1
    assert scene.expected_active_game_seat_summary=={'claimed_seats':2,'remaining_seats':2}
    assert 'create_game' in scene.forbidden_tool_names and 'create_invite_drafts' in scene.forbidden_tool_names
    expected=owner.reply_semantic_expectation(scene)
    ids={item['id'] for item in expected['required_meaning']}
    assert 'scenario_reply_alternatives_0' in ids
    assert not {'scenario_reply_alternatives_1','scenario_reply_alternatives_2'}&ids
    assert 'count_query_optional_known_context_20261103' in expected['product_contract_revision_ids']


@pytest.mark.parametrize('fault',['remove_people','change_count','remove_privacy','change_source','change_original'])
def test_count_only_revision_cannot_relax_number_tools_or_privacy(fault):
    records=deepcopy(read_product_records());record=next(r for r in records if r.get('id')=='count_query_no_redundant_anchors_review_20261103');revision=record['revisions'][0]
    if fault=='remove_people':revision['current_scenario_contract']['required_reply_any']=[]
    elif fault=='change_count':revision['current_scenario_contract']['required_reply_any']=[['三个人']]
    elif fault=='remove_privacy':revision['current_scenario_contract']['forbidden_reply_contains']=[]
    elif fault=='change_source':revision['source_case_sha256']='0'*64
    else:revision['current_expectation']['should_not_create_game']=False
    with pytest.raises(ValueError):validate_reply_presentation_policy(records)


@pytest.mark.parametrize('scenario_id,removed',[
    ('accept_existing_offer_marks_game_ready',{'七点','无烟','0.5'}),
    ('duration_limit_update_should_persist',{'5小时','五小时'}),
    ('resume_status_after_casual_chat',{'0.5','无烟'})])
def test_accurate_echo_is_not_a_literal_hard_failure_but_business_state_remains(scenario_id,removed):
    scene=next(s for s in owner.live_eval_scenarios() if s.scenario_id==scenario_id)
    assert not removed&set(scene.forbidden_reply_contains)
    expected=owner.reply_semantic_expectation(scene)
    assert scenario_id+'_semantic_over_literal_echo_20261103' in expected['product_contract_revision_ids']
    if scenario_id=='duration_limit_update_should_persist':
        assert scene.expected_task_memory_contains==[{'field':'max_duration_hours','value':4,
            'conversation_id':scene.message.conversation_id,'customer_id':scene.message.sender_id,'status':'active'}]
        assert scene.require_task_memory_current_context
        assert any('五小时也行' in x['description'] for x in expected['forbidden_meaning'])
    else:
        assert scene.expected_active_game_count==1
        assert scene.forbidden_tool_names


@pytest.mark.parametrize('field',['must_use_existing_context','should_not_treat_as_new_game'])
def test_presentation_revision_cannot_replace_boolean_business_invariants_with_numbers(field):
    records=deepcopy(read_product_records());record=next(r for r in records if r.get('id')=='literal_echo_business_fact_review_20261103')
    revision=next(r for r in record['revisions'] if r['live_scenario_id']=='resume_status_after_casual_chat')
    assert revision['current_expectation'][field] is True
    revision['current_expectation'][field]=1
    with pytest.raises(ValueError):validate_reply_presentation_policy(records)
