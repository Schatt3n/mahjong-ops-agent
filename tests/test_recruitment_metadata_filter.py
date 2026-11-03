"""新增权限与方案元数据不能外发；普通组局话术不因内部过滤被误杀。"""
import pytest
from mahjong_agent_runtime.customer_visible_contract import customer_visible_text_contract_violations

MARKERS = ['registered_by_customer_id','option_root_game_id','alternative_to_game_id',
           'record_registered_member_withdrawal','bound_assertions','binding_call_ids','scope.call_id','scope.game_id']

@pytest.mark.parametrize('marker',MARKERS)
@pytest.mark.parametrize('encoding',['plain','fullwidth','zero_width'])
def test_internal_recruitment_identifiers_reject_normalization_bypasses(marker,encoding):
    if encoding=='fullwidth':marker=''.join(chr(ord(c)+0xFEE0) if 33<=ord(c)<=126 else c for c in marker)
    elif encoding=='zero_width':marker='\u200b'.join(marker)
    assert 'internal_recruitment_metadata.identifier' in customer_visible_text_contract_violations('这次的'+marker+'已经记下')

@pytest.mark.parametrize('text',['七点半和八点两边都先找。','你只是先看看，我先不给你定。','D不来了，原局还差一个。','现在只有一个，还差三个。','scope'])
def test_normal_recruitment_facts_and_general_words_remain_allowed(text):
    assert customer_visible_text_contract_violations(text)==[]
