"""纠错事实来源仅供主模型核实，不进入对客审查；末端规则拦截字段泄露并允许普通事实。"""

from copy import deepcopy
import json

import pytest

from mahjong_agent_runtime.customer_visible_contract import customer_visible_text_contract_violations
from mahjong_agent_runtime import AgentAction, UserMessage
from mahjong_agent_runtime.customer_visible_review import (
    build_reply_self_review_payload, expand_review_transport_references,
)


def internal_feedback():
    return {'name': 'search_current_games', 'called': False, 'allowed': False,
            'error': 'tool argument consistency violation', 'result': {}}


def make_review(context):
    message = UserMessage(conversation_id='fact-private', sender_id='customer', sender_name='客户', text='今晚有局吗')
    action = AgentAction(goal='回答查询', objective_status='completed', reasoning_summary='', reply_to_user='有一桌。')
    return build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope='reply_to_user', review_items=[{'item_id': 'reply_to_user', 'recipient_id': 'customer', 'text': action.reply_to_user}])


@pytest.mark.parametrize("path", ["turn_tool_evidence", "previous_tool_results", "recent_conversation"])
def test_source_references_are_projected_out_without_losing_business_reference_or_mutating_trace(path):
    feedback = internal_feedback()
    feedback['result']['reference_fact_sources'] = {'stake': {'source_message_id': 'PRIVATE_CUSTOMER_MESSAGE_ID', 'role': 'current'}}
    feedback['result']['validation_error'] = {'code': 'explicit_task_fact_consistency_rejected', 'message': 'PRIVATE_NORMALIZATION_DETAIL'}
    feedback['result']['reference_requirement'] = {'stake': '2'}
    if path == 'recent_conversation':
        context = {path: [{'role': 'tool', 'content': json.dumps([feedback])}]}
    else:
        context = {path: [feedback]}
    before = deepcopy(context)
    payload = make_review(context)
    serialized = json.dumps(payload)
    assert 'PRIVATE_CUSTOMER_MESSAGE_ID' not in serialized and 'PRIVATE_NORMALIZATION_DETAIL' not in serialized
    # 传输层可将最新反馈的精确副本替换为本轮证据引用；按生产严格合同解引用，
    # 再检查业务事实和失败历史，而不要求相同证据在 prompt 中存两份。
    expanded = expand_review_transport_references(payload)
    if path == 'previous_tool_results':
        assert 'result_reference' in payload[path][0]
        assert expanded[path][0] == expanded['turn_tool_evidence'][0]
    projected = json.loads(expanded['recent_conversation_tail'][0]['content'])[0] if path == 'recent_conversation' else expanded[path][0]
    assert projected['result']['reference_requirement'] == {'stake': '2'}
    assert projected['called'] is False and projected['allowed'] is False and projected['error']
    assert 'reference_fact_sources' not in projected['result']
    assert 'validation_error' not in projected['result']
    assert context == before


@pytest.mark.parametrize('text', ['reference_fact_sources', 'ＶＡＬＩＤＡＴＩＯＮ＿ＥＲＲＯＲ', 'reference fact sources'])
def test_last_filter_blocks_internal_feedback_fields_even_after_separator_or_width_changes(text):
    assert 'internal_task_fact_feedback.field' in customer_visible_text_contract_violations(text)


@pytest.mark.parametrize('text', ['七点三缺一，你打吗？', '1到3块可以，先给你记下。', '这会儿还差一个。'])
def test_last_filter_does_not_block_normal_game_facts(text):
    assert 'internal_task_fact_feedback.field' not in customer_visible_text_contract_violations(text)
