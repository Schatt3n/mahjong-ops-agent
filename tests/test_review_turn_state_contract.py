"""本轮状态合同只解释枚举，话语是否真正待答仍由审查模型结合上下文判断。"""

from copy import deepcopy
from pathlib import Path

import pytest

from mahjong_agent_runtime import AgentAction, UserMessage
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload


@pytest.mark.parametrize('status,answer_requirement', [('waiting_user', 'required'), ('completed', 'absent')])
@pytest.mark.parametrize('text', [
    '查过了，下午暂时没有登记的局。',
    '要我帮你问问吗？',
    '说说你喜欢的口味，我再推荐凉菜。',
    '这么热的天气，谁还想出门呢？',
])
def test_review_state_semantics_bind_enum_without_deciding_from_punctuation(status, answer_requirement, text):
    """同文案可用于正反状态对照；后台未成局和可能继续聊天都不能改写待审枚举。"""
    message = UserMessage(conversation_id='state-check', sender_id='customer', sender_name='客户',
        text='下午有人打吗？', message_id='state-source')
    action = AgentAction(goal='回应本轮输入', objective_status=status, reasoning_summary='状态协议测试',
        reply_to_user=text)
    context = {'active_games': [{'game_id': 'still-forming', 'status': 'forming'}],
        'turn_tool_evidence': [{'name': 'search_current_games', 'called': True, 'allowed': True,
            'error': None, 'result': {'matches': []}}]}
    original = deepcopy(context)
    payload = build_reply_self_review_payload(message=message, action=action,
        review_items=[{'item_id': 'reply_to_user', 'recipient_id': message.sender_id, 'text': text}],
        context_payload=context, review_scope='reply_to_user')
    semantics = payload['action_boundary'].get('turn_state_semantics')
    assert semantics == {'scope': 'current_conversation_turn',
        'current_user_answer_requirement': answer_requirement,
        'business_goal_completion_implied': False, 'mismatch_violation': 'semantic_contradiction'}
    assert payload['action_boundary']['objective_status'] == status
    assert payload['review_items'][0]['text'] == text and payload['current_message'] == message.to_dict()
    assert context == original


@pytest.mark.parametrize('status,recipient,scope', [
    ('needs_tool', 'customer', 'reply_to_user'),
    ('needs_human', 'customer', 'reply_to_user'),
    ('completed', 'candidate', 'tool_calls'),
])
def test_current_reply_turn_semantics_do_not_reclassify_other_stages_or_recipients(status, recipient, scope):
    """当前客户等待状态不能套到工具预检或发给另一人的邀约。"""
    message = UserMessage(conversation_id='state-check', sender_id='customer', sender_name='客户', text='帮我看看')
    action = AgentAction(goal='本轮目标', objective_status=status, reasoning_summary='协议范围', reply_to_user='好。')
    payload = build_reply_self_review_payload(message=message, action=action,
        review_items=[{'item_id': 'reply_to_user', 'recipient_id': recipient, 'text': '好。'}],
        context_payload={}, review_scope=scope)
    assert payload['action_boundary'].get('turn_state_semantics') is None


def test_review_prompt_requires_both_directions_and_separates_turn_from_business_lifecycle():
    """固定状态错配不是文风选项，普通话题的真正问题也可形成待答。"""
    prompt = (Path(__file__).resolve().parents[1] / 'src/mahjong_agent_runtime/prompts/agent_runtime_reply_self_review.md').read_text()
    assert 'action_boundary.turn_state_semantics' in prompt
    assert '没有真实待答问题的纯告知必须 completed' in prompt
    assert '生活、工作等非业务交流' in prompt
    assert '反问、礼貌收尾、可能继续聊天或原局未成' in prompt
