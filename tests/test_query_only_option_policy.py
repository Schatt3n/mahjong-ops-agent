"""隔离合同提案：查询可告知或征询，已有委托/授权/无局服务默认均不撤销。"""
from pathlib import Path
import json

import pytest

import mahjong_agent_runtime
from mahjong_agent_runtime import (AgentAction, AgentRuntime, AgentContextBuilder, InMemoryAgentStore,
    InMemoryTraceRecorder, StaticAgentClient, ToolCall, ToolGateway, UserMessage)
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract

ROOT = Path(__file__).resolve().parents[1]


def test_query_only_contract_is_consistent_in_main_prompt_and_context_without_requiring_a_reply_template():
    """只验策略合同的范围和两种终态，不把生成回复绑定某一句自然语言。"""
    assert str(Path(mahjong_agent_runtime.__file__).resolve()).startswith(str(ROOT / 'src'))
    store = InMemoryAgentStore()
    message = UserMessage(conversation_id='query-policy', sender_id='customer', sender_name='客户', text='只看看有没有现成局')
    built = AgentContextBuilder(store, ToolGateway(store)).build(message, trace_id='query-policy')
    prompt = built.messages[0]['content']
    matching_rule = next(line for line in prompt.splitlines() if '如果有一个或多个匹配局' in line)
    hint = built.payload['planning_contract']['tool_selection_hints']['new_matching_goal']
    for contract in (matching_rule, hint):
        assert '仅查询且无有效找人或组局委托' in contract
        assert '纯告知' in contract and 'completed' in contract
        assert '征询选择' in contract and 'waiting_user' in contract
        assert '已有有效委托按原目标推进' in contract
        assert '实际待答选择' in contract
    assert '查局后等待新组确认' not in prompt
    assert '回复可选现成局并问用户是否加入' not in prompt
    assert '仍需先征询是否帮忙才能转入招募或新组局' in prompt


def test_query_option_flexibility_preserves_authorized_recruitment_and_empty_help_defaults():
    """灵活性只针对无有效委托的查询，不撤销原有招募责任或把查询升级为授权。"""
    strategy = game_query_strategy_contract()
    empty = strategy['empty_personal_matching_default']
    assert '尚未授权招募' in empty and '默认自然' in empty and 'waiting_user' in empty
    assert '明确不需招募或已有招募委托' in empty
    assert '不授权建局、占座、招募或外发' in strategy['side_effect_boundary']
    prompt = (ROOT/'src/mahjong_agent_runtime/prompts/agent_runtime_system.md').read_text()
    assert '同一尚未完成目标中的明确委托' in prompt
    assert '对于已获组局授权的目标，如果没有匹配局，继续按组局目标调用' in prompt
    assert 'create_game.result.recruitment_policy.status=scheduled' in prompt
    review = (ROOT/'src/mahjong_agent_runtime/prompts/agent_runtime_reply_self_review.md').read_text()
    assert '纯结果告知且无待答选择可 completed' in review
    assert '不按问号或单个关键词判断' in review


@pytest.mark.parametrize('status,reply', [
    ('completed', '有个国标2块三缺一的。'),
    ('waiting_user', '国标2块有个三缺一的，你想打这桌就回我一声。'),
])
def test_query_only_runtime_keeps_read_only_boundary_for_both_communication_paths(status, reply):
    """固定动作仅验证两条合法路径、审查与无写入边界，不冒充真实模型理解通过。"""
    store, trace = InMemoryAgentStore(), InMemoryTraceRecorder()
    game, _ = store.create_game(conversation_id='other', organizer_id='host', organizer_name='牌友',
        requirement={'game_type': 'national_standard_mahjong', 'stake': '2', 'smoke_preference': 'any',
                     'start_time_kind': 'asap_when_full'},
        known_players=[{'customer_id': 'host', 'display_name': '牌友', 'seat_count': 3}], trace_id='seed')
    before = [value.to_dict() for value in store.games.values()]
    query = AgentAction(goal='仅查询现成局', objective_status='needs_tool', reasoning_summary='按已知条件查询',
        tool_calls=[ToolCall(name='search_current_games', call_id='read-current-games', arguments={'requirement': {
            'game_type': 'national_standard_mahjong', 'stake': '2', 'smoke_preference': 'any',
            'requesting_party': {'seat_count': 1}}}, reason='只读查询')],
        stop_reason={'can_stop': False, 'why': '先查局', 'pending_work': ['查询'], 'depends_on_tool_results': True})
    final = AgentAction(goal='回答现成局查询', objective_status=status, reasoning_summary='依据查询结果回复',
        reply_to_user=reply, stop_reason={'can_stop': True, 'why': '按实际告知或待答决定结束同步轮',
        'pending_work': ['客户选择'] if status == 'waiting_user' else [], 'depends_on_tool_results': True})
    main = StaticAgentClient([json.dumps(action.to_dict(), ensure_ascii=False) for action in (query, final)])
    reviewer = StaticAgentClient([json.dumps({'approved': True, 'needs_human': False, 'reasoning_summary': '原稿安全',
        'violations': [], 'item_reviews': [{'item_id': 'reply_to_user', 'approved': True,
            'suggested_safe_text': reply, 'reasoning_summary': '忠于查询与用户范围', 'violations': []}]}, ensure_ascii=False)])
    runtime = AgentRuntime(llm_client=main, store=store, trace_recorder=trace,
        reply_self_review_enabled=True, reply_self_review_client=reviewer)
    result = runtime.handle_user_message(UserMessage(conversation_id='query-policy', sender_id='customer',
        sender_name='客户', text='我一个人，国标2块烟不限，只看看有没有现成局', message_id='query-only'))
    assert result.status == status and result.final_reply == reply
    assert len(main.calls) == 2 and len(reviewer.calls) == 1
    assert not [event.content.get('errors') for event in trace.get_trace(result.trace_id) if event.step == 'action_contract_error']
    assert [item.name for item in result.tool_results if item.called and item.allowed] == [
        'search_current_games', 'customer_visible_content_review'], [item.to_dict() for item in result.tool_results]
    search = result.tool_results[0]
    assert search.result['matches'] and search.result['matches'][0]['game']['game_id'] == game.game_id
    assert [value.to_dict() for value in store.games.values()] == before
    assert not store.invite_drafts and not store.outbound_message_drafts and not store.scheduled_tasks
