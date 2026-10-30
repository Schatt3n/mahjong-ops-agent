"""共享策略合同接入真实规划与检索载荷，不以合同字符串冒充真实模型语义验收。"""

import pytest

from mahjong_agent_runtime import (
    AgentContextBuilder, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract


@pytest.fixture(params=['memory', 'sqlite'])
def option_store(request, tmp_path):
    """用不同玩法与底注的真实牌局验证未知个人需求下的公开库存查询能力。"""
    store = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path/'strategy.sqlite3')
    for index, (game_type, stake) in enumerate([('red_center_mahjong', '1'), ('hangzhou_mahjong', '2')]):
        store.create_game(conversation_id=f'other-{index}', organizer_id=f'owner-{index}', organizer_name='发起人',
            requirement={'game_type':game_type, 'stake':stake, 'start_time_kind':'asap_when_full'},
            known_players=[{'customer_id':f'owner-{index}', 'display_name':'发起人', 'seat_count':3}], trace_id='setup')
    yield store
    if request.param == 'sqlite':
        store._connection.close()


@pytest.mark.parametrize('requirement,expected_count', [({}, 2), ({'stake':'1'}, 1)])
def test_unknown_need_inventory_reads_and_planning_share_one_strategy_contract(option_store, requirement, expected_count):
    """无画像与部分条件都可真实只读查局，规划和结果使用同一策略且不修改状态。"""
    gateway = ToolGateway(option_store)
    message = UserMessage(conversation_id='customer-chat', sender_id='new-customer', sender_name='客户', text='下午有人打吗')
    before = [game.to_dict() for game in option_store.games.values()]
    built = AgentContextBuilder(option_store, gateway).build(message, trace_id='strategy-context')
    result = gateway.execute(ToolCall(name='search_current_games', arguments={'requirement':requirement}),
        trace_id='strategy-query', conversation_id=message.conversation_id, sender_id=message.sender_id,
        sender_name=message.sender_name, step_index=1)
    assert result.called and result.allowed and result.error is None
    assert len(result.result['matches']) == expected_count
    assert result.result['requirement'].get('stake') == requirement.get('stake')
    assert not {'game_type', 'smoke_preference', 'requesting_party', 'start_time_kind'} & result.result['requirement'].keys()
    strategy = game_query_strategy_contract()
    assert built.payload['planning_contract']['game_query_strategy_contract'] == strategy
    assert result.result['customer_reply_contract']['game_query_strategy_contract'] == strategy
    assert result.result['customer_reply_contract']['evidence_scope']['criteria_completeness'] == 'not_assessed_by_retrieval'
    assert [game.to_dict() for game in option_store.games.values()] == before
    assert not option_store.invite_drafts and not option_store.outbound_message_drafts and not option_store.scheduled_tasks


def test_strategy_contract_instances_do_not_share_mutable_consumer_state():
    """审查或生成阶段修改自己的载荷不能污染下一次规划的权威策略。"""
    original = game_query_strategy_contract()
    consumer_copy = game_query_strategy_contract()
    consumer_copy['reasonable_strategy_examples'].clear()
    consumer_copy['known_fact_reuse'] = '错误消费方改写'
    assert game_query_strategy_contract() == original


def test_empty_personal_matching_default_and_reporting_share_the_same_authority():
    """条件充分的个人咨询保留自然服务默认，回复顺序与事实真实性分开约束。"""
    from mahjong_agent_runtime.domains.context_builders.contracts import planning_contract
    strategy=game_query_strategy_contract()
    assert planning_contract()['game_query_strategy_contract']==strategy
    reporting=strategy['query_result_reporting']
    assert '不强制每轮先复述空结果' in reporting
    assert '用户明确要求结果核对' in reporting
    assert '表达出来的查询结论不得虚构、反转或超出真实范围' in reporting
    default=strategy['empty_personal_matching_default']
    assert '个人找局条件足够、真实查询无可用现成局、尚未授权招募' in default
    assert '用户未拒绝继续服务' in default and 'waiting_user' in default
    assert '不要求在征询前复述空结果' in default
    assert '不套用纯公开盘点、明确不需招募或已有招募委托' in default
    assert 'empty_personal_matching_default' in planning_contract()['tool_selection_hints']['new_matching_goal']
