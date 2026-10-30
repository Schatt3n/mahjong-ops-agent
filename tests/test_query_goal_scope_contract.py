"""公开查询与个人适配先判断目标再判断缺项，合同存在不代替真实模型回放。"""

from pathlib import Path

from mahjong_agent_runtime.domains.context_builders.contracts import planning_contract
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract


ROOT = Path(__file__).resolve().parents[1]


def test_public_query_scope_precedes_personal_matching_preconditions():
    """公开查询保留范围边界，个人需求不足时也按共享合同选择查询或澄清。"""
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    public_rule = prompt.index("查询前先判断本次目标是读取公开库存/已有局况")
    strategy_rule = prompt.index("查询、澄清与公开选项协商以 `planning_contract.game_query_strategy_contract`")
    assert public_rule < strategy_rule
    assert prompt.count("公开公告板语境的局况浏览") == 1
    assert "未给出的个人玩法、档位等保持通配" in prompt
    assert "不能因这些个人槽位未给出就把公开查询改成适配问卷" in prompt
    assert "不把缺画像或个人条件不足当作禁止只读查局的理由" in prompt
    assert "不承诺适合该客户或替其占座" in prompt
    scope = planning_contract()["context_scope"]["requested_inventory_scope"]
    assert "game_query_strategy_contract" in scope
    strategy = planning_contract()["game_query_strategy_contract"]
    assert strategy == game_query_strategy_contract()
    assert strategy["strategy_owner"] == "main_agent"
    assert "未知值保持未知" in strategy["unknown_fact_boundary"]
    assert "不阻断公开库存查询" in strategy["unknown_fact_boundary"]
    assert "不能声称已经适配" in strategy["evidence_boundary"]


def test_group_source_neither_forces_query_nor_grants_personal_side_effects():
    """群消息仍区分公开查询、实际个人请求和非查询，不靠来源字段决定业务意图。"""
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    assert "`source=group` 只是消息来源，不自动决定查询目标或要求检索" in prompt
    assert "群里的个人约局或加入需求仍按必要事实和授权处理" in prompt
    assert "非查询消息不顺手查局" in prompt
    scope = planning_contract()["context_scope"]["requested_inventory_scope"]
    assert "消息来自群聊不自动要求查询" in scope
    assert "个人约局或加入仍需必要事实和授权" in scope
    assert "非查询不顺手检索" in scope
