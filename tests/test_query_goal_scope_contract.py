"""公开查询与个人适配先判断目标再判断缺项，合同存在不代替真实模型回放。"""

from pathlib import Path

from mahjong_agent_runtime.domains.context_builders.contracts import planning_contract


ROOT = Path(__file__).resolve().parents[1]


def test_public_query_scope_precedes_personal_matching_preconditions():
    """公开局况查询不因缺个人槽位而变成适配问卷，缺项前置只作用于实际个人需求。"""
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    public_rule = prompt.index("查询前先判断本次目标是读取公开库存/已有局况")
    personal_rule = prompt.index("若目标是个人推荐，再判断匹配依据是否成立")
    assert public_rule < personal_rule
    assert prompt.count("公开公告板语境的局况浏览") == 1
    assert "未给出的个人玩法、档位等保持通配" in prompt
    assert "不能因这些个人槽位未给出就把公开查询改成适配问卷" in prompt
    assert "关键玩法或档位未知且影响个人适配判断时仍先澄清" in prompt
    assert "不承诺适合该客户或替其占座" in prompt
    scope = planning_contract()["context_scope"]["requested_inventory_scope"]
    assert "先判断要读取公开库存/已有局况，还是做个人推荐" in scope
    assert "未知个人槽位保持通配" in scope
    assert "个人推荐的必要缺项仍须澄清" in scope


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
