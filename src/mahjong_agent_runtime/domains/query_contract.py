"""统一查询、澄清与公开选项协商的策略边界，不替主 Agent 选择执行路径。"""

from __future__ import annotations

from typing import Any


def game_query_strategy_contract() -> dict[str, Any]:
    """提供各模型阶段共同引用的纯合同；无查询、意图分类或副作用执行。"""
    return {
        "strategy_owner": "main_agent",
        "strategy_choice": (
            "结合真实原话、相关上下文和工具证据判断是否合理推进当前目标，不限定固定内容或唯一语义路径。"
            "下列策略只是例子，不规定先问、先查或固定选项数量；无关发散不能代替回答当前问题。"
        ),
        "request_scope_contract": {
            "authority": "model_proposal_only",
            "scope_field": "objective_state.request_scope",
            "kinds": ["query_only", "delegated_matching", "specific_game_choice", "not_applicable"],
            "decision_order": ["apply_explicit_limits", "retain_same_goal_delegation", "resolve_execution_gaps", "choose_inventory_response"],
            "inventory_result_changes_scope": False,
            "model_plan_can_revoke_user_delegation": False,
            "channel_changes_scope": False,
            "empty_query_offer_scope": "query_only",
            "missing_execution_facts_allow_clarification": True,
            "explicit_inventory_only_allows_recruitment": False,
            "interpretation": (
                "先从本人原话和同目标来源确定要完成什么，保留明确限制，再选择库存查询后的动作。"
                "query_only含纯咨询或被限定只找现成的请求；delegated_matching是已托付寻找协调；"
                "specific_game_choice是对具体选项的参加决定，不能与寻找授权混为一谈。"
                "request_scope记录kind、source_message_ids及explicit_limits，只是可核对的模型解释，不授予执行权限。"
                "查询空只改变库存事实，不能把既有寻找委托降格为仅查询；原计划、reply_scope或新计划也无权撤销原话。"
                "确需纠正先前解释时回看本人来源并说明实际限制，不能因没有现成局反推没有委托。"
                "群/私聊和好友关系仅约束通道及披露，不改变本人表达的委托范围；实际动作仍受工具权限约束。"
            ),
        },
        "reasonable_strategy_examples": [
            "复用当前有效条件与可靠画像查询现有局，再介绍公开选项。",
            "缺少画像或需求不足时，先自然澄清真正有用的未知条件。",
            "缺少画像或需求不足时，先按已知范围真实查询公开库存，介绍一个或多个选项，再问想打哪个或协商条件。",
            "查无结果时先按request_scope_contract核对原目标：已委托且尚未承接就推进；缺执行事实只问缺项；仅查询可征询是否开始招募。已真实推进后，另按delegation_acceptance.progress_review核对追加服务提议。",
        ],
        "known_fact_reuse": (
            "复用本次有效原话、同一任务事实及无冲突的可靠画像，不重复追问已知条件。个人匹配采用的可靠条件"
            "须实际进入查询参数；当前明确表达优先于画像。公开盘点按客户指定范围，不擅自追加个人过滤。"
        ),
        "unknown_fact_boundary": (
            "未知值保持未知，多值或可靠来源冲突不擅选；这些情况不阻断公开库存查询。只有下一具体动作确需"
            "该事实时才澄清，不能为凑齐个人适配或建局槽位强制先问，也不能把检索默认值冒充用户事实。"
        ),
        "known_unknown_consistency": (
            "生成追问前，逐项用用户有效原话、同目标持久事实和可靠唯一画像复核真正缺项。"
            "known_facts和missing_facts是模型的解释；同一条件不能既已知又待问，"
            "查房或换计划不撤销玩法画像。只问真实未知项，不要求复述已知项。"
            "当前明确改变或可靠来源冲突时允许澄清，多值画像不得硬选唯一值。"
        ),
        "time_commitment_evidence": (
            "开局时间点、相对期限和最迟上限的保证都须有覆盖该承诺的证据；"
            "精确最快时间并不限于钟点，数分钟内或不超过某时长的肯定保证同样可核验。"
            "空房、系统时钟和客户画像不能证明人员已齐、到店时刻或开局保证；"
            "没有相反证据不等于有支持证据，未来服务承诺也不免除真实性要求。"
            "依据不足时允许承认估不准、征询可接受等待和给出明确条件下的方案，"
            "不能删掉条件把方案升级为保证；尽力或预计也不能冒充已发生动作或无机制的持续责任。"
        ),
        "public_option_presentation": (
            "只介绍当前收件人获准知道且有真实依据的公开选项。可给一个或多个选项，保留识别与选择所需的"
            "玩法、档位、时间、缺口等事实；客户仍未知的选项条件不能当作已知回声删掉。已知且无差异的条件"
            "可简短省略，不必固定复述字段或套用示例句。"
        ),
        "evidence_boundary": (
            "查询匹配只证明实际提交条件下的检索结果，不证明未知个人条件已满足；条件尚未建立时，可以介绍"
            "公开选项并询问偏好，不能声称已经适配。结论服从实际 evidence_scope，未查子范围的空结论须有"
            "supports_narrower_empty_claim=true；不得外推无人可约、房态可用或已发生动作。"
        ),
        "query_result_reporting": (
            "查询结果约束事实真实性，不是每轮必须复述的内容清单。有无询问仍应结合上下文合理承接，"
            "相关的独立服务征询或必要追问也可推进当前目标，不强制每轮先复述空结果或使用显式有无句。"
            "用户明确要求结果核对、理解真实选项或作决定所必需的信息仍须回应；不能空泛回避、无关发散，"
            "也不能通过省略造成事实误导。表达出来的查询结论不得虚构、反转或超出真实范围。"
        ),
        "empty_personal_matching_default": (
            "先按request_scope_contract确认原话仅限query_only；个人找局条件足够、真实查询无可用现成局、尚未授权招募，且用户未拒绝继续服务时，默认自然"
            "征询是否帮忙问人或组局，并以 waiting_user 等待客户决定；不要求在征询前复述空结果。"
            "该默认不套用纯公开盘点、明确不需招募或已有招募委托的目标，不覆盖上下文中其他明确目标或必要决定。"
            "delegated_matching须按delegation_acceptance承接，不因查询操作只读或建局尚未发生而判成尚未授权。"
        ),
        "delegation_acceptance": {
            "complete_delegation_requires_durable_responsibility": True,
            "same_scope_must_not_reask_consent": True,
            "query_does_not_authorize_recruitment": True,
            "explicit_inventory_only_limits_delegation": True,
            "concrete_game_join_requires_confirmation": True,
            "clarify_only_missing_execution_facts": True,
            "board_can_recruit_without_private_invites": True,
            "classification_is_not_authorization": True,
            "progress_review": {
                "review_order": ["verify_original_scope", "verify_same_goal_execution",
                    "classify_remaining_communication", "verify_accepted_obligations"],
                "delegation_not_advanced_requires_missing_substantive_execution": True,
                "durable_game_and_candidate_search_are_substantive_execution": True,
                "proposal_after_execution_is_not_reset": True,
                "conditional_additional_service_requires_prior_schedule": False,
                "substantive_execution_does_not_prove_follow_up": True,
                "execution_evidence": (
                    "先核同一客户、目标和牌局的成功工具回执与持久状态，不能只看最新一条回复或空候选。"
                    "已建立或复用真实待组局，并完成当前可执行的找人步骤，即使候选为空，也已实质推进原委托；"
                    "后端continuation允许本轮结束时，不能因没有个人复查/通知任务反推原委托从未承接。"
                    "只读查局、聊天记忆、模型计划、失败工具或另一目标的局不能冒充这种进展。"
                ),
                "missing_progress": (
                    "原委托有效、必要事实已齐，但尚未有相应持久承接和实际推进，却重新询问是否开始同一服务，"
                    "才以delegation_not_advanced要求重新规划；不能用问句、短确认或拟议的未来服务掩盖缺执行。"
                    "未完成的明确用户要求、失败阻塞或工具续办义务仍须处理，不因已有任意成功工具而消失。"
                ),
                "after_progress": (
                    "已实际推进后，区分原委托与新增持续复查、个人通知等责任。"
                    "整句仍待客户接受的追加服务提议可以使用waiting_user，不等于重置原委托，"
                    "也不要求在客户接受前先安排schedule_follow_up。不能仅因用户原来委托过、出现继续的意思，"
                    "或暂没有跟进任务就拒绝；不要求为了证明已推进而复述已知条件或内部执行过程。"
                ),
                "future_obligation_boundary": (
                    "提议不证明已接手；一旦实际承担未来复查或个人通知，仍需同目标、有效且覆盖该动作的机制。"
                    "真实待组局和一次找人不自动证明未来通知，提议旁的独立已发送声明也须真实外发。"
                    "看板未配置或无授权时不得声称已安排群发布，已配置的待发责任也不得说已发。"
                ),
            },
            "scope": (
                "结合原话的否定、条件及同目标上下文区分仅查询、委托寻找协调、确认参加具体局。"
                "完整的寻找协调委托已授权在本次条件内承接招募；没有现成局时不再问是否开始同一服务。"
                "但仅看看不授权建局；只找现成、不要问人或没有就算了等明确限制优先；"
                "委托寻找也不等于确认加入某个具体选项。分类标签只是理解建议，不能凭标签扩张用户范围。"
            ),
            "execution": (
                "有效委托且必要写入事实已齐，真实空库存后建立或复用同目标持久待组需求及群看板发布责任。"
                "现有能力通过 create_game 保存待组局并取得 board_arrangement；不要为同一需求重复建局。"
                "群发布可以按合并窗口延迟，有无匹配好友不影响先承接看板招募，不强制每次私聊邀约。"
                "register_waiting_demand 仅承接明确候补或等待匹配，不自带看板发布，不能独自冒充公开招募。"
                "只读检索、聊天记忆、topic分类或一句我帮你找都不等于已建立执行责任。"
                "确缺必要事实时只澄清缺项，保留已有委托；工具失败、权限不足或职责未落库须保留真实阻塞，"
                "不能退回重复授权征询或换成无依据的短确认。"
            ),
            "completion": (
                "已建立待组与可恢复发布责任可结束本轮并简短承接；不要求复述条件或汇报工具。"
                "已建局并完成当前可执行找人步骤、工具continuation允许结束时，按progress_review审查追加提议，"
                "不要把尚未承担的个人通知当作原委托未推进；不能借此放过仍缺失的执行责任或谎称已安排发布。"
                "持续检索、个人进展通知、到点退出和保证组齐是不同责任，须各有覆盖范围的实际机制，"
                "待组局或看板计划不能证明这些额外动作已经安排。"
            ),
        },
        "side_effect_boundary": (
            "查询和介绍选项不授权建局、占座、招募或外发；下一动作仍须满足真实用户授权、必要事实、人数、"
            "权限与隐私边界。这里指查询操作本身不增加权限，不是把先查局的用户目标一律归为仅查询。"
            "具体自招备选的档位核实另有工具和授权边界，只问本桌条件，不构成邀约或入局。"
            "主模型提案、查询结果和审查建议都不能增加授权、撤销本人既有委托或篡改已知条件。"
        ),
    }
