"""构建contracts上下文，并执行作用域裁剪与隐私隔离。

"""

from __future__ import annotations

from typing import Any

from ..query_contract import game_query_strategy_contract


def output_contract() -> dict[str, Any]:
    """声明动作字段的类型和嵌套位置，避免状态元字段挤断根对象或重复计划。

    这里只描述输出协议，不提供业务计划；实际动作仍须经过合同和 Gateway 校验。
    """
    return {
        "format": "json_object",
        "structure_rules": {
            "root_object_count": 1,
            "unique_member_names_at_every_level": True,
            "root_plan_path": "objective_plan",
            "state_metadata_paths": ["objective_state.active_game_id", "objective_state.reply_scope"],
            "root_close_rule": "Close the root object once, after every root field; do not append fields or another plan after that closing brace.",
        },
        "required_keys": [
            "goal",
            "objective_status",
            "reasoning_summary",
            "objective_state",
            "objective_plan",
            "plan_revision_reason",
            "reply_to_user",
            "tool_calls",
            "needs_human",
            "stop_reason",
        ],
        "objective_status_values": ["needs_tool", "waiting_user", "completed", "needs_human", "unknown"],
        "field_types": {
            "goal": "string",
            "objective_status": "string",
            "reasoning_summary": "string",
            "objective_state": "object; structured current task state, including known facts, missing facts, current phase, active IDs, blockers",
            "objective_plan": "array; ordered plan steps. each step should include step_id, title, status, tool, depends_on, decision_rule",
            "plan_revision_reason": "string; why this plan is created or changed after reading current message/tool results",
            "reply_to_user": "string; empty for explicit silent decisions",
            "response_mode": "optional reply|silent; default reply",
            "silence_reason": "required non-empty string when response_mode=silent",
            "tool_calls": "array",
            "needs_human": "boolean",
            "stop_reason": "object",
            "self_assessment": (
                "optional object|null: {progress: advancing|stalled|regressing, should_escalate: boolean}; "
                "should_escalate=true only with stalled"
            ),
            "badcase": "null; deprecated side-channel, call record_badcase tool instead",
        },
        "objective_state_contract": {
            "current_phase": "recommended string: understand_intent | query_existing_games | collect_missing_info | create_game | search_customers | draft_invites | record_feedback | answer_user | wait_user | human_review",
            "known_facts": (
                "recommended JSON object mapping stable keys to values; facts already safe to use for this objective. "
                "Every object member must use key:value syntax, for example "
                "{\"requested_game\": \"hangzhou_mahjong\", \"known_player_count\": 2}. "
                "Never write set-like JSON such as {\"fact A\", \"fact B\"}; use an array when no keys are needed."
            ),
            "fact_assertions": (
                "optional array of {field,value,role:current|condition|quoted|uncertain,source_message_id}; "
                "current means a still-applicable user assertion in this task, including prior user turns with their original source_message_id; "
                "not current_message-only, a lexical match or a desired future state"
            ),
            "missing_facts": "recommended array of strings; facts still needed before state writes or drafts",
            "active_game_id": "optional string|null",
            "blockers": "recommended array of strings",
            "reply_scope": (
                "recommended object for terminal replies: requested_information, allowed_response_facts, "
                "background_facts_to_withhold. Context facts may support reasoning without becoming customer-visible."
            ),
        },
        "objective_plan_contract": {
            "step_status_values": ["pending", "in_progress", "done", "blocked", "skipped"],
            "required_step_keys": ["step_id", "title", "status"],
            "recommended_step_keys": ["tool", "depends_on", "decision_rule"],
            "tool_step_rule": "Any step that needs system state should map to one available tool. Use objective_status=needs_tool while such steps are still in_progress.",
            "revision_rule": "Use turn_tool_evidence as the complete current-turn history and previous_tool_results as the latest feedback. Mark completed tool steps done, update known facts/blockers, and choose the next step instead of restarting from scratch.",
        },
        "stop_reason_contract": {
            "can_stop": "required boolean; false when objective_status=needs_tool, true for terminal statuses",
            "why": "required non-empty string explaining why the agent can stop now or why it must continue with tools",
            "pending_work": "required array of strings; non-empty when can_stop=false",
            "depends_on_tool_results": "required boolean; true if the decision depends on turn_tool_evidence, previous_tool_results, or system state",
            "waiting_decision": (
                "征询是否继续找人、是否加入等业务选择时，用 objective_status=waiting_user，"
                "在 objective_state 和计划保留待选问题；局池查询已回答不等于当前任务已完成。"
                "stop_reason.why 说明等待客户决定时，不能同时把顶层状态标为 completed。"
            ),
        },
        "tool_call_contract": {
            "call_id": (
                "optional non-empty string in legacy single-call mode; required and unique for every call when an "
                "action contains an explicit dependency graph"
            ),
            "depends_on": (
                "optional array in legacy mode; array of prerequisite call_id strings in dependency graph mode. "
                "Every identifier must reference another call_id inside the same tool_calls array; never reference "
                "objective_plan.step_id or a tool from an earlier loop step. Omitted/null is normalized to []; "
                "use [] only when the call is truly independent"
            ),
            "name": "required non-empty string",
            "arguments": "required object, validated again by ToolGateway schema",
            "reason": "required non-empty string explaining why this tool is needed now",
            "idempotency_key": "optional string|null; backend still derives authoritative idempotency key",
        },
        "invariants": [
            "objective_status=needs_tool requires at least one tool_call",
            "objective_status=needs_tool requires empty reply_to_user",
            "objective_status=waiting_user|completed|needs_human|unknown must not include tool_calls",
            "terminal replies require non-empty reply_to_user unless response_mode=silent; silent requires completed|waiting_user, empty text and a silence_reason",
            "objective_status=needs_human requires needs_human=true",
            "needs_human=true requires objective_status=needs_human",
            "objective_status=needs_tool requires stop_reason.can_stop=false and non-empty pending_work",
            "objective_status=waiting_user|completed|needs_human|unknown requires stop_reason.can_stop=true",
            "when completing the current user objective requires their answer or confirmation, use waiting_user; asking the question completes this model turn, not the user objective",
            "invalid contract means backend will not execute any tool",
            "a turn_tool_evidence continuation with can_stop=false rejects terminal actions unless the current objective_status is explicitly allowed by that continuation",
            "multiple independent parallel_safe read-only calls should declare unique call_id and depends_on=[]; dependent calls must list their prerequisites",
            "objective_plan.step_id and tool_calls.call_id are separate namespaces; tool depends_on may reference only call_id values from the same action",
            "missing graph metadata keeps backward-compatible sequential execution; the model cannot mark a tool parallel-safe because that permission comes from ToolGateway",
            "badcase must be null; badcase/eval writes must use record_badcase tool_call",
            "terminal reply must answer only current_message or an explicitly unresolved confirmation; do not append adjacent active-game facts, shortage, time, or calls to action that the user did not ask for",
            "casual-chat replies may use business state for continuity but must not surface that state unless current_message explicitly refers to it",
        ],
    }


def planning_contract() -> dict[str, Any]:
    """声明查询证据范围和读写前置条件，业务目标与缺项判断仍由主模型完成。"""
    return {
        "purpose": "把每轮用户输入转成一个可执行目标，然后用工具结果持续修订计划。",
        "game_query_strategy_contract": game_query_strategy_contract(),
        "loop_rule": (
            "每一轮先更新 objective_state，再给出 objective_plan。"
            "如果计划中的下一步依赖系统事实，必须通过 tool_calls 调用工具；"
            "工具返回后以 turn_tool_evidence 保留本轮完整证据，并结合 previous_tool_results 的最近反馈修订计划。"
        ),
        "state_progression": [
            "理解意图和上下文",
            "确认已知槽位、画像默认值和缺失槽位",
            "需要事实时查询当前局、房态或候选人",
            "需要写入时创建/更新局、记录候选人反馈或生成待审批草稿",
            "根据工具结果决定继续调用工具、追问用户、短句回复或转人工",
        ],
        "context_scope": {
            "active_games": (
                "仅包含当前会话、发送者组织/参与、邀约或候补通知相关局，不是全局局池。"
                "空列表不能证明没有可拼局；全局匹配结论须来自实际 search_current_games 结果。"
            ),
            "requested_inventory_scope": (
                "查询、澄清与选项协商策略以 game_query_strategy_contract 为准。"
                "消息来自群聊不自动要求查询，个人约局或加入仍需必要事实和授权，非查询不顺手检索。画像补全只用于个人适配；"
                "全店盘点不能追加客户未要求的画像过滤，显式限定玩法等范围的盘点则保留该限制。"
                "结论须服从真实查询的 evidence_scope，不能以窄查询回答全店概况。"
            ),
        },
        "tool_selection_hints": {
            "new_matching_goal": (
                "新的撮合/组局目标在条件足以检索时，默认先 search_current_games；"
                "用户明确排除拼现成局或要求独立新开时，尊重该目标，按新开需求规划，"
                "不强制检索或推荐现成局，仍须满足写入条件与权限。"
                "本轮同条件且仍有效的成功查询应复用，不重复。"
                "默认检索有匹配先提供公开选项并等待加入确认；检索无匹配或明确独立新开，且已授权组局、写入条件足够时，"
                "才 create_game、search_customers 并按候选结果 create_invite_drafts。"
                "无现成局时按 game_query_strategy_contract 决定回复、必要追问或服务提议；"
                "符合 empty_personal_matching_default 时按其有条件默认征询，不把空查询当成招募授权。"
                "纯盘点、明确拒绝继续服务或目标已结束时不追加提议。"
                "已有局的参加、拒绝、协商或状态反馈继续按已有局规则处理，不因此重查局池。"
            ),
            "read_vs_write_preconditions": (
                "查询与缺项策略引用 game_query_strategy_contract；未知己方人数不阻断只读检索，"
                "不得默认人数，需占座/建局等依赖人数的写动作前再补齐。"
                "先按句法指向、对话主体与委托独立解释人数来源：独立陈述桌况再委托补齐通常是己方进度，"
                "寻找对象、参加条件或他人引语里的桌况通常是目标或他局，仍须结合上下文；工具字段含义不决定原话角色。"
                "个人匹配已知己方人数时须显式传 arguments.requirement.requesting_party.seat_count；"
                "查询的 requesting_party 嵌套在 requirement 内，create_game 才放 arguments 顶层，"
                "由后端按整组容量筛选；目标桌的 known_player_count/needed_seats 不能代替己方人数。"
                "smoke_preference=any 表示已知不限，不能当未知重问。"
            ),
            "known_fact_retention": (
                "本次有效条件是同一任务仍成立的用户事实合集，不限于 current_message。"
                "补充按字段增量更新，未重提不撤销、不清空、不套用默认值；用户新改、撤销或目标变化优先，事实与动作授权分开判断。"
                "个人匹配的唯一可靠画像玩法默认须实际写入 requirement.game_type。"
                "本次新改优先，盘点按其显式范围，多值或真实冲突不硬选；查询默认不授权加入、成局或外发。"
                "查房、查询或审查等辅助结果不撤销原有事实，修订计划不能将仍有效的已知条件改回缺项。"
                "同一未完成目标的委托范围与来源也须保留；老板追问条件、空查询或参数纠错不撤销原委托，"
                "不能因最后一轮是补槽就要求客户重复授权。用户改目标或撤销时再重判。"
                "保留多值和冲突来源，是否澄清按 game_query_strategy_contract 判断。"
            ),
            "source_interpretation_consistency": (
                "同一来源的已知己方人数须贯穿查询和建局：先解释人数角色，再分别投影查询座位与建局参与者，"
                "先查现成局不使独立的己方陈述变成条件或未知，工具层级不同不代表用户人数变化。"
                "空查询不提供人数新事实，不能查询时unknown而建局时无依据成为明确己方人数。"
                "来源澄清、新输入或重新核实原话可合法重判，须说明依据并重评已有步骤；采用的条件未被旧检索覆盖时先重查再写入。"
                "具体动作依赖该人数时，真正未知或存在歧义时澄清，不硬猜人数，也不把已发现的错误解释锁死。"
            ),
            "previous_decision_authority": (
                "根字段 previous_decision 是前一步模型提案，authority=model_proposal_only，"
                "其中 goal/objective_state/objective_plan/plan_revision_reason 用于保持计划连续、核对来源解释及说明纠错。"
                "用户原话、数据库状态和真实工具结果优先；提案不能凭空增加用户事实、授权或已执行动作。"
                "发现错误可回看原话有据纠正，须说明来源并重核受影响步骤，不把先前解释永久锁死。"
            ),
            "game_lookup_vs_room_availability": (
                "search_current_games 查询可拼局；check_room_availability 查询物理房态，二者不能互替。"
                "人齐开或未指定固定时段可能暂不查房，不能据此省略新撮合的现成局检索。"
            ),
            "time_constraint_revision": (
                "根据最新答复重算可用时间；无时段限定的放宽不能自动继承旧试探时段，"
                "只在某时段内灵活则保留范围。宽时段不自动选钟点或 scheduled。"
                "sent_at/occurred_at/task_context.started_at 是事件时间，不是客户指定开场；"
                "它们只能辅助解释相对日期，不能作为开场时刻的事实依据。"
                "已确认的固定时刻须把日期和时区实际写入工具 planned_start_at，不能只存在模型已知事实中。"
            ),
            "search_current_games 返回空 + 用户明确表示愿意等": (
                "调用 register_waiting_demand 登记本次需求；登记只代表等待后续匹配，不代表已经入局。"
            ),
            "用户明确表示算了、不打了或不再等待": (
                "调用 cancel_waiting_demand 取消当前客户拥有的等待需求。"
            ),
        },
        "system_trigger_handling": {
            "match_found": {
                "必须": "向用户展示局的基本公开信息（档位、烟况、时间、当前人数或缺口），并征求是否参加。",
                "禁止": "不得直接调用 join_game，不得替用户确认参加，不得泄露参与者身份或后台过程。",
                "话术参考": "有个X点X档X烟的三缺一，你要打吗？",
                "下一步": "本轮以 waiting_user 结束；只有客户后续明确确认，普通用户消息才可调用 join_game。",
            }
        },
        "historical_task_recovery": {
            "source": "Only same-conversation quoted messages and trusted scheduled tasks may populate recovered_task_contexts.",
            "evidence_order": "Use checkpoint first, then bounded raw task turns when no checkpoint exists.",
            "state_boundary": "Recovered history explains intent but never replaces read tools for mutable game, room, participant, or invite state.",
        },
        "do_not": [
            "不要只用一句自然语言承诺代替应执行的工具步骤",
            "不要在工具结果回来后丢掉上一轮已确认的计划和槽位",
            "不要因为最近一步是写记忆、审查或其他辅助动作，就重复 turn_tool_evidence 中已经成功完成且仍有效的查询",
            "不要把计划、工具名或后台细节暴露给客户",
        ],
    }


__all__ = ["output_contract", "planning_contract"]
