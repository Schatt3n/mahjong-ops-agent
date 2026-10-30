"""构建并解析客户可见内容审查结果，将模型审查与后端安全合同衔接。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .customer_visible_contract import (
    customer_visible_action_claim_violations,
    customer_visible_text_contract_violations,
)
from .models import AgentAction, ToolResult, UserMessage, now
from .domains.query_contract import game_query_strategy_contract


CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME = "customer_visible_content_review"


def external_action_evidence_from_tool_results(
    tool_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """从后端 ToolResults 提取交付事实，用于最终回复审查。"""

    relevant_names = {
        "create_invite_drafts",
        "create_outbound_message_drafts",
        "update_invite_delivery_status",
    }
    source_tool_names: list[str] = []
    draft_statuses: list[str] = []
    for raw in tool_results:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        if name not in relevant_names or not raw.get("called") or not raw.get("allowed") or raw.get("error"):
            continue
        source_tool_names.append(name)
        result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        for draft in result.get("drafts") or []:
            if isinstance(draft, dict) and draft.get("status"):
                draft_statuses.append(str(draft["status"]))
        recorded = result.get("recorded_status")
        if recorded:
            draft_statuses.append(str(recorded))
    if not source_tool_names:
        return None
    contact_started_statuses = {"sent", "confirmed", "declined", "negotiating", "no_reply"}
    return {
        "source": "backend_tool_results",
        "source_tool_names": source_tool_names,
        "draft_statuses": draft_statuses,
        "contact_started": any(status in contact_started_statuses for status in draft_statuses),
    }


def customer_visible_content_review_approved(result: ToolResult) -> bool:
    """返回审查工具是否批准了所有客户可见项目。"""

    return (
        result.name == CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
        and result.called
        and result.allowed
        and result.error is None
        and bool(result.result.get("approved"))
    )


def customer_visible_items_for_action(action: AgentAction) -> list[dict[str, Any]]:
    """在执行前从工具参数中提取出站文本。"""

    items: list[dict[str, Any]] = []
    for call_index, call in enumerate(action.tool_calls, start=1):
        if call.name == "create_invite_drafts":
            for item_index, raw in enumerate(call.arguments.get("invitations") or [], start=1):
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("message_text") or "").strip()
                if text:
                    items.append(
                        {
                            "item_id": f"tool_calls[{call_index}].arguments.invitations[{item_index}].message_text",
                            "source": "create_invite_drafts",
                            "recipient_id": str(raw.get("customer_id") or ""),
                            "recipient_name": str(raw.get("display_name") or raw.get("customer_id") or ""),
                            "text": text,
                        }
                    )
        if call.name == "create_outbound_message_drafts":
            for item_index, raw in enumerate(call.arguments.get("drafts") or [], start=1):
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("message_text") or "").strip()
                if text:
                    items.append(
                        {
                            "item_id": f"tool_calls[{call_index}].arguments.drafts[{item_index}].message_text",
                            "source": "create_outbound_message_drafts",
                            "recipient_id": str(raw.get("recipient_id") or ""),
                            "recipient_name": str(raw.get("recipient_name") or raw.get("recipient_id") or ""),
                            "channel": str(raw.get("channel") or ""),
                            "purpose": str(raw.get("purpose") or ""),
                            "text": text,
                        }
                    )
    return items


def normalize_item_reviews(
    review: dict[str, Any], review_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """规范化itemreviews，消除等价输入之间的表示差异。

"""

    raw_reviews = review.get("item_reviews")
    if not isinstance(raw_reviews, list):
        legacy_safe_text = str(review.get("final_reply") or "").strip()
        normalized: list[dict[str, Any]] = []
        for item in review_items:
            original = str(item.get("text") or "")
            suggested = legacy_safe_text or original
            violations = review.get("violations")
            normalized.append(
                normalize_review_item_contract(
                    item_id=str(item.get("item_id") or ""),
                    approved=bool(review.get("approved"))
                    and (not legacy_safe_text or legacy_safe_text == original),
                    suggested_safe_text=suggested,
                    reasoning_summary=str(review.get("reasoning_summary") or ""),
                    violations=list(violations) if isinstance(violations, list) else [],
                    action_evidence=(
                        dict(item.get("action_evidence") or {})
                        if isinstance(item.get("action_evidence"), dict)
                        else None
                    ),
                )
            )
        return normalized

    by_item_id = {
        str(item.get("item_id") or ""): item for item in raw_reviews if isinstance(item, dict)
    }
    normalized = []
    for item in review_items:
        item_id = str(item.get("item_id") or "")
        raw = by_item_id.get(item_id, {})
        suggested = str(
            raw.get("suggested_safe_text")
            or raw.get("safe_text")
            or raw.get("final_text")
            or item.get("text")
            or ""
        )
        violations = raw.get("violations")
        normalized.append(
            normalize_review_item_contract(
                item_id=item_id,
                approved=bool(raw.get("approved")),
                suggested_safe_text=suggested,
                reasoning_summary=str(raw.get("reasoning_summary") or ""),
                violations=list(violations) if isinstance(violations, list) else [],
                action_evidence=(
                    dict(item.get("action_evidence") or {})
                    if isinstance(item.get("action_evidence"), dict)
                    else None
                ),
            )
        )
    return normalized


def normalize_review_item_contract(
    *,
    item_id: str,
    approved: bool,
    suggested_safe_text: str,
    reasoning_summary: str,
    violations: list[str],
    action_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """规范化审查item结构化合同，消除等价输入之间的表示差异。

"""

    contract_violations = customer_visible_text_contract_violations(suggested_safe_text)
    action_claim_violations = customer_visible_action_claim_violations(
        suggested_safe_text,
        action_evidence,
    )
    normalized_violations = list(violations)
    normalized_violations.extend(
        f"customer_visible_contract:{violation}" for violation in contract_violations
    )
    normalized_violations.extend(
        f"customer_visible_contract:{violation}" for violation in action_claim_violations
    )
    return {
        "item_id": item_id,
        "approved": bool(approved) and not contract_violations and not action_claim_violations,
        "suggested_safe_text": suggested_safe_text,
        "reasoning_summary": reasoning_summary,
        "violations": normalized_violations,
    }


def item_reviews_approved(
    item_reviews: list[dict[str, Any]], review_items: list[dict[str, Any]]
) -> bool:
    """要求每个项目都有一对一、未更改、无违规的批准。"""

    original_by_id = {
        str(item.get("item_id") or ""): str(item.get("text") or "") for item in review_items
    }
    evidence_by_id = {
        str(item.get("item_id") or ""): (
            dict(item.get("action_evidence") or {})
            if isinstance(item.get("action_evidence"), dict)
            else None
        )
        for item in review_items
    }
    if len(item_reviews) != len(review_items):
        return False
    for item in item_reviews:
        item_id = str(item.get("item_id") or "")
        suggested = str(item.get("suggested_safe_text") or "")
        if item_id not in original_by_id or not bool(item.get("approved")):
            return False
        if suggested != original_by_id[item_id]:
            return False
        if customer_visible_text_contract_violations(suggested):
            return False
        if customer_visible_action_claim_violations(suggested, evidence_by_id.get(item_id)):
            return False
    return True


def _current_follow_up_summaries(context_payload: dict[str, Any], message: UserMessage) -> list[dict[str, Any]]:
    """仅投影当前客户、会话和业务目标仍有效的复查证据，不复制任务私有载荷。

    旧目标的任务不能替新承诺作证；无明确目标标识时不猜测归属。这里只校验结构和
    身份边界，承诺是否与记录的目标一致仍交给模型审查，不解析客户自然语言。
    """
    window = context_payload.get("task_context_window")
    context_id = window.get("task_context_id") if isinstance(window, dict) else None
    tasks = context_payload.get("active_follow_ups")
    if not isinstance(context_id, str) or not context_id or not isinstance(tasks, list):
        return []
    summaries: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        detail = task.get("payload")
        if (not isinstance(task.get("task_id"), str) or not task["task_id"]
                or task.get("task_type") != "conversation_follow_up"
                or task.get("conversation_id") != message.conversation_id
                or task.get("subject_id") != message.sender_id
                or task.get("status") not in {"pending", "processing"}
                or not isinstance(detail, dict) or detail.get("task_context_id") != context_id):
            continue
        try:
            expires_at = datetime.fromisoformat(detail["expires_at"])
            if expires_at.tzinfo is None or expires_at <= now():
                continue
        except (KeyError, TypeError, ValueError):
            continue
        summary = {key: task[key] for key in (
            "task_id", "task_type", "status", "due_at", "aggregate_type", "aggregate_id",
        ) if isinstance(task.get(key), str)}
        summary.update({key: detail[key] for key in (
            "task_context_id", "goal", "waiting_for", "expires_at",
        ) if isinstance(detail.get(key), str)})
        summaries.append(summary)
    return summaries


def build_reply_self_review_payload(
    *,
    message: UserMessage,
    action: AgentAction,
    review_items: list[dict[str, Any]],
    context_payload: dict[str, Any],
    review_scope: str,
) -> dict[str, Any]:
    """投影审查所需的原话、执行证据与当前需求解释，避免把旧查询当成新约束。

    主模型解释仅帮助审查理解回复目的，不替代用户原话或数据库事实；不在这里
    解析自然语言、决定查询范围或发起运营动作。
    """

    turn_tool_evidence = (
        context_payload.get("turn_tool_evidence")
        or context_payload.get("previous_tool_results")
        or []
    )
    current_customer_reply = review_scope == "reply_to_user" and all(
        item.get("recipient_id") == message.sender_id for item in review_items
    )
    allow_known_echo_omission = current_customer_reply
    objective_state = action.objective_state
    return {
        "current_message": message.to_dict(),
        "game_query_strategy_contract": game_query_strategy_contract(),
        "customer_visibility_contract": context_payload.get("customer_visibility_contract") or {},
        "sender_profile": context_payload.get("sender_profile"),
        "sender_relationships": context_payload.get("sender_relationships") or [],
        "active_games": context_payload.get("active_games") or [],
        "active_follow_up_summaries": (
            _current_follow_up_summaries(context_payload, message) if current_customer_reply else []
        ),
        "active_game_visible_summaries": context_payload.get("active_game_visible_summaries") or [],
        "previous_tool_results": context_payload.get("previous_tool_results") or [],
        "turn_tool_evidence": turn_tool_evidence,
        "recent_conversation_tail": list(context_payload.get("recent_conversation") or [])[-8:],
        "action_boundary": {
            "objective_status": action.objective_status,
            "needs_human": action.needs_human,
            "tool_call_names": [call.name for call in action.tool_calls],
            "has_reply_to_user": bool(action.reply_to_user.strip()),
            "goal": action.goal,
            "reply_scope": action.objective_state.get("reply_scope") or {},
            "known_facts": objective_state.get("known_facts") if isinstance(objective_state.get("known_facts"), dict) else {},
            "missing_facts": objective_state.get("missing_facts") if isinstance(objective_state.get("missing_facts"), list) else [],
            "fact_assertions": objective_state.get("fact_assertions") if isinstance(objective_state.get("fact_assertions"), list) else [],
            "customer_visible_item_count": len(review_items),
        },
        "customer_service_review_contract": {
            "offer_is_not_execution": True,
            "proposal_scope_is_whole_utterance": True,
            "separate_assertions_need_independent_evidence": True,
            "commitment_evidence_requires_accepted_responsibility": True,
            "customer_acceptance_condition_differs_from_event_trigger": True,
            "communication_act_review_order": [
                "先保留整句疑问、条件、否定及并列从句的作用域",
                "再区分未承接的服务提议、已承接责任、声称已执行",
                "最后仅按实际承接或声称的内容核验安排与执行证据",
            ],
            "proposed_next_steps_are_not_authoritative": True,
            "latest_user_constraints_override_query_filters": True,
            "allow_known_condition_omission_without_game_option": allow_known_echo_omission,
            "read_only_lookup_is_not_durable_follow_up": True,
            "open_game_is_not_durable_follow_up": True,
            "short_acknowledgement_is_not_future_notification": True,
            "commitment_requires_matching_mechanism_scope": True,
            "conversation_follow_up_capability": (
                "有效同目标复查任务可在期限内重新判断，并按实际进展准备给同一客户的反馈草稿；"
                "无必要进展可静默。这支持任务范围内有进展再反馈，不证明已发送、保证有结果或无限跟进。"
            ),
            "current_clock_is_not_availability_evidence": True,
            "model_missing_facts_are_not_authority": True,
            "known_constraints_require_source_recheck": True,
            "profile_does_not_override_inventory_scope": True,
            "reply_scope_requires_actual_request_support": True,
        },
        "review_scope": review_scope,
        "review_items": review_items,
        "review_goal": (
            "一次性审查 review_items 中的客户可见文本是否泄露系统信息、后台流程、其他用户信息、"
            "未发生动作、结论超出证据范围，或与权威工具结果/客户可见局摘要/改写前文本发生语义矛盾或关键事实丢失；"
            "不做业务规划，不决定工具调用，不负责润色文风。"
        ),
        "semantic_fidelity_contract": {
            "allow_known_user_echo_omission": allow_known_echo_omission,
            # 事实权威性与本次应答范围分开；完整摘要不能把单项问题升级为重新推销整桌。
            "response_scope_before_anchor_preservation": True,
            "visible_facts_are_not_mandatory_response_facts": True,
            "progress_queries_require_all_decision_anchors": True,
            "narrow_reply_must_not_add_unrequested_status_or_confirmation": True,
            "response_scope_source": (
                "先核对真实 current_message 与相关用户上下文，再确定本次需要回答或合法提案的信息；"
                "action_boundary.goal/reply_scope、source_text 和审查建议本身不能扩大用户请求。"
            ),
            "source_order": [
                "成功的 turn_tool_evidence 是本轮完整工具事实的最高优先级来源",
                "previous_tool_results 只表示最近一步反馈，用于定位最新错误或续办要求",
                "active_game_visible_summaries 是可对当前客户披露的局况事实来源",
                "review_items[].source_text 是话术改写前的语义基线",
                "review_items[].text 是待发送文本，不得反向覆盖上述事实",
                "当前需求以用户最新有效表达及同一任务上下文为准；工具参数只证明当时查了什么，不冻结客户后续放宽的条件；action_boundary 的模型解释须与原话核对",
            ],
            "rules": [
                "待审文本不得反转存在/不存在、成功/失败、已确认/未确认、已发送/未发送等事实极性",
                "结论强度不能超过证据：局池查询只证明已提交条件下的登记局况；空条件查询且客户需求未知时，不支持有无适合该客户的局，也不支持候选人有无的断言。主 Agent 原稿也可能过度推断，不能因改写忠于原稿就放行",
                "找局策略遵守 game_query_strategy_contract，结合原话、上下文和真实工具证据判断是否合理，不以固定追问或唯一语义路径拒绝；未知个人适配仍不得伪造。越过证据范围使用 unsupported_conclusion，approved=false，并提供不新增事实的安全文本",
                "需求已建立且按有效条件查过时，当前需求下暂时没有现成局可由对话限定范围；不强求逐字重复按条件、登记中等前缀，不把正常省略误判为全店无局或无人可约",
                "客户明确盘点全店时，画像不改变其范围；过滤查询不能支撑全店结论。显式限定的盘点保留限制，范围不符使用 unsupported_conclusion 回灌纠正查询，不因主模型自称全局目标而放行",
                "只有真实查询的 customer_reply_contract.evidence_scope.supports_narrower_empty_claim=true，且本次成功查询确为空时，才可依据后端已证实的未过滤可用局池为空推断更窄范围为空。该字段为 false 或缺失时，空列表只证明本次检索未返回；评分阈值可能使新增过滤后出现结果，不能凭条件包含关系自动推断未查子范围为空。本轮是否显式呈现结果按 game_query_strategy_contract.query_result_reporting 审查；非空、截断、不完整或失败结果均不能当空集，不得外推未知个人适配、潜在客户或房态",
                "missing_facts 和 reply_scope 是待核解释；原话、任务事实或可靠唯一画像仍确认的条件，不能因本条未重复或中间查房而重问。无真实冲突的重复澄清以 known_constraint_reask 拒绝",
                "当前目标由真实原话和相关上下文确定；非业务交流附带招揽、业务提醒或业务回流尾句属于 unsolicited_goal_expansion，即使无工具也须拒绝，安全改写可去掉未请求背景；正常找局咨询的服务提议仍允许",
                "明确名单询问可使用 active_game_visible_summaries.public_participant_names 的后端私聊授权白名单，不把已授权公开值当私有备注；字段缺失时不能从全局画像/内部参与记录补名，角色/邀约状态/关系仍保密，不能顺带招揽或确认",
                "未知个人适配不能先宣判有无适合的局再补问；如表达空结论，缺少 supports_narrower_empty_claim=true 时只支持本次没查到，不能自动宣判未查范围为空。不得因相关服务征询或必要追问未复述空结果就拒绝，按 game_query_strategy_contract.query_result_reporting 核对上下文合理性；不得扩大查询范围、冒充个人适配或由登记局为空推断没有潜在客户；active_games 仅是客户相关局，不是全店库存",
                "不得篡改仍在表达的业务事实；要区分必需决策信息和用户已知条件的冗余回声",
                "allow_known_user_echo_omission=true 时，收讫、澄清或无具体局选项的服务提议可省去客户已知条件；候选邀约、真实局选项、条件差异及必要确认仍须完整",
                "本次真实问题或已授权目标需要介绍可选局、说明条件差异、回答局况、人数、缺口或进展时，人数、缺口、时间、档位、烟况、时长、玩法和必要确认问题必须保留",
                "status_query_reply_contract 的 all_decision_anchors 先核对其 applicability；局况、人数、缺口、进展查询和真实选项均以完整决策锚点为基线，不能将人数或缺口当作可省时间等锚点的窄属性；无需用户逐字要求完整摘要",
                "仅名单及不表示局况进展的窄属性可省未问邻接背景，省去无关时间、缺口等不构成 semantic_fact_loss；原稿或审查建议曾加入背景不产生保留义务。为识别用户正在问的对象确实必要的信息仍须保留，不能机械按字段删减",
                "可披露事实不是本轮必答清单；名单或窄属性待审文本擅自附带未请求的局况、进展、招揽或参加确认，且不是理解所问对象必要的信息时，必须以 unsolicited_goal_expansion 拒绝，安全改写仅保留所问信息。背景真实、原稿含背景或主模型自述扩大后的目标都不能豁免",
                "成功查询返回非空 matches 时，待审文本不得声称没有匹配结果",
                "审查发现语义不保真时 approved=false，并在 violations 标注 semantic_contradiction 或 semantic_fact_loss",
                "当 review_items[].action_evidence.contact_started=false 时，不得声称已经问了、联系了、发送了或通知了；外部动作只能以该后端证据为准",
                "按完整句子的疑问或条件作用域先识别各独立沟通行为，再分类为服务提议、已承接责任、声称执行，最后核验证据；不能先按动作内容索要安排再倒推它已经承诺",
                "整句征询是否需要服务时，逗号后的打听、关注或将来通知仍可处在提议范围；尚待客户接受的条件不等于有消息就通知等业务事件触发条件；后者不自动豁免已接手责任。尚未承接不要求后续安排，不适用 unsupported_follow_up_commitment，不能因没有任务或含持续动作就拒绝提议",
                "本轮真实建局、找候选人或准备邀约可正常短承接，不等于承担持续复查或未来回信。仅已承接持续责任才需覆盖当前客户、目标、有效期及所承诺动作的真实机制，缺少时使用 unsupported_follow_up_commitment。开放局、一次查询、已有草稿或任意任务都不自动支持持续通知；无安排不能反向证明征询已变成承诺",
                "active_follow_up_summaries 中的有效同目标任务支持有限复查，并可按实际进展准备给同一客户的反馈草稿，覆盖该范围内有进展再反馈的承接；真实有效候补登记支持相应匹配事件后的通知准备；招募窗口任务只支持届时开启招募，不自动支持通知发起人。有限安排不证明无限跟进、一定凑齐或已经外发",
                "独立的已执行声明或无条件承诺分别核验，不因邻接提议放行；声称联系/发送/通知须真实外发事实，实际接手有安排但尚未发送不能误判成虚构已联系",
                "同为无外发无安排的对照：需不需要替你打听，消息有了再告诉你？为提议；我替你打听，消息有了再告诉你。为已承接，后者需拒绝。条件式征询可通过，但另起我已经联系过其他人了须证据；提议后另起我也会持续替你盯着仍是独立责任，无安排拒绝",
                "安全改写只修违规，不新增换档位、换时间等业务安排；原稿里的下一步建议不是权威事实，不能因为它曾出现就强制保留；若业务依据有问题，反馈原因给主模型重新规划",
            ],
        },
        "review_contract": {
            "format": "json_object",
            "required_keys": ["approved", "needs_human", "reasoning_summary", "violations", "item_reviews"],
            "field_types": {
                "approved": "boolean",
                "needs_human": "boolean",
                "reasoning_summary": "string",
                "violations": "array of string labels",
                "item_reviews": "array; one item per review_items entry",
                "item_reviews[].item_id": "must equal original review_items[].item_id",
                "item_reviews[].approved": "boolean",
                "item_reviews[].suggested_safe_text": "string; same as original text when approved=true, safe rewrite when approved=false",
            },
            "invariants": [
                "approved=true means every review item has no information leakage, unverified external action, semantic contradiction, or required fact loss",
                "approved=true requires every item_reviews[].approved=true and suggested_safe_text equals the original text",
                "the top-level approved value must equal the aggregate of item_reviews; an unchanged, violation-free, approved item must not be rejected only because it safely refuses a user's requested disclosure format",
                "approved=false and needs_human=false means at least one item_reviews entry contains a safe rewrite",
                "needs_human=true means the main agent should use a short human-handoff reply or stop the unsafe tool action",
                "suggested_safe_text must not expose tool names, JSON, traceId, internal process, draft/approval state, or other customer identities/roles",
            ],
            "available_tools": [],
        },
    }


def parse_reply_self_review(raw_response: str) -> tuple[dict[str, Any], list[str]]:
    """解析replyself审查，并把非结构化输入转换为稳定合同。

"""

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return {}, [f"reply self review is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return {}, ["reply self review JSON root must be object"]
    errors: list[str] = []
    for key in ["approved", "needs_human", "reasoning_summary", "violations"]:
        if key not in payload:
            errors.append(f"missing required review key: {key}")
    if "approved" in payload and not isinstance(payload.get("approved"), bool):
        errors.append("approved must be boolean")
    if "needs_human" in payload and not isinstance(payload.get("needs_human"), bool):
        errors.append("needs_human must be boolean")
    if "reasoning_summary" in payload and not isinstance(payload.get("reasoning_summary"), str):
        errors.append("reasoning_summary must be string")
    if "violations" in payload and not isinstance(payload.get("violations"), list):
        errors.append("violations must be array")
    if "item_reviews" in payload and not isinstance(payload.get("item_reviews"), list):
        errors.append("item_reviews must be array")
    if "item_reviews" in payload and isinstance(payload.get("item_reviews"), list):
        for index, item in enumerate(payload.get("item_reviews") or [], start=1):
            if not isinstance(item, dict):
                errors.append(f"item_reviews[{index}] must be object")
                continue
            if not isinstance(item.get("item_id"), str) or not item.get("item_id"):
                errors.append(f"item_reviews[{index}].item_id is required")
            if "approved" in item and not isinstance(item.get("approved"), bool):
                errors.append(f"item_reviews[{index}].approved must be boolean")
            if not isinstance(item.get("suggested_safe_text"), str):
                errors.append(f"item_reviews[{index}].suggested_safe_text must be string")
    return payload, errors


__all__ = [
    "CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME",
    "build_reply_self_review_payload",
    "customer_visible_content_review_approved",
    "customer_visible_items_for_action",
    "item_reviews_approved",
    "normalize_item_reviews",
    "normalize_review_item_contract",
    "parse_reply_self_review",
]
