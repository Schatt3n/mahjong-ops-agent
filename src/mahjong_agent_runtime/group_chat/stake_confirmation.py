"""自招局档位核实和补位协商：只在各自私聊传递同一桌的必要事实。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256

from ..budget import TokenBudget
from ..customer_visible_contract import customer_visible_text_contract_violations
from ..customer_visible_review import customer_visible_content_review_approved
from ..domains.customer_domain import HOST_CONFIRMATION_REQUIRED
from ..domains.stake_values import format_number, stake_options
from ..models import AgentAction, AgentRuntimeResult, GameStatus, MessageReference, ScheduledAgentTask, ScheduledTaskStatus, ToolResult, UserMessage, now
from ..scheduled_tasks import ScheduledTaskDeferred
from ..stores.task_store import BOARD_CANDIDATE_DECISION_TASK_TYPE, BOARD_CANDIDATE_UPDATE_TASK_TYPE
from .projections import public_game_start_display


BOARD_STAKE_CONFIRMATION_TASK_TYPE = "group_board_stake_confirmation"
BOARD_STAKE_UPDATE_TASK_TYPE = "group_board_stake_update"


def _current_board_inquiry(store, conversation_id, sender_id, trace_id, game_id, *, needs_stake):
    """只认本人当轮持久化的条件备选；另一轮、他人会话或模型猜出的局号不能触发联系。"""
    turns = [turn for turn in store.recent_turns(conversation_id, limit=40)
             if turn.trace_id == trace_id and not turn.metadata.get("context_invalidated")]
    source = next((turn for turn in reversed(turns) if turn.role.value == "user" and turn.sender_id == sender_id
                   and not turn.metadata.get("system_trigger") and not turn.metadata.get("internal_event")), None)
    if source is None:
        return None
    for turn in turns:
        if turn.role.value != "tool":
            continue
        try:
            results = json.loads(turn.content)
        except (TypeError, ValueError):
            continue
        for result in results if isinstance(results, list) else []:
            if not isinstance(result, dict):
                continue
            if result.get("name") != "search_current_games" or not result.get("called") or not result.get("allowed") or result.get("error"):
                continue
            for option in result.get("result", {}).get("alternatives", []):
                if (option.get("match_kind") == HOST_CONFIRMATION_REQUIRED
                        and (not needs_stake or "stake" in option.get("decision_required_fields", []))
                        and option.get("game", {}).get("game_id") == game_id):
                    return source
    return None


def _missing_current_search(call):
    """查询授权未满足时只返回可恢复指令，不登记、不联系，也不允许借用上一轮结果。"""
    return ToolResult(name=call.name, called=False, allowed=False,
        error="需要本轮本人查到的待发局人核实选项，请按更新后的条件先重新查局。",
        result={"retryable_no_effect": True, "next_step_policy": {"required_tool": "search_current_games"}})


def _contactable_board_host(store, game_id, *, require_unknown_stake=True):
    """在排队和发送两处重核真实缺位、原群授权与发局人身份。"""
    try:
        game = store.require_game(game_id)
    except ValueError:
        return None
    if (game.requirement.get("recruitment_mode") != "board_only"
            or (require_unknown_stake and stake_options(game.requirement))
            or game.status not in {GameStatus.FORMING, GameStatus.INVITING} or game.remaining_seats() < 1
            or (game.expires_at and game.expires_at <= now())):
        return None
    organizer = store.customers.get(game.organizer_id)
    if organizer and organizer.no_contact:
        return None
    for link in store.game_conversation_links(game_id=game_id):
        if link.link_type != "origin":
            continue
        policy = store.get_group_room_policy(link.room_id)
        if not policy or not policy.managed or not policy.board_enabled or not policy.outbound_enabled:
            continue
        identity = store.get_channel_identity_for_customer(game.organizer_id, channel=policy.channel)
        if identity and identity.can_private_message:
            return game, policy, identity
    return None


def _soft_preference_mismatch(game, gender):
    """只在确有相反性别的询局者时协商软偏好，不把硬性限制或未知画像当成候选。"""
    preferred = game.requirement.get("preferred_gender")
    if (preferred in {"男", "女"} and gender in {"男", "女"} and preferred != gender
            and game.requirement.get("required_gender") not in {"男", "女"}):
        return gender
    return None


@dataclass(frozen=True)
class _BoardCandidateInquiry:
    """把画像匹配和本次本人自报统一为同一桌的匿名候选证据。"""

    customer_id: str
    conversation_id: str
    trace_id: str
    expires_at: str
    gender: str
    declaration_message_id: str | None = None


def _active_fit_candidate(store, game, policy, *, bound_candidate_id=None):
    """从同局的真实询局责任选择一个可私聊的候选，只将性别用于发局人话术。"""
    for task in store.scheduled_tasks.values():
        if (task.task_type != BOARD_STAKE_UPDATE_TASK_TYPE or task.aggregate_id != game.game_id
                or task.payload.get("room_id") != policy.room_id or task.subject_id == game.organizer_id
                or (bound_candidate_id is not None and task.subject_id != bound_candidate_id)
                or datetime.fromisoformat(task.payload["expires_at"]) <= now()):
            continue
        profile = store.customers.get(task.subject_id)
        gender = _soft_preference_mismatch(game, getattr(profile, "gender", None))
        identity = store.get_channel_identity_for_customer(task.subject_id, channel=policy.channel)
        if (gender and not profile.no_contact and identity and identity.can_private_message
                and identity.private_conversation_id == task.conversation_id
                and _current_board_inquiry(store, task.conversation_id, task.subject_id,
                                           task.payload["inquiry_trace_id"], game.game_id, needs_stake=True)):
            return _BoardCandidateInquiry(task.subject_id, task.conversation_id, task.payload["inquiry_trace_id"],
                                          task.payload["expires_at"], gender)
    return None


def _board_recipient_key(customer_id):
    """稳定标识同桌同候选的责任，供后端去重，避免把对方身份带入桌主上下文。"""
    return sha256(customer_id.encode()).hexdigest()[:24]


def _candidate_decision_id(game_id, candidate_id):
    """候选协商沿用原询局责任的去重范围，同桌不同候选分别记录。"""
    inquiry_id = f"board-stake-update:{game_id}:{_board_recipient_key(candidate_id)}"
    digest = sha256(f"board-candidate-decision:{inquiry_id}".encode()).hexdigest()[:24]
    return f"board-candidate-decision:{digest}"


def _register_candidate_question(store, question, candidate, trace_id):
    """先记录绑定关系再发送，且只有对应的真实私聊回执能开启这项待答。"""
    task_id = _candidate_decision_id(question.aggregate_id, candidate.customer_id)
    decision = ScheduledAgentTask(task_id=task_id, task_type=BOARD_CANDIDATE_DECISION_TASK_TYPE,
        aggregate_type="game", aggregate_id=question.aggregate_id, conversation_id=question.conversation_id,
        subject_id=question.subject_id, subject_name=question.subject_name,
        due_at=datetime.fromisoformat(candidate.expires_at), idempotency_key=task_id,
        payload={"room_id": question.payload["room_id"], "expires_at": candidate.expires_at,
                 "candidate_id": candidate.customer_id, "candidate_conversation_id": candidate.conversation_id,
                 "inquiry_trace_id": candidate.trace_id, "candidate_gender": candidate.gender,
                 "question_message_id": question.idempotency_key,
                 **({"inquiry_source_message_id": candidate.declaration_message_id}
                    if candidate.declaration_message_id else {})})
    saved, _ = store.ensure_board_candidate_decision_task(decision, trace_id=trace_id)
    return saved.task_id if saved.payload["question_message_id"] == question.idempotency_key else None


def pending_board_candidate_decisions(store, *, conversation_id, sender_id, now):
    """为桌主提供本人已收到的待答项，不投影候选身份、会话或原话。"""
    pending = []
    for task in store.scheduled_tasks.values():
        if (task.task_type != BOARD_CANDIDATE_DECISION_TASK_TYPE or task.subject_id != sender_id
                or task.conversation_id != conversation_id or task.status != ScheduledTaskStatus.PENDING
                or task.due_at <= now):
            continue
        question = store.resolve_message_reference(conversation_id=conversation_id,
                                                  message_id=task.payload["question_message_id"])
        if (question is None or question.conversation_id != conversation_id or question.recipient_id != sender_id
                or not question.metadata.get("delivery_id")
                or question.metadata.get("candidate_decision_task_id") != task.task_id):
            continue
        contact = _contactable_board_host(store, task.aggregate_id, require_unknown_stake=False)
        if contact is None or contact[1].room_id != task.payload["room_id"] or contact[2].customer_id != sender_id:
            continue
        game = contact[0]
        pending.append({"decision_task_id": task.task_id, "game_id": task.aggregate_id,
                        "candidate_gender": task.payload["candidate_gender"], "question_message_id": question.message_id,
                        "question_text": question.text,
                        "question_delivered_at": question.created_at.isoformat(),
                        "time": public_game_start_display(game),
                        "game": game.requirement.get("ruleset") or game.requirement.get("game_type")})
    return pending


def record_board_candidate_decision(store, call, trace_id, conversation_id, sender_id, sender_name):
    """记录桌主对一个已送达问题的答复；回告由事务排队，双方仍无新增席位。"""
    args = call.arguments
    task = store.scheduled_tasks.get(args["decision_task_id"])
    if (task is None or task.task_type != BOARD_CANDIDATE_DECISION_TASK_TYPE
            or task.subject_id != sender_id or task.conversation_id != conversation_id):
        return ToolResult(name=call.name, called=False, allowed=False, error="需要当前桌主本人私聊中的补位待答项。")
    stamp = now()
    update = None
    if args["decision"] == "accepted":
        update_id = task.task_id.replace("board-candidate-decision:", "board-candidate-update:", 1)
        update = ScheduledAgentTask(task_id=update_id, task_type=BOARD_CANDIDATE_UPDATE_TASK_TYPE,
            aggregate_type="game", aggregate_id=task.aggregate_id,
            conversation_id=task.payload["candidate_conversation_id"], subject_id=task.payload["candidate_id"],
            subject_name="", due_at=stamp, idempotency_key=update_id,
            payload={"room_id": task.payload["room_id"], "expires_at": task.payload["expires_at"],
                     "inquiry_trace_id": task.payload["inquiry_trace_id"], "decision_task_id": task.task_id})
    try:
        saved, transitions = store.resolve_board_candidate_decision(task.task_id, host_id=sender_id,
            conversation_id=conversation_id, decision=args["decision"], evidence_message_id=args["evidence_message_id"],
            update_task=update, trace_id=trace_id, at=stamp)
    except ValueError as exc:
        return ToolResult(name=call.name, called=False, allowed=False, error=str(exc))
    result = {"candidate_decision": {"status": saved.payload["decision"], "game_id": task.aggregate_id}}
    if update is not None and transitions:
        result["candidate_update"] = {"task_id": update.task_id, "status": "scheduled"}
    return ToolResult(name=call.name, called=True, allowed=True, state_transitions=transitions, result=result)


def _queue_board_questions(store, game, policy, host, requester, trace_id):
    """在同一写入登记发局人问题与原询局人回告；后者保留真实询局轨迹。"""
    game_id = game.game_id
    task_id = f"board-stake:{game_id}"
    stamp = now()
    expires_at = min(game.expires_at or stamp + timedelta(hours=4), stamp + timedelta(hours=4))
    recipient_key = _board_recipient_key(requester.customer_id)
    update_id = f"board-stake-update:{game_id}:{recipient_key}"
    update = ScheduledAgentTask(task_id=update_id, task_type=BOARD_STAKE_UPDATE_TASK_TYPE,
        aggregate_type="game", aggregate_id=game_id, conversation_id=requester.private_conversation_id,
        subject_id=requester.customer_id, subject_name=requester.public_name, due_at=expires_at, idempotency_key=update_id,
        payload={"room_id": policy.room_id, "expires_at": expires_at.isoformat(), "inquiry_trace_id": trace_id})
    question = ScheduledAgentTask(task_id=task_id, task_type=BOARD_STAKE_CONFIRMATION_TASK_TYPE,
        aggregate_type="game", aggregate_id=game_id, conversation_id=host.private_conversation_id,
        subject_id=host.customer_id, subject_name=host.public_name, due_at=stamp, idempotency_key=task_id,
        payload={"room_id": policy.room_id})
    saved, transitions = store.ensure_board_stake_confirmation_task(question, trace_id=trace_id, update_task=update)
    return saved, update, transitions


def request_board_stake_confirmation(store, call, trace_id, conversation_id, sender_id, sender_name):
    """一桌只核实一次；保存本人私聊的回告责任，模型不能指定收件人。"""
    game_id = call.arguments["game_id"]
    if not _current_board_inquiry(store, conversation_id, sender_id, trace_id, game_id, needs_stake=True):
        return _missing_current_search(call)
    contact = _contactable_board_host(store, game_id)
    if contact is None:
        return ToolResult(name=call.name, called=True, allowed=True,
                          result={"stake_confirmation": {"status": "unavailable"}})
    game, policy, host = contact
    requester = store.get_channel_identity_for_customer(sender_id, channel=policy.channel)
    if (sender_id == host.customer_id or requester is None
            or requester.private_conversation_id != conversation_id):
        return ToolResult(name=call.name, called=False, allowed=False,
                          error="此工具只承接其他客户本人的私聊询局。")
    customer = store.customers.get(sender_id)
    if not requester.can_private_message or (customer and customer.no_contact):
        return ToolResult(name=call.name, called=True, allowed=True,
                          result={"stake_confirmation": {"status": "unavailable"}})
    task, update, transitions = _queue_board_questions(store, game, policy, host, requester, trace_id)
    return ToolResult(name=call.name, called=True, allowed=True, state_transitions=transitions,
        result={"stake_confirmation": {"task_id": task.task_id, "status": task.status.value},
                "stake_update": {"task_id": update.task_id, "status": "registered"},
                "instruction": "先回复本人有这桌、档位待核实；确认后会另行回告。排队不等于已送达、已回复或已占座。"})


def request_board_candidate_confirmation(store, call, trace_id, conversation_id, sender_id, sender_name):
    """承接询局人明确授权的软偏好协商，只安排桌主私聊，不重复核档或修改画像。"""
    game_id = call.arguments["game_id"]
    gender = call.arguments["candidate_gender"]
    source = _current_board_inquiry(store, conversation_id, sender_id, trace_id, game_id, needs_stake=False)
    if not source:
        return _missing_current_search(call)
    if not source.metadata.get("source_message_id") or not _is_private_candidate_input(source):
        return ToolResult(name=call.name, called=False, allowed=False, error="补位征询需要客户本人当前的私聊消息。")
    unavailable = ToolResult(name=call.name, called=True, allowed=True,
                             result={"candidate_confirmation": {"status": "unavailable"}})
    contact = _contactable_board_host(store, game_id, require_unknown_stake=False)
    if contact is None:
        return unavailable
    game, policy, host = contact
    requester = store.get_channel_identity_for_customer(sender_id, channel=policy.channel)
    profile = store.customers.get(sender_id)
    if (sender_id == host.customer_id or requester is None or requester.private_conversation_id != conversation_id
            or not requester.can_private_message or (profile and profile.no_contact)
            or not stake_options(game.requirement) or not _soft_preference_mismatch(game, gender)):
        return unavailable
    existing = store.scheduled_tasks.get(_candidate_decision_id(game_id, sender_id))
    if existing is not None:
        if existing.payload.get("candidate_conversation_id") != conversation_id:
            return unavailable
        return ToolResult(name=call.name, called=True, allowed=True,
                          result={"candidate_confirmation": {"status": "already_requested"}})
    stamp = now()
    expires_at = min(game.expires_at or stamp + timedelta(hours=4), stamp + timedelta(hours=4))
    task_id = f"board-host-fit:{game_id}:{_board_recipient_key(sender_id)}"
    question = ScheduledAgentTask(task_id=task_id, task_type=BOARD_STAKE_CONFIRMATION_TASK_TYPE,
        aggregate_type="game", aggregate_id=game_id, conversation_id=host.private_conversation_id,
        subject_id=host.customer_id, subject_name=host.public_name, due_at=stamp, idempotency_key=task_id,
        payload={"room_id": policy.room_id, "question_kind": "candidate_fit_only", "candidate_id": sender_id,
                 "candidate_conversation_id": conversation_id, "candidate_gender": gender,
                 "inquiry_trace_id": trace_id, "inquiry_source_message_id": source.metadata["source_message_id"],
                 "expires_at": expires_at.isoformat()})
    try:
        saved, transitions = store.ensure_board_stake_confirmation_task(question, trace_id=trace_id)
    except ValueError:
        return unavailable
    return ToolResult(name=call.name, called=True, allowed=True, state_transitions=transitions,
        result={"candidate_confirmation": {"task_id": saved.task_id, "status": saved.status.value},
                "instruction": "只告知已经安排核实能否补位。排队不等于送达、桌主答应或已占座。"})


def _is_private_candidate_input(source):
    """核实私聊入站来源；公开群消息即使借用私聊上下文也不能授权转述个人条件。"""
    metadata = source.metadata
    return not (metadata.get("source") in ("group", "group_private_continuation")
                or metadata.get("is_room") or metadata.get("room_id") or metadata.get("group_topic_source")
                or metadata.get("self_message") or metadata.get("context_invalidated")
                or metadata.get("system_trigger") or metadata.get("internal_event")
                or metadata.get("input_source") == "system_trigger")


def _candidate_declaration(store, payload):
    """只接受绑定到原询局轨迹的本人有效消息，自报性别只在本次协商生效。"""
    source = store.find_conversation_turn(payload["candidate_conversation_id"],
                                          message_id=payload["inquiry_source_message_id"])
    return (source is not None and source.role.value == "user" and source.sender_id == payload["candidate_id"]
            and source.trace_id == payload["inquiry_trace_id"] and _is_private_candidate_input(source))


def _explicit_fit_candidate(store, task, game, policy):
    """外发前重查这次私聊授权、候选收件能力以及发局人的软偏好。"""
    payload = task.payload
    sender_id = payload["candidate_id"]
    profile = store.customers.get(sender_id)
    identity = store.get_channel_identity_for_customer(sender_id, channel=policy.channel)
    if (not stake_options(game.requirement) or datetime.fromisoformat(payload["expires_at"]) <= now()
            or not _soft_preference_mismatch(game, payload["candidate_gender"])
            or not identity or not identity.can_private_message
            or identity.private_conversation_id != payload["candidate_conversation_id"]
            or (profile and profile.no_contact) or not _candidate_declaration(store, payload)):
        return None
    return _BoardCandidateInquiry(sender_id, identity.private_conversation_id, payload["inquiry_trace_id"],
                                  payload["expires_at"], payload["candidate_gender"],
                                  payload["inquiry_source_message_id"])


def _write_stake_message(runtime, game, recipient, trace_id, *, question, candidate_gender=None, ask_stake=True,
                         fit_pending=False, candidate_update=False):
    """用最小公开投影写一句话并独立审查；后台任务不再进入工具规划。"""
    facts = {"time": public_game_start_display(game),
             "game": game.requirement.get("ruleset") or game.requirement.get("game_type")}
    if question:
        goal = "核实发局人本桌的待定条件"
        requirements = []
        examples = []
        if ask_stake:
            requirements.append("直接问这桌打多大。")
            examples.append("七点那桌打多大的？")
        if candidate_gender:
            facts["candidate_gender"] = candidate_gender
            requirements.append(f"说有个{candidate_gender}生想打，问这桌能不能接受。")
            examples.append(f"有个{candidate_gender}生想打，行不？")
        instruction = ("像老板给熟人发微信。一件事一句短话，省掉称呼、铺垫和解释。用实际钟点或玩法指代这桌。"
                       + "".join(requirements) + f"口气参考：{' '.join(examples)}。说完就结束。")
        event = "board_stake_question"
    elif candidate_update:
        goal = "回告询局人这桌愿意接纳他补位"
        facts["host_accepts_candidate"] = True
        facts["stake_confirmed"] = bool(stake_options(game.requirement))
        instruction = ("自然告诉先前询局的人，这桌愿意让他补位。用钟点或玩法指代。"
                       "档位由单独的核档回告负责，这条只说补位进展，正文不要报档位数字。"
                       "档位已确认时问还想不想打；档位待定时说明档位待确认。"
                       "不能说已经给对方留座或牌局已成。")
        event = "board_candidate_update"
    else:
        goal = "回告询局人已核实的档位"
        # 规范化需求可能同时保留 stake 和 base_stake；一组底注只投影一次。
        facts["stake_options"] = [{"min": format_number(lower), "max": format_number(upper),
                                   "cap": format_number(cap) if cap is not None else None}
                                  for lower, upper, cap in dict.fromkeys(stake_options(game.requirement))]
        instruction = ("给先前询局的人写一句纯告知的档位更新，用给定钟点或玩法指出这桌并准确报档位。"
                       "min=max 就按店里口语说这桌打几块，min<max 就说区间；cap 是另列的封顶。"
                       "说清档位就结束，不问是否合适、是否过来或要不要参加，也不请对方确认。")
        if fit_pending:
            facts["host_acceptance_pending"] = True
            instruction += "同时说能不能补位还待确认。"
        event = "board_stake_update"
    # 把本次要说的内容和内部约束分开，避免模型把核实规则念给客户听。
    visible_boundary = ("像熟客微信私聊，直接说与本次沟通相关的牌局事实和问题。事实只用给定信息，"
                        "不可透露姓名、身份标识或私聊原文，也不可把询局说成已报名或已占座。"
                        "以上是写作边界，正文不复述写作要求，也不解释自己的判断规则或处理流程。")
    messages = [{"role": "system", "content": f"你是麻将馆老板。{instruction}{visible_boundary}"
                 "只输出 JSON 对象，唯一字段 text 为待发送正文。"},
                {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]
    stake_notice = event == "board_stake_update"
    message = UserMessage(conversation_id=recipient.private_conversation_id, sender_id=recipient.customer_id,
        sender_name=recipient.public_name, text=f"系统任务：{goal}。{instruction}")
    status = "waiting_user" if question or facts.get("stake_confirmed") else "completed"
    scope = "只发相关牌局事实和指定的问题。正文复述生成要求、解释内部判断流程或追加无关问题，都应拒绝。"
    if candidate_update:
        scope += "本次是补位回告，未提供档位金额；正文报档位数字应拒绝。"
    if stake_notice:
        scope += "本次只告知已核实档位，没有待答事项；任何追问、邀约或请求确认都应拒绝。"
    for attempt in range(2 if stake_notice else 1):
        runtime.trace_recorder.record(trace_id, f"{event}_prompt", {"messages": messages, "attempt": attempt + 1})
        raw = runtime.llm_client.complete(messages, trace_id=trace_id, timeout_seconds=min(20, runtime.llm_timeout_seconds))
        payload = json.loads(str(raw))
        text = payload.get("text") if isinstance(payload, dict) and set(payload) == {"text"} else None
        limit = 42 if question else 100
        if not isinstance(text, str) or not 0 < len(text.strip()) <= limit or customer_visible_text_contract_violations(text):
            raise RuntimeError("board stake message failed the text contract")
        text = text.strip()
        if question and text.count("?") + text.count("？") > len(requirements):
            raise RuntimeError("board stake message includes extra questions")
        action = AgentAction(goal=goal, objective_status=status, reasoning_summary=goal, reply_to_user=text)
        review = runtime.action_processor.review_customer_visible_content(message=message, trace_id=trace_id, action=action,
            review_items=[{"item_id": event, "source": "reply_to_user", "recipient_id": recipient.customer_id, "text": text}],
            context_payload={"active_game_visible_summaries": [facts], "customer_visibility_contract": {
                "audience": "private", "private_identities_allowed": False, "message_scope": scope}},
            turn_budget=TokenBudget(max_calls_per_turn=1), review_scope="reply_to_user")
        if review is not None and customer_visible_content_review_approved(review):
            return AgentRuntimeResult(trace_id=trace_id, conversation_id=recipient.private_conversation_id,
                                      final_reply=text, status=status, actions=[action], tool_results=[review])
        rejection = review.result if review is not None else {}
        if (not stake_notice or attempt or rejection.get("needs_human")
                or rejection.get("violations") != ["semantic_contradiction"]):
            break
        # 审查结论只用于纠错。重写仍从权威事实生成并独立复审，不能把审查器的建议直接当成可发送正文。
        messages = [*messages, {"role": "user", "content": json.dumps({
            "violations": rejection["violations"],
            "instruction": "重写成仅告知准确档位的陈述句，保留钟点或玩法，不要求回复、确认或参加。"
        }, ensure_ascii=False)}]
    raise RuntimeError("board stake message content review did not approve delivery")


def _board_update_target(store, task, *, require_known_stake):
    """过期、已满、已关闭或无私聊许可的订阅不能发，也不切到群里通知。"""
    stamp = now()
    if datetime.fromisoformat(task.payload["expires_at"]) <= stamp:
        return None
    try:
        game = store.require_game(task.aggregate_id)
    except ValueError:
        return None
    if (game.requirement.get("recruitment_mode") != "board_only"
            or (require_known_stake and not stake_options(game.requirement))
            or game.status not in {GameStatus.FORMING, GameStatus.INVITING} or game.remaining_seats() < 1
            or (game.expires_at and game.expires_at <= stamp) or game.organizer_id == task.subject_id):
        return None
    room_id = task.payload["room_id"]
    if not any(link.link_type == "origin" and link.room_id == room_id
               for link in store.game_conversation_links(game_id=game.game_id)):
        return None
    policy = store.get_group_room_policy(room_id)
    if not policy or not policy.managed or not policy.board_enabled or not policy.outbound_enabled:
        return None
    customer = store.customers.get(task.subject_id)
    recipient = store.get_channel_identity_for_customer(task.subject_id, channel=policy.channel)
    if (not recipient or recipient.private_conversation_id != task.conversation_id
            or not recipient.can_private_message or (customer and customer.no_contact)):
        return None
    # 档位可能在原回复送达前已确认。只依据本人私聊的交付证据放行回告。
    turns = [turn for turn in store.recent_turns(task.conversation_id, limit=80)
             if turn.trace_id == task.payload["inquiry_trace_id"]]
    if not any(turn.role.value == "user" and turn.sender_id == task.subject_id
               and not turn.metadata.get("context_invalidated") for turn in turns):
        return None
    replies = [turn for turn in turns if turn.role.value == "assistant"]
    if any(turn.metadata.get("delivery_status") == "delivered" for turn in replies):
        return game, policy, recipient
    if not replies or any(turn.metadata.get("delivery_status") in {"pending_operator_send", "dispatching"}
                          for turn in replies):
        raise ScheduledTaskDeferred("board_stake_initial_reply_not_delivered", delay_seconds=5)
    return None


def _candidate_update_target(store, task):
    """回告仅使用持久的桌主同意，并重查原局、原询局人及当前硬性条件。"""
    decision = store.scheduled_tasks.get(task.payload["decision_task_id"])
    if (decision is None or decision.payload.get("decision") != "accepted"
            or decision.aggregate_id != task.aggregate_id or decision.payload.get("candidate_id") != task.subject_id
            or decision.payload.get("candidate_conversation_id") != task.conversation_id):
        return None
    evidence = store.find_conversation_turn(decision.conversation_id,
                                           message_id=decision.payload["evidence_message_id"])
    if evidence is None or evidence.metadata.get("context_invalidated"):
        return None
    contact = _board_update_target(store, task, require_known_stake=False)
    if contact is None or contact[0].organizer_id != decision.subject_id:
        return None
    profile = store.customers.get(task.subject_id)
    if decision.payload.get("inquiry_source_message_id"):
        if not _candidate_declaration(store, decision.payload):
            return None
        gender = decision.payload["candidate_gender"]
    else:
        gender = getattr(profile, "gender", None)
    required_gender = contact[0].requirement.get("required_gender")
    if gender != decision.payload["candidate_gender"] or (required_gender in {"男", "女"} and required_gender != gender):
        return None
    return contact


def _deliver_stake_message(runtime, messenger, task, trace_id, *, target, question, candidate_update=False):
    """共同的交付边界：审查后复核权限、游戏版本、会话水位与当前领取凭证。"""
    contact = target()
    if contact is None:
        return None
    game, policy, recipient = contact
    if (policy.room_id != task.payload["room_id"] or recipient.customer_id != task.subject_id
            or recipient.private_conversation_id != task.conversation_id):
        return None
    if runtime.store.resolve_message_reference(conversation_id=task.conversation_id,
                                               message_id=task.idempotency_key) is not None:
        return None
    version = game.updated_at
    conversation_version = runtime.store.conversation_version(task.conversation_id)
    fit_only = question and task.payload.get("question_kind") == "candidate_fit_only"
    bound_candidate_id = None
    if question:
        # 发送重试可能复用通道的旧回执；同一条问话必须始终绑定原候选。
        bound = next((item for item in runtime.store.scheduled_tasks.values()
                      if item.task_type == BOARD_CANDIDATE_DECISION_TASK_TYPE and item.aggregate_id == task.aggregate_id
                      and item.conversation_id == task.conversation_id
                      and item.payload.get("question_message_id") == task.idempotency_key), None)
        if bound is not None:
            bound_candidate_id = bound.payload["candidate_id"]

    def projected_fit(current_game, current_policy):
        """发局人只看到匿名候选类型，询局人只看到参加资格仍待确认。"""
        if question:
            if fit_only:
                return _explicit_fit_candidate(runtime.store, task, current_game, current_policy)
            return _active_fit_candidate(runtime.store, current_game, current_policy,
                                         bound_candidate_id=bound_candidate_id)
        if candidate_update:
            return None
        decision = runtime.store.scheduled_tasks.get(_candidate_decision_id(task.aggregate_id, task.subject_id))
        if decision is not None and decision.payload.get("decision") == "accepted":
            return False
        profile = runtime.store.customers.get(task.subject_id)
        return bool(_soft_preference_mismatch(current_game, getattr(profile, "gender", None)))

    fit = projected_fit(game, policy)
    if (fit_only or bound_candidate_id is not None) and not fit:
        return None
    result = _write_stake_message(runtime, game, recipient, trace_id, question=question,
        candidate_gender=fit.gender if question and fit else None, ask_stake=not fit_only,
        fit_pending=bool(fit) and not question, candidate_update=candidate_update)
    decision_id = None
    with runtime.input_receipt_lock(result.conversation_id):
        latest = target()
        if latest is None:
            return None
        if latest[0].updated_at != version or latest[1:] != contact[1:] or projected_fit(latest[0], latest[1]) != fit:
            raise ScheduledTaskDeferred("board_stake_facts_changed_during_review", delay_seconds=5)
        claim = runtime.store.scheduled_tasks.get(task.task_id)
        if (claim is None or task.lease_until is None or task.lease_until <= now()
                or not claim.matches_claim(expected_attempts=task.attempts, expected_lease_until=task.lease_until)):
            return None
        if runtime.store.conversation_version(result.conversation_id) != conversation_version:
            raise ScheduledTaskDeferred("board_stake_conversation_changed_during_review", delay_seconds=5)
        if question and fit:
            decision_id = _register_candidate_question(runtime.store, task, fit, trace_id)
            if not decision_id:
                # 同一候选已绑定另一条实际问话时，不伪造本次回执或重复打扰桌主。
                return None
        runtime.store.append_assistant_turn(result.conversation_id, result.final_reply, trace_id,
            metadata={"delivery_status": "pending_operator_send", "conversation_version": conversation_version,
                      "source_message_id": task.idempotency_key})
        if not runtime.store.claim_assistant_reply_delivery(result.conversation_id, trace_id=result.trace_id):
            raise ScheduledTaskDeferred("board_stake_reply_not_claimed", delay_seconds=5)
    receipt = messenger.send_private_message(recipient.external_user_id, result.final_reply,
        metadata={"trace_id": trace_id, "draft_id": task.idempotency_key, "source_message_id": task.idempotency_key,
                  "origin_room_id": policy.room_id, "conversation_id": task.conversation_id})
    if not receipt or receipt.startswith("suppressed:"):
        raise RuntimeError("board stake message delivery was not acknowledged")
    runtime.store.register_message_reference(MessageReference(
        message_id=task.idempotency_key, conversation_id=task.conversation_id,
        business_ref_type=task.task_type, business_ref_id=game.game_id,
        text=result.final_reply, channel=policy.channel, recipient_id=recipient.customer_id,
        recipient_name=recipient.public_name, metadata={"delivery_id": receipt, "trace_id": trace_id,
            **({"candidate_fit_gender": fit.gender, "candidate_decision_task_id": decision_id} if decision_id else {})},
    ))
    runtime.store.mark_assistant_reply_delivered(result.conversation_id, trace_id=result.trace_id)
    runtime.trace_recorder.record(trace_id, f"{task.task_type}_sent", {
        "task_id": task.task_id, "game_id": game.game_id, "delivery_id": receipt,
    })
    return result


def handle_board_stake_confirmation(runtime, messenger, task, trace_id):
    """向桌主核实档位和实际候选的软偏好，只传匿名的必要信息。"""
    return _deliver_stake_message(runtime, messenger, task, trace_id,
        target=lambda: _contactable_board_host(runtime.store, task.aggregate_id,
            require_unknown_stake=task.payload.get("question_kind") != "candidate_fit_only"), question=True)


def handle_board_stake_update(runtime, messenger, task, trace_id):
    """在桌主确认后只回告原询局人，不自动推荐为适配、邀约或占座。"""
    return _deliver_stake_message(runtime, messenger, task, trace_id,
        target=lambda: _board_update_target(runtime.store, task, require_known_stake=True), question=False)


def handle_board_candidate_update(runtime, messenger, task, trace_id):
    """桌主确认后另行回告候选人，领取和外发独立于档位任务，且不写席位。"""
    return _deliver_stake_message(runtime, messenger, task, trace_id,
        target=lambda: _candidate_update_target(runtime.store, task), question=False, candidate_update=True)


def permits_board_stake_answer(store, game, *, conversation_id, sender_id, patch):
    """发局人在收到的私聊中只能补本桌档位；该收据不授权跨会话改人、时间或取消。"""
    if (game.requirement.get("recruitment_mode") != "board_only" or game.organizer_id != sender_id
            or not patch or set(patch) - {"stake", "base_stake", "stake_range", "cap_score", "stake_label"}):
        return False
    reference = store.resolve_message_reference(conversation_id=conversation_id,
                                               message_id=f"board-stake:{game.game_id}")
    return (reference is not None and reference.business_ref_type == BOARD_STAKE_CONFIRMATION_TASK_TYPE
            and reference.business_ref_id == game.game_id and reference.recipient_id == sender_id
            and bool(reference.metadata.get("delivery_id")))
