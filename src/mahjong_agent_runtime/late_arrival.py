"""把晚到征询、原桌联系人答复和回告绑定，禁止征询过程写席位或开场。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Any

from .domains.game_domain import START_KIND_ASAP_WHEN_FULL
from .domains.game_participants import participation_table_terms
from .domains.model_context import has_agreed_game_duration
from .models import DEFAULT_TZ, ConversationRole, Game, GameStatus, ScheduledAgentTask, ScheduledTaskStatus, StateTransition, ToolResult, now

if TYPE_CHECKING:
    from .stores import AgentStore


LATE_ARRIVAL_QUESTION_TASK_TYPE = "late_arrival_question"
LATE_ARRIVAL_UPDATE_TASK_TYPE = "late_arrival_update"
_ACTIVE_QUESTION_STATUSES = {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING, ScheduledTaskStatus.COMPLETED}
_ORIGIN_LINK_TYPES = {"origin", "private_switch_created", "private_board_created"}


class LateArrivalLookupRequired(ValueError):
    """本轮尚无准确查桌凭据，事务计划在安排任何征询之前退出。"""


def _aware(value: Any) -> datetime | None:
    """只接受明确时区的时间，不由服务端为模型猜测日期或时区。"""
    try:
        at = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return at.astimezone(DEFAULT_TZ) if at.utcoffset() is not None else None


def _private_input(turn, sender_id: str) -> bool:
    """只以存库的本人有效私聊作证，系统任务和撤回消息不能发起或同意协商。"""
    if turn is None or turn.role != ConversationRole.USER or turn.sender_id != sender_id:
        return False
    meta = turn.metadata
    return bool(meta.get("source_message_id") and not (
        meta.get("source") in {"group", "group_private_continuation"} or meta.get("is_room")
        or meta.get("room_id") or meta.get("group_topic_source") or meta.get("context_invalidated")
        or meta.get("self_message") or meta.get("system_trigger") or meta.get("internal_event")
        or meta.get("input_source") == "system_trigger"
    ))


def _same_turn_search(store, conversation_id: str, sender_id: str, trace_id: str, game_id: str):
    """查原始工具轮的最新成功精确查局，历史、备选和模型自述都不建立联系资格。"""
    source = None
    latest_search = None
    for turn in reversed(store.recent_turns(conversation_id, limit=60)):
        if turn.trace_id != trace_id or turn.metadata.get("context_invalidated"):
            continue
        if source is None and _private_input(turn, sender_id):
            source = turn
        if latest_search is not None or turn.role != ConversationRole.TOOL:
            continue
        try:
            results = json.loads(turn.content)
        except (ValueError, TypeError):
            continue
        if isinstance(results, list):
            latest_search = next((row for row in reversed(results) if isinstance(row, dict)
                and row.get("name") == "search_current_games" and row.get("called") is True
                and row.get("allowed") is True and not row.get("error")), None)
    result = latest_search.get("result") if latest_search else None
    matches = result.get("matches") if isinstance(result, dict) and not result.get("selection_truncated") else None
    match = matches[0] if isinstance(matches, list) and len(matches) == 1 else None
    game = match.get("game") if isinstance(match, dict) and match.get("match_kind") == "exact" else None
    return (source, game) if source is not None and isinstance(game, dict) and game.get("game_id") == game_id else None


def _contacts(game: Game) -> list[str]:
    """每个已承诺座位方由其登记联系人答复；桌主始终负责自己的决定。"""
    return sorted({game.organizer_id} | {
        party.contact_id for party in game.parties if party.status in {"joined", "confirmed"} and party.contact_id
    })


def _terms(game: Game) -> dict:
    """同一轮征询固定牌局条件与现有承诺方，任何改期或承诺变动都会使旧同意失效。"""
    value = participation_table_terms(game)
    value["planned_end_at"] = game.planned_end_at.isoformat() if game.planned_end_at else None
    value["claimed_seats"] = game.seats_total - game.remaining_seats()
    value["roster"] = sorted(
        [party.contact_id, party.seat_count, sorted(party.known_member_ids)]
        for party in game.parties if party.status in {"joined", "confirmed"}
    )
    for field in ("duration_hours", "duration"):
        if field in game.requirement:
            value["terms"][field] = game.requirement[field]
    return value


def _actionable(game: Game, arrival_at: datetime, at: datetime) -> bool:
    """仍在招募且确实需要商量晚到，过期与实际开打由权威牌局状态保护。"""
    return bool(game.status in {GameStatus.FORMING, GameStatus.INVITING}
        and not game.actual_started_at and game.remaining_seats() > 0
        and game.requirement.get("recruitment_mode") != "board_only"
        and game.requirement.get("start_time_kind") != START_KIND_ASAP_WHEN_FULL
        and game.planned_start_at and game.planned_start_at < arrival_at
        and game.planned_end_at and arrival_at < game.planned_end_at
        and (game.expires_at is None or game.expires_at > at))


def _contact(store, customer_id: str, channel: str, conversation_id: str | None = None):
    """在既有可私聊身份内选联系人，不接受模型提供的外部地址。"""
    profile = store.customers.get(customer_id)
    identity = store.get_channel_identity_for_customer(customer_id, channel=channel)
    if (not identity or identity.customer_id != customer_id or not identity.can_private_message
            or not identity.private_conversation_id or profile and profile.no_contact
            or conversation_id is not None and identity.private_conversation_id != conversation_id):
        return None
    return identity


def _route(store, game: Game, candidate_id: str, candidate_conversation_id: str, *, room_id: str | None = None):
    """同一授权牌局来源渠道须能联系所有责任方，联系不齐便不能给出整桌同意。"""
    if not game.organizer_id or candidate_id in _contacts(game):
        return None
    for link in sorted(store.game_conversation_links(game_id=game.game_id), key=lambda link: link.room_id):
        if link.link_type not in _ORIGIN_LINK_TYPES or room_id is not None and link.room_id != room_id:
            continue
        policy = store.get_group_room_policy(link.room_id)
        if not policy or not policy.managed or not policy.board_enabled or not policy.outbound_enabled:
            continue
        candidate = _contact(store, candidate_id, policy.channel, candidate_conversation_id)
        contacts = {account: _contact(store, account, policy.channel) for account in _contacts(game)}
        if candidate and all(contacts.values()):
            return policy, candidate, contacts
    return None


def _base_payload(game, policy, candidate, contacts, source, arrival_at: datetime) -> dict:
    """响应的稳定锚点覆盖候选人、整桌条件和到店约定，同方案重投复用原沟通。"""
    value = {
        "room_id": policy.room_id,
        "candidate_id": candidate.customer_id,
        "candidate_conversation_id": candidate.private_conversation_id,
        "arrival_at": arrival_at.isoformat(),
        "table_terms": _terms(game),
        "required_contact_ids": sorted(contacts),
    }
    digest = sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    expiry = min(arrival_at, game.planned_end_at, game.expires_at or game.planned_end_at)
    return {**value, "negotiation_id": f"late-arrival:{digest}", "source_message_id": source.metadata["source_message_id"],
            "source_trace_id": source.trace_id, "expires_at": expiry.isoformat()}


def _question(payload, game, contact, at) -> ScheduledAgentTask:
    """每位联系人单独收到最小问题，后端任务键承担重复发起的幂等。"""
    suffix = sha256(contact.customer_id.encode()).hexdigest()[:24]
    task_id = f"{payload['negotiation_id']}:question:{suffix}"
    return ScheduledAgentTask(
        task_id=task_id, task_type=LATE_ARRIVAL_QUESTION_TASK_TYPE, aggregate_type="game",
        aggregate_id=game.game_id, conversation_id=contact.private_conversation_id, subject_id=contact.customer_id,
        subject_name=contact.public_name, due_at=at, idempotency_key=task_id, payload=dict(payload))


def _question_tasks(store, negotiation_id: str) -> list[ScheduledAgentTask]:
    """同一时间提案的所有联系人问题共享一个不可变锚点。"""
    return [task for task in store.scheduled_tasks.values()
            if task.task_type == LATE_ARRIVAL_QUESTION_TASK_TYPE and task.payload.get("negotiation_id") == negotiation_id]


def _live_scope(store, task: ScheduledAgentTask, at: datetime, *, require_unexpired: bool = True):
    """执行要求有效期限；本人查旧指代时仍核对原桌、联系边界和当前群体。"""
    payload = task.payload
    arrival, expiry = _aware(payload.get("arrival_at")), _aware(payload.get("expires_at"))
    if task.aggregate_type != "game" or arrival is None or expiry is None or (require_unexpired and at >= expiry):
        return None
    try:
        game = store.require_game(task.aggregate_id)
    except ValueError:
        return None
    if not _actionable(game, arrival, at) or payload.get("table_terms") != _terms(game):
        return None
    latest = _latest_candidate_tasks(store, game_id=game.game_id, candidate_id=payload["candidate_id"])
    if not latest or latest[0].payload["negotiation_id"] != payload["negotiation_id"]:
        return None
    route = _route(store, game, payload["candidate_id"], payload["candidate_conversation_id"], room_id=payload["room_id"])
    if route is None or sorted(route[2]) != payload.get("required_contact_ids"):
        return None
    source = store.find_conversation_turn(payload["candidate_conversation_id"], message_id=payload["source_message_id"])
    if not _private_input(source, payload["candidate_id"]) or source.trace_id != payload.get("source_trace_id"):
        return None
    return game, arrival, route


def late_arrival_delivery_receipt(store, task: ScheduledAgentTask, *, at: datetime):
    """已登记的渠道回执须匹配这个人、会话、牌局和提案，排队不能替代送达。"""
    reference = store.resolve_message_reference(conversation_id=task.conversation_id, message_id=task.idempotency_key)
    if (reference is None or reference.conversation_id != task.conversation_id
            or reference.recipient_id != task.subject_id or reference.business_ref_type != task.task_type
            or reference.business_ref_id != task.aggregate_id or reference.created_at > at
            or reference.metadata.get("negotiation_id") != task.payload.get("negotiation_id")
            or not reference.metadata.get("delivery_id")):
        return None
    return reference


def _verified_decision(store, task, receipt, at: datetime) -> str | None:
    """答复撤回或对应送达变化后，已写入的答复也不足以继续占席。"""
    payload = task.payload
    if payload.get("decision") not in {"accepted", "declined"} or not receipt:
        return None
    if payload.get("question_delivery_id") != receipt.metadata["delivery_id"]:
        return None
    evidence = store.find_conversation_turn(task.conversation_id, message_id=payload["evidence_message_id"])
    if (not _private_input(evidence, task.subject_id) or evidence.trace_id != payload.get("evidence_trace_id")
            or evidence.occurred_at.utcoffset() is None or not receipt.created_at <= evidence.occurred_at <= at):
        return None
    return payload["decision"]


def _status(store, questions: list[ScheduledAgentTask], at: datetime) -> tuple[str, bool]:
    """拒绝立即生效，同意须全体责任方有已送达问题的可靠答复。"""
    receipts = [late_arrival_delivery_receipt(store, task, at=at) for task in questions]
    decisions = [_verified_decision(store, task, receipt, at) for task, receipt in zip(questions, receipts)]
    if "declined" in decisions:
        return "declined", any(receipts)
    if decisions and all(decision == "accepted" for decision in decisions):
        return "accepted", True
    if any(task.status not in _ACTIVE_QUESTION_STATUSES for task in questions):
        return "unavailable", any(receipts)
    return ("pending" if any(receipts) else "queued"), any(receipts)


def late_arrival_delivery_context(store: AgentStore, task: ScheduledAgentTask, *, at: datetime) -> dict | None:
    """只向工作器提供此任务的收件人与业务事实，已决或过期问题不得迟发。"""
    live = _live_scope(store, task, at)
    if live is None or task.task_type not in {LATE_ARRIVAL_QUESTION_TASK_TYPE, LATE_ARRIVAL_UPDATE_TASK_TYPE}:
        return None
    game, arrival, (policy, candidate, contacts) = live
    questions = _question_tasks(store, task.payload["negotiation_id"])
    if not questions or sorted(question.subject_id for question in questions) != sorted(contacts):
        return None
    status, _ = _status(store, questions, at)
    if task.task_type == LATE_ARRIVAL_QUESTION_TASK_TYPE:
        recipient = contacts.get(task.subject_id)
        if status in {"accepted", "declined", "unavailable"} or task.payload.get("decision"):
            return None
    else:
        recipient = candidate
        if task.payload.get("decision") != status or status not in {"accepted", "declined"}:
            return None
    if recipient is None or (recipient.customer_id, recipient.private_conversation_id) != (task.subject_id, task.conversation_id):
        return None
    return {"game": game, "policy": policy, "recipient": recipient, "arrival_at": arrival,
            "decision": task.payload.get("decision"), "status": status}


def plan_late_arrival_request(store: AgentStore, *, game_id: str, arrival_at: datetime, sender_id: str,
                              conversation_id: str, trace_id: str, at: datetime):
    """在调用方写锁内校验真实来源、牌局及私聊授权，然后生成全部责任任务。"""
    query = _same_turn_search(store, conversation_id, sender_id, trace_id, game_id)
    if query is None:
        raise LateArrivalLookupRequired("请先按这桌原条件查到同一桌，再安排晚到核实。")
    source, observed = query
    if source.occurred_at.utcoffset() is None or source.occurred_at > at:
        raise ValueError("请先按这桌原条件查到同一桌，再安排晚到核实。")
    try:
        game = store.require_game(game_id)
    except ValueError as exc:
        raise ValueError("该桌目前无法安排晚到核实。") from exc
    route = _route(store, game, sender_id, conversation_id)
    if arrival_at <= at or not _actionable(game, arrival_at, at) or route is None:
        raise ValueError("该桌当前局况、到店时间或联系人授权不足，暂不能完成征询。")
    if _aware(observed.get("planned_start_at")) != game.planned_start_at or _aware(observed.get("planned_end_at")) != game.planned_end_at:
        raise ValueError("该桌时间有变，请重新核实当前局况。")
    policy, candidate, contacts = route
    payload = _base_payload(game, policy, candidate, contacts, source, arrival_at)
    questions = _question_tasks(store, payload["negotiation_id"])
    previous = _latest_candidate_tasks(store, game_id=game_id, candidate_id=sender_id)
    if questions:
        if previous and previous[0].payload["negotiation_id"] != payload["negotiation_id"]:
            raise ValueError("另一个到店时间正在核实，请以最近的协商为准。")
        return questions, []
    if previous:
        payload["previous_negotiation_id"] = previous[0].payload["negotiation_id"]
    questions = [_question(payload, game, contact, at) for contact in contacts.values()]
    return questions, [StateTransition("scheduled_agent_task", task.task_id, None, task.status.value,
        "late_arrival_question_queued", trace_id) for task in questions]


def _pending_decision_tasks(store, *, conversation_id: str, sender_id: str, at: datetime):
    """待答只来自已私聊送达的问题，候选人身份、原话和同桌其他人的答复不进入投影。"""
    for task in store.scheduled_tasks.values():
        if (task.task_type != LATE_ARRIVAL_QUESTION_TASK_TYPE or task.subject_id != sender_id
                or task.conversation_id != conversation_id or task.payload.get("decision")):
            continue
        context = late_arrival_delivery_context(store, task, at=at)
        receipt = late_arrival_delivery_receipt(store, task, at=at)
        if context and receipt:
            yield task, context, receipt


def _validate_reply_binding(store, task, receipt, evidence, *, at: datetime) -> None:
    """多件待答并存时使用原生问题引用，避免把同一句同意写入多份协商。"""
    quoted = evidence.metadata.get("input_receipt_quoted_message")
    if isinstance(quoted, dict):
        if (quoted.get("conversation_id") not in {None, task.conversation_id}
                or quoted.get("message_id") not in {receipt.message_id, receipt.metadata["delivery_id"]}):
            raise ValueError("请使用对当前晚到问题的原始答复。")
        return
    pending = list(_pending_decision_tasks(store, conversation_id=task.conversation_id, sender_id=task.subject_id, at=at))
    reused = any(other.task_id != task.task_id and other.subject_id == task.subject_id
        and other.conversation_id == task.conversation_id
        and other.payload.get("evidence_message_id") == evidence.metadata.get("source_message_id")
        for other in store.scheduled_tasks.values() if other.task_type == LATE_ARRIVAL_QUESTION_TASK_TYPE)
    if len(pending) != 1 or pending[0][0].task_id != task.task_id or reused:
        raise ValueError("当前存在多条待答，请引用所答复的那条晚到问题。")


def _update_task(saved, questions, status, at):
    """原子登记给原询局人的结果回告，接受与拒绝不改原桌和席位。"""
    payload = saved.payload
    task_id = f"{payload['negotiation_id']}:update"
    private = {k: payload[k] for k in ("negotiation_id", "room_id", "candidate_id", "candidate_conversation_id",
        "arrival_at", "expires_at", "table_terms", "required_contact_ids", "source_message_id", "source_trace_id")}
    return ScheduledAgentTask(
        task_id=task_id, task_type=LATE_ARRIVAL_UPDATE_TASK_TYPE, aggregate_type="game",
        aggregate_id=saved.aggregate_id, conversation_id=payload["candidate_conversation_id"],
        subject_id=payload["candidate_id"], subject_name="", due_at=at, idempotency_key=task_id,
        payload={**private, "decision": status, "decision_task_ids": sorted(task.task_id for task in questions)},
    )


def plan_late_arrival_decision(store: AgentStore, task_id: str, *, sender_id: str, conversation_id: str,
                               decision: str, evidence_message_id: str, trace_id: str, at: datetime):
    """在调用方事务中验证同一私聊、交付后本人原始答复和原桌未变，生成原子写入。"""
    task = store.scheduled_tasks.get(task_id)
    if (task is None or task.task_type != LATE_ARRIVAL_QUESTION_TASK_TYPE
            or task.subject_id != sender_id or task.conversation_id != conversation_id
            or decision not in {"accepted", "declined"}):
        raise ValueError("需要本人收到的晚到问题。")
    if task.payload.get("decision"):
        if task.payload["decision"] == decision and task.payload.get("evidence_message_id") == evidence_message_id:
            group_status, _ = _status(store, _question_tasks(store, task.payload["negotiation_id"]), at)
            return task, None, [], group_status
        raise ValueError("该晚到问题已经完成。")
    context = late_arrival_delivery_context(store, task, at=at)
    receipt = late_arrival_delivery_receipt(store, task, at=at)
    evidence = store.find_conversation_turn(conversation_id, message_id=evidence_message_id)
    if not context or not receipt or not _private_input(evidence, sender_id) or evidence.trace_id != trace_id:
        raise ValueError("需要在私聊答复本人收到的有效晚到问题。")
    if evidence.occurred_at.utcoffset() is None or not receipt.created_at <= evidence.occurred_at <= at:
        raise ValueError("晚到答复必须发生在问题送达后。")
    _validate_reply_binding(store, task, receipt, evidence, at=at)
    saved = replace(task, updated_at=at, payload={**task.payload, "decision": decision,
        "evidence_message_id": evidence_message_id, "evidence_trace_id": evidence.trace_id,
        "question_delivery_id": receipt.metadata["delivery_id"]})
    questions = [saved if other.task_id == task_id else other
                 for other in _question_tasks(store, task.payload["negotiation_id"])]
    status, _ = _status(store, questions, at)
    update = _update_task(saved, questions, status, at) if status in {"accepted", "declined"} else None
    changes = [StateTransition("scheduled_agent_task", saved.task_id, task.status.value, saved.status.value,
                               f"late_arrival_{decision}", trace_id)]
    if update is not None:
        changes.append(StateTransition("scheduled_agent_task", update.task_id, None, update.status.value,
                                       "late_arrival_update_queued", trace_id))
    return saved, update, changes, status


def _latest_candidate_tasks(store, *, game_id: str, candidate_id: str, conversation_id: str | None = None):
    """候选人最新提出的有效期版本优先，不能改用之前同一人的较早同意。"""
    proposals = [task for task in store.scheduled_tasks.values()
        if task.task_type == LATE_ARRIVAL_QUESTION_TASK_TYPE and task.aggregate_id == game_id
        and task.payload.get("candidate_id") == candidate_id]
    superseded = {task.payload.get("previous_negotiation_id") for task in proposals}
    latest = max((task for task in proposals if task.payload["negotiation_id"] not in superseded),
                 key=lambda task: (task.created_at, task.task_id), default=None)
    if latest is None or (conversation_id is not None and latest.payload["candidate_conversation_id"] != conversation_id):
        return []
    return [task for task in proposals if task.payload["negotiation_id"] == latest.payload["negotiation_id"]]


def _candidate_status(store, questions: list[ScheduledAgentTask], at: datetime) -> str:
    """全部原桌答复和候选人回告的回执齐备后，才允许申请人使用该同意。"""
    question = questions[0]
    if _live_scope(store, question, at) is None:
        return "stale"
    status, _ = _status(store, questions, at)
    if status != "accepted":
        return status
    update = store.scheduled_tasks.get(f"{question.payload['negotiation_id']}:update")
    if (update is None or update.task_type != LATE_ARRIVAL_UPDATE_TASK_TYPE
            or update.payload.get("decision") != "accepted"
            or update.payload.get("decision_task_ids") != sorted(task.task_id for task in questions)
            or not late_arrival_delivery_receipt(store, update, at=at)):
        return "pending"
    return "accepted"


def late_arrival_seat_status(store: AgentStore, *, game: Game, customer_id: str, at: datetime) -> str | None:
    """在席位事务中复查旧同意与新条件，不让后台征询本身成为报名授权。"""
    questions = _latest_candidate_tasks(store, game_id=game.game_id, candidate_id=customer_id)
    if not questions:
        return None
    arrival = _aware(questions[0].payload["arrival_at"])
    if game.planned_start_at is not None and arrival is not None and at <= game.planned_start_at and arrival <= game.planned_start_at:
        return None  # 原来申报的到店时间已赶得上新开场，按正常报名重新核实。
    return _candidate_status(store, questions, at)


def late_arrival_result_for_candidate(store: AgentStore, *, game_id: str, conversation_id: str, sender_id: str,
                                      at: datetime) -> dict | None:
    """返回本人协商状态所需的最小桌况，不包含桌主身份、私聊或投票详情。"""
    questions = _latest_candidate_tasks(store, game_id=game_id, candidate_id=sender_id, conversation_id=conversation_id)
    if not questions:
        return None
    question = questions[0]
    payload = question.payload
    status = _candidate_status(store, questions, at)
    # 约定到店时间过去后，原请求仍可定位同一桌。它不能恢复同意、继续发话或授权占席。
    arrival = _aware(payload.get("arrival_at"))
    if status == "stale" and arrival is not None and at >= arrival:
        if _live_scope(store, question, at, require_unexpired=False) is not None:
            status = "arrival_elapsed"
    return _candidate_request_summary(question, status)


def _candidate_request_summary(question: ScheduledAgentTask, status: str) -> dict[str, Any]:
    """提供本人与原桌的协商事实。只对同桌已约定的散场时间计算本人可参与的剩余时长。"""
    payload = question.payload
    table = payload["table_terms"]
    summary = {"game_id": question.aggregate_id, "arrival_at": payload["arrival_at"],
               "planned_start_at": table["planned_start_at"], "status": status}
    arrival = _aware(payload["arrival_at"])
    agreed_end = _aware(table["planned_end_at"]) if has_agreed_game_duration(table["terms"]) else None
    if arrival and agreed_end and arrival < agreed_end:
        summary["planned_end_at"] = agreed_end.isoformat()
        summary["remaining_play_minutes"] = int((agreed_end - arrival).total_seconds() // 60)
    return summary


def late_arrival_context(store: AgentStore, *, conversation_id: str, sender_id: str, at: datetime) -> dict:
    """仅装配自己的待答项以及自己提出的晚到请求，不共享其他会话内容。"""
    pending = [
        {"decision_task_id": task.task_id, "game_id": task.aggregate_id,
         "planned_start_at": task.payload["table_terms"]["planned_start_at"], "arrival_at": task.payload["arrival_at"],
         "question_message_id": receipt.message_id, "question_delivered_at": receipt.created_at.isoformat()}
        for task, _, receipt in _pending_decision_tasks(store, conversation_id=conversation_id, sender_id=sender_id, at=at)
    ]
    game_ids = {task.aggregate_id for task in store.scheduled_tasks.values()
        if task.task_type == LATE_ARRIVAL_QUESTION_TASK_TYPE and task.payload.get("candidate_id") == sender_id
        and task.payload.get("candidate_conversation_id") == conversation_id}
    current = [result for game_id in sorted(game_ids) if (result := late_arrival_result_for_candidate(
        store, game_id=game_id, conversation_id=conversation_id, sender_id=sender_id, at=at
    )) is not None and result["status"] != "stale"]
    return {**({"pending_late_arrival_decisions": pending} if pending else {}),
            **({"current_late_arrival_requests": current} if current else {})}


def request_late_arrival(store: AgentStore, call, trace_id, conversation_id, sender_id, sender_name) -> ToolResult:
    """登记本人查实的同桌晚到征询，工具回执只证明已落库的任务和渠道实际送达状态。"""
    arrival = _aware(call.arguments["arrival_at"])
    if arrival is None:
        return ToolResult(name=call.name, called=False, allowed=False,
                          error="需要带日期和时区的明确到店时间。")
    stamp = now()
    try:
        questions, changes = store.ensure_late_arrival_questions(game_id=call.arguments["game_id"],
            arrival_at=arrival, sender_id=sender_id, conversation_id=conversation_id, trace_id=trace_id, at=stamp)
    except ValueError as exc:
        details = {"late_arrival_request": {"game_id": call.arguments["game_id"], "arrival_at": arrival.isoformat(),
                    "status": "unavailable"}, "contact_started": False}
        if isinstance(exc, LateArrivalLookupRequired):
            details.update(
                retryable_no_effect=True,
                validation_error={"code": "late_arrival_current_table_lookup_required", "message": str(exc)},
                instruction="先按原条件查到同一张桌，再用本轮凭据重试。已明确委托的晚到征询无需再问。"
                            "等待真实回执前不要说已经联系或留位。",
            )
        return ToolResult(name=call.name, called=False, allowed=False, error=str(exc), result=details)
    status, started = _status(store, questions, stamp)
    return ToolResult(name=call.name, called=True, allowed=True, state_transitions=changes,
        result={"late_arrival_request": _candidate_request_summary(questions[0], status),
            "contact_started": started,
            "instruction": "按该桌实际状态说明进展。排队可以告知仍待确认，不能说已经问到；"
                           "等牌友答复不要求申请人继续回答。原开场与空席不变。"})


def record_late_arrival_decision(store: AgentStore, call, trace_id, conversation_id, sender_id, sender_name) -> ToolResult:
    """记录一位原桌联系人的本人答复；整体结果需要已送达的问题和全体同意。"""
    args = call.arguments
    try:
        saved, changes, status = store.resolve_late_arrival_question(args["decision_task_id"], decision=args["decision"],
            evidence_message_id=args["evidence_message_id"], sender_id=sender_id, conversation_id=conversation_id,
            trace_id=trace_id, at=now())
    except ValueError as exc:
        return ToolResult(name=call.name, called=False, allowed=False, error=str(exc),
                          result={"late_arrival_decision": {"status": "unavailable"}})
    return ToolResult(name=call.name, called=True, allowed=True, state_transitions=changes,
        result={"late_arrival_decision": {"game_id": saved.aggregate_id, "status": status},
                "instruction": "只确认当前联系人的答复，回告未送达前不说已通知。未改开场或占席。"})
