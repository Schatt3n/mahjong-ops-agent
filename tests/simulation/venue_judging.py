"""从真实交付记录构造有明确引用范围的精简营业多轮评测。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import json
from time import monotonic
from typing import Any

from mahjong_agent_runtime import customer_visible_contract

from .venue_acceptance import _delivered_replies
from .venue_actors import JudgeResult, SemanticJudge
from .venue_channel import AGENT_ID, OWNER_ID


Row = Mapping[str, Any]
ROLE_CHECK = "顾客像真人一样处理自己的目标，询问范围和续轮有实际理由"
AGENT_CHECK = "Agent 依据已送达对话承接需求，不重复追问已知条件且不误报或无依据承诺"
FIRST_CONTACT_CHECK = "首次联系店家的顾客说话自然，未无故报姓名、照列字段或复述条件"
DURATION_CHECK = "确认现成局席位时，顾客的时长意愿和店家交代的牌桌时长有证据"
ROLLING_REVIEW_INSTRUCTIONS = (
    "这是同一位顾客的两日回合。区分每天的日历变化、实际引用和已送达私聊。"
    "无参与意愿、天气不合适或话题与本人无关时，等待可以是有效决定。"
)
_CHECKS = (ROLE_CHECK, AGENT_CHECK, FIRST_CONTACT_CHECK, DURATION_CHECK)
ROLLING_CHECK_IDS = {
    ROLE_CHECK: "role_behavior",
    AGENT_CHECK: "shop_behavior",
    FIRST_CONTACT_CHECK: "first_contact",
    DURATION_CHECK: "duration_at_join",
}
_PROFILE_FIELDS = ("preferred_game", "stake_min", "stake_max", "smoking", "available_periods", "speaking_style", "temperament")
_TERM_FIELDS = ("game_type", "stake", "base_stake", "start_time", "planned_start_at", "planned_end_at", "duration_kind", "duration_hours",
                "min_duration_hours", "max_duration_hours",
                "smoke", "smoking", "smoke_preference", "smoking_policy", "party_size", "case_goal", "case_intent",
                "earliest_start_time", "latest_start_time", "knows_shopkeeper", "start_time_range")
_WORLD_FIELDS = ("work_status", "location", "travel_minutes", "free_from", "free_until", "earliest_arrival",
                 "delay_minutes", "previous_session_end", "can_arrive", "event_condition")
_GAME_FIELDS = ("game_id", "status", "planned_start_at", "planned_end_at", "remaining_seats", "seats_total")
_STATE_FIELDS = ("status", "game_id", "room_id", "reservation_id", "demand_id", "start_at", "end_at", "expires_at")
_PERSISTED_FIELDS = ("game_id", "customer_seat_before", "customer_seat_after", "game_start_before", "game_start_after")
_WRITING_TOOLS = {"customer_visible_content_review", "customer_visible_text_generation"}
_ARCHIVED_BOARD_PREFIX = "outbound:board:"
# 发言形态来自真人脱敏时间线中的 availability_then_smoke_preference、intermediary_role_correction
# 和 existing_game_join_then_duration_cancel。引用它们的互动方式，不把原话设为通过条件。
_CITATION_RULES = """逐字抄录 evidence_id 中简短的 E 编号作为引用。顾客项引用 customer_message（顾客等待或离开时可引用 customer_action），可补充本人的 customer_action、当时实际看到的 agent_reply。画像是倾向，允许自然变动。顾客项给 pass 要引用每条实际顾客消息与结束时的 wait/leave 等动作（若发生）；判失败要引用暴露问题的各条消息。
顾客项先看这是不是顾客随手发的微信。普通首次问局主动向不熟的老板报姓名且无需对号，应判 fail；已知熟人、转介绍或实际询问身份有识别理由时，可按对话辨别。按资料把玩法、时间、档位、烟况、人数、时长逐项报齐，或应答时照单复述已经确认的条件，像替店家做录入且无沟通理由，也应判 fail。开场只问眼下最关心的一个问题、等对方问了再补确有用的条件，或一次自然说清几个当时重要的条件，都可以通过；勿按字数、字段数或固定句式判定。
顾客项另核对这句在收件人视角是否好懂。群看板转私聊时，引用真实来话和可核实的看板，判断对方能否凭该编号或实际特征分辨哪桌。看板没写的“一缺”等叫法、仅角色知道的事件或他人私聊不能当成收件人也知道的线索。若一个消息同时问原桌与另找一桌，两件事应能区分。共享上下文中自然说“那桌”、只问眼下最关心的一件事、收到追问后才补条件，都可通过。若指代本可以带一个现成线索却含糊得让人难以辨认、把两问说成同一桌、或编造看板细节，应在顾客项引用原话及相关证据，说明是什么造成困难。缺少可判断的先前上下文时勿猜测。
如果 checks 有“首次联系店家”项，knows_shopkeeper=false 是已记录的场景事实，不要用报出的姓名反推两人认识。逐条看整段顾客消息，这是一项独立的验收：业务条件最终完整或成功占座都不能抵消不自然的顾客来话。普通问局无理由主动报全名，或面前只有一桌、直接答应就行却整串复读店家刚报的条件，都应判此项 fail。仅重提一项来消除歧义、点出一个关键限制、或为区分多桌而确认，不属于照抄清单。此项 pass 必须引用所有 customer_message；fail 引用暴露问题的消息。
顾客项按整段对话判断效率与收尾。已知都可接受的连续时间应一次问成范围；没有逐轮出现的新约束、具体备选或先前范围缺口，却按相同条件一小时一小时探问，必须 fail。即使旧场景或画像写着愿意追问，也不能把逐点探测当作合理聊天。确实只剩离散时点、后来才改了安排、店家给出具体备选、或明确只查了范围的一部分，可以继续问；要指出新依据。开局时间范围和打牌时长要分清。
顾客项单独核对知道多少。这桌几点开与打多久不同；本人一般打多久、最多能打多久，也不等于桌子的时长。顾客知道桌子时长，必须能指向该轮之前实际看到的文字。顾客若在意时长但没听说，问打到几点或带条件表达兴趣合理；把自己的时长当成桌子的已知事实，或明有时间限制却在未核实下把条件说成全都合适并直接报名，应判 fail。顾客说自己能打多久并不等于他声称知道这桌时长，要看它是在说明自身情况、设条件还是无条件报名。对时长确实无所谓或时长早已说清的顾客，可以直接报名。
校准参考为脱敏真人聊天时间线：有人先说下午能来，后来才补无烟；有人把钟点拆在连续几个气泡里；有人先认领，听到时长不合又退出。这样的碎片输入、短答、合理改口和自然收尾可以通过；不要要求逐字复现参考，也不要把漏消息、机械拆字、无理由改条件或刻意扮演人设当成真实。连续气泡的 input_batch 指向共用的入站锚点，连起来理解，同时引用每条真实气泡。天气、通行、本地位置、私人安排等变化只能用当时记录的世界事件或实际对话解释；顾客可早说、晚说或保持沉默，不必泄露全部私有细节。店家不能在顾客告知前猜出这些细节。
没有合适桌时，按本人意愿明确请老板组、请留意、婉拒或不再发消息都可能合理；一轮已问清即可收尾。为了继续对话而把同一问题换个钟点重问、回避已送达的组局提议且没有新信息，不能因 Agent 查得准或口吻贴合画像而通过。
owner_invitation 是真实送达的经营者模拟邀约，agent_initiative 是送达的 Agent 主动消息，可用于理解顾客是主动来话还是答复邀约。无回复也可能正常，不能把店主夹具发出的邀约说成 Agent 的执行结果。每条 agent_reply 已在聊天账本核对送达给本人，末轮顾客不再发言也不影响这个事实。店家项必须引用适用的 agent_reply，并按需补充 customer_message 与 agent_execution。店家永远不知道 customer_action 中的私有设定，店家项的理由和引用均不得用该类证据。无送达时不能判店家通过，可引用入站、执行错误指出失败或证据不足。
店家项单独核对对歧义的处理。指代、授权或事实确实无法判断，针对性追问、说明已核实的范围并回答能答的部分，或确实需店员查证权限或现场事实时说明要人工确认，都可通过。两件事中一件不清楚，另一件能独立安全查答时，应正常推进；只说“稍等”或泛泛请重述、转人工，不能代替可回答的部分。已明确、可查的普通问题被反复追问或直接推给人工也属于店家处理问题。真实人工交接或承诺已有谁在处理须有相应执行及交付证据；仅建议人工确认不等于已交接。工具或模型执行不通时以记录说明阻塞来源，不能归成顾客问法，也不能因为顾客说得含糊就判定店家处理失败。店家猜中桌子并不消除无依据的猜测。
确认占座需要客户明确同意参加，接受单项条件或询问房间、流程不能单独授权占座。明确报名并同时询问房间仍可占座。区分席位确认、空房查询和房间预订，每项只用对应的实际证据。若 persisted_game_state 核实顾客来话前后都已确认席位，只告知位置照旧不需要重新写席位；该证据不能证明同桌已同意等待或店家已经转达。查现成牌桌、委托新组局、登记候补的授权各自独立。“帮我问问有没有人打”本身只授权询问，开新局还需本人明确委托或确认已送达的开局提议。forming 是仍在招募的待组局，已有席位和还缺人的描述可以同时为真。
顾客将自己的时长带进参加决定时，店家项必须同时引用该条顾客消息、桌子真实时长或未知的执行依据、实际回复。顾客说能打多久，不能当成牌桌时长的来源。若牌桌已核实且符合来话，顾客态度明确并且没有待答的条件，可以在说明该桌实际时长的同时确认留位，不必再问一次。若明确以时长为条件或仍在询问时长、实际时长未知或不匹配，先说明或核实并等顾客确认。直接写入占座、或只说“留好了”却没有向在意时长的顾客交代桌子时长，应判店家项 fail。不要用自动安排的过期时间替代约定时长。
如果 checks 有“确认现成局席位时”项，单列时长核对，不以已写入的席位当作时长通过。引用真正决定留位的 customer_message 和 agent_execution。此项 pass 还须引用实际送达的 agent_reply，并在 reason 用一对「」逐字分别引用该轮完整顾客来话和说明时长的完整店家回复（时长无关时引用正常确认回复，并说明为何无关）。如果顾客本次表示“打三个钟头左右”，店家邀约只给开局点，后来也只有“留好了”，即使后台这桌正好三小时，本项仍 fail。否则，若顾客只做无条件报名且全程没提及或依赖时长，可以通过，无须追问或主动补报时长。
沿 answers 连接检查当轮入站，只有该回复实际送达之前的来话才算已知。按来话所问范围核对每轮查询。顾客明确接受一个连续范围时，单个钟点的查询不能证明整段无局；即使结果为空，店家用范围首点替代全区间后直接答无局，也必须判店家项 fail。能涵盖整个时段的查局证据（或明确只报子范围结果并继续核对其余条件）才支持对应结论。最终状态只描述会话结束时，不能倒推前一轮。没有适配牌局时，如实告知或在同意后候补也可通过。"""
_PEER_CITATION_RULES = (
    "peer_delivery 是当前顾客实际收到的搭子私信，peer_message 是本人发给搭子的原话。"
    "只在顾客项引用，可据此判断邀约、应答和转向其他会话的理由。"
    "顾客项通过须逐条引用实际发送的 peer_message。店家和首次联系店家项均不能引用这两类证据。"
)
_EVIDENCE_ID_FIELDS = ("event_id", "trace_id", "message_id")
_EVIDENCE_LINK_FIELDS = ("input", "caused_by", "answers", "responds_to", "input_batch", "quoted_message_id")


@dataclass(slots=True)
class _Evidence:
    """区分可引用的上下文与每项判定必须命中的评价对象。"""

    rows: list[dict[str, Any]]
    allowed: dict[str, set[str]]
    anchors: dict[str, set[str]]

    def add(self, key: str, identifier: str, fields: dict[str, Any], *, context=(), targets=()) -> None:
        """证据 ID 直接关联归档消息或 trace，分别限制各检查项的引用。"""
        self.rows.append({key: identifier, **fields})
        for check in (*context, *targets):
            self.allowed[check].add(identifier)
        for check in targets:
            self.anchors[check].add(identifier)


def _pick(row: Row, fields: Sequence[str]) -> dict[str, Any]:
    return {field: row[field] for field in fields if field in row and row[field] is not None}


def _byte_count(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _terms(row: Row) -> dict[str, Any]:
    """投影查局条件和开局区间，只保留区间合同里的两个边界。"""
    terms = _pick(row, _TERM_FIELDS)
    if "start_time_range" in terms:
        interval = terms.pop("start_time_range")
        if isinstance(interval, Mapping) and (bounds := _pick(interval, ("start_at", "end_at"))):
            terms["start_time_range"] = bounds
    return terms


def _game(row: Row) -> dict[str, Any]:
    game = row.get("game") or row
    terms = _terms(game.get("requirement") or {})
    return (_pick(game, _GAME_FIELDS) | ({"terms": terms} if terms else {})
            | _pick(row, ("match_kind", "relative_to_window", "offset_minutes")))


def _executions(turn: Row) -> list[dict[str, Any]]:
    """保留业务动作与关键返回值，原始客户列表、话术模型和供应商诊断留在本地归档。"""
    summaries = []
    for tool in turn.get("tool_results") or ():
        name = tool.get("name")
        if not tool.get("called") or not name or name in _WRITING_TOOLS:
            continue
        allowed = tool.get("allowed") is True
        error = tool.get("error")
        summary: dict[str, Any] = {"name": name, "execution": "ok" if allowed and not error else "error" if allowed else "blocked"}
        if summary["execution"] != "ok":
            summaries.append(summary)
            continue
        result = tool.get("result") or {}
        returned = _pick(result, ("configured", "start_at", "end_at", "available_count", "cancelled_count", "recorded_status"))
        if result.get("requirement"):
            returned["query"] = _terms(result["requirement"])
        for label in ("matches", "alternatives", "nearby_options"):
            if isinstance(result.get(label), list):
                options = result[label]
                returned[label] = {"count": len(options), "options": [_game(item) for item in options if isinstance(item, Mapping)]}
        if isinstance(result.get("game"), Mapping):
            returned["game"] = _game(result["game"])
        for label in ("reservation", "waiting_demand"):
            if isinstance(result.get(label), Mapping):
                returned[label] = _pick(result[label], _STATE_FIELDS)
        if returned:
            summary["result"] = returned
        summaries.append(summary)
    return summaries


def _role(turn: Row, *, include_profile: bool, visible_text: Sequence[str]) -> dict[str, Any]:
    """只提取顾客本人的本轮动机、动作及与已见人物相关的偏好。"""
    decision = turn.get("decision") or {}
    context = decision.get("context") or {}
    actions = decision.get("action") or {}
    events = context.get("events") or ()
    role = _pick(decision, ("source", "status"))
    if actions.get("action"):
        role["action"] = actions["action"]
    if events:
        event = events[-1]
        details = event.get("details") or {}
        if details:
            role["situation"] = _terms(details)
        if world := _pick(details, _WORLD_FIELDS):
            role["private_world"] = _pick(event, ("kind", "at")) | world
    profile = context.get("persona") or {}
    if include_profile and profile:
        role["private_profile"] = _pick(profile, _PROFILE_FIELDS)
    names = "\n".join(visible_text)
    relationships = [
        _pick(item, ("target_name", "kind")) for item in profile.get("relationships", ())
        if item.get("target_name") and item["target_name"] in names
    ]
    if relationships:
        role["private_relationships"] = relationships
    return role


def _is_peer_message(row: Row) -> bool:
    """核对档案方向和两个顾客端点，店家账号不能作为搭子证据。"""
    sender, recipient = row.get("sender_id"), row.get("recipient_id")
    return (row.get("direction") == "peer" and row.get("kind") == "private"
            and all(isinstance(account, str) and account and account not in {AGENT_ID, OWNER_ID}
                    for account in (sender, recipient)) and sender != recipient)


def _turn_messages(turn: Row, ledger: Mapping[str, Row], positions: Mapping[str, int], customer_id: str,
                   *, include_peers: bool) -> list[Row]:
    """从账本认领同人、同会话且不晚于事件锚点的顾客原始气泡。"""
    primary_id = turn.get("message_id")
    primary = ledger.get(primary_id) if isinstance(primary_id, str) else None
    if not primary or primary.get("sender_id") != customer_id:
        return []
    if include_peers and _is_peer_message(primary):
        return [primary] if _at_or_before_turn(primary, turn) else []
    if primary.get("direction") != "inbound":
        return []
    fragment_ids = turn.get("message_ids")
    fragments = fragment_ids if isinstance(fragment_ids, (list, tuple)) else ()
    ids = dict.fromkeys([primary_id, *(key for key in fragments if isinstance(key, str))])
    return [
        item for message_id in sorted((key for key in ids if key in ledger), key=positions.__getitem__)
        if (item := ledger[message_id]).get("direction") == "inbound"
        and positions[message_id] <= positions[primary_id]
        and all(item.get(key) == primary.get(key) for key in ("sender_id", "recipient_id", "conversation_id", "kind"))
    ]


def _invitation_precedes_turn(invite: Row, turn: Row, messages: Sequence[Row], positions: Mapping[str, int]) -> bool:
    """实际气泡顺序优先。顾客沉默时以虚拟事件时刻核对邀约已送达。"""
    if messages:
        return (invite.get("conversation_id") == messages[0].get("conversation_id")
                and positions[invite["message_id"]] < positions[messages[0]["message_id"]])
    return _at_or_before_turn(invite, turn)


def _at_or_before_turn(message: Row, turn: Row) -> bool:
    """在无新的来话时也可用虚拟时刻核对触发消息已经送达。"""
    try:
        return datetime.fromisoformat(message["sent_at"]) <= datetime.fromisoformat(turn["virtual_at"])
    except (KeyError, TypeError, ValueError):
        return False


def _same_visible_message(archived: Row, visible: Row) -> bool:
    """核对模型看到的气泡与账本原文，排除后补或第三方消息。"""
    return all(archived.get(key) == visible.get(key) for key in ("text", "kind", "sender_id", "recipient_id", "sent_at"))


def _project_dialogue(customer_id: str, turns: Sequence[Row], records: Iterable[Row], outcome: Row | None,
                      *, first_contact: bool, delivered_history: bool = False) -> _Evidence:
    """根据同一顾客的实际来话和投递建立最小可审计证据。"""
    evidence = _Evidence([], {check: set() for check in _CHECKS}, {check: set() for check in _CHECKS})
    ledger = {row["message_id"]: row for row in records}
    positions = {message_id: index for index, message_id in enumerate(ledger)}
    replies = _delivered_replies(ledger, positions)

    profile_added = False
    execution_rows = []
    own_reply_ids = set()
    history_ids = set()
    peer_delivery_ids = set()
    seen_messages = set()
    for turn in turns:
        if turn.get("actor_id") != customer_id:
            continue
        message_id = turn.get("message_id")
        inbound = _turn_messages(turn, ledger, positions, customer_id, include_peers=delivered_history)
        peer_turn = bool(inbound and _is_peer_message(inbound[0]))
        cause = turn.get("caused_by")
        peer_cause = False
        if inbound or delivered_history:
            history = (turn.get("decision", {}).get("context") or {}).get("history", ())
            first_input_position = positions[inbound[0]["message_id"]] if inbound else None
            quoted = {row.get("quoted_message_id") for row in inbound if row.get("kind") == "group"}
            # 私聊没有群引用时，保留最近真正可见的正式看板，避免评审只能猜对方看到哪条公告。
            latest_board = max((visible["message_id"] for visible in history
                                if inbound and visible.get("kind") == "group"
                                and str(visible.get("message_id") or "").startswith(_ARCHIVED_BOARD_PREFIX)
                                and positions.get(visible["message_id"], first_input_position) < first_input_position
                                and _same_visible_message(ledger[visible["message_id"]], visible)),
                               key=positions.__getitem__, default=None)
            for visible in history:
                visible_id = visible.get("message_id")
                archived = ledger.get(visible_id, {})
                room = archived.get("kind") == "group"
                own = archived.get("kind") == "private" and customer_id in (archived.get("sender_id"), archived.get("recipient_id"))
                historical = delivered_history and archived.get("direction") == "history" and (room or own)
                public_context = room and (visible_id == latest_board or visible_id in quoted or visible_id == cause)
                ordered = bool(inbound and visible_id in positions and positions[visible_id] < first_input_position) or (
                    not inbound and _at_or_before_turn(archived, turn))
                if (delivered_history and visible_id == cause and ordered and _is_peer_message(archived)
                        and archived["recipient_id"] == customer_id and _at_or_before_turn(archived, turn)
                        and _same_visible_message(archived, visible)):
                    peer_cause = True
                    if visible_id not in peer_delivery_ids:
                        evidence.add("message_id", visible_id, {
                            "kind": "peer_delivery",
                            **_pick(archived, ("text", "sent_at", "sender_id")),
                        }, context=(ROLE_CHECK,))
                        peer_delivery_ids.add(visible_id)
                if ((historical or public_context) and ordered and visible_id not in history_ids
                        and visible_id not in own_reply_ids and visible_id not in seen_messages
                        and _same_visible_message(archived, visible)):
                    label = "private_history" if own else "group_history" if historical else "group_delivery"
                    evidence.add("message_id", visible_id, {
                        "kind": label,
                        **_pick(archived, ("text", "sent_at", "sender_id", "quoted_message_id")),
                    }, context=(ROLE_CHECK,) if room else (ROLE_CHECK, AGENT_CHECK))
                    history_ids.add(visible_id)
        if cause not in own_reply_ids:
            invite = ledger.get(cause or "", {})
            if (invite.get("kind") == "private" and invite.get("recipient_id") == customer_id
                    and (invite.get("direction"), invite.get("sender_id")) in {("owner", OWNER_ID), ("outbound", AGENT_ID)}
                    and not invite.get("source_message_id") and _invitation_precedes_turn(invite, turn, inbound, positions)):
                evidence.add("message_id", invite["message_id"], {
                    "kind": "owner_invitation" if invite["direction"] == "owner" else "agent_initiative",
                    **_pick(invite, ("text", "sent_at")),
                }, context=(ROLE_CHECK, AGENT_CHECK, FIRST_CONTACT_CHECK))
                own_reply_ids.add(cause)
        safe_cause = cause if peer_cause or cause in own_reply_ids or cause in history_ids else None
        text = [str(ledger[safe_cause].get("text", ""))] if safe_cause else []
        text.extend(str(row.get("text", "")) for row in inbound)
        role = _role(turn, include_profile=not profile_added, visible_text=text)
        if first_contact and not profile_added:
            role.setdefault("situation", {})["knows_shopkeeper"] = False
        profile_added = profile_added or "private_profile" in role
        valid_action = role.get("action") if role.get("status") == "valid" else None
        if role:
            decision = turn.get("decision") or {}
            role_trace = decision.get("trace_id") or turn.get("event_id")
            if role_trace:
                evidence.add("trace_id" if decision.get("trace_id") else "event_id", role_trace,
                             {"kind": "customer_action", "input": message_id if inbound else None,
                              "caused_by": safe_cause, **role},
                             targets=(ROLE_CHECK,) if valid_action in {"wait", "leave", "arrive"} else (),
                             context=(ROLE_CHECK, FIRST_CONTACT_CHECK))
        if not inbound:
            continue
        if peer_turn:
            # 搭子会话从实际出账取证，切断店家入站、回复、工具和首次联系的引用范围。
            if message_id not in seen_messages:
                evidence.add("message_id", message_id, {
                    "kind": "peer_message", **_pick(inbound[0], ("text", "sent_at", "recipient_id")),
                }, targets=(ROLE_CHECK,) if valid_action == "send" else (), context=(ROLE_CHECK,))
                seen_messages.add(message_id)
            continue
        chat = [(row, False) for row in inbound]
        for row in inbound:
            chat.extend((reply, True) for reply in replies.get(row["message_id"], ()))
        for row, is_reply in sorted(chat, key=lambda item: positions[item[0]["message_id"]]):
            identifier = row["message_id"]
            if identifier in seen_messages:
                continue
            seen_messages.add(identifier)
            text_and_time = _pick(row, ("text", "sent_at"))
            if is_reply:
                evidence.add("message_id", identifier, {"kind": "agent_reply", "answers": row["source_message_id"],
                             **text_and_time}, targets=(AGENT_CHECK,), context=(ROLE_CHECK, FIRST_CONTACT_CHECK, DURATION_CHECK))
                own_reply_ids.add(identifier)
            else:
                batch = {"input_batch": message_id} if len(inbound) > 1 else {}
                evidence.add("message_id", identifier, {"kind": "customer_message", "channel_kind": row["kind"],
                             **text_and_time, **_pick(row, ("quoted_message_id",)), **batch},
                             targets=(ROLE_CHECK, FIRST_CONTACT_CHECK) if valid_action == "send" else (), context=_CHECKS)
        if trace_id := turn.get("trace_id"):
            tools = _executions(turn)
            confirmed = any(tool["name"] in {"join_game", "record_candidate_reply"} and tool["execution"] == "ok"
                            and tool.get("result", {}).get("recorded_status") == "confirmed" for tool in tools)
            execution = {"kind": "agent_execution", "responds_to": message_id, "tools": tools,
                         **_pick(turn, ("runtime_status", "error_type", "failure_stage"))}
            if persisted := _pick(turn.get("persisted_game_state") or {}, _PERSISTED_FIELDS):
                execution["persisted_game_state"] = persisted
            if trace_id not in evidence.allowed[AGENT_CHECK]:
                evidence.add("trace_id", trace_id, execution, context=(AGENT_CHECK, DURATION_CHECK),
                             targets=(DURATION_CHECK,) if confirmed else ())
                execution_rows.append((message_id, evidence.rows[-1]))

    if outcome:
        sources = outcome.get("source_message_ids") or ()
        final_execution = next((row for message_id, row in reversed(execution_rows) if message_id in sources), None)
        if final_execution is not None:
            final_execution["final_outcome"] = _pick(outcome, ("kind", "status"))
    return evidence


def _judge_evidence(rows: Sequence[Row], *, keep_external_links: bool = False
                    ) -> tuple[list[dict[str, Any]], dict[str, str], tuple[dict[str, Any], ...]]:
    """模型复制短编号，原始消息和记录链接保留在本地报告，禁止猜测或修复未知编号。"""
    keyed_rows = [(next(key for key in _EVIDENCE_ID_FIELDS if key in row), row) for row in rows]
    aliases = {row[key]: f"E{index:02d}" for index, (key, row) in enumerate(keyed_rows, start=1)}
    request, archive = [], []
    for key, row in keyed_rows:
        original = row[key]
        alias = aliases[original]
        links = {link: aliases.get(row[link], row[link] if keep_external_links else None)
                 for link in _EVIDENCE_LINK_FIELDS if link in row}
        request.append(dict(row) | {key: alias} | links)
        archive.append(dict(row) | {"evidence_id": original, "evidence_alias": alias})
    return request, {alias: identifier for identifier, alias in aliases.items()}, tuple(archive)


def judge_with_short_evidence_ids(
    judge: SemanticJudge, case_id: str, rows: Sequence[Row], *, trace_id: str, checks: Sequence[str], instructions: str,
) -> JudgeResult:
    """让评审逐字引用短标签，原始账本编号只从严格通过校验的引用表恢复。"""
    source_ids = [next((row.get(key) for key in _EVIDENCE_ID_FIELDS if key in row), None) for row in rows]
    if not source_ids or any(not isinstance(key, str) or not key for key in source_ids) or len(set(source_ids)) != len(rows):
        return JudgeResult(case_id, trace_id, "invalid", "inconclusive", (), None, (), {},
                           error="judge evidence needs nonempty, unique event, trace, or message IDs")
    request, aliases, archive = _judge_evidence(rows, keep_external_links=True)
    instructions += ("\n引用时逐字复制每行 evidence_id 里的短标签，不能用消息编号或自行加前缀。"
                     "涉及店家实际回复时引用含 customer_message 和 outputs 的交付行。"
                     "开场、散场等现场行只证明现场事实，不能代替聊天交付。")
    result = judge.judge(case_id, request, trace_id=trace_id, checks=checks, instructions=instructions)
    findings = tuple(replace(finding, evidence_ids=tuple(aliases[label] for label in finding.evidence_ids))
                     for finding in result.findings) if result.status == "valid" else ()
    return replace(result, evidence=archive, findings=findings)


def judge_reply_with_verified_seats(
    judge: SemanticJudge, case_id: str, rows: Sequence[Row], *, trace_id: str, checks: Sequence[str],
    instructions: str, seat_check: str, delivered_reply_id: str, seat_source_id: str,
) -> dict[str, Any]:
    """用实际送达原话和同轮核实的局快照比较明确余位，其余项目继续独立评审。"""
    evidence = {next((row[key] for key in _EVIDENCE_ID_FIELDS if key in row), None): row for row in rows}
    if (len(evidence) != len(rows) or None in evidence or delivered_reply_id == seat_source_id
            or delivered_reply_id not in evidence or seat_source_id not in evidence or seat_check not in checks
            or len(set(checks)) != len(checks) or len(checks) < 2):
        raise ValueError("seat review requires separate, unique reply and fact evidence and a distinct numeric check")
    reply = evidence[delivered_reply_id].get("text")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("seat review requires the exact delivered reply text")
    game_facts = evidence[seat_source_id].get("verified_seat_candidates")
    if not isinstance(game_facts, list) or any(not isinstance(row, dict) for row in game_facts):
        raise ValueError("seat review needs verified candidate game facts in the cited evidence row")

    proof = customer_visible_contract.customer_visible_seat_claim_evidence(reply, game_facts)
    fact_verdict = {"matched": "pass", "mismatch": "fail"}.get(proof["status"])
    review_checks = tuple(check for check in checks if fact_verdict is None or check != seat_check)
    instructions += ("\n总判通过时，每条标有 citation_required=true 的真实来话、查局事实和实际回复"
                     "都须在相关判项中引用其短编号。")
    if fact_verdict is not None:
        instructions += ("\n仅评价 checks 列出的项目。每桌的余位数字由独立事实核验负责，"
                         "不用重新判定或将席位理由记入其他项。")
    result = judge_with_short_evidence_ids(judge, case_id, rows, trace_id=trace_id,
                                           checks=review_checks, instructions=instructions)
    cited = {identifier for finding in result.findings for identifier in finding.evidence_ids}
    required = {identifier for identifier, row in evidence.items() if row.get("citation_required")}
    # 硬事实与独立判定各有原始引用。独立判定缺少本轮业务证据时，即使报数正确也不能通过。
    verdict = "inconclusive"
    if fact_verdict == "fail" or (result.status == "valid" and result.verdict == "fail"):
        verdict = "fail"
    elif result.status == "valid" and result.verdict == "pass" and required <= cited:
        verdict = "pass"
    return {"verdict": verdict, "independent_judgment": result.to_dict(), "seat_fact_check": {
        "check": seat_check, "evidence_ids": [delivered_reply_id, seat_source_id], "verdict": fact_verdict, **proof,
    }}


@dataclass(frozen=True, slots=True)
class PreparedDialogueReview:
    """同一套请求、归档及引用边界同时服务原评审与只读补判。"""

    case_id: str
    projected: _Evidence
    checks: tuple[str, ...]
    request: list[dict[str, Any]]
    aliases: dict[str, str]
    archive: tuple[dict[str, Any], ...]
    source_turn_bytes: int


def prepare_dialogue_review(*, customer_id: str, turns: Sequence[Row], records: Iterable[Row],
                            outcome: Row | None = None, first_contact: bool = False, role_only: bool = False,
                            delivered_history: bool = False) -> PreparedDialogueReview:
    """从本地归档构造判决证据，无需客户端或任何外部请求。"""
    projected = _project_dialogue(customer_id, turns, records, outcome, first_contact=first_contact,
                                  delivered_history=delivered_history)
    checks = [ROLE_CHECK]
    if not role_only or any(row["kind"] in {"agent_reply", "agent_execution", "agent_initiative"} for row in projected.rows):
        checks.append(AGENT_CHECK)
    if any(row.get("situation", {}).get("knows_shopkeeper") is False for row in projected.rows):
        checks.append(FIRST_CONTACT_CHECK)
    if projected.anchors[DURATION_CHECK]:
        checks.append(DURATION_CHECK)
    request, aliases, archive = _judge_evidence(projected.rows)
    return PreparedDialogueReview(f"dialogue:{customer_id}", projected, tuple(checks), request, aliases, archive,
                                  _byte_count(turns))


def judge_prepared_dialogue(judge: SemanticJudge, prepared: PreparedDialogueReview, *, trace_id: str,
                            additional_instructions: str = "", rolling_check_ids: bool = False,
                            retry_invalid: bool = False) -> dict[str, Any]:
    """发送已组装且可预先核验的证据，判决引用仍必须命中原评审边界。"""
    instructions = _CITATION_RULES
    if any(row["kind"] in {"peer_message", "peer_delivery"} for row in prepared.request):
        instructions += f"\n{_PEER_CITATION_RULES}"
    wire_checks = {(ROLLING_CHECK_IDS[check] if rolling_check_ids else check): check for check in prepared.checks}
    if rolling_check_ids:
        instructions += f"\n{_rolling_citation_instructions(prepared)}"
    if additional_instructions:
        instructions += f"\n{additional_instructions}"
    deadline = monotonic() + judge.timeout_seconds if retry_invalid else None
    first = judge.judge(prepared.case_id, prepared.request, trace_id=trace_id,
                        checks=tuple(wire_checks), instructions=instructions)
    result = _validate_dialogue_judgment(first, prepared, wire_checks, rolling_check_ids=rolling_check_ids)
    if (deadline is None or result["status"] != "invalid" or not (first.response_source or {}).get("http_response_complete")
            or (remaining := deadline - monotonic()) <= 0):
        return result
    # 对完整返回但不能验实的判决只允许再独立评审一次，使用同一证据和剩余时限。
    # 初判原文只留在归档，避免补判照抄尚未验实的结论或引用。
    instructions += (
        f"\n上一份评审未通过验证：{result['error']}。重新根据同一证据独立逐项判断。"
        "顾客项通过时须覆盖所有实际发言和等待等评价对象。材料不足就判待定。"
        "完成各判项后按 fail、inconclusive、pass 的优先级核对总判定。"
    )
    revised = judge.judge(prepared.case_id, prepared.request, trace_id=trace_id,
                          checks=tuple(wire_checks), instructions=instructions, timeout_seconds=remaining, step_index=2)
    final = _validate_dialogue_judgment(revised, prepared, wire_checks, rolling_check_ids=rolling_check_ids)
    final["prior_attempts"] = ({key: result[key] for key in (
        "status", "verdict", "error", "raw", "diagnostics", "response_source",
    )},)
    return final


def _validate_dialogue_judgment(judgment: JudgeResult, prepared: PreparedDialogueReview,
                                wire_checks: Mapping[str, str], *, rolling_check_ids: bool) -> dict[str, Any]:
    """分别验证每个独立响应，原引用边界、通过证据和时长校验不随重审变化。"""
    projected, aliases = prepared.projected, prepared.aliases
    result = judgment.to_dict()
    result["evidence_aliases"] = aliases
    result["projection"] = {
        "version": 7, "rows": len(prepared.request), "evidence_bytes": _byte_count(result["evidence"]),
        "source_turn_bytes": prepared.source_turn_bytes,
    }
    if rolling_check_ids:
        result["projection"]["rolling_check_ids"] = True
    result["evidence"] = prepared.archive
    if result["status"] != "valid":
        return result
    for finding in result["findings"]:
        finding["check"] = wire_checks[finding["check"]]
        finding["evidence_ids"] = tuple(aliases[alias] for alias in finding["evidence_ids"])
        check, ids, verdict = finding["check"], set(finding["evidence_ids"]), finding["verdict"]
        # Agent 的通过必须指向实际回复。店家未交付或无回复时可以判失败，引用实际执行或入站证据。
        need_target = verdict == "pass" or verdict == "fail" and (check != AGENT_CHECK or projected.anchors[check])
        if not ids <= projected.allowed[check] or need_target and not ids.intersection(projected.anchors[check]):
            result.update(status="invalid", verdict="inconclusive", findings=(), error="judgment does not cite its review subject")
            break
        if check in {ROLE_CHECK, FIRST_CONTACT_CHECK} and verdict == "pass" and not projected.anchors[check] <= ids:
            result.update(status="invalid", verdict="inconclusive", findings=(), error="customer pass must cite every archived action")
            break
        if check == DURATION_CHECK and verdict != "inconclusive" and not _has_duration_evidence(projected, finding, ids):
            result.update(status="invalid", verdict="inconclusive", findings=(), error="duration review needs linked join and delivered proof")
            break
    return result


def review_dialogue(judge: SemanticJudge, *, customer_id: str, turns: Sequence[Row], records: Iterable[Row],
                    outcome: Row | None = None, first_contact: bool = False, role_only: bool = False,
                    additional_instructions: str = "", delivered_history: bool = False,
                    rolling_check_ids: bool = False, retry_invalid: bool = False) -> dict[str, Any]:
    """评审实际顾客动作与店家回复；只有独立活动未触发店家时可单验顾客。"""
    prepared = prepare_dialogue_review(customer_id=customer_id, turns=turns, records=records, outcome=outcome,
        first_contact=first_contact, role_only=role_only, delivered_history=delivered_history)
    return judge_prepared_dialogue(judge, prepared, trace_id=f"judge:{customer_id}",
        additional_instructions=additional_instructions, rolling_check_ids=rolling_check_ids, retry_invalid=retry_invalid)


def shop_reply_verdicts(reviews: Iterable[Row]) -> dict[str, str]:
    """只从有效店家判项抽取明确引用的送达回复，以便逐轮核对待处理答复。"""
    verdicts = {}
    for review in reviews:
        if review["status"] != "valid":
            continue
        delivered = {row["evidence_id"] for row in review.get("evidence", ()) if row["kind"] == "agent_reply"}
        for finding in review["findings"]:
            if finding["check"] == AGENT_CHECK:
                verdicts.update(dict.fromkeys(delivered.intersection(finding["evidence_ids"]), finding["verdict"]))
    return verdicts


def _duration_sources(rows: Mapping[str, Row], confirmed: set[str]) -> dict[str, tuple[str, tuple[str, ...]]]:
    """把已确认席位的执行关联到原来话和不晚于该轮的实际回复。"""
    positions = {key: index for index, key in enumerate(rows)}
    sources = {}
    for trace_id, execution in rows.items():
        if trace_id not in confirmed:
            continue
        inbound = execution["responds_to"]
        replies = tuple(key for key, row in rows.items() if row["kind"] == "agent_reply"
                        and (positions[key] < positions[inbound] or row["answers"] == inbound))
        sources[trace_id] = (inbound, replies)
    return sources


def _rolling_citation_instructions(prepared: PreparedDialogueReview) -> str:
    """在滚动评审请求中公开后端核对的证据范围，固定编号仍由严格合同校验。"""
    projected, aliases = prepared.projected, prepared.aliases
    rules = {
        ROLLING_CHECK_IDS[check]: {
            "definition": check,
            "allowed_evidence_ids": [alias for alias, source in aliases.items() if source in projected.allowed[check]],
            "subject_evidence_ids": [alias for alias, source in aliases.items() if source in projected.anchors[check]],
        }
        for check in prepared.checks
    }
    for check in (ROLE_CHECK, FIRST_CONTACT_CHECK):
        if check in prepared.checks:
            rule = rules[ROLLING_CHECK_IDS[check]]
            rule["required_for_pass"] = rule["subject_evidence_ids"]
    duration_sources = _duration_sources({row["evidence_id"]: row for row in prepared.archive},
                                         projected.anchors[DURATION_CHECK])
    if duration_sources:
        labels = {source: alias for alias, source in aliases.items()}
        rules[ROLLING_CHECK_IDS[DURATION_CHECK]]["confirmed_joins"] = [
            {"execution": labels[trace], "customer": labels[inbound], "candidate_replies": [labels[key] for key in replies]}
            for trace, (inbound, replies) in duration_sources.items()
        ]
    return (
        "checks 中每个短编号原样复制到 check 字段。以下 definition 说明含义，不是输出编号。\n"
        "各判项只引用该项 allowed_evidence_ids。subject_evidence_ids 标出实际评价对象。"
        "判 pass 时必须列出 required_for_pass 中的每一个编号，包括实际等待等动作。"
        "店家项不能引用顾客所见但未进入店家评审范围的群看板，要引用店家实际送达的本人回复。\n"
        "只在给定 duration_at_join 时评审现成局留位。每笔 confirmed_joins 都要核对真实入席执行和同桌送达回复。"
        "通过时逐笔引用 execution、customer 和说明时长的 candidate_replies，在 reason 中分别逐字用「」摘录两条完整聊天。"
        "只有同一桌在留位前或该轮实际送达的文字才能证明已交代时长；找不到确切依据不要判通过。\n"
        "区分约定时长、顾客可接受的上限和系统默认排期。max_duration_hours 只表示上限，"
        "planned_end_at 是计划排期，仅凭它与开场的差值不能推断已经约定或实际打了多久。"
        "顾客说自己已报名也不能替代真实席位执行或落库状态。\n"
        f"{json.dumps(rules, ensure_ascii=False, separators=(',', ':'))}"
    )


def _has_duration_evidence(projected: _Evidence, finding: Row, cited: set[str]) -> bool:
    """时长通过时要求可逐字回查的来话和已送达回复，内部时长字段不能冒充已交代。"""
    rows = {next(row[key] for key in _EVIDENCE_ID_FIELDS if key in row): row for row in projected.rows}
    for trace_id, (inbound, replies) in _duration_sources(rows, projected.anchors[DURATION_CHECK]).items():
        if finding["verdict"] != "pass" and trace_id not in cited:
            continue
        if not {trace_id, inbound} <= cited:
            return False
        if finding["verdict"] == "pass":
            reason = finding["reason"]
            if f"「{rows[inbound]['text']}」" not in reason or not any(
                f"「{rows[key]['text']}」" in reason for key in replies if key in cited
            ):
                return False
    return True
