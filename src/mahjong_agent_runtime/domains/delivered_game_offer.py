"""为真实送达的私聊讨论保存最小牌局引用，供任务空闲轮转后查局。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..models import DEFAULT_TZ, AgentAction, ConversationRole, ConversationTaskContext, Game, GameStatus, UserMessage, now
from ..stores import AgentStore
from .model_context import has_agreed_game_duration
from .stake_values import parse_number
from .value_utils import first_present_value, normalize_smoke_preference


DELIVERED_GAME_OFFER = "delivered_game_offer"
_RECENT_TURNS = 60
_OFFER_REQUIREMENT_FIELDS = (
    "game_type", "stake", "stake_range", "base_stake", "cap_score", "stake_label",
    "smoke_preference", "start_time_kind", "duration_kind", "duration_hours", "duration", "max_duration_hours",
)


def seat_facts_for_private_reply(store: AgentStore, message: UserMessage,
                                 tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只从本轮末次成功查局和当前库读出可核对的同桌局况，不继承旧查询或模型摘要。"""
    if not ordinary_private_message(message):
        return []
    query_index = next((index for index in range(len(tool_results) - 1, -1, -1)
                        if isinstance(tool_results[index], dict)
                        and tool_results[index].get("name") == "search_current_games"), None)
    if query_index is None:
        return []
    query = tool_results[query_index]
    if (query.get("called") is not True or query.get("allowed") is not True or query.get("error")
            or not isinstance(query.get("result"), dict)):
        return []
    result = query["result"]
    if (result.get("selection_truncated")
            or isinstance(result.get("match_count"), int) and result["match_count"] > len(result.get("matches") or [])):
        return []
    options = []
    for key in ("matches", "alternatives", "nearby_options"):
        rows = result.get(key) or []
        if not isinstance(rows, list):
            return []
        options.extend(rows)
    # 查局后另建的桌可能有相同钟点，不能用既有候选核对新桌回复。
    if any(entry.get("name") == "create_game" and entry.get("called") is True and not entry.get("error")
           for entry in tool_results[query_index + 1:] if isinstance(entry, dict)):
        return []
    facts: dict[str, dict[str, Any]] = {}
    for option in options:
        observed = option.get("game") if isinstance(option, dict) else None
        game_id = observed.get("game_id") if isinstance(observed, dict) else None
        if not isinstance(game_id, str) or not game_id:
            return []
        try:
            game = store.require_game(game_id)
        except ValueError:
            return []
        start = game.planned_start_at.astimezone(DEFAULT_TZ).isoformat() if game.planned_start_at else None
        if (game.status not in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY} or game.actual_started_at
                or _time_anchor(observed.get("planned_start_at")) != start):
            return []
        facts[game_id] = {"game_id": game_id, "status": game.status.value,
                          "planned_start_at": start, "actual_started_at": None,
                          "game_type": game.requirement.get("game_type") or game.requirement.get("requested_game"),
                          "remaining_seats": game.remaining_seats(), "seats_total": game.seats_total}
    return list(facts.values())


def game_offer_for_review(
    store: AgentStore, message: UserMessage, action: AgentAction, tool_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """用本轮最新成功查询与当前局况生成待审候选，不把规划器选择当成已送达推荐。"""
    if (not ordinary_private_message(message) or action.needs_human
            or action.objective_status not in {"waiting_user", "completed"}):
        return None
    latest_query = next((entry for entry in reversed(tool_results)
                         if isinstance(entry, dict) and entry.get("name") == "search_current_games"), None)
    if (not latest_query or latest_query.get("called") is not True or latest_query.get("allowed") is not True
            or latest_query.get("error")):
        return None
    result = latest_query.get("result")
    if not isinstance(result, dict):
        return None
    matches = result.get("matches") or []
    if not isinstance(matches, list) or len(matches) != 1 or result.get("alternatives") or result.get("nearby_options"):
        return None
    match = matches[0]
    if not isinstance(match, dict) or match.get("requires_timing_confirmation") is True:
        return None
    observed = match.get("game")
    if not isinstance(observed, dict):
        return None
    game_id = observed.get("game_id")
    # 只有查询和独立审查看到的具体一桌能建立引用；主模型未填可选槽位不构成证据缺失。
    proposed_id = action.objective_state.get("active_game_id")
    if not isinstance(game_id, str) or not game_id or (proposed_id is not None and proposed_id != game_id):
        return None
    game = _available_game(store, game_id, message)
    if game is None or _offer_terms(observed) != _offer_terms(game.to_dict()):
        return None
    return {"game_id": game_id, "terms": _offer_terms(observed)}


def table_duration_facts_for_private_reply(store: AgentStore, table_facts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """只给当前唯一查实的一桌提供约定时长，不把内部预计散场或招募失效算作约定。"""
    if len(table_facts) != 1:
        return None
    table, = table_facts
    try:
        game = store.require_game(table["game_id"])
    except ValueError:
        return None
    if game.status not in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY} or game.actual_started_at:
        return None
    agreed = has_agreed_game_duration(game.requirement)
    duration = parse_number(first_present_value(game.requirement, "duration_hours", "duration")) if agreed else None
    return {"game_id": game.game_id,
            "planned_start_at": game.planned_start_at.isoformat() if game.planned_start_at else None, "agreed_duration_hours": duration,
            "agreed_end_at": game.planned_end_at.isoformat() if agreed and game.planned_end_at else None}


def own_seat_fact_for_private_reply(
    store: AgentStore, message: UserMessage, visible_games: list[dict[str, Any]],
    tool_results: list[dict[str, Any]], current_seat_facts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """唯一有据的桌才能核对本人席位，实际写入和当下席位均由 Store 核实。"""
    metadata = message.metadata or {}
    if (not ordinary_private_message(message) or metadata.get("source") == "group_private_continuation"
            or metadata.get("group_topic_source") or metadata.get("input_source") == "system_trigger"):
        return None
    successful = [entry for entry in tool_results if entry.get("called") is True and entry.get("allowed") is True
                  and not entry.get("error") and isinstance(entry.get("result"), dict)]
    writes = [entry["result"]["game"] for entry in successful if isinstance(entry["result"].get("game"), dict)
              and (entry.get("name") in {"create_game", "join_game"}
                   or (entry.get("name") == "record_candidate_reply"
                       and entry["result"].get("recorded_status") in {"joined", "confirmed", "accepted", "arrived"}))]
    if (not current_seat_facts and not writes
            and any(entry.get("name") == "search_current_games" for entry in tool_results)):
        return None  # 失败、空、截断或已变化的搜索不足以限定本次回复指向哪桌。
    games = {row["game_id"] for row in (*visible_games, *current_seat_facts, *writes)
             if isinstance(row, dict) and isinstance(row.get("game_id"), str) and row["game_id"]}
    if len(games) != 1:
        return None
    try:
        game = store.require_game(next(iter(games)))
    except ValueError:
        return None
    if game.status not in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY} or game.actual_started_at:
        return None
    if any(participant.customer_id == message.sender_id and participant.status in {"accepted", "arrived"}
           for participant in game.participants):
        return None  # 历史状态已记录参与，交给原有审查处理。
    game_type = first_present_value(game.requirement, "game_type", "requested_game")
    smoke = normalize_smoke_preference(game.requirement.get("smoke_preference"))
    stake_label = game.requirement.get("stake_label")
    lookup_terms = {}
    if isinstance(game_type, str) and game_type:
        lookup_terms["game_type"] = game_type
        if smoke in {"no_smoking", "smoking"}:
            lookup_terms["smoke_preference"] = smoke
        if isinstance(stake_label, str):
            lookup_terms["stake_label"] = stake_label
    return {
        "game_id": game.game_id,
        "planned_start_at": game.planned_start_at.isoformat() if game.planned_start_at else None,
        **lookup_terms,
        "participation_already_recorded": any(message.sender_id in {claim["contact_id"], *claim["known_member_ids"]}
                                              for claim in game.seat_claims()),
    }


def delivered_offer_matches_game(offer: Any, *, game: Game, recipient_id: str) -> bool:
    """正式推荐才可作为本人接受提议的证据；调用方仍须核实送达与报名权限。"""
    return (isinstance(offer, dict) and "game_reference_only" not in offer
            and delivered_reference_matches_game(offer, game=game, recipient_id=recipient_id))


def delivered_reference_matches_game(reference: dict[str, Any], *, game: Game, recipient_id: str) -> bool:
    """定位也须对齐收件人、牌局和条款，防止旧条件或另一位顾客的引用被沿用。"""
    return (reference.get("recipient_id") == recipient_id and reference.get("game_id") == game.game_id
            and reference.get("terms") == _offer_terms(game.to_dict()))


def pending_game_offer_after_idle_reset(
    store: AgentStore, message: UserMessage, *, task_context: ConversationTaskContext | None, current_version: int,
) -> dict[str, Any] | None:
    """只投影本会话上一轮送达且仍有效的一桌；原话、名单、旧任务和写权限都不恢复。"""
    if (not ordinary_private_message(message) or message.quoted_message is not None or task_context is None
            or task_context.reset_reason != "idle_task_timeout" or not task_context.previous_task_context_id):
        return None
    preceding_reply = next((turn for turn in reversed(store.recent_turns(message.conversation_id, _RECENT_TURNS))
                            if turn.role == ConversationRole.ASSISTANT
                            and turn.metadata.get("delivery_status") == "delivered"), None)
    if preceding_reply is None:
        return None
    metadata = preceding_reply.metadata
    offer = metadata.get(DELIVERED_GAME_OFFER)
    if (metadata.get("context_invalidated") or metadata.get("task_context_id") != task_context.previous_task_context_id
            or metadata.get("conversation_version") != current_version - 1 or not isinstance(offer, dict)
            or offer.get("recipient_id") != message.sender_id):
        return None
    game_id = offer.get("game_id")
    if not isinstance(game_id, str) or not game_id:
        return None
    game = _available_game(store, game_id, message)
    if game is None:
        return None
    if not delivered_reference_matches_game(offer, game=game, recipient_id=message.sender_id):
        return None
    reference_only = offer.get("game_reference_only") is True
    return {
        "game_id": game_id,
        "terms": offer["terms"],
        **({"game_reference_only": True} if reference_only else {}),
        "usage": (
            ("这是店家上一轮与本人谈过的唯一一桌。此前没有确认可直接报名，也不能证明桌上同意等候。"
             if reference_only else "这是店家上一轮实际送达给本人的单一牌局推荐。")
            + "当前来话若明显接续这桌，先用它的原开场、玩法及档位调用 search_current_games，核对同一个 game_id，"
            "无需再问顾客哪桌。顾客说的到店时间与能打到几点分别是个人条件，不能拿来代替这桌的开场或散场。"
            "同一桌仍可参加，顾客本轮明确授权且相关条件已核实，才按正常流程确认。"
            "计划开场已过需核对实际局况；若顾客提到的到店时间也已过去且未说明已经到店，澄清现在何时能到，"
            "不能假定已经到店或桌上已同意等候。当前来话明显在聊其他事或改找别桌时不主动提这桌。"
        ),
    }


def ordinary_private_message(message: UserMessage) -> bool:
    """隔离群聊投影、群转私和系统主动通知；这些流程有自己的业务引用。"""
    metadata = message.metadata or {}
    return bool(message.sender_id and not (
        metadata.get("source") == "group" or metadata.get("is_room") or metadata.get("room_id")
        or "reply_constraints" in metadata or metadata.get("internal_event") or metadata.get("system_trigger")
        or metadata.get("_trusted_source_task_context_id")
    ))


def _available_game(store: AgentStore, game_id: str, message: UserMessage) -> Game | None:
    """复查当前报名窗口；计划开场已过但实际尚未开打的局仍可作为查找线索。"""
    try:
        game = store.require_game(game_id)
    except ValueError:
        return None
    at = max(message.sent_at, now())
    if (game.status not in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY} or game.actual_started_at
            or not game.remaining_seats() or game.requirement.get("recruitment_mode") == "board_only"
            or not game.planned_start_at or not any((game.planned_end_at, game.expires_at))
            or any(limit and limit <= at for limit in (game.planned_end_at, game.expires_at))):
        return None
    return game


def _offer_terms(game: dict[str, Any]) -> dict[str, Any]:
    """对齐查询投影与库内局况，只比较开场、确定的时长及允许披露的条件。"""
    requirement = game.get("requirement") or {}
    terms = {key: requirement[key] for key in _OFFER_REQUIREMENT_FIELDS if requirement.get(key) is not None}
    terms["planned_start_at"] = _time_anchor(game.get("planned_start_at"))
    if "duration_hours" in terms or "duration" in terms:
        terms["planned_end_at"] = _time_anchor(game.get("planned_end_at"))
    return terms


def _time_anchor(value: str | None) -> str | None:
    """比较实际时间点，避免相同时间因采用不同的区时表示而被判改期。"""
    return datetime.fromisoformat(value).astimezone(DEFAULT_TZ).isoformat() if value else None
