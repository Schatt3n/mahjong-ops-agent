"""筛选与当前客户相关的牌局，并生成内部与客户可见两种投影。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...models import CustomerProfile, Game
from ...stores import AgentStore
from ..model_context import game_for_model_context


@dataclass(slots=True)
class GameContextBundle:
    """与单个发送者相关的牌局事实，与全局存储状态分离。"""

    games: list[Game]
    model_contexts: list[dict[str, Any]]
    visible_summaries: list[dict[str, Any]]
    sender_memberships: list[dict[str, Any]]
    active_parties: list[dict[str, Any]]


def build_game_context(
    store: AgentStore,
    *,
    conversation_id: str,
    sender_id: str,
    allow_private_public_roster: bool = False,
) -> GameContextBundle:
    """读取客户相关局，而不是把全局局池整体交给模型。

    相关性来源包括当前会话、本人组织/参与、本人邀约和本人候补通知。
    ``model_contexts`` 供内部决策，``visible_summaries`` 遵守客户可见边界。
    公开名单须另有私聊授权，且仅同会话、本人组织或有效占座才可披露；
    邀约或候补通知只用于内部相关性，不随公开昵称登记扩大名单权限。
    """

    all_active_games = store.active_games()
    related_game_ids = {
        game.game_id
        for game in all_active_games
        if game.conversation_id == conversation_id
        or game.organizer_id == sender_id
        or any(participant.customer_id == sender_id for participant in game.participants)
    }
    related_game_ids.update(
        draft.game_id
        for draft in store.invite_drafts.values()
        if draft.customer_id == sender_id
    )
    related_game_ids.update(
        str(draft.metadata.get("game_id") or "")
        for draft in store.outbound_message_drafts.values()
        if draft.recipient_id == sender_id
        and draft.purpose == "waiting_match_notification"
        and draft.metadata.get("game_id")
    )
    games = [game for game in all_active_games if game.game_id in related_game_ids]
    model_contexts = [compact_game(game_for_model_context(game, store.customers)) for game in games]
    # 名单披露与内部决策相关性分开：未发草稿、旧拒绝及其他客户关系均不授予名单权限。
    roster_game_ids = {
        game.game_id for game in games
        if game.conversation_id == conversation_id
        or game.organizer_id == sender_id
        or any(claim["contact_id"] == sender_id for claim in game.seat_claims())
    }
    visible_summaries = [
        active_game_visible_summary(
            game,
            registered_public_names=(
                _registered_public_participant_names(game, store.customers)
                if allow_private_public_roster and game.game_id in roster_game_ids else None
            ),
        ) for game in games
    ]
    sender_memberships = sender_active_game_memberships(games, sender_id)
    active_parties = [
        {
            "game_id": game_context["game_id"],
            "seat_summary": dict(game_context.get("seat_summary") or {}),
        }
        for game_context in model_contexts
    ]
    return GameContextBundle(
        games=games,
        model_contexts=model_contexts,
        visible_summaries=visible_summaries,
        sender_memberships=sender_memberships,
        active_parties=active_parties,
    )


def sender_active_game_memberships(games: list[Game], sender_id: str) -> list[dict[str, Any]]:
    memberships: list[dict[str, Any]] = []
    for game in games:
        for participant in game.participants:
            if participant.customer_id != sender_id:
                continue
            memberships.append(
                {
                    "game_id": game.game_id,
                    "participant_status": participant.status,
                    "seat_count": participant.seat_count,
                    "participation_already_recorded": participant.status
                    in {"joined", "confirmed", "accepted", "arrived"},
                    "write_instruction": (
                        "Do not call record_candidate_reply with the same participation meaning unless the current "
                        "message explicitly changes status or seat_count."
                    ),
                }
            )
    return memberships


def compact_game(game: Any) -> dict[str, Any]:
    """压缩牌局，保留决策所需字段并限制载荷大小。"""
    if not isinstance(game, dict):
        return {}
    return {
        "game_id": game.get("game_id"),
        "conversation_id": game.get("conversation_id"),
        "organizer_id": game.get("organizer_id"),
        "organizer_name": game.get("organizer_name"),
        "status": game.get("status"),
        "requirement": compact_requirement(game.get("requirement")),
        "seat_summary": game.get("seat_summary"),
        "remaining_seats": game.get("remaining_seats"),
        "planned_start_at": game.get("planned_start_at"),
        "planned_end_at": game.get("planned_end_at"),
        "expires_at": game.get("expires_at"),
        "participants": [
            {
                "customer_id": item.get("customer_id"),
                "display_name": item.get("display_name"),
                "status": item.get("status"),
                "seat_count": item.get("seat_count"),
                "source": item.get("source"),
            }
            for item in list(game.get("participants") or [])[:8]
            if isinstance(item, dict)
        ],
        "parties": [compact_party(item) for item in list(game.get("parties") or [])[:8]],
    }


def compact_party(party: Any) -> dict[str, Any]:
    """压缩party，保留决策所需字段并限制载荷大小。"""
    if not isinstance(party, dict):
        return {}
    return {
        "party_id": party.get("party_id"),
        "contact_id": party.get("contact_id"),
        "contact_name": party.get("contact_name"),
        "seat_count": party.get("seat_count"),
        "anonymous_seat_count": party.get("anonymous_seat_count"),
        "status": party.get("status"),
        "source": party.get("source"),
    }


def compact_requirement(requirement: Any) -> dict[str, Any]:
    """压缩组局条件，保留决策所需字段并限制载荷大小。

"""

    if not isinstance(requirement, dict):
        return {}
    structural_duplicates = {
        "requesting_party",
        "seat_claims",
        "parties",
        "participants",
        "known_players",
    }
    return {
        key: compact_context_value(value)
        for key, value in requirement.items()
        if key not in structural_duplicates
    }


def compact_context_value(value: Any, *, max_list_items: int = 12, max_string_chars: int = 1000) -> Any:
    """压缩上下文value，保留决策所需字段并限制载荷大小。

"""

    if isinstance(value, dict):
        return {
            str(key): compact_context_value(
                item,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            compact_context_value(
                item,
                max_list_items=max_list_items,
                max_string_chars=max_string_chars,
            )
            for item in value[:max_list_items]
        ]
    if isinstance(value, str) and len(value) > max_string_chars:
        return value[:max_string_chars] + "...[truncated]"
    return value


def _registered_public_participant_names(game: Game, customers: dict[str, CustomerProfile]) -> list[str]:
    """仅从真实占座者的显式公开昵称登记提取值，不输出身份键、状态或备注。

    调用方须先确定当前收件人与牌局及私聊通道的可见关系；这里不扫描无关客户、
    不包含候选人或已拒绝者，也不使用 visible_name 的显示名/内部编号回退。
    """
    names: list[str] = []
    for claim in game.seat_claims():
        profile = customers.get(str(claim["contact_id"]))
        raw_name = profile.public_name if profile is not None else None
        public_name = raw_name.strip() if isinstance(raw_name, str) else ""
        if public_name and public_name not in names:
            names.append(public_name)
    return names


def active_game_visible_summary(
    game: Game, *, registered_public_names: list[str] | None = None,
) -> dict[str, Any]:
    """构建允许对当前客户披露的局事实；名单只接受已完成作用域校验的登记投影。

    默认不带名单，防止其他直接调用者或群聊路径凭 Game 对象获得披露授权。
    可见摘要提供事实与条件性回复合同，不把全部可披露字段变成本轮必答内容。
    """

    requirement = dict(game.requirement or {})
    public_requirement_keys = (
        "user_visible_summary",
        "game_type",
        "stake",
        "stake_range",
        "base_stake",
        "cap_score",
        "stake_label",
        "smoke_preference",
        "start_time_kind",
        "start_time",
        "duration_kind",
        "duration_hours",
        "known_player_count",
        "needed_seats",
    )
    visible_summary = str(requirement.get("user_visible_summary") or "")
    summary = {
        "game_id": game.game_id,
        "status": game.status.value,
        "user_visible_summary": visible_summary,
        "status_query_reply_contract": {
            "when_to_use": "用户问当前局况、现在几个人、还差几人、有没有进展时使用。",
            "applicability": (
                "先按用户真实问题和已授权目标确定本次响应范围；本轮回答该局局况、人数、缺口或进展，"
                "或介绍供客户决策的真实选项时，须保留摘要的全部决策锚点，不要求用户逐字说完整局况。"
                "仅名单及不表示局况进展的窄属性可省未问的邻接背景；人数、缺口和进展不属于这种豁免。"
                "可披露不等于必须披露；单问名单不擅自附带局况或参加确认，但识别所问对象确实必要的信息"
                "仍保留。主模型的 goal/reply_scope 及审查建议都不能自行扩大用户请求。"
            ),
            "preferred_reply_source": "user_visible_summary",
            "preferred_reply_text": visible_summary,
            "preservation_mode": "all_decision_anchors",
            "required_semantic_source": "preferred_reply_text",
            "invalid_rewrite": "本次在回答局况、人数、缺口或进展，或提供决策选项时，只保留人数或缺口，丢失 preferred_reply_text 中的时间、公开昵称、局名或缺口短码。",
            "rule": "仅在 applicability 成立且 user_visible_summary 非空时，优先原样使用或轻微口语化；不要只根据 seat_summary 重新概括而丢掉本次决策所需的时间、公开昵称、局名或缺口短码。",
        },
        "seat_summary": game.seat_summary(),
        "public_requirement": {
            key: requirement.get(key)
            for key in public_requirement_keys
            if requirement.get(key) is not None
        },
    }
    if registered_public_names is not None:
        # 名单是可按需使用的公开值，不附带谁组织、谁确认、是否被邀约等内部解释。
        summary["public_participant_names"] = list(registered_public_names)
    return summary


__all__ = [
    "GameContextBundle",
    "active_game_visible_summary",
    "build_game_context",
    "compact_context_value",
    "compact_game",
    "compact_party",
    "compact_requirement",
    "sender_active_game_memberships",
]
