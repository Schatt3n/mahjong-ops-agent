"""实现工具域中的shared能力，供可信工具网关调用。

"""

from __future__ import annotations

from typing import Any
from ..customer_domain import scored_game_requirement_fields
from ..query_contract import game_query_strategy_contract

CANDIDATE_REPLY_STATUSES = ["accepted", "confirmed", "arrived", "declined", "negotiating", "no_reply"]
GAME_STATUSES = ["forming", "inviting", "ready", "cancelled", "finished"]

CANDIDATE_REPLY_NEXT_STEP_POLICIES: dict[str, dict[str, Any]] = {
    "declined": {
        "terminal_for_current_offer": True,
        "requires_explicit_user_request_to_search_alternatives": True,
        # 结束当前邀约协调不能吞掉同条消息新增事实；是否有新差异由主模型核对持久态。
        "complete_necessary_fact_recording_before_stop": True,
        "instruction": (
            "This tool has recorded that the current user declined or rejected the current offer. "
            "Complete any necessary recording of new or changed constraints from the same message before stopping; "
            "equivalent facts already persisted with no new difference do not require duplicate writes. "
            "Unless the same user message explicitly asks to continue looking for another game, then stop this turn "
            "with a short acknowledgement. Do not call search_current_games, search_customers, create_game, or "
            "create_invite_drafts just because the user explained a preference while declining."
        ),
    },
    "negotiating": {
        "terminal_for_current_offer": False,
        "requires_coordination_before_confirmation": True,
        "instruction": (
            "This tool has recorded a negotiation on the current offer. Continue by coordinating the current game's "
            "open question or by replying that you will ask; do not switch to a new search unless the user explicitly "
            "asks for another game."
        ),
    },
    "no_reply": {
        "terminal_for_current_offer": True,
        "instruction": "This tool has recorded no reply. Avoid customer-visible claims that the user confirmed.",
    },
    "accepted": {
        "terminal_for_current_offer": True,
        "instruction": (
            "This tool has recorded acceptance of the current offer. Reply with a minimal acknowledgement like ok/好/okk. "
            "Do not restate time, stake, smoke, ready/full status, or arrival instructions unless the user explicitly asked for status."
        ),
    },
    "confirmed": {
        "terminal_for_current_offer": True,
        "instruction": (
            "This tool has recorded confirmation of the current offer. Reply with a minimal acknowledgement like ok/好/okk. "
            "Do not restate time, stake, smoke, ready/full status, or arrival instructions unless the user explicitly asked for status."
        ),
    },
    "arrived": {
        "terminal_for_current_offer": True,
        "instruction": "This tool has recorded arrival. Reply briefly; no further search is needed from this fact alone.",
    },
}


def cross_game_commitment_summary(transitions: list[Any]) -> dict[str, Any]:
    winner_game_ids = sorted(
        {
            transition.entity_id
            for transition in transitions
            if transition.entity_type == "game"
            and transition.to_status == "ready"
            and transition.reason == "seats_full"
        }
    )
    released = []
    for transition in transitions:
        if transition.entity_type != "game_participant" or transition.to_status != "superseded":
            continue
        game_id, _, customer_id = transition.entity_id.partition(":")
        committed_game_id = transition.reason.partition("participant_committed_to_game:")[2]
        released.append(
            {
                "customer_id": customer_id,
                "released_from_game_id": game_id,
                "committed_to_game_id": committed_game_id,
            }
        )
    return {
        "winner_game_ids": winner_game_ids,
        "released_participations": released,
        "affected_game_ids": sorted(
            {
                item["released_from_game_id"]
                for item in released
            }
            | set(winner_game_ids)
        ),
        "instruction": (
            "A participant may be provisionally present in many options. When the first overlapping game becomes "
            "ready, the backend atomically commits that participant there and releases every conflicting option. "
            "Use released_participations to coordinate follow-up messages; never re-add the participant to a losing "
            "overlapping game unless the winning commitment is cancelled first."
        ),
    }


def current_game_search_reply_contract(
    requirement: dict[str, Any],
    matches: list[dict[str, Any]],
    alternatives: list[dict[str, Any]] | None = None,
    *,
    unfiltered_available_pool_empty: bool | None = None,
) -> dict[str, Any]:
    """按后端空池证据限定缺失结论，不从评分查询空返回推导任意子范围也为空。"""
    applied_filters = scored_game_requirement_fields(requirement)
    unfiltered = not applied_filters
    alternatives = list(alternatives or [])
    # 未提供未过滤局池证据时保持未知；有返回项时也不能被外部空池标记覆盖。
    pool_empty = unfiltered_available_pool_empty if not matches and not alternatives else False
    supports_narrower_empty_claim = pool_empty is True
    match_summaries = [
        str(item.get("game", {}).get("requirement", {}).get("user_visible_summary") or "").strip()
        for item in matches
    ]
    match_summaries = [item for item in match_summaries if item]
    has_matches = bool(matches)
    alternative_summaries = [
        str(item.get("game", {}).get("requirement", {}).get("user_visible_summary") or "").strip()
        for item in alternatives
    ]
    alternative_summaries = [item for item in alternative_summaries if item]
    status = "actionable_matches" if has_matches else (
        "decision_required_alternatives" if alternatives else "no_actionable_match"
    )
    if unfiltered:
        status = "unfiltered_pool_listing" if has_matches or alternatives else "unfiltered_pool_empty"
    return {
        "source_tool": "search_current_games",
        "game_query_strategy_contract": game_query_strategy_contract(),
        "matched_query_requirement": requirement,
        "matched_result_summaries": match_summaries,
        "alternative_result_summaries": alternative_summaries,
        "evidence_scope": {
            "kind": "unfiltered_inventory" if unfiltered else "submitted_filters_only",
            "applied_filter_fields": sorted(applied_filters),
            "criteria_completeness": "not_assessed_by_retrieval",
            # 返回空可能来自评分门槛；只有同请求者未过滤可用局池也空才足以证明任意子范围为空。
            "unfiltered_available_pool_empty": pool_empty,
            "available_pool_scope": "same_requester_and_conversation",
            "supports_narrower_empty_claim": supports_narrower_empty_claim,
            "claim_boundary": (
                "Inventory is not personal suitability. An empty unfiltered result only describes currently available "
                "registered open games for the same requester and conversation; it establishes neither this customer's "
                "needs nor the availability of unqueried customers. "
                "Personal fit, unknown criteria and public option selection follow game_query_strategy_contract. "
                "Whether to state a query result follows game_query_strategy_contract.query_result_reporting. Only when "
                "supports_narrower_empty_claim=true has the backend confirmed an empty unfiltered available pool "
                "for the same requester and conversation; that evidence supports absence in its narrower scopes. "
                "Otherwise an empty filtered result establishes only that this retrieval returned no game: adding "
                "filters may raise a candidate's score, so do not infer that every narrower query would also be empty. "
                "Non-empty or truncated results cannot establish absence in an unqueried narrower scope."
            ),
        },
        "search_result_semantics": {
            "status": status,
            "backend_retrieval_policy_applied": True,
            "actionable_match_count": 0 if unfiltered else len(matches),
            "returned_game_count": len(matches) + len(alternatives),
            "alternative_count": len(alternatives),
            "instruction": (
                "This is an unfiltered inventory read, not a personalized match. Describe established public inventory "
                "facts and choose the next response under game_query_strategy_contract. Do not turn no available "
                "registered game into no suitable game or no available player."
                if unfiltered else
                "The non-empty matches list is the backend-selected actionable candidate set. Do not recompute "
                "eligibility, reject these candidates, or repeat the same semantic search. Matching applies to the "
                "submitted filters; personal fit and public option presentation follow game_query_strategy_contract."
                if has_matches
                else (
                    "No exact actionable game was found. alternatives contains only profile-supported options whose "
                    "decision_required_fields conflict with the current explicit request. You may offer one only by "
                    "clearly disclosing those differences and asking the customer; never describe it as an exact match "
                    "and do not repeat the same search."
                    if alternatives
                else
                "No game was returned under the submitted filters. This is not proof that the customer's relevant "
                "criteria are complete. Follow game_query_strategy_contract without announcing an unsupported "
                "personal fit/no-fit conclusion. Result reporting and any conditional service default follow "
                "game_query_strategy_contract; only supports_narrower_empty_claim=true additionally "
                "supports unqueried narrower empty scopes. Otherwise do not turn a scoring-based empty return "
                "into proof that registered games do not exist. Ask unknown facts only when the selected next action "
                "needs them. An expressed query fact does not require creating or joining a game. "
                "No registered actionable game does not mean no potential "
                "players. This read does not decide whether the main "
                "agent should offer help finding players. Updated user constraints may justify a new broader query; "
                "the empty result neither grants nor revokes a prior user delegation. Keep its scope and source "
                "across tool steps; when recruitment was already delegated and required facts are known, continue "
                "that goal through tools rather than ask for the same permission again. "
                "do not repeat the same search without new evidence."
                )
            ),
        },
        "reply_shape": "Choose a contextually reasonable response under game_query_strategy_contract; no fixed option count or clarification order.",
        "customer_visible_rule": (
            "Use actual customer-visible game facts under game_query_strategy_contract.public_option_presentation. "
            "For a requester, 可以不/可以吗 is a natural confirmation; 打吗/来吗 is mainly for candidate invitations. "
            "Do not expose backend search reasons or private profile sources."
        ),
        "reply_examples_scope": "以下是客户已知个人条件且无差异时的简短话术参考，不限制未知条件下公开选项的必要事实或合法回复路径。",
        "good_reply_examples": ["七点三缺一，可以不", "七点三缺一，可以吗"],
        "bad_reply_examples": [
            "七点三缺一，0.5无烟杭麻，打吗",
            "已按你的画像找到七点三缺一",
            "六点半没有，七点有个三缺一，0.5无烟，可以不",
        ],
    }


def known_players_with_requesting_party(
    *,
    known_players: list[dict[str, Any]],
    requesting_party: Any,
) -> list[dict[str, Any]]:
    players = [dict(item) for item in known_players if isinstance(item, dict)]
    if not isinstance(requesting_party, dict):
        return players
    contact_id = str(requesting_party.get("contact_id") or requesting_party.get("customer_id") or "").strip()
    if not contact_id:
        return players
    payload = {
        "customer_id": contact_id,
        "display_name": str(requesting_party.get("contact_name") or requesting_party.get("display_name") or contact_id),
        "source": str(requesting_party.get("source") or "requesting_party"),
        "seat_count": requesting_party.get("seat_count") or requesting_party.get("party_size") or 1,
        "known_member_ids": list(requesting_party.get("known_member_ids") or [contact_id]),
        "anonymous_seat_count": requesting_party.get("anonymous_seat_count"),
    }
    for index, item in enumerate(players):
        if str(item.get("customer_id") or "").strip() != contact_id:
            continue
        merged = {**item}
        for key, value in payload.items():
            if value is None:
                continue
            if key not in merged or merged.get(key) in (None, "", [], {}):
                merged[key] = value
        players[index] = merged
        break
    else:
        players.insert(0, payload)
    return players
