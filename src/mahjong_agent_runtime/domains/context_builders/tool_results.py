"""构建工具results上下文，并执行作用域裁剪与隐私隔离。

"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any

from ...models import ConversationTurn, ToolResult
from ...llm import compact_model_response_source
from ...token_estimation import estimate_tokens
from ..delivered_game_offer import DELIVERED_GAME_OFFER
from ..model_context import game_requirement_for_conversation
from .customer_context import compact_candidate, compact_draft
from .game_context import compact_game
from .sanitization import sanitize_message_metadata_for_context
from ..model_context import public_game_lookup_projection


PASSTHROUGH_TOOL_RESULT_KEYS = (
    "participation_deadline",
    "retryable_no_effect",
    "cleared",
    "follow_up",
    # Store已投影为不含群标识的安排证据；必须随真实工具结果回灌主模型和话术审查。
    "board_arrangement",
    "stake_confirmation",
    "closed",
    "requirement",
    "lookup_scope",
    "table_reference_evidence",
    "reference_requirement",
    # 规范化冲突的拒绝原因与同客户来源标识必须回灌，供主模型有据修正断言后重新规划。
    "validation_error",
    "table_availability_evidence",
    "reference_fact_sources",
    "customer_reply_contract",
    "selection_truncated",
    "seat_claim_evidence",
    "same_table_membership_evidence",
    "table_duration_fit_evidence",
    "configured",
    "start_at",
    "end_at",
    "room_count",
    "available_room_ids",
    "occupied_room_ids",
    "available_count",
    "recorded_status",
    "next_step_policy",
    "approved",
    "needs_human",
    "raw_approved",
    "reasoning_summary",
    "violations",
    "item_reviews",
    "instruction",
    "review_scope",
    "items",
    "exclude_customer_ids",
    "continuation",
    "stale_run",
    "current_version",
    "run_version",
)

_PARTICIPATION_GAME_RECEIPTS = frozenset({"join_game", "record_candidate_reply", "record_registered_member_withdrawal"})

# 只对后端生成的查询口径和业务需求副本做逐值引用。结果、来源和身份/权限外壳照常展开。
_REPEATED_TOOL_VALUE_PATHS = (
    ("requirement",),
    ("continuation", "authoritative_facts", "requirement"),
    ("board_arrangement",),
    ("continuation", "authoritative_facts", "board_arrangement"),
    ("customer_reply_contract", "matched_query_requirement"),
    ("customer_reply_contract", "search_result_semantics"),
    ("customer_reply_contract", "evidence_scope", "claim_boundary"),
    ("customer_reply_contract", "customer_visible_rule"),
    ("customer_reply_contract", "reply_shape"),
)


def compact_tool_feedback_for_transport(payload: dict[str, Any]) -> int:
    """在独立的发送副本中去除已投影工具反馈的空白和重复值，返回替换数量。

    先按执行顺序索引本轮证据，每个引用只指向仍完整展开的 `turn_tool_evidence` 值。
    历史工具反馈保留 JSON 字符串类型；用户和助手消息不会被解析或改写。
    """

    canonical_values: dict[str, dict[str, Any]] = {}

    def reference_values(results: list[Any], *, is_turn_evidence: bool = False) -> int:
        """比较保留类型的紧凑 JSON，避免将 True 与 1 等不同证据折叠。"""

        count = 0
        for index, item in enumerate(results):
            result = item.get("result") if isinstance(item, dict) else None
            for path in _REPEATED_TOOL_VALUE_PATHS:
                parent = result
                for key in path[:-1]:
                    parent = parent.get(key) if isinstance(parent, dict) else None
                key = path[-1]
                if not isinstance(parent, dict) or key not in parent:
                    continue
                value = json.dumps(parent[key], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                canonical = canonical_values.get(value)
                if canonical is None and is_turn_evidence:
                    canonical_values[value] = {"index": index, "path": ".".join(("result", *path))}
                elif canonical is not None:
                    reference = {"tool_evidence_value_ref": canonical}
                    if estimate_tokens(reference) < estimate_tokens(value):
                        parent[key] = reference
                        count += 1
        return count

    count = reference_values(payload.get("turn_tool_evidence") or [], is_turn_evidence=True)
    count += reference_values(payload.get("previous_tool_results") or [])
    for turn in payload.get("recent_conversation") or []:
        if not isinstance(turn, dict) or turn.get("role") != "tool":
            continue
        metadata = turn.get("metadata")
        content = turn.get("content")
        if (not isinstance(metadata, dict) or metadata.get("compacted_for_context") is not True
                or not isinstance(content, str)):
            continue
        try:
            results = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(results, list):
            count += reference_values(results)
            turn["content"] = json.dumps(results, ensure_ascii=False, separators=(",", ":"))
    return count


def reference_shared_tool_schemas(payload: dict[str, Any]) -> int:
    """为发送副本中的完全相同参数子合同保留一份定义，返回实际替换数。

    只遍历模型可用工具的 schema 节点，不更改工具清单、权限、事实或服务端校验器。
    已有引用或标识的节点依赖原作用域，不能移动；共享定义均保持展开，不形成引用链。
    """

    tools = payload.get("available_tools") or []
    if "shared_tool_schemas" in payload:
        return 0

    def slots(container: dict[str, Any], key: str):
        """只枚举对象参数属性和数组元素使用的 JSON Schema 子节点。"""

        if not isinstance(container.get(key), dict):
            return
        yield container, key
        schema = container[key]
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name in properties:
                yield from slots(properties, name)
        yield from slots(schema, "items")

    def packed_schema(schema: dict[str, Any]) -> str:
        """按类型编码以比较完整结构和说明；不合并已有引用或相对地址。"""

        encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return encoded if '"$' not in encoded else ""

    counts = Counter(
        serialized for tool in tools if isinstance(tool, dict)
        for container, key in slots(tool, "schema")
        if (serialized := packed_schema(container[key]))
    )
    # 小对象展开更便于阅读；只有节省量覆盖指针和共享键的预留成本时才替换。
    eligible = {serialized for serialized, count in counts.items()
                if count >= 2 and (count - 1) * estimate_tokens(serialized) > 22 * count + 8}
    definitions: dict[str, Any] = {}
    keys: dict[str, str] = {}
    count = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        for container, key in slots(tool, "schema"):
            serialized = packed_schema(container[key])
            if serialized not in eligible:
                continue
            definition_id = keys.get(serialized)
            if definition_id is None:
                definition_id = keys[serialized] = f"s{len(keys)}"
                definitions[definition_id] = json.loads(serialized)
            container[key] = {"$ref": f"#/shared_tool_schemas/{definition_id}"}
            count += 1
    if definitions:
        payload["shared_tool_schemas"] = definitions
    return count


def turn_payload_for_context(turn: ConversationTurn) -> dict[str, Any]:
    """将一轮对话及已持久化的工具结果精确地序列化压缩一次。"""

    payload = turn.to_dict()
    if payload.get("role") == "assistant":
        # 原始交付引用只由送达后的作用域校验读取，不随任意历史轮次进入模型。
        payload["metadata"].pop(DELIVERED_GAME_OFFER, None)
    if payload.get("role") == "user":
        payload["metadata"] = sanitize_message_metadata_for_context(payload.get("metadata"))
    if payload.get("role") != "tool":
        return payload
    try:
        raw_results = json.loads(str(payload.get("content") or "[]"))
    except json.JSONDecodeError:
        return payload
    if not isinstance(raw_results, list):
        return payload
    payload["content"] = json.dumps(
        [compact_tool_result_dict(item) for item in raw_results],
        ensure_ascii=False,
    )
    payload["metadata"] = {**dict(payload.get("metadata") or {}), "compacted_for_context": True}
    return payload


def tool_result_for_context(result: ToolResult) -> dict[str, Any]:
    return compact_tool_result_dict(result.to_dict())


def reference_duplicate_latest_tool_results(
    latest_results: list[dict[str, Any]],
    turn_evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """用指向完整轮次证据的指针替换重复的最新反馈。

    ``previous_tool_results`` 是最近反馈视图，``turn_tool_evidence`` 是完整的有序记录。
    不论当前预算是否紧张，重复序列化都不增加事实。指针保留最新结果顺序和执行状态，
    被引用项保留完整载荷；相同 call_id 不能代替完整内容相等。
    """

    referenced: list[dict[str, Any]] = []
    compacted_count = 0
    claimed_indexes: set[int] = set()
    for latest in latest_results:
        evidence_index = _matching_evidence_index(
            latest,
            turn_evidence,
            claimed_indexes=claimed_indexes,
        )
        if evidence_index is None:
            referenced.append(latest)
            continue
        claimed_indexes.add(evidence_index)
        referenced.append(
            {
                "name": latest.get("name"),
                "called": latest.get("called"),
                "allowed": latest.get("allowed"),
                "call_id": latest.get("call_id"),
                "error": latest.get("error"),
                "deduplicated": latest.get("deduplicated", False),
                "result_reference": {
                    "source": "turn_tool_evidence",
                    "index": evidence_index,
                    "call_id": latest.get("call_id"),
                },
            }
        )
        compacted_count += 1
    return referenced, compacted_count


def _matching_evidence_index(
    latest: dict[str, Any],
    turn_evidence: list[dict[str, Any]],
    *,
    claimed_indexes: set[int],
) -> int | None:
    """查找最新等效证据项，而不折叠重复项。"""

    for index in range(len(turn_evidence) - 1, -1, -1):
        if index in claimed_indexes:
            continue
        candidate = turn_evidence[index]
        if _same_json_value(candidate, latest):
            return index
    return None


def reference_shared_query_contracts(
    payload: dict[str, Any],
    *,
    contract_path: tuple[str, ...] = ("planning_contract", "game_query_strategy_contract"),
    conversation_field: str = "recent_conversation",
) -> tuple[dict[str, Any], int]:
    """仅在发送副本中引用完全一致的查询合同，审计和持久化证据保持原样。

    只访问后端定义的工具结果路径，不遍历或解析用户正文；查询范围、失败信息和来源等
    动态字段均保留。历史合同不同于当前定义时原样展开，不能用新规则替换旧证据。
    contract_path 与 conversation_field 由调用方指定固定投影结构，供主规划和审查共用。
    """

    transport = deepcopy(payload)
    canonical: Any = transport
    for key in contract_path:
        canonical = canonical.get(key) if isinstance(canonical, dict) else None
    if not isinstance(canonical, dict) or not canonical:
        return transport, 0
    reference_path = ".".join(contract_path)
    count = 0
    for field in ("previous_tool_results", "turn_tool_evidence"):
        results = transport.get(field)
        if isinstance(results, list):
            count += _reference_query_contract_in_results(results, canonical, reference_path)
    for turn in transport.get(conversation_field) or []:
        if not isinstance(turn, dict) or turn.get("role") != "tool":
            continue
        content = turn.get("content")
        if isinstance(content, str):
            try:
                results = json.loads(content)
            except json.JSONDecodeError:
                continue
        else:
            results = content
        if not isinstance(results, (list, dict)):
            continue
        replaced = _reference_query_contract_in_results(
            results if isinstance(results, list) else [results], canonical, reference_path,
        )
        if replaced:
            turn["content"] = json.dumps(results, ensure_ascii=False) if isinstance(content, str) else results
            count += replaced
    return transport, count


def _reference_query_contract_in_results(
    results: list[Any], canonical: dict[str, Any], reference_path: str,
) -> int:
    """替换受信任工具结果中的静态副本；不改变工具外壳、动态事实或列表顺序。"""

    count = 0
    for item in results:
        result = item.get("result") if isinstance(item, dict) else None
        reply = result.get("customer_reply_contract") if isinstance(result, dict) else None
        if isinstance(reply, dict) and _same_json_value(reply.get("game_query_strategy_contract"), canonical):
            reply["game_query_strategy_contract"] = {
                "contract_reference": reference_path,
            }
            count += 1
    return count


def _same_json_value(left: Any, right: Any) -> bool:
    """逐层保留 JSON 类型再比较，避免 Python 把 True/1、False/0 当作相同证据。"""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json_value(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json_value(a, b) for a, b in zip(left, right))
    return left == right


def compact_tool_result_dict(raw: Any) -> dict[str, Any]:
    """压缩工具结果dict，保留决策所需字段并限制载荷大小。"""
    if not isinstance(raw, dict):
        return {
            "name": "unknown",
            "called": False,
            "allowed": False,
            "result": {},
            "error": "invalid tool result payload",
        }
    payload = model_game_tool_evidence(raw.get("name"), raw.get("result") or {})
    is_inventory = raw.get("name") == "search_current_games"
    compact: dict[str, Any] = {
        "name": raw.get("name"),
        "called": raw.get("called"),
        "allowed": raw.get("allowed"),
        "call_id": raw.get("call_id"),
        "error": raw.get("error"),
        "deduplicated": raw.get("deduplicated", False),
        "result": compact_tool_payload(payload, public_inventory=is_inventory,
                                       public_participation=raw.get("name") in _PARTICIPATION_GAME_RECEIPTS),
    }
    # 话术与审查已在本轮执行器消费过完整输入；成功后不再把每份原文、改写和逐项
    # 解释重复送回主模型。失败详情保留，原始 trace 和实际邀约工具结果也不改写。
    if raw.get("allowed") and not raw.get("error"):
        if raw.get("name") == "customer_visible_content_review" and payload.get("approved") is True:
            compact["result"] = {"approved": True, "review_scope": payload.get("review_scope"),
                                 "item_reviews": [{"item_id": item.get("item_id"), "approved": item.get("approved"), "violations": item.get("violations", [])}
                                                  for item in payload.get("item_reviews") or []]}
        rewrites = payload.get("item_rewrites") or []
        if raw.get("name") == "customer_visible_text_generation" and rewrites and all(item.get("semantic_preserved") is True for item in rewrites):
            compact["result"] = {"generation_scope": payload.get("generation_scope"),
                                 "item_rewrites": [{"item_id": item.get("item_id"), "final_text": item.get("final_text"), "semantic_preserved": True}
                                                   for item in rewrites]}
    if raw.get("state_transitions"):
        # 模型只需知道通知的安排状态，另一会话的任务标识留在原始审计证据。
        private_task_details = (payload.get("candidate_confirmation"), payload.get("candidate_update"))
        private_task_ids = {
            detail["task_id"] for detail in private_task_details
            if isinstance(detail, dict) and isinstance(detail.get("task_id"), str)
        }
        compact["state_transitions"] = [
            {
                "entity_type": item.get("entity_type"),
                "entity_id": None if (item.get("entity_id") in private_task_ids
                    or (raw.get("name") in {"request_late_arrival", "record_late_arrival_decision"}
                        and item.get("entity_type") == "scheduled_agent_task")) else item.get("entity_id"),
                "from_status": item.get("from_status"),
                "to_status": item.get("to_status"),
                "reason": item.get("reason"),
            }
            for item in raw.get("state_transitions") or []
            if isinstance(item, dict)
        ][:8]
    return compact


def model_game_tool_evidence(tool_name: str | None, payload: Any) -> Any:
    """投影查局和报名回执中的公共桌况，原始证据仍供 Store 和权限校验使用。"""
    if not isinstance(payload, dict):
        return payload
    if tool_name in _PARTICIPATION_GAME_RECEIPTS and "game" in payload:
        return {**payload, "game": public_game_lookup_projection(payload["game"])}
    if tool_name != "search_current_games":
        return payload
    # 同一投影也用于历史库存回执，参与方与自由摘要不因存档较早而跨过边界。
    projected = dict(payload)
    for key in ("matches", "nearby_options", "alternatives"):
        if isinstance(items := payload.get(key), list):
            projected[key] = [
                {**item, "game": public_game_lookup_projection(item.get("game"))}
                for item in items if isinstance(item, dict)
            ]
    if isinstance(contract := payload.get("customer_reply_contract"), dict):
        projected["customer_reply_contract"] = {**contract, **{
            key: [] for key in ("matched_result_summaries", "alternative_result_summaries") if key in contract
        }}
    return projected


def compact_tool_payload(payload: Any, *, public_inventory: bool = False, public_participation: bool = False) -> dict[str, Any]:
    """压缩工具载荷，保留决策所需字段并限制载荷大小。"""
    if not isinstance(payload, dict):
        return {}
    compact: dict[str, Any] = {
        key: payload[key]
        for key in PASSTHROUGH_TOOL_RESULT_KEYS
        if key in payload
    }
    # 核档和接受候选是不同事实。对当前客户只回灌结论与已安排状态，不回灌外发任务标识。
    decision = payload.get("candidate_decision")
    if (isinstance(decision, dict) and decision.get("status") in ("accepted", "declined")
            and isinstance(decision.get("game_id"), str)):
        compact["candidate_decision"] = {"status": decision["status"], "game_id": decision["game_id"]}
    for key, statuses in (
        ("candidate_confirmation", ("pending", "processing", "completed", "already_requested", "unavailable")),
        ("candidate_update", ("scheduled",)),
        ("stake_update", ("registered",)),
    ):
        update = payload.get(key)
        if isinstance(update, dict) and update.get("status") in statuses:
            compact[key] = {"status": update["status"]}
    # 晚到安排只携带同桌状态、时间和本人可参与时间。外发名单、关联任务和原话不随工具回执传播。
    late_request = payload.get("late_arrival_request")
    if isinstance(late_request, dict) and late_request.get("status") in {
        "queued", "pending", "accepted", "declined", "unavailable",
    }:
        compact["late_arrival_request"] = {key: late_request[key] for key in (
            "status", "game_id", "arrival_at", "arrival_is_estimate", "planned_start_at", "planned_end_at",
            "remaining_play_minutes",
        ) if key in late_request}
        if type(payload.get("contact_started")) is bool:
            compact["contact_started"] = payload["contact_started"]
    late_decision = payload.get("late_arrival_decision")
    if isinstance(late_decision, dict) and late_decision.get("status") in {"pending", "accepted", "declined", "unavailable"}:
        compact["late_arrival_decision"] = {key: late_decision[key] for key in ("status", "game_id") if key in late_decision}
    # 来源只保留固定身份与哈希；完整调用诊断、正文及推理都不能随错误反馈回灌。
    response_source = compact_model_response_source(payload.get("response_source"))
    if response_source is not None:
        compact["response_source"] = response_source
    if "nearby_options" in payload:
        compact["nearby_options"] = [
            compact_match(item, public_inventory=public_inventory) | {"match_kind": item.get("match_kind"),
                                   "relative_to_window": item.get("relative_to_window"),
                                   "offset_minutes": item.get("offset_minutes")}
            for item in (payload.get("nearby_options") or [])[:2] if isinstance(item, dict)
        ]
    if "matches" in payload:
        compact["matches"] = [compact_match(item, public_inventory=public_inventory) for item in list(payload.get("matches") or [])[:5]]
        compact["match_count"] = len(payload.get("matches") or [])
    # 自招选项已有公开投影。时长待核实选项仍是实际牌局，保留其身份与未决条件才可继续协商。
    decision_options = [item for item in payload.get("alternatives") or []
                        if isinstance(item, dict) and item.get("match_kind") in (
                            "host_confirmation_required", "duration_confirmation_required",
                        )]
    if decision_options:
        compact["alternatives"] = [
            {"match_kind": item["match_kind"], "decision_required_fields": item.get("decision_required_fields", []),
             **({"requires_timing_confirmation": True} if item.get("requires_timing_confirmation") is True else {}),
             "game": item.get("game") if public_inventory or item["match_kind"] == "host_confirmation_required"
             else compact_game(item.get("game"))}
            for item in decision_options[:5]
        ]
    requirement = compact.get("requirement")
    if payload.get("requirement_source") == "active_game_aggregate" and isinstance(requirement, dict):
        compact["requirement"] = game_requirement_for_conversation(requirement)
    continuation = compact.get("continuation")
    facts = continuation.get("authoritative_facts") if isinstance(continuation, dict) else None
    if isinstance(facts, dict) and facts.get("game_id") and isinstance(facts.get("requirement"), dict):
        compact["continuation"] = {**continuation, "authoritative_facts": {
            **facts, "requirement": game_requirement_for_conversation(facts["requirement"]),
        }}
    if "candidates" in payload:
        compact["candidates"] = [
            compact_candidate(item)
            for item in list(payload.get("candidates") or [])[:12]
        ]
        compact["candidate_count"] = len(payload.get("candidates") or [])
    if "game" in payload:
        compact["game"] = payload["game"] if public_participation else compact_game(payload.get("game"))
    if "drafts" in payload:
        compact["drafts"] = [compact_draft(item) for item in list(payload.get("drafts") or [])[:20]]
        compact["draft_count"] = len(payload.get("drafts") or [])
    if "checkpoint" in payload and isinstance(payload.get("checkpoint"), dict):
        checkpoint = payload["checkpoint"]
        compact["checkpoint"] = {
            "summary": checkpoint.get("summary"),
            "facts": checkpoint.get("facts"),
            "open_questions": checkpoint.get("open_questions"),
        }
    if "badcase" in payload:
        compact["badcase"] = payload["badcase"]
    return compact


def compact_match(match: Any, *, public_inventory: bool = False) -> dict[str, Any]:
    """把候选桌况与加入后的座位变化同时交给模型。"""
    if not isinstance(match, dict):
        return {}
    return {
        "score": match.get("score"),
        "reasons": match.get("reasons"),
        "join_projection": match.get("join_projection"),
        **({"requires_timing_confirmation": True} if match.get("requires_timing_confirmation") is True else {}),
        "game": match.get("game") if public_inventory else compact_game(match.get("game")),
    }


__all__ = [
    "PASSTHROUGH_TOOL_RESULT_KEYS",
    "compact_match",
    "compact_tool_payload",
    "compact_tool_result_dict",
    "reference_duplicate_latest_tool_results",
    "reference_shared_query_contracts",
    "tool_result_for_context",
    "turn_payload_for_context",
]
