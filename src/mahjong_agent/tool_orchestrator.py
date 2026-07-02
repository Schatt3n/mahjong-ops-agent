from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .core import AgentCore
from .customer_repository import CustomerProfileRepository
from .models import DEFAULT_TZ, CustomerProfile as LegacyCustomerProfile
from .observability import to_trace_payload
from .profile_observation_contract import normalize_profile_observation_for_storage
from .tool_result_contract import (
    expected_side_effect_for_mode,
    tool_result_audit_policy,
    with_tool_result_contract,
)
from .tool_permissions import tool_allowed_for_action
from .tools import CandidateSearchTool, CurrentGameSearchTool, PendingOutboxTool
from .workflow_models import (
    ActionName,
    ConversationContext,
    EntityType,
    GameRequirement,
    GameWorkflowStatus,
    RiskLevel,
    SemanticResolution,
    SlotSource,
    SlotValue,
    ToolCallRequest,
    ToolExecutionMode,
    ToolName,
    ToolResult,
    ValidatedAction,
)


@dataclass(slots=True)
class ToolOrchestratorConfig:
    allow_read_only: bool = True
    allow_create_pending: bool = True
    allow_state_write: bool = False
    allow_direct_send: bool = False


@dataclass(slots=True)
class ToolOrchestrationResult:
    tool_results: list[ToolResult] = field(default_factory=list)
    skipped_tools: list[str] = field(default_factory=list)

    def result_for(self, tool_name: ToolName) -> ToolResult | None:
        for result in reversed(self.tool_results):
            if result.request.tool_name == tool_name:
                return result
        return None


class ToolExecutionLedger(Protocol):
    def lookup(self, idempotency_key: str) -> ToolResult | None:
        ...

    def record(self, result: ToolResult) -> ToolResult:
        ...

    def history(self, *, tool_name: ToolName | None = None) -> list[ToolResult]:
        ...


class InMemoryToolExecutionLedger:
    """Auditable idempotency ledger for backend-approved tool calls."""

    def __init__(self) -> None:
        self._by_idempotency_key: dict[str, ToolResult] = {}
        self._history: list[ToolResult] = []

    def lookup(self, idempotency_key: str) -> ToolResult | None:
        return self._by_idempotency_key.get(str(idempotency_key))

    def record(self, result: ToolResult) -> ToolResult:
        self._history.append(result)
        key = result.request.idempotency_key
        if key and result.called and result.allowed and key not in self._by_idempotency_key:
            self._by_idempotency_key[key] = result
        return result

    def history(self, *, tool_name: ToolName | None = None) -> list[ToolResult]:
        history = list(self._history)
        if tool_name is not None:
            history = [item for item in history if item.request.tool_name == tool_name]
        return history


class SQLiteToolExecutionLedger:
    """SQLite-backed tool execution ledger for durable idempotency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def lookup(self, idempotency_key: str) -> ToolResult | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM controlled_tool_execution_history
                WHERE idempotency_key = ? AND called = 1 AND allowed = 1
                ORDER BY id ASC
                LIMIT 1
                """,
                (str(idempotency_key),),
            ).fetchone()
        return self._result_from_row(row) if row else None

    def record(self, result: ToolResult) -> ToolResult:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO controlled_tool_execution_history (
                    tool_name, idempotency_key, called, allowed, deduplicated,
                    error, request_json, result_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.request.tool_name.value,
                    result.request.idempotency_key,
                    1 if result.called else 0,
                    1 if result.allowed else 0,
                    1 if result.deduplicated else 0,
                    result.error,
                    _dump_json(to_trace_payload(result.request)),
                    _dump_json(to_trace_payload(result.result)),
                    datetime.now(DEFAULT_TZ).isoformat(),
                ),
            )
        return result

    def history(self, *, tool_name: ToolName | None = None) -> list[ToolResult]:
        sql = "SELECT * FROM controlled_tool_execution_history"
        params: list[str] = []
        if tool_name is not None:
            sql += " WHERE tool_name = ?"
            params.append(_coerce_tool_name(tool_name).value)
        sql += " ORDER BY id ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._result_from_row(row) for row in rows]

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS controlled_tool_execution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    idempotency_key TEXT,
                    called INTEGER NOT NULL,
                    allowed INTEGER NOT NULL,
                    deduplicated INTEGER NOT NULL,
                    error TEXT,
                    request_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_controlled_tool_execution_idempotency
                    ON controlled_tool_execution_history(idempotency_key, called, allowed, id);

                CREATE INDEX IF NOT EXISTS idx_controlled_tool_execution_tool
                    ON controlled_tool_execution_history(tool_name, id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _result_from_row(self, row: sqlite3.Row) -> ToolResult:
        request_payload = _loads_dict(str(row["request_json"] or "{}"))
        result_payload = _loads_dict(str(row["result_json"] or "{}"))
        arguments = request_payload.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        request = ToolCallRequest(
            tool_name=str(row["tool_name"]),
            arguments=arguments,
            risk_level=str(request_payload.get("risk_level") or RiskLevel.LOW.value),
            execution_mode=str(request_payload.get("execution_mode") or ToolExecutionMode.READ_ONLY.value),
            idempotency_key=str(row["idempotency_key"]) if row["idempotency_key"] is not None else None,
            reason=str(request_payload.get("reason") or ""),
        )
        return ToolResult(
            request=request,
            called=bool(row["called"]),
            allowed=bool(row["allowed"]),
            result=result_payload,
            error=str(row["error"]) if row["error"] is not None else None,
            deduplicated=bool(row["deduplicated"]),
        )


class ToolOrchestrator:
    """Runs backend-approved tools and normalizes results.

    The orchestrator enforces permissions and idempotency keys. It does not let
    LLM output call arbitrary tools or directly send messages.
    """

    def __init__(
        self,
        core: AgentCore,
        config: ToolOrchestratorConfig | None = None,
        current_games_tool: CurrentGameSearchTool | None = None,
        candidate_tool: CandidateSearchTool | None = None,
        outbox_tool: PendingOutboxTool | None = None,
        execution_ledger: ToolExecutionLedger | None = None,
        customer_repository: CustomerProfileRepository | None = None,
    ) -> None:
        self.core = core
        self.config = config or ToolOrchestratorConfig()
        self.current_games_tool = current_games_tool or CurrentGameSearchTool()
        self.candidate_tool = candidate_tool or CandidateSearchTool(core)
        self.outbox_tool = outbox_tool or PendingOutboxTool()
        self.execution_ledger = execution_ledger or InMemoryToolExecutionLedger()
        self.customer_repository = customer_repository

    def run(
        self,
        *,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        validated_action: ValidatedAction,
        now: datetime | None = None,
    ) -> ToolOrchestrationResult:
        results: list[ToolResult] = []
        scratch: dict[str, Any] = {}
        for tool_name in validated_action.required_tools:
            request = self._request_for_tool(
                tool_name,
                context=context,
                semantic_resolution=semantic_resolution,
                validated_action=validated_action,
                scratch=scratch,
            )
            permission_error = self._permission_error(request, validated_action)
            if permission_error:
                results.append(
                    self.execution_ledger.record(
                        ToolResult(
                            request=request,
                            called=False,
                            allowed=False,
                            error=permission_error,
                        )
                    )
                )
                continue
            deduplicated = self._deduplicated_result(request)
            if deduplicated is not None:
                deduplicated = self.execution_ledger.record(deduplicated)
                results.append(deduplicated)
                self._update_scratch_from_result(deduplicated, scratch)
                continue
            result = self._execute(
                request,
                context=context,
                semantic_resolution=semantic_resolution,
                scratch=scratch,
                now=now,
            )
            result = self.execution_ledger.record(result)
            self._update_scratch_from_result(result, scratch)
            results.append(result)
        return ToolOrchestrationResult(tool_results=results)

    def _request_for_tool(
        self,
        tool_name: ToolName,
        *,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        validated_action: ValidatedAction,
        scratch: dict[str, Any],
    ) -> ToolCallRequest:
        mode = self._mode_for_tool(tool_name)
        arguments: dict[str, Any] = {
            "effective_action": validated_action.effective_action.value,
            "requirement": semantic_resolution.game_requirement.to_prompt_dict(),
        }
        if tool_name == ToolName.CREATE_PENDING_OUTBOX:
            arguments["candidate_count"] = len(scratch.get("candidates") or [])
            arguments["conversation_id"] = context.current_message.conversation_id
        if tool_name in {ToolName.CREATE_GAME, ToolName.CLOSE_GAME}:
            arguments["game_id"] = self._state_entity_id(
                context=context,
                semantic_resolution=semantic_resolution,
                validated_action=validated_action,
            )
            arguments["conversation_id"] = context.current_message.conversation_id
            arguments["trace_id"] = context.current_message.trace_id
        if tool_name == ToolName.RECORD_SEAT_ACCEPTANCE:
            arguments["game_id"] = self._active_game_id(context, semantic_resolution)
            arguments["candidate_id"] = context.current_message.sender_id
            arguments["candidate_name"] = context.current_message.sender_name
            arguments["conversation_id"] = context.current_message.conversation_id
            arguments["trace_id"] = context.current_message.trace_id
        if tool_name == ToolName.PROFILE_UPDATE:
            arguments["target_customer_id"] = context.current_message.sender_id
            arguments["target_display_name"] = context.current_message.sender_name
            arguments["conversation_id"] = context.current_message.conversation_id
            arguments["trace_id"] = context.current_message.trace_id
            arguments["observations"] = _profile_observations_from_resolution(semantic_resolution)
        return ToolCallRequest(
            tool_name=tool_name,
            arguments=arguments,
            risk_level=self._risk_for_tool(tool_name, validated_action),
            execution_mode=mode,
            idempotency_key=f"{validated_action.idempotency_key}:{tool_name.value}"
            if validated_action.idempotency_key
            else None,
            reason=validated_action.reason,
        )

    def _execute(
        self,
        request: ToolCallRequest,
        *,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        scratch: dict[str, Any],
        now: datetime | None,
    ) -> ToolResult:
        if request.tool_name == ToolName.SEARCH_CURRENT_OPEN_GAMES:
            payload = self.current_games_tool.search(context, semantic_resolution.game_requirement)
            scratch["current_game_matches"] = payload.get("matches") or []
            payload = self._with_contract(
                request,
                payload,
                result_type="current_game_matches",
            )
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        if request.tool_name == ToolName.SEARCH_CANDIDATE_CUSTOMERS:
            payload = self.candidate_tool.search(semantic_resolution.game_requirement, now=now)
            scratch["candidates"] = payload.get("candidates") or []
            payload = self._with_contract(
                request,
                payload,
                result_type="candidate_recommendations",
            )
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        if request.tool_name == ToolName.CREATE_PENDING_OUTBOX:
            candidates = list(scratch.get("candidates") or [])
            if not candidates:
                return ToolResult(
                    request=request,
                    called=False,
                    allowed=False,
                    error="CREATE_PENDING_OUTBOX requires candidate search results.",
                )
            missing_terms = _missing_public_invite_terms(semantic_resolution.game_requirement)
            if missing_terms:
                return ToolResult(
                    request=request,
                    called=False,
                    allowed=False,
                    error=(
                        "CREATE_PENDING_OUTBOX requires confirmed public invite terms: "
                        + ", ".join(missing_terms)
                    ),
                )
            payload = self.outbox_tool.create_pending_invites(
                semantic_resolution.game_requirement,
                candidates,
                conversation_id=context.current_message.conversation_id,
                trace_id=context.current_message.trace_id,
                base_idempotency_key=request.idempotency_key,
            )
            scratch["outbox_drafts"] = payload.get("drafts") or []
            payload = self._with_contract(
                request,
                payload,
                result_type="pending_outbox_drafts",
            )
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        if request.tool_name == ToolName.CREATE_GAME:
            payload = self._create_game_state_write_intent(request, semantic_resolution, scratch)
            scratch.setdefault("state_write_intents", []).append(payload["state_write_intent"])
            payload = self._with_contract(
                request,
                payload,
                result_type="state_write_intent",
                side_effect="state_write_intent_only",
            )
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        if request.tool_name == ToolName.CLOSE_GAME:
            payload = self._close_game_state_write_intent(request, semantic_resolution)
            scratch.setdefault("state_write_intents", []).append(payload["state_write_intent"])
            payload = self._with_contract(
                request,
                payload,
                result_type="state_write_intent",
                side_effect="state_write_intent_only",
            )
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        if request.tool_name == ToolName.RECORD_SEAT_ACCEPTANCE:
            payload, error = self._record_seat_acceptance_state_write_intent(request, context)
            if error:
                return ToolResult(request=request, called=False, allowed=False, error=error)
            scratch.setdefault("state_write_intents", []).append(payload["state_write_intent"])
            payload = self._with_contract(
                request,
                payload,
                result_type="state_write_intent",
                side_effect="state_write_intent_only",
            )
            return ToolResult(request=request, called=True, allowed=True, result=payload)
        if request.tool_name == ToolName.PROFILE_UPDATE:
            payload = self._apply_profile_update_intent(request)
            payload = self._with_contract(
                request,
                payload,
                result_type="profile_observation_update",
                side_effect="low_risk_profile_observation_write",
            )
            return ToolResult(
                request=request,
                called=bool(payload["applied_count"] or payload["rejected_count"]),
                allowed=True,
                result=payload,
            )
        return ToolResult(
            request=request,
            called=False,
            allowed=False,
            error=f"Tool {request.tool_name.value} is not implemented in controlled orchestrator.",
        )

    def _deduplicated_result(self, request: ToolCallRequest) -> ToolResult | None:
        if not self._should_deduplicate(request):
            return None
        existing = self.execution_ledger.lookup(str(request.idempotency_key))
        if existing is None:
            return None
        return replace(
            existing,
            request=request,
            deduplicated=True,
        )

    def _should_deduplicate(self, request: ToolCallRequest) -> bool:
        if not request.idempotency_key:
            return False
        return request.execution_mode in {
            ToolExecutionMode.CREATE_PENDING,
            ToolExecutionMode.STATE_WRITE,
            ToolExecutionMode.DIRECT_SEND,
        }

    def _update_scratch_from_result(self, result: ToolResult, scratch: dict[str, Any]) -> None:
        if not result.allowed or not result.result:
            return
        if result.request.tool_name == ToolName.SEARCH_CURRENT_OPEN_GAMES:
            scratch["current_game_matches"] = result.result.get("matches") or []
        elif result.request.tool_name == ToolName.SEARCH_CANDIDATE_CUSTOMERS:
            scratch["candidates"] = result.result.get("candidates") or []
        elif result.request.tool_name == ToolName.CREATE_PENDING_OUTBOX:
            scratch["outbox_drafts"] = result.result.get("drafts") or []
        elif result.request.tool_name in {
            ToolName.CREATE_GAME,
            ToolName.CLOSE_GAME,
            ToolName.RECORD_SEAT_ACCEPTANCE,
        }:
            intent = result.result.get("state_write_intent")
            if intent:
                scratch.setdefault("state_write_intents", []).append(intent)

    def _with_contract(
        self,
        request: ToolCallRequest,
        payload: dict[str, Any],
        *,
        result_type: str,
        side_effect: str | None = None,
    ) -> dict[str, Any]:
        return with_tool_result_contract(
            payload,
            request=request,
            result_type=result_type,
            side_effect=side_effect or expected_side_effect_for_mode(request.execution_mode),
            audit_policy=tool_result_audit_policy(request.tool_name),
        )

    def _permission_error(self, request: ToolCallRequest, validated_action: ValidatedAction) -> str | None:
        if request.risk_level == RiskLevel.HIGH:
            return "High risk tool call requires human review."
        action_error = self._tool_action_error(request, validated_action)
        if action_error:
            return action_error
        if request.execution_mode == ToolExecutionMode.READ_ONLY and not self.config.allow_read_only:
            return "Read-only tools are disabled."
        if request.execution_mode == ToolExecutionMode.CREATE_PENDING and not self.config.allow_create_pending:
            return "Create-pending tools are disabled."
        if request.execution_mode == ToolExecutionMode.STATE_WRITE and not self.config.allow_state_write:
            return "State-write tools are disabled in this orchestrator."
        if request.execution_mode == ToolExecutionMode.DIRECT_SEND:
            return "Direct-send tools are not allowed without explicit human approval." if not self.config.allow_direct_send else None
        if validated_action.risk_level == RiskLevel.HIGH:
            return "Validated action is high risk and cannot call tools automatically."
        return None

    def _tool_action_error(self, request: ToolCallRequest, validated_action: ValidatedAction) -> str | None:
        if not tool_allowed_for_action(request.tool_name, validated_action.effective_action):
            return (
                f"Tool {request.tool_name.value} is not allowed for effective action "
                f"{validated_action.effective_action.value}."
            )
        return None

    def _mode_for_tool(self, tool_name: ToolName) -> ToolExecutionMode:
        if tool_name in {ToolName.SEARCH_CURRENT_OPEN_GAMES, ToolName.SEARCH_CANDIDATE_CUSTOMERS}:
            return ToolExecutionMode.READ_ONLY
        if tool_name == ToolName.CREATE_PENDING_OUTBOX:
            return ToolExecutionMode.CREATE_PENDING
        if tool_name in {
            ToolName.CREATE_GAME,
            ToolName.CLOSE_GAME,
            ToolName.RECORD_SEAT_ACCEPTANCE,
            ToolName.PROFILE_UPDATE,
            ToolName.RECORD_APPROVAL_DECISION,
        }:
            return ToolExecutionMode.STATE_WRITE
        if tool_name == ToolName.SEND_MESSAGE:
            return ToolExecutionMode.DIRECT_SEND
        return ToolExecutionMode.NOT_CALLED

    def _risk_for_tool(self, tool_name: ToolName, validated_action: ValidatedAction) -> RiskLevel:
        if tool_name == ToolName.SEND_MESSAGE:
            return RiskLevel.HIGH
        if tool_name in {
            ToolName.CREATE_PENDING_OUTBOX,
            ToolName.CREATE_GAME,
            ToolName.CLOSE_GAME,
            ToolName.RECORD_SEAT_ACCEPTANCE,
            ToolName.PROFILE_UPDATE,
            ToolName.RECORD_APPROVAL_DECISION,
        }:
            return RiskLevel.MEDIUM
        return validated_action.risk_level if validated_action.risk_level == RiskLevel.HIGH else RiskLevel.LOW

    def _state_entity_id(
        self,
        *,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
        validated_action: ValidatedAction,
    ) -> str:
        action_game_id = semantic_resolution.proposed_action.arguments.get("game_id")
        if action_game_id:
            return str(action_game_id)
        return validated_action.idempotency_key or f"pending_game:{context.current_message.trace_id}"

    def _create_game_state_write_intent(
        self,
        request: ToolCallRequest,
        semantic_resolution: SemanticResolution,
        scratch: dict[str, Any],
    ) -> dict[str, Any]:
        outbox_created = bool(scratch.get("outbox_drafts"))
        game_id = str(request.arguments.get("game_id") or request.idempotency_key or "pending_game")
        intent = {
            "kind": "create_game",
            "entity_type": EntityType.GAME.value,
            "entity_id": game_id,
            "target_status": GameWorkflowStatus.NEGOTIATING.value
            if outbox_created
            else GameWorkflowStatus.OPEN.value,
            "enter_negotiating_if_outbox_created": outbox_created,
            "reason": request.reason,
            "requirement": semantic_resolution.game_requirement.to_prompt_dict(),
        }
        return {
            "state_write_intent": intent,
            "game_id": game_id,
            "policy": "只生成状态写入意图，由 StateMachine 校验并由 StateStore 落库。",
        }

    def _close_game_state_write_intent(
        self,
        request: ToolCallRequest,
        semantic_resolution: SemanticResolution,
    ) -> dict[str, Any]:
        game_id = str(request.arguments.get("game_id") or request.idempotency_key or "pending_game")
        intent = {
            "kind": "close_game",
            "entity_type": EntityType.GAME.value,
            "entity_id": game_id,
            "target_status": GameWorkflowStatus.CANCELLED.value,
            "reason": request.reason,
            "requirement": semantic_resolution.game_requirement.to_prompt_dict(),
        }
        return {
            "state_write_intent": intent,
            "game_id": game_id,
            "policy": "只生成关闭局状态写入意图，由 StateMachine 校验并由 StateStore 落库。",
        }

    def _record_seat_acceptance_state_write_intent(
        self,
        request: ToolCallRequest,
        context: ConversationContext,
    ) -> tuple[dict[str, Any], str | None]:
        game_id = str(request.arguments.get("game_id") or "").strip()
        requirement = self._active_game_requirement(context, game_id=game_id or None)
        if requirement is None:
            return {}, "RECORD_SEAT_ACCEPTANCE requires an active game in conversation context."
        if not game_id:
            game_id = self._game_id_from_requirement(requirement) or ""
        if not game_id:
            return {}, "RECORD_SEAT_ACCEPTANCE requires a game_id."

        seats_total = int(requirement.seats_total or 4)
        current_count = _coerce_int_slot(requirement.slot("current_player_count"))
        missing_count = _coerce_int_slot(requirement.slot("missing_count"))
        if current_count is None and missing_count is not None:
            current_count = max(seats_total - missing_count, 0)
        if missing_count is None and current_count is not None:
            missing_count = max(seats_total - current_count, 0)
        if current_count is None or missing_count is None:
            return {}, "RECORD_SEAT_ACCEPTANCE requires current_player_count or missing_count."
        if missing_count <= 0:
            return {}, "RECORD_SEAT_ACCEPTANCE rejected because the game has no missing seats."

        new_current_count = min(current_count + 1, seats_total)
        new_missing_count = max(missing_count - 1, 0)
        updated_requirement = _copy_requirement(requirement)
        updated_requirement.slots["current_player_count"] = _tool_slot(
            "current_player_count",
            new_current_count,
            evidence=f"{context.current_message.sender_name} 确认加入",
        )
        updated_requirement.slots["missing_count"] = _tool_slot(
            "missing_count",
            new_missing_count,
            evidence=f"{context.current_message.sender_name} 确认加入后更新缺口",
        )
        updated_requirement.slots["party_size"] = _tool_slot(
            "party_size",
            {
                "current_player_count": new_current_count,
                "missing_count": new_missing_count,
                "seats_total": seats_total,
            },
            evidence="候选人确认入局后由后端工具更新",
        )
        updated_requirement.notes.extend(
            [
                f"accepted_candidate_id={context.current_message.sender_id}",
                f"accepted_candidate_name={context.current_message.sender_name}",
            ]
        )

        participant = {
            "customer_id": context.current_message.sender_id,
            "display_name": context.current_message.sender_name,
            "status": "confirmed",
            "source_message_id": context.current_message.message_id,
            "trace_id": context.current_message.trace_id,
        }
        target_status = (
            GameWorkflowStatus.CONFIRMED.value
            if new_missing_count == 0
            else GameWorkflowStatus.NEGOTIATING.value
        )
        intent = {
            "kind": "record_seat_acceptance",
            "entity_type": EntityType.GAME.value,
            "entity_id": game_id,
            "target_status": target_status,
            "reason": request.reason,
            "requirement": updated_requirement.to_prompt_dict(),
            "participant": participant,
            "seat_delta": {
                "previous_current_player_count": current_count,
                "previous_missing_count": missing_count,
                "current_player_count": new_current_count,
                "missing_count": new_missing_count,
                "seats_total": seats_total,
            },
        }
        return {
            "state_write_intent": intent,
            "game_id": game_id,
            "candidate": participant,
            "policy": "只生成候选人确认入局状态写入意图，由 StateMachine 校验并由 StateStore 落库。",
        }, None

    def _apply_profile_update_intent(self, request: ToolCallRequest) -> dict[str, Any]:
        customer_id = str(request.arguments.get("target_customer_id") or "").strip()
        display_name = str(request.arguments.get("target_display_name") or customer_id or "未知客户")
        raw_observations = request.arguments.get("observations")
        observations = raw_observations if isinstance(raw_observations, list) else []
        profile = self.core.store.customers.get(customer_id)
        if profile is None:
            profile = LegacyCustomerProfile(id=customer_id, display_name=display_name)
            self.core.upsert_customer(profile)

        applied: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for item in observations:
            normalized, error = _normalize_profile_observation(item)
            if error:
                rejected.append({"observation": to_trace_payload(item), "reason": error})
                continue
            if _profile_observation_exists(profile, normalized):
                rejected.append({"observation": normalized, "reason": "duplicate_observation"})
                continue
            _append_profile_observation(profile, normalized)
            applied.append(normalized)
        if applied and self.customer_repository is not None:
            self.customer_repository.save(profile)

        return {
            "target_customer_id": customer_id,
            "applied": applied,
            "rejected": rejected,
            "applied_count": len(applied),
            "rejected_count": len(rejected),
            "policy": "只写入低风险画像观察事实，不直接覆盖强画像字段。",
        }

    def _active_game_id(
        self,
        context: ConversationContext,
        semantic_resolution: SemanticResolution,
    ) -> str | None:
        action_game_id = semantic_resolution.proposed_action.arguments.get("game_id")
        if action_game_id:
            return str(action_game_id)
        if context.active_game is not None:
            active_id = self._game_id_from_requirement(context.active_game)
            if active_id:
                return active_id
        for requirement in context.open_games:
            game_id = self._game_id_from_requirement(requirement)
            if game_id and self._has_available_seat(requirement):
                return game_id
        return None

    def _active_game_requirement(
        self,
        context: ConversationContext,
        *,
        game_id: str | None,
    ) -> GameRequirement | None:
        candidates = [item for item in [context.active_game, *context.open_games] if item is not None]
        if game_id:
            for requirement in candidates:
                if self._game_id_from_requirement(requirement) == game_id:
                    return requirement
            return None
        for requirement in candidates:
            if self._has_available_seat(requirement):
                return requirement
        return None

    def _game_id_from_requirement(self, requirement: GameRequirement) -> str | None:
        for note in requirement.notes:
            text = str(note)
            if text.startswith("controlled_state_entity_id="):
                return text.split("=", 1)[1]
            if text.startswith("game_id="):
                return text.split("=", 1)[1]
        return None

    def _has_available_seat(self, requirement: GameRequirement) -> bool:
        missing_count = _coerce_int_slot(requirement.slot("missing_count"))
        if missing_count is None:
            return True
        return missing_count > 0


def _copy_requirement(requirement: GameRequirement) -> GameRequirement:
    copied = GameRequirement(
        seats_total=int(requirement.seats_total or 4),
        organizer_id=requirement.organizer_id,
        organizer_name=requirement.organizer_name,
        candidate_composition_preference=dict(requirement.candidate_composition_preference),
        notes=list(requirement.notes),
    )
    for slot in requirement.slots.values():
        copied.set_slot(
            SlotValue(
                name=slot.name,
                value=slot.value,
                source=slot.source,
                confidence=slot.confidence,
                confirmed=slot.confirmed,
                needs_confirmation=slot.needs_confirmation,
                evidence=slot.evidence,
                updated_at=slot.updated_at,
                metadata=dict(slot.metadata),
            ),
            prefer_confirmed=False,
        )
    return copied


def _tool_slot(name: str, value: Any, *, evidence: str) -> SlotValue:
    return SlotValue(
        name=name,
        value=value,
        source=SlotSource.TOOL,
        confidence=1.0,
        confirmed=True,
        needs_confirmation=False,
        evidence=evidence,
    )


def _coerce_int_slot(slot: SlotValue | None) -> int | None:
    if slot is None or slot.value in (None, "", "unknown"):
        return None
    try:
        return int(slot.value)
    except (TypeError, ValueError):
        return None


def _coerce_tool_name(tool_name: ToolName | str) -> ToolName:
    if isinstance(tool_name, ToolName):
        return tool_name
    try:
        return ToolName(str(tool_name))
    except ValueError:
        return ToolName.UNKNOWN


def _missing_public_invite_terms(requirement: GameRequirement) -> list[str]:
    return requirement.missing_required_slots(
        (
            "stake",
            "start_time_mode",
            "party_size",
            "smoke",
            "duration_mode",
        )
    )


def _dump_json(value: Any) -> str:
    return json.dumps(to_trace_payload(value), ensure_ascii=False, sort_keys=True)


def _loads_dict(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"raw_json": raw}
    return payload if isinstance(payload, dict) else {"value": payload}


def _profile_observations_from_resolution(semantic_resolution: SemanticResolution) -> list[dict[str, Any]]:
    model_output = semantic_resolution.raw_response.get("model_output")
    if not isinstance(model_output, dict):
        return []
    observations = model_output.get("profile_observations")
    if not isinstance(observations, list):
        return []
    return [item for item in observations if isinstance(item, dict)]


def _normalize_profile_observation(raw: Any) -> tuple[dict[str, Any], str | None]:
    return normalize_profile_observation_for_storage(raw)


def _profile_observation_exists(profile: LegacyCustomerProfile, observation: dict[str, Any]) -> bool:
    key = _profile_observation_key(observation)
    return any(
        _profile_observation_key(item) == key
        for item in _stored_profile_observations(profile)
        if isinstance(item, dict)
    )


def _append_profile_observation(profile: LegacyCustomerProfile, observation: dict[str, Any]) -> None:
    stored = _stored_profile_observations(profile)
    stored.append(observation)
    profile.metadata["controlled_profile_observations"] = stored[-50:]


def _stored_profile_observations(profile: LegacyCustomerProfile) -> list[dict[str, Any]]:
    raw = profile.metadata.get("controlled_profile_observations")
    return list(raw) if isinstance(raw, list) else []


def _profile_observation_key(observation: dict[str, Any]) -> str:
    payload = {
        "field": observation.get("field"),
        "value": observation.get("value"),
        "evidence": observation.get("evidence"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
