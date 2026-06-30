from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .models import CandidateRecommendation, GameRequest


@dataclass(slots=True)
class TrialToolOrchestrationInput:
    trace_id: str
    sender_id: str
    sender_name: str
    source_text: str
    effective_text: str
    workflow_followup_context: dict[str, Any]
    decision: Any
    game: GameRequest | None
    missing_fields: list[str]
    decision_action: str
    pool_inquiry: bool
    now: datetime


@dataclass(slots=True)
class TrialToolOrchestrationResult:
    action_plans: list[dict[str, Any]] = field(default_factory=list)
    pool_tool_result: dict[str, Any] = field(default_factory=dict)
    candidate_tool_result: dict[str, Any] = field(default_factory=dict)
    send_tool_result: dict[str, Any] = field(default_factory=dict)
    tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    pool_matches: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[CandidateRecommendation] = field(default_factory=list)
    outbox: list[dict[str, Any]] = field(default_factory=list)
    use_existing_pool: bool = False
    explicit_grouping_request: bool = False
    critical_missing_fields: set[str] = field(default_factory=set)
    user_action_record: dict[str, Any] = field(default_factory=dict)
    user_action_validation: dict[str, Any] = field(default_factory=dict)
    effective_user_action: str = ""
    proposed_user_action: str = ""
    create_game_followup_attempt: bool = False
    should_materialize_game: bool = False
    inquiry_without_materialized_game: bool = False
    response_missing_fields: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TrialToolOrchestrationCallbacks:
    llm_tool_plan: Callable[..., dict[str, Any]]
    action_plan_view: Callable[[dict[str, Any]], dict[str, Any]]
    single_action_plan_view: Callable[..., dict[str, Any]]
    tool_requested: Callable[[dict[str, Any], str], bool]
    replace_action_plan_view: Callable[[list[dict[str, Any]], dict[str, Any]], None]
    search_current_open_games_tool: Callable[..., dict[str, Any]]
    has_start_time_ambiguity: Callable[[GameRequest | None], bool]
    is_explicit_grouping_request: Callable[..., bool]
    user_semantic_action_record: Callable[..., dict[str, Any]]
    is_grouping_confirmation_followup: Callable[[dict[str, Any], str], bool]
    stable_request_game_id: Callable[[str], str]
    should_search_existing_pool: Callable[[str, str, GameRequest | None], bool]
    skipped_tool_result: Callable[..., dict[str, Any]]
    rejected_tool_result: Callable[..., dict[str, Any]]
    search_candidate_customers_tool: Callable[..., dict[str, Any]]
    candidate_recommendations_from_tool: Callable[[dict[str, Any]], list[CandidateRecommendation]]
    send_message_tool: Callable[..., dict[str, Any]]


@dataclass(slots=True)
class TrialToolOrchestrationService:
    """Runs the legacy trial-page tool planning sequence.

    This is a migration adapter around the existing trial-page behavior. It
    owns tool ordering and result aggregation, while actual LLM planning,
    backend validation, tool execution, and state writes remain injected
    callbacks until the legacy path is fully replaced by the controlled runtime.
    """

    callbacks: TrialToolOrchestrationCallbacks
    critical_fields: set[str]

    def run(self, data: TrialToolOrchestrationInput) -> TrialToolOrchestrationResult:
        action_plans: list[dict[str, Any]] = []

        initial_tool_plan = self.callbacks.llm_tool_plan(
            trace_id=data.trace_id,
            stage="before_open_game_search",
            sender_id=data.sender_id,
            sender_name=data.sender_name,
            source_text=data.source_text,
            effective_text=data.effective_text,
            workflow_followup_context=data.workflow_followup_context,
            game=data.game,
            missing_fields=data.missing_fields,
            decision_action=data.decision_action,
            tool_results={},
            now=data.now,
        )
        action_plans.append(self.callbacks.action_plan_view(initial_tool_plan))
        pool_requested_by_llm = self.callbacks.tool_requested(initial_tool_plan, "search_current_open_games")
        pool_tool_result = self.callbacks.search_current_open_games_tool(
            trace_id=data.trace_id,
            query_game=data.game,
            source_text=data.source_text,
            effective_text=data.effective_text,
            sender_id=data.sender_id,
            decision_action=data.decision_action,
            llm_requested=pool_requested_by_llm,
            tool_plan=initial_tool_plan,
            now=data.now,
        )
        self.callbacks.replace_action_plan_view(action_plans, initial_tool_plan)

        pool_matches = list(pool_tool_result.get("matches") or [])
        use_existing_pool = bool(pool_matches) and not self.callbacks.has_start_time_ambiguity(data.game)
        explicit_grouping_request = self.callbacks.is_explicit_grouping_request(
            source_text=data.source_text,
            effective_text=data.effective_text,
            game=data.game,
        )
        critical_missing_fields = set(data.missing_fields) & self.critical_fields
        user_action_record = self.callbacks.user_semantic_action_record(
            trace_id=data.trace_id,
            decision=data.decision,
            game=data.game,
            missing_fields=data.missing_fields,
            explicit_grouping_request=explicit_grouping_request,
            use_existing_pool=use_existing_pool,
            pool_tool_result=pool_tool_result,
            now=data.now,
        )
        action_plans.append(
            self.callbacks.single_action_plan_view(
                stage="user_semantic_action",
                source=str(user_action_record.get("source") or "unknown"),
                action=user_action_record,
            )
        )

        user_action_validation = (
            user_action_record.get("validation") if isinstance(user_action_record.get("validation"), dict) else {}
        )
        effective_user_action = str(user_action_validation.get("effective_action") or "")
        proposed_user_action = str((user_action_record.get("arguments") or {}).get("proposed_action") or "")
        create_game_followup_attempt = bool(
            proposed_user_action == "create_game"
            and (
                explicit_grouping_request
                or self.callbacks.is_grouping_confirmation_followup(
                    data.workflow_followup_context,
                    data.source_text,
                )
            )
        )
        should_materialize_game = bool(
            data.game
            and not use_existing_pool
            and effective_user_action == "create_game"
            and not critical_missing_fields
        )
        if should_materialize_game and data.game:
            data.game.id = self.callbacks.stable_request_game_id(data.trace_id)

        inquiry_without_materialized_game = bool(
            not use_existing_pool
            and not should_materialize_game
            and not explicit_grouping_request
            and not create_game_followup_attempt
            and (
                data.pool_inquiry
                or pool_tool_result.get("called") is True
                or (
                    data.game
                    and self.callbacks.should_search_existing_pool(data.source_text, data.effective_text, data.game)
                )
            )
        )
        response_missing_fields = [] if (use_existing_pool or inquiry_without_materialized_game) else data.missing_fields

        candidate_tool_result: dict[str, Any] = self.callbacks.skipped_tool_result(
            "search_candidate_customers",
            "已有可拼局或关键信息不足，暂不搜索候选人。",
        )
        send_tool_result: dict[str, Any] = self.callbacks.skipped_tool_result(
            "send_message",
            "没有待审批消息发送请求。",
            risk_level="high",
            approval_required=True,
        )
        recommendations: list[CandidateRecommendation] = []
        outbox: list[dict[str, Any]] = []

        if should_materialize_game and data.game:
            candidate_tool_result, send_tool_result, recommendations, outbox = self._run_materialized_game_tools(
                data=data,
                action_plans=action_plans,
                pool_tool_result=pool_tool_result,
                game=data.game,
            )

        tool_results = {
            "search_current_open_games": pool_tool_result,
            "search_candidate_customers": candidate_tool_result,
            "send_message": send_tool_result,
        }
        return TrialToolOrchestrationResult(
            action_plans=action_plans,
            pool_tool_result=pool_tool_result,
            candidate_tool_result=candidate_tool_result,
            send_tool_result=send_tool_result,
            tool_results=tool_results,
            pool_matches=pool_matches,
            recommendations=recommendations,
            outbox=outbox,
            use_existing_pool=use_existing_pool,
            explicit_grouping_request=explicit_grouping_request,
            critical_missing_fields=critical_missing_fields,
            user_action_record=user_action_record,
            user_action_validation=user_action_validation,
            effective_user_action=effective_user_action,
            proposed_user_action=proposed_user_action,
            create_game_followup_attempt=create_game_followup_attempt,
            should_materialize_game=should_materialize_game,
            inquiry_without_materialized_game=inquiry_without_materialized_game,
            response_missing_fields=response_missing_fields,
        )

    def _run_materialized_game_tools(
        self,
        *,
        data: TrialToolOrchestrationInput,
        action_plans: list[dict[str, Any]],
        pool_tool_result: dict[str, Any],
        game: GameRequest,
    ) -> tuple[dict[str, Any], dict[str, Any], list[CandidateRecommendation], list[dict[str, Any]]]:
        second_tool_plan = self.callbacks.llm_tool_plan(
            trace_id=data.trace_id,
            stage="after_open_game_search",
            sender_id=data.sender_id,
            sender_name=data.sender_name,
            source_text=data.source_text,
            effective_text=data.effective_text,
            workflow_followup_context=data.workflow_followup_context,
            game=game,
            missing_fields=data.missing_fields,
            decision_action=data.decision_action,
            tool_results={"search_current_open_games": pool_tool_result},
            now=data.now,
        )
        action_plans.append(self.callbacks.action_plan_view(second_tool_plan))
        candidate_requested_by_llm = self.callbacks.tool_requested(second_tool_plan, "search_candidate_customers")
        send_requested_by_llm = self.callbacks.tool_requested(second_tool_plan, "send_message")
        critical_missing_fields = set(data.missing_fields) & self.critical_fields
        candidate_tool_result: dict[str, Any] = self.callbacks.skipped_tool_result(
            "search_candidate_customers",
            "已有可拼局或关键信息不足，暂不搜索候选人。",
        )
        send_tool_result: dict[str, Any] = self.callbacks.skipped_tool_result(
            "send_message",
            "没有待审批消息发送请求。",
            risk_level="high",
            approval_required=True,
        )
        recommendations: list[CandidateRecommendation] = []
        outbox: list[dict[str, Any]] = []

        if candidate_requested_by_llm and critical_missing_fields:
            candidate_tool_result = self.callbacks.rejected_tool_result(
                data.trace_id,
                "search_candidate_customers",
                "组局关键信息不足，后端拒绝候选人搜索，避免误邀约。",
            )
        elif candidate_requested_by_llm:
            candidate_tool_result = self.callbacks.search_candidate_customers_tool(
                trace_id=data.trace_id,
                game=game,
                now=data.now,
                tool_plan=second_tool_plan,
            )
            self.callbacks.replace_action_plan_view(action_plans, second_tool_plan)
            recommendations = self.callbacks.candidate_recommendations_from_tool(candidate_tool_result)
            if recommendations and not send_requested_by_llm:
                third_tool_plan = self.callbacks.llm_tool_plan(
                    trace_id=data.trace_id,
                    stage="after_candidate_search",
                    sender_id=data.sender_id,
                    sender_name=data.sender_name,
                    source_text=data.source_text,
                    effective_text=data.effective_text,
                    workflow_followup_context=data.workflow_followup_context,
                    game=game,
                    missing_fields=data.missing_fields,
                    decision_action=data.decision_action,
                    tool_results={
                        "search_current_open_games": pool_tool_result,
                        "search_candidate_customers": candidate_tool_result,
                    },
                    now=data.now,
                )
                action_plans.append(self.callbacks.action_plan_view(third_tool_plan))
                send_requested_by_llm = self.callbacks.tool_requested(third_tool_plan, "send_message")
                if send_requested_by_llm:
                    second_tool_plan = third_tool_plan

        if send_requested_by_llm and critical_missing_fields:
            send_tool_result = self.callbacks.rejected_tool_result(
                data.trace_id,
                "send_message",
                "组局关键信息不足，后端拒绝创建外发草稿。",
                risk_level="high",
                approval_required=True,
            )
        elif send_requested_by_llm and recommendations:
            send_tool_result = self.callbacks.send_message_tool(
                trace_id=data.trace_id,
                game=game,
                recommendations=recommendations[:8],
                now=data.now,
                tool_plan=second_tool_plan,
            )
            self.callbacks.replace_action_plan_view(action_plans, second_tool_plan)
            outbox = list(send_tool_result.get("outbox") or [])

        return candidate_tool_result, send_tool_result, recommendations, outbox
