from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from mahjong_agent.models import CandidateRecommendation, GameRequest
from mahjong_agent.trial_tool_orchestration import (
    TrialToolOrchestrationCallbacks,
    TrialToolOrchestrationInput,
    TrialToolOrchestrationService,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 1, 16, 0, tzinfo=TZ)
CRITICAL_FIELDS = {"start_time", "known_players", "stake", "smoke", "duration"}


def make_game() -> GameRequest:
    return GameRequest(
        id="game_temp",
        organizer_id="zhang",
        organizer_name="张哥",
        channel_id="boss_trial",
        game_type="杭麻",
        level="0.5",
        missing_count=3,
        current_player_count=1,
        rules=["无烟", "人齐开"],
        duration_hours=4,
    )


def make_input(game: GameRequest | None = None) -> TrialToolOrchestrationInput:
    return TrialToolOrchestrationInput(
        trace_id="trace_tool",
        sender_id="zhang",
        sender_name="张哥",
        source_text="人齐开0.5无烟，组",
        effective_text="人齐开0.5无烟，组",
        workflow_followup_context={},
        decision=SimpleNamespace(action=SimpleNamespace(value="find_players")),
        game=game or make_game(),
        missing_fields=[],
        decision_action="find_players",
        pool_inquiry=False,
        now=NOW,
    )


class FakeToolCallbacks:
    def __init__(self, *, pool_matches=None, after_open_calls=None, after_candidate_calls=None) -> None:
        self.pool_matches = list(pool_matches or [])
        self.after_open_calls = list(after_open_calls or [])
        self.after_candidate_calls = list(after_candidate_calls or [])
        self.calls: list[str] = []

    def callbacks(self) -> TrialToolOrchestrationCallbacks:
        return TrialToolOrchestrationCallbacks(
            llm_tool_plan=self.llm_tool_plan,
            action_plan_view=self.action_plan_view,
            single_action_plan_view=self.single_action_plan_view,
            tool_requested=self.tool_requested,
            replace_action_plan_view=self.replace_action_plan_view,
            search_current_open_games_tool=self.search_current_open_games_tool,
            has_start_time_ambiguity=lambda game: False,
            is_explicit_grouping_request=lambda **kwargs: True,
            user_semantic_action_record=self.user_semantic_action_record,
            is_grouping_confirmation_followup=lambda context, text: False,
            stable_request_game_id=lambda trace_id: "game_stable",
            should_search_existing_pool=lambda source_text, effective_text, game: False,
            skipped_tool_result=self.skipped_tool_result,
            rejected_tool_result=self.rejected_tool_result,
            search_candidate_customers_tool=self.search_candidate_customers_tool,
            candidate_recommendations_from_tool=self.candidate_recommendations_from_tool,
            send_message_tool=self.send_message_tool,
        )

    def llm_tool_plan(self, *, stage: str, **kwargs):
        self.calls.append(f"plan:{stage}")
        calls_by_stage = {
            "before_open_game_search": [{"tool_name": "search_current_open_games"}],
            "after_open_game_search": self.after_open_calls,
            "after_candidate_search": self.after_candidate_calls,
        }
        return {"stage": stage, "source": "llm", "tool_calls": calls_by_stage.get(stage, [])}

    def action_plan_view(self, plan):
        return {"stage": plan.get("stage"), "source": plan.get("source"), "tool_count": len(plan.get("tool_calls") or [])}

    def single_action_plan_view(self, *, stage: str, source: str, action):
        return {"stage": stage, "source": source, "validated_actions": [action]}

    def tool_requested(self, plan, tool_name: str) -> bool:
        return any(item.get("tool_name") == tool_name for item in plan.get("tool_calls") or [])

    def replace_action_plan_view(self, action_plans, tool_plan) -> None:
        stage = tool_plan.get("stage")
        for index in range(len(action_plans) - 1, -1, -1):
            if action_plans[index].get("stage") == stage:
                action_plans[index] = self.action_plan_view(tool_plan)
                return
        action_plans.append(self.action_plan_view(tool_plan))

    def search_current_open_games_tool(self, **kwargs):
        self.calls.append("tool:search_current_open_games")
        return {"tool_name": "search_current_open_games", "called": True, "matches": self.pool_matches}

    def user_semantic_action_record(self, **kwargs):
        self.calls.append("action:user_semantic")
        return {
            "source": "llm",
            "arguments": {"proposed_action": "create_game", "confidence": 0.9},
            "validation": {"allowed": True, "effective_action": "create_game"},
        }

    def skipped_tool_result(self, tool_name, reason, **kwargs):
        return {"tool_name": tool_name, "called": False, "call_reason": reason, **kwargs}

    def rejected_tool_result(self, trace_id, tool_name, reason, **kwargs):
        return {"tool_name": tool_name, "called": False, "rejected": True, "validation_error": reason, **kwargs}

    def search_candidate_customers_tool(self, **kwargs):
        self.calls.append("tool:search_candidate_customers")
        return {"tool_name": "search_candidate_customers", "called": True, "candidates": [{"customer_id": "ran"}]}

    def candidate_recommendations_from_tool(self, tool_result):
        return [CandidateRecommendation(customer_id="ran", display_name="冉姐", score=100)]

    def send_message_tool(self, **kwargs):
        self.calls.append("tool:send_message")
        return {"tool_name": "send_message", "called": True, "outbox": [{"id": "out_1", "customer_id": "ran"}]}


def test_trial_tool_orchestration_uses_existing_pool_without_candidate_search() -> None:
    fake = FakeToolCallbacks(pool_matches=[{"game_id": "pool_1", "summary": "18:00 0.5无烟"}])

    result = TrialToolOrchestrationService(fake.callbacks(), CRITICAL_FIELDS).run(make_input())

    assert result.use_existing_pool is True
    assert result.should_materialize_game is False
    assert result.pool_matches == [{"game_id": "pool_1", "summary": "18:00 0.5无烟"}]
    assert result.response_missing_fields == []
    assert result.recommendations == []
    assert result.outbox == []
    assert "tool:search_candidate_customers" not in fake.calls
    assert result.tool_results["search_candidate_customers"]["called"] is False


def test_trial_tool_orchestration_searches_candidates_then_creates_outbox() -> None:
    fake = FakeToolCallbacks(
        after_open_calls=[{"tool_name": "search_candidate_customers"}],
        after_candidate_calls=[{"tool_name": "send_message"}],
    )
    game = make_game()

    result = TrialToolOrchestrationService(fake.callbacks(), CRITICAL_FIELDS).run(make_input(game))

    assert game.id == "game_stable"
    assert result.use_existing_pool is False
    assert result.should_materialize_game is True
    assert [item.customer_id for item in result.recommendations] == ["ran"]
    assert result.outbox == [{"id": "out_1", "customer_id": "ran"}]
    assert result.tool_results["search_candidate_customers"]["called"] is True
    assert result.tool_results["send_message"]["called"] is True
    assert fake.calls == [
        "plan:before_open_game_search",
        "tool:search_current_open_games",
        "action:user_semantic",
        "plan:after_open_game_search",
        "tool:search_candidate_customers",
        "plan:after_candidate_search",
        "tool:send_message",
    ]
