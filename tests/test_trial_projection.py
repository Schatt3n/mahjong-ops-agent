from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from mahjong_agent.context_builder import WorkflowContextBuilder
from mahjong_agent.controlled_workflow import ControlledWorkflowService
from mahjong_agent.core import AgentCore
from mahjong_agent.memory import InMemoryShortTermMemoryStore
from mahjong_agent.models import ChannelType, CustomerProfile, Message, PlayPreference
from mahjong_agent.semantic_resolver import SemanticResolver
from mahjong_agent.trial_projection import project_controlled_result_for_trial
from mahjong_agent.trial_response import TrialControlledResponseAdapter, merge_controlled_trial_response


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 30, 16, 0, tzinfo=TZ)


class FakeSemanticLLMClient:
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "intent": "find_players",
            "proposed_action": "create_game",
            "confidence": 0.91,
            "reasoning_summary": "用户确认新组局。",
            "slots": {
                "game_type": {
                    "value": "hangzhou_mahjong",
                    "source": "explicit",
                    "confidence": 0.9,
                    "confirmed": True,
                    "needs_confirmation": False,
                },
                "variant": {
                    "value": "caiqiao",
                    "source": "explicit",
                    "confidence": 0.88,
                    "confirmed": True,
                    "needs_confirmation": False,
                },
                "stake": {
                    "value": "0.5",
                    "source": "explicit",
                    "confidence": 0.9,
                    "confirmed": True,
                    "needs_confirmation": False,
                },
                "start_time_mode": {
                    "value": "people_ready",
                    "source": "explicit",
                    "confidence": 0.86,
                    "confirmed": True,
                    "needs_confirmation": False,
                },
                "missing_count": {
                    "value": 3,
                    "source": "explicit",
                    "confidence": 0.9,
                    "confirmed": True,
                    "needs_confirmation": False,
                },
                "smoke": {
                    "value": "any",
                    "source": "explicit",
                    "confidence": 0.9,
                    "confirmed": True,
                    "needs_confirmation": False,
                },
                "duration_hours": {
                    "value": 4,
                    "source": "explicit",
                    "confidence": 0.9,
                    "confirmed": True,
                    "needs_confirmation": False,
                },
            },
        }


def seed_customers(core: AgentCore) -> None:
    core.upsert_customer(
        CustomerProfile(
            id="ran",
            display_name="冉姐",
            preferred_levels=["0.5"],
            smoke_free_preference=True,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["0.5"])],
            usual_start_hours=[16],
        )
    )
    core.upsert_customer(
        CustomerProfile(
            id="liu",
            display_name="刘姐",
            preferred_levels=["0.5"],
            smoke_free_preference=False,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["0.5"])],
            usual_start_hours=[16],
        )
    )


def make_result():
    core = AgentCore()
    seed_customers(core)
    memory = InMemoryShortTermMemoryStore()
    service = ControlledWorkflowService(
        core=core,
        context_builder=WorkflowContextBuilder(core, memory),
        semantic_resolver=SemanticResolver(FakeSemanticLLMClient()),
        memory_store=memory,
    )
    message = Message(
        text="0.5财敲人齐开，173，4h，烟都可",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_projection",
        metadata={"conversation_id": "boss_trial"},
    )
    return service.handle_message(message, now=NOW, trace_id="trace_projection")


def test_trial_projection_exposes_legacy_shape_without_redeciding_business_flow() -> None:
    projected = project_controlled_result_for_trial(make_result())

    assert projected["workflow"]["engine"] == "controlled_workflow.v1"
    assert projected["workflow"]["trace_id"] == "trace_projection"
    assert projected["parsed"]["conversation_id"] == "boss_trial"
    assert projected["parsed"]["user_intent"] == "find_players"
    assert projected["parsed"]["raw_action"] == "create_game"
    assert projected["parsed"]["intent_action"] == "queue_invites"
    assert projected["parsed"]["level"] == "0.5"
    assert projected["parsed"]["start_time"] == "people_ready"
    assert projected["parsed"]["missing_count"] == 3
    assert projected["parsed"]["semantic_action"]["required_tools"] == [
        "search_current_open_games",
        "search_candidate_customers",
        "create_pending_outbox",
        "create_game",
    ]

    assert projected["suggested_reply"]["text"] == "好的，我帮你问问。"
    assert projected["suggested_reply"]["status"] == "待审批"
    assert projected["outbox"]
    assert projected["outbox"][0]["status"] == "待审批"
    assert projected["outbox"][0]["approval_required"] is True
    assert projected["outbox"][0]["direct_send_executed"] is False
    assert "打吗" in projected["outbox"][0]["message_text"]

    assert projected["tool_results"]["search_current_open_games"]["called"] is True
    assert projected["tool_results"]["search_candidate_customers"]["called"] is True
    assert projected["tool_results"]["create_pending_outbox"]["called"] is True
    assert projected["tool_results"]["create_game"]["called"] is True
    assert projected["state"]["games"][0]["status"] == "negotiating"
    assert projected["agent_actions"][0]["protocol"] == "controlled_workflow.v1"
    assert projected["trace"]
    assert any(event["step"] == "llm_prompt" for event in projected["trace"])


def test_trial_response_adapter_delegates_persistence_and_merges_trial_shape() -> None:
    workflow_result = make_result()

    class FakePersistenceAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def persist(self, **kwargs) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {
                "persisted": True,
                "game": {"id": "game_persisted", "status": "邀约中"},
                "outbox": [{"id": "out_persisted", "approval_required": True}],
                "agent_actions": [{"stage": "create_game", "validated_actions": []}],
            }

    persistence = FakePersistenceAdapter()
    response = TrialControlledResponseAdapter(persistence).build(
        workflow_result=workflow_result,
        source_text="0.5财敲人齐开，173，4h，烟都可",
        sender_id="zhang",
        sender_name="张哥",
        trace_id="trace_response",
        now=NOW,
    )

    assert len(persistence.calls) == 1
    assert persistence.calls[0]["workflow_result"] is workflow_result
    assert persistence.calls[0]["projected"]["workflow"]["engine"] == "controlled_workflow.v1"
    assert response["controlled_workflow_enabled"] is True
    assert response["legacy_path"] is False
    assert response["api_trace_id"] == "trace_response"
    assert response["persistence"]["persisted"] is True
    assert response["state"]["games"] == [{"id": "game_persisted", "status": "邀约中"}]
    assert response["outbox"] == [{"id": "out_persisted", "approval_required": True}]
    assert response["agent_actions"][-1]["stage"] == "create_game"


def test_merge_controlled_trial_response_keeps_projected_data_when_persistence_skips() -> None:
    projected = {
        "state": {"games": [{"id": "projected_game"}]},
        "outbox": [{"id": "projected_outbox"}],
        "agent_actions": [{"stage": "action_validation"}],
    }
    response = merge_controlled_trial_response(
        projected,
        {"persisted": False, "reason": "no_state_write_for_action"},
        trace_id="trace_skip",
    )

    assert response["state"]["games"] == [{"id": "projected_game"}]
    assert response["outbox"] == [{"id": "projected_outbox"}]
    assert response["agent_actions"] == [{"stage": "action_validation"}]
    assert response["persistence"]["reason"] == "no_state_write_for_action"
    assert response["trace_id"] == "trace_skip"
