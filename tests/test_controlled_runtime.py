from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.controlled_runtime import ControlledRuntimeConfig, build_controlled_runtime
from mahjong_agent.models import ChannelType, Message
from mahjong_agent.state_machine import InMemoryWorkflowStateStore
from mahjong_agent.workflow_models import ActionName, GameWorkflowStatus


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 30, 17, 0, tzinfo=TZ)


def test_controlled_runtime_fails_closed_without_llm_and_writes_jsonl_trace(tmp_path, monkeypatch) -> None:
    for key in ("MAHJONG_LLM_API_KEY", "MAHJONG_LLM_MODEL", "MAHJONG_LLM_PROVIDER", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    trace_path = tmp_path / "logs" / "controlled_trace.jsonl"
    runtime = build_controlled_runtime(
        config=ControlledRuntimeConfig(
            trace_jsonl_path=trace_path,
            fail_closed_without_llm=True,
        )
    )
    message = Message(
        text="老板，今天有人打吗",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime")

    assert result.run.semantic_resolution is not None
    assert result.run.semantic_resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert result.final_text == "这个我先转人工确认一下。"
    assert trace_path.exists()
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "trace_runtime-2026-06-30 17:00:00-INFO:" in trace_text
    assert "LLM 未配置" in trace_text
    assert runtime.memory_store.load("boss_trial", "zhang", now=NOW)[0].system_reply == "这个我先转人工确认一下。"


def test_controlled_runtime_can_be_built_with_injected_llm_client(tmp_path) -> None:
    class FakeLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "inquire_existing_game",
                "proposed_action": "search_existing_games",
                "confidence": 0.88,
                "reasoning_summary": "用户只是咨询现有局。",
                "slots": {
                    "stake": {
                        "value": "0.5",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    }
                },
            }

    runtime = build_controlled_runtime(
        llm_client=FakeLLMClient(),
        config=ControlledRuntimeConfig(trace_jsonl_path=tmp_path / "trace.jsonl"),
    )
    message = Message(
        text="现在有0.5的吗",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_fake",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_fake")

    assert result.run.validated_action is not None
    assert result.run.validated_action.effective_action == ActionName.ASK_CREATE_CONFIRMATION
    assert result.final_text == "现在没有合适的，要组一个吗？"


def test_controlled_runtime_exposes_state_store_for_applied_transitions(tmp_path) -> None:
    class CreateGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "find_players",
                "proposed_action": "create_game",
                "confidence": 0.92,
                "reasoning_summary": "用户明确要组局，信息齐全。",
                "slots": {
                    "game_type": {
                        "value": "hangzhou_mahjong",
                        "source": "explicit",
                        "confidence": 0.9,
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
                        "confidence": 0.9,
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
                    "duration_mode": {
                        "value": "overnight",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    },
                },
            }

    state_store = InMemoryWorkflowStateStore()
    runtime = build_controlled_runtime(
        llm_client=CreateGameLLMClient(),
        state_store=state_store,
        config=ControlledRuntimeConfig(trace_jsonl_path=tmp_path / "trace_state.jsonl"),
    )
    message = Message(
        text="通宵0.5人齐开，173，烟都可",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_state",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_state")

    assert runtime.state_store is state_store
    assert result.run.state_transitions
    game_id = result.run.state_transitions[-1].entity_id
    assert runtime.state_store.current_status("game", game_id) == GameWorkflowStatus.OPEN.value
    assert runtime.state_store.transition_history(entity_type="game", entity_id=game_id)[0].metadata["store_applied"] is True
