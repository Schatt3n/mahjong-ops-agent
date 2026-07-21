from __future__ import annotations

import json

from tests.simulation.behavior_policy import (
    DIALOG_PHASE_BUSINESS,
    BehaviorPolicy,
    MessageGenerationRequest,
    MessageGenerationResult,
)
from tests.simulation.message_generation import (
    GLMSimulationMessageGenerator,
    SimulationGeneratorConfig,
)
from tests.simulation.sim_factory import VirtualUser


class FakeCompletionClient:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.calls: list[dict] = []

    def complete(self, messages, *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append(
            {
                "messages": messages,
                "trace_id": trace_id,
                "timeout_seconds": timeout_seconds,
            }
        )
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def _request() -> MessageGenerationRequest:
    return MessageGenerationRequest(
        sender_id="sim_user_081",
        sender_name="王哥",
        persona="active_gambler",
        preferred_game="sichuan_mahjong",
        channel="group",
        conversation_id="sim:group:sim_group_001",
        turn_count=1,
        last_agent_reply="你这边几个人？",
        fallback_text="我一个人",
        is_follow_up=True,
        dialog_phase=DIALOG_PHASE_BUSINESS,
        business_anchor="帮我约个川麻局",
    )


def test_glm_generator_returns_contract_text_and_audit_metadata() -> None:
    client = FakeCompletionClient(json.dumps({"text": "就我一个"}, ensure_ascii=False))
    generator = GLMSimulationMessageGenerator(
        client,
        config=SimulationGeneratorConfig(model="glm-4.7-flash", timeout_seconds=7),
    )

    result = generator.generate(_request())

    assert result.text == "就我一个"
    assert result.source == "glm"
    assert result.model == "glm-4.7-flash"
    assert result.trace_id.startswith("trace_sim_gen_")
    assert result.latency_ms is not None
    assert result.error is None
    assert client.calls[0]["timeout_seconds"] == 7
    prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert prompt_payload["last_agent_reply"] == "你这边几个人？"
    assert prompt_payload["fallback_text"] == "我一个人"
    assert prompt_payload["dialog_phase"] == DIALOG_PHASE_BUSINESS
    assert prompt_payload["business_anchor"] == "帮我约个川麻局"


def test_glm_generator_falls_back_without_breaking_simulation() -> None:
    client = FakeCompletionClient("not-json")
    generator = GLMSimulationMessageGenerator(
        client,
        config=SimulationGeneratorConfig(model="glm-4.7-flash"),
    )

    result = generator.generate(_request())

    assert result.text == "我一个人"
    assert result.source == "rule_fallback"
    assert result.model == "glm-4.7-flash"
    assert "ValueError" in str(result.error)


def test_behavior_policy_only_calls_generator_when_action_is_dispatched() -> None:
    class CountingGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request: MessageGenerationRequest) -> MessageGenerationResult:
            self.calls += 1
            return MessageGenerationResult(
                text=f"模型生成：{request.fallback_text}",
                source="glm",
                model="glm-4.7-flash",
                trace_id="trace_lazy_generation",
            )

    user = VirtualUser(
        customer_id="sim_user_081",
        display_name="王哥",
        balance=100,
        preferred_game="sichuan_mahjong",
        persona="active_gambler",
    )
    generator = CountingGenerator()
    policy = BehaviorPolicy([user], seed=7, message_generator=generator)

    scheduled = policy.first_action(user, sequence=1)

    assert scheduled is not None
    assert generator.calls == 0
    dispatched = policy.materialize_action(scheduled, user=user)
    assert generator.calls == 1
    assert dispatched.generation_source == "glm"
    assert dispatched.text.startswith("模型生成：")
