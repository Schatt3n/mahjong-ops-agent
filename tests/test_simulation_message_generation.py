from __future__ import annotations

import json
from pathlib import Path

from tests.simulation.behavior_policy import (
    DIALOG_PHASE_BUSINESS,
    BehaviorPolicy,
    MessageGenerationRequest,
    MessageGenerationResult,
)
from tests.simulation.message_generation import (
    ModelSimulationMessageGenerator,
    SimulationGeneratorConfig,
)
from tests.simulation.real_group_case_library import RealGroupCaseLibrary
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
    generator = ModelSimulationMessageGenerator(
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
    assert result.reference_case_ids
    assert client.calls[0]["timeout_seconds"] == 7
    prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert prompt_payload["last_agent_reply"] == "你这边几个人？"
    assert prompt_payload["fallback_text"] is None
    assert prompt_payload["dialog_phase"] == DIALOG_PHASE_BUSINESS
    assert prompt_payload["business_anchor"] == "帮我约个川麻局"
    assert {
        item["case_id"] for item in prompt_payload["real_group_chat_references"]
    } == set(result.reference_case_ids)


def test_real_case_library_only_loads_anonymized_approved_gold(tmp_path: Path) -> None:
    dataset = tmp_path / "real_cases.jsonl"
    records = [
        {
            "id": "approved_query",
            "case_type": "member_query",
            "quality_tier": "gold",
            "review_status": "approved",
            "source": {"anonymized": True, "source_refs": ["must-not-enter-prompt"]},
            "tags": ["real_group_chat", "query"],
            "messages": [
                {"role": "customer", "content_type": "text", "text": "1块杭麻有人吗"}
            ],
            "expected": {"intent": "query", "search_requirement": {"stake": "1"}},
        },
        {
            "id": "not_reviewed",
            "case_type": "member_query",
            "quality_tier": "gold",
            "review_status": "pending",
            "source": {"anonymized": True},
            "messages": [{"role": "customer", "content_type": "text", "text": "不该进入"}],
        },
        {
            "id": "not_anonymized",
            "case_type": "member_query",
            "quality_tier": "gold",
            "review_status": "approved",
            "source": {"anonymized": False},
            "messages": [{"role": "customer", "content_type": "text", "text": "也不该进入"}],
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )

    library = RealGroupCaseLibrary.from_jsonl(dataset)

    assert [item.case_id for item in library.examples] == ["approved_query"]
    payload = library.examples[0].to_prompt_payload()
    assert payload["turns"][0]["text"] == "1块杭麻有人吗"
    assert "source" not in payload
    assert "source_refs" not in json.dumps(payload, ensure_ascii=False)


def test_glm_group_prompt_uses_real_cases_and_records_provenance(tmp_path: Path) -> None:
    dataset = tmp_path / "real_cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "real_fragment",
                "case_type": "fragmented_input",
                "quality_tier": "gold",
                "review_status": "approved",
                "source": {"anonymized": True},
                "tags": ["fragmented_input", "constraint_relaxation"],
                "messages": [
                    {"role": "customer", "content_type": "text", "text": "现在一块无烟还有嘛"},
                    {"role": "customer", "content_type": "text", "text": "0.5无烟也可以"},
                ],
                "expected": {"intent": "query", "same_session": True},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeCompletionClient(json.dumps({"text": "五毛无烟也行"}, ensure_ascii=False))
    generator = ModelSimulationMessageGenerator(
        client,
        config=SimulationGeneratorConfig(
            model="glm-4.7-flash",
            timeout_seconds=7,
            real_case_dataset_path=dataset,
        ),
        case_library=RealGroupCaseLibrary.from_jsonl(dataset),
    )

    result = generator.generate(_request())

    assert result.reference_case_ids == ("real_fragment",)
    prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert prompt_payload["real_group_chat_references"][0]["case_id"] == "real_fragment"
    assert prompt_payload["real_group_chat_references"][0]["turns"][1]["text"] == "0.5无烟也可以"
    assert "请模仿表达习惯但不要逐字复制" in client.calls[0]["messages"][0]["content"]


def test_real_group_examples_are_not_injected_into_private_or_chitchat_turns(tmp_path: Path) -> None:
    dataset = tmp_path / "real_cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "real_query",
                "case_type": "member_query",
                "quality_tier": "gold",
                "review_status": "approved",
                "source": {"anonymized": True},
                "messages": [{"role": "customer", "content_type": "text", "text": "1块有人吗"}],
                "expected": {"intent": "query"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    library = RealGroupCaseLibrary.from_jsonl(dataset)

    private_refs = library.select_for_generation(
        channel="private",
        fallback_text="一块有人吗",
        dialog_phase="business",
        is_follow_up=False,
        seed_material="private",
    )
    chitchat_refs = library.select_for_generation(
        channel="group",
        fallback_text="今天也太热了",
        dialog_phase="chitchat",
        is_follow_up=False,
        seed_material="chitchat",
    )

    assert private_refs == ()
    assert chitchat_refs == ()


def test_customer_model_failure_is_recorded_without_sending_a_fixed_replacement() -> None:
    client = FakeCompletionClient("not-json")
    generator = ModelSimulationMessageGenerator(
        client,
        config=SimulationGeneratorConfig(model="glm-4.7-flash"),
    )

    result = generator.generate(_request())

    assert result.text == ""
    assert result.source == "generation_failed"
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


def test_audit_context_is_kept_for_review_but_never_sent_to_the_agent() -> None:
    user = VirtualUser(customer_id='sim_user_081', display_name='王哥', balance=100, preferred_game='sichuan_mahjong', persona='active_gambler')
    policy = BehaviorPolicy([user], seed=42)
    action = policy.first_action(user, sequence=1)
    assert action is not None
    dispatched = policy.materialize_action(action, user=user)
    assert dispatched.generation_context['scenario_text'] == action.text
    assert dispatched.generation_context['persona'] == user.persona
    metadata = dispatched.to_wechat_payload()['metadata']
    assert 'simulation_thread_id' not in metadata
    assert 'simulation_dialog_phase' not in metadata
    assert 'simulation_generation' not in metadata
    assert 'generation_context' not in metadata


def test_model_customer_gets_only_own_fixed_facts_and_visible_dialogue() -> None:
    from dataclasses import replace
    request = replace(_request(), customer_facts={"game_type": "national_standard_mahjong", "party_size": 2},
                      visible_history=({"role": "owner", "text": "你这边几个人？"},))
    client = FakeCompletionClient('{"text":"我们两个，打国标的"}')
    result = ModelSimulationMessageGenerator(client, config=SimulationGeneratorConfig(min_interval_seconds=0)).generate(request)
    payload = json.loads(client.calls[0]['messages'][1]['content'])
    assert payload['customer_facts']['party_size'] == 2
    assert payload['visible_history'] == [{"role": "owner", "text": "你这边几个人？"}]
    assert 'objective_state' not in payload and 'tool_results' not in payload
    assert '不是只润色 fallback_text' in client.calls[0]['messages'][0]['content']
    assert result.text == '我们两个，打国标的'


def test_model_customer_can_explain_waiting_instead_of_being_forced_to_keep_talking() -> None:
    client = FakeCompletionClient('{"text":"好，那我先等着","wants_to_continue":false,"reason":"老板已确认继续找人，这轮选择等待"}')
    result = ModelSimulationMessageGenerator(client, config=SimulationGeneratorConfig(min_interval_seconds=0)).generate(_request())
    assert result.wants_to_continue is False
    assert result.decision_reason == '老板已确认继续找人，这轮选择等待'
    assert result.text == '好，那我先等着'


def test_rule_customer_answers_with_stable_own_conditions_instead_of_random_confirmation() -> None:
    from tests.simulation.sim_orchestrator import DialogState
    user = VirtualUser(customer_id='sim_user_093', display_name='测试客户', balance=100,
                       preferred_game='national_standard_mahjong', persona='active_gambler', party_size=2)
    policy = BehaviorPolicy([user], seed=7)
    for sequence, question in enumerate(('杭麻还是川麻，你几个人？', '打多大？有烟没烟？', '几点开，几个人？'), 1):
        state = DialogState(turn_count=sequence, pending_response_to='user', last_agent_reply=question,
                            last_conversation_id='sim:private:test', last_channel='private')
        action = policy.get_next_action(user, sequence=sequence, due_simulated_seconds=10*sequence, dialog_state=state)
        assert action is not None
        assert '国标' in action.text and '2个人' in action.text
        assert '0.5' in action.text and '无烟' in action.text and '人齐开' in action.text
        dispatched = policy.materialize_action(action, user=user, dialog_state=state)
        assert dispatched.generation_context['customer_facts']['party_size'] == 2
        assert 'customer_facts' not in dispatched.to_wechat_payload()['metadata']
