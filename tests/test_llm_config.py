from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
import json
import os
import pathlib
import tempfile
from types import SimpleNamespace

from mahjong_agent import ChannelType, Message
from mahjong_agent.budget import LLMBudgetLimits, LLMBudgetManager
from mahjong_agent.llm import LLMConfig, OpenAICompatibleLLMResolver


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_qwen_provider_defaults_to_dashscope_compatible_endpoint() -> None:
    with patched_env(
        MAHJONG_LLM_PROVIDER="qwen",
        MAHJONG_LLM_API_KEY="test-key",
        MAHJONG_LLM_MODEL=None,
        MAHJONG_LLM_BASE_URL=None,
        DASHSCOPE_API_KEY=None,
        OPENAI_API_KEY=None,
    ):
        config = LLMConfig.from_env()

    assert config is not None
    assert config.provider == "qwen"
    assert config.model == "qwen-plus"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_dashscope_api_key_implies_qwen_provider() -> None:
    with patched_env(
        DASHSCOPE_API_KEY="test-key",
        MAHJONG_LLM_PROVIDER=None,
        MAHJONG_LLM_API_KEY=None,
        MAHJONG_LLM_MODEL=None,
        MAHJONG_LLM_BASE_URL=None,
        OPENAI_API_KEY=None,
    ):
        config = LLMConfig.from_env()

    assert config is not None
    assert config.provider == "qwen"
    assert config.model == "qwen-plus"


def test_zai_provider_defaults_to_glm_flash_endpoint() -> None:
    with patched_env(
        MAHJONG_LLM_PROVIDER="zai",
        MAHJONG_LLM_API_KEY="test-key",
        MAHJONG_LLM_MODEL=None,
        MAHJONG_LLM_BASE_URL=None,
        DASHSCOPE_API_KEY=None,
        OPENAI_API_KEY=None,
    ):
        config = LLMConfig.from_env()

    assert config is not None
    assert config.provider == "zai"
    assert config.model == "glm-4.7-flash"
    assert config.base_url == "https://api.z.ai/api/paas/v4"
    assert config.max_completion_tokens == 1024


def test_deepseek_provider_defaults_to_v4_flash_json_mode() -> None:
    with patched_env(
        MAHJONG_LLM_PROVIDER="deepseek",
        MAHJONG_LLM_API_KEY="test-key",
        MAHJONG_LLM_MODEL=None,
        MAHJONG_LLM_BASE_URL=None,
        MAHJONG_LLM_THINKING_ENABLED=None,
        MAHJONG_LLM_RESPONSE_FORMAT=None,
        DASHSCOPE_API_KEY=None,
        OPENAI_API_KEY=None,
    ):
        config = LLMConfig.from_env()

    assert config is not None
    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-flash"
    assert config.base_url == "https://api.deepseek.com"
    assert config.max_completion_tokens == 1024
    assert config.thinking_enabled is False
    assert config.response_format == "json_object"


def test_deepseek_integration_runner_forces_deepseek_provider_without_network() -> None:
    module = load_script(ROOT / "scripts" / "run_deepseek_integration_test.py")
    args = argparse.Namespace(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        timeout_seconds=30.0,
        max_completion_tokens=128,
        max_calls=2,
        max_tokens=12000,
        max_cost=1.0,
        max_tokens_per_call=8000,
        input_price_per_1k=0.0,
        output_price_per_1k=0.0,
    )

    with patched_env(
        MAHJONG_DEEPSEEK_API_KEY="test-key",
        DEEPSEEK_API_KEY=None,
        MAHJONG_LLM_API_KEY=None,
        MAHJONG_LLM_PROVIDER="openai",
        MAHJONG_LLM_MODEL="wrong-model",
    ):
        resolver, key_source = module.build_resolver(args)

    assert key_source == "MAHJONG_DEEPSEEK_API_KEY"
    assert resolver.config.provider == "deepseek"
    assert resolver.config.model == "deepseek-v4-flash"
    assert resolver.config.base_url == "https://api.deepseek.com"
    assert resolver.config.response_format == "json_object"
    assert resolver.config.thinking_enabled is False


def test_deepseek_integration_runner_requires_usage_to_prove_real_response() -> None:
    module = load_script(ROOT / "scripts" / "run_deepseek_integration_test.py")

    class FakeResolver:
        config = SimpleNamespace(
            provider="deepseek",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
        )

        def resolve(self, message, context=None):
            return SimpleNamespace(
                is_mahjong_related=True,
                intent="find_players",
                confidence=0.95,
                normalized_text="今天下班想打麻将，0.5或者1都行，烟都可",
                reply_text="我帮你看看。",
                needs_human_review=False,
                usage=None,
                budget={"allowed": True},
                notes=["模拟无 usage 的响应"],
            )

    original_build_resolver = module.build_resolver
    module.build_resolver = lambda args: (FakeResolver(), "MAHJONG_DEEPSEEK_API_KEY")
    try:
        args = argparse.Namespace(text="老板，今天有人打麻将吗", model="deepseek-v4-flash", min_confidence=0.45)
        with contextlib.redirect_stdout(io.StringIO()):
            assert module.run_semantic_smoke(args) == 1
    finally:
        module.build_resolver = original_build_resolver


def test_controlled_acceptance_gate_adds_deepseek_only_when_requested() -> None:
    module = load_script(ROOT / "scripts" / "run_controlled_agent_acceptance.py")

    offline_steps = module.build_steps(with_deepseek=False)
    integration_steps = module.build_steps(with_deepseek=True)

    assert all(step.name != "deepseek_integration" for step in offline_steps)
    assert any(step.name == "deepseek_integration" for step in integration_steps)


def test_controlled_acceptance_secret_scan_reports_paths_only() -> None:
    module = load_script(ROOT / "scripts" / "run_controlled_agent_acceptance.py")

    with tempfile.TemporaryDirectory() as temp_dir:
        path = pathlib.Path(temp_dir) / "leaked.txt"
        fake_key = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
        path.write_text(f"OPENAI_API_KEY={fake_key}\n", encoding="utf-8")
        hits = module.secret_scan([path])

    assert hits == [path]


def test_llm_payload_uses_context_message_view_instead_of_raw_metadata() -> None:
    resolver = OpenAICompatibleLLMResolver(
        LLMConfig(api_key="test-key", model="test-model")
    )
    message = Message(
        text="原始消息 13812345678",
        sender_id="wxid_raw_secret",
        sender_name="张哥",
        channel_id="group",
        channel_type=ChannelType.WECHAT_GROUP,
        metadata={"raw_private_token": "should-not-be-sent"},
    )
    context = {
        "current_message": {
            "text": "脱敏消息 [手机号]",
            "sender_ref": "customer_hash",
            "sender_display_name": "张哥",
            "channel_type": "wechat_group",
            "modalities": ["text"],
            "source": {"message_ref": "message_hash"},
        },
        "text_normalization": {
            "raw_text": "脱敏消息 [手机号]",
            "normalized_text": "脱敏消息 [手机号]",
            "changed_rule_ids": [],
        },
    }

    payload = resolver._message_payload(message, context)

    assert payload["text"] == "脱敏消息 [手机号]"
    assert payload["text_normalization"]["normalized_text"] == "脱敏消息 [手机号]"
    assert payload["sender_ref"] == "customer_hash"
    assert "13812345678" not in str(payload)
    assert "wxid_raw_secret" not in str(payload)
    assert "raw_private_token" not in str(payload)


def test_llm_budget_manager_denies_when_call_budget_is_used() -> None:
    manager = LLMBudgetManager(LLMBudgetLimits(max_calls_per_day=1, max_tokens_per_day=10_000))

    first = manager.reserve(
        key="shop-a",
        model="test-model",
        prompt={"message": "今天下班有人打麻将吗"},
        max_completion_tokens=32,
    )
    second = manager.reserve(
        key="shop-a",
        model="test-model",
        prompt={"message": "还有一条消息"},
        max_completion_tokens=32,
    )

    assert first.allowed is True
    assert second.allowed is False
    assert "调用次数预算" in second.reason


def test_llm_resolver_fails_closed_when_budget_denies() -> None:
    resolver = OpenAICompatibleLLMResolver(
        LLMConfig(api_key="test-key", model="test-model"),
        budget_manager=LLMBudgetManager(LLMBudgetLimits(max_calls_per_day=0)),
    )
    message = Message(
        text="老地方搭子",
        sender_id="u1",
        sender_name="张哥",
        channel_id="group",
        channel_type=ChannelType.WECHAT_GROUP,
    )

    resolution = resolver.resolve(message, context={"current_message": {"text": "老地方搭子"}})

    assert resolution.needs_human_review is True
    assert resolution.budget["allowed"] is False
    assert any("预算不足" in note for note in resolution.notes)


def test_llm_resolver_retries_truncated_json_with_compact_schema() -> None:
    calls: list[dict] = []
    audits: list[tuple[str, str, dict]] = []

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            if len(calls) == 1:
                content = (
                    '{"is_mahjong_related":true,"intent":"find_players",'
                    '"proposed_action":"create_game","confidence":0.86,'
                    '"reasoning_summary":"用户确认要组局","slots":{"duration_mode":{"value":"overnight"'
                )
                finish_reason = "length"
            else:
                content = json.dumps(
                    {
                        "is_mahjong_related": True,
                        "intent": "find_players",
                        "proposed_action": "create_game",
                        "confidence": 0.86,
                        "normalized_text": "通宵0.5帮我组一桌",
                        "reply_text": "好的，我帮你问问。",
                        "needs_human_review": False,
                        "reasoning_summary": "紧凑重试成功。",
                        "slots": {"duration_mode": "overnight"},
                    },
                    ensure_ascii=False,
                )
                finish_reason = "stop"
            return json.dumps(
                {
                    "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = __import__("mahjong_agent.llm", fromlist=["urllib"]).urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(payload)
        return FakeResponse(payload)

    llm_module = __import__("mahjong_agent.llm", fromlist=["urllib"])
    llm_module.urllib.request.urlopen = fake_urlopen
    try:
        resolver = OpenAICompatibleLLMResolver(
            LLMConfig(
                api_key="test-key",
                model="test-model",
                base_url="https://example.invalid/v1",
                max_completion_tokens=512,
                parse_retry_max_tokens=128,
            ),
            budget_manager=LLMBudgetManager(LLMBudgetLimits(max_calls_per_day=5, max_tokens_per_day=20_000)),
            audit_logger=lambda trace_id, event, payload: audits.append((trace_id, event, payload)),
        )
        message = Message(
            text="可以",
            sender_id="zhang",
            sender_name="张哥",
            channel_id="boss_trial",
            channel_type=ChannelType.MANUAL,
            metadata={"trace_id": "trace_retry_parse"},
        )
        resolution = resolver.resolve(
            message,
            context={
                "runtime": {"trace_id": "trace_retry_parse"},
                "current_message": {"text": "可以", "channel_type": "manual", "modalities": ["text"]},
                "workflow_followup_context": {"previous_system_suggested_reply": "0.5的暂时没有诶，要组一个吗？"},
            },
        )

        assert len(calls) == 2
        assert calls[1]["max_tokens"] == 128
        assert resolution.proposed_action == "create_game"
        assert resolution.confidence == 0.86
        assert any("重试成功" in note for note in resolution.notes)
        assert any(event == "llm_retry_request" for _, event, _ in audits)
        assert any(event == "llm_retry_parsed" for _, event, _ in audits)
    finally:
        llm_module.urllib.request.urlopen = original_urlopen


def test_llm_resolver_records_timeout_as_interrupted() -> None:
    audits: list[tuple[str, str, dict]] = []
    llm_module = __import__("mahjong_agent.llm", fromlist=["urllib"])
    original_urlopen = llm_module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        raise TimeoutError("simulated timeout")

    llm_module.urllib.request.urlopen = fake_urlopen
    try:
        resolver = OpenAICompatibleLLMResolver(
            LLMConfig(api_key="test-key", model="test-model", timeout_seconds=0.01),
            budget_manager=LLMBudgetManager(LLMBudgetLimits(max_calls_per_day=5, max_tokens_per_day=20_000)),
            audit_logger=lambda trace_id, event, payload: audits.append((trace_id, event, payload)),
        )
        message = Message(
            text="老板，有人打吗",
            sender_id="zhang",
            sender_name="张哥",
            channel_id="boss_trial",
            channel_type=ChannelType.MANUAL,
            metadata={"trace_id": "trace_timeout"},
        )

        resolution = resolver.resolve(message, context={"runtime": {"trace_id": "trace_timeout"}, "current_message": {"text": "老板，有人打吗"}})

        assert resolution.needs_human_review is True
        assert any("已中断" in note for note in resolution.notes)
        assert any(event == "llm_timeout" for _, event, _ in audits)
    finally:
        llm_module.urllib.request.urlopen = original_urlopen


class patched_env:
    def __init__(self, **values: str | None) -> None:
        self.values = values
        self.original: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, value in self.original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_script(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
