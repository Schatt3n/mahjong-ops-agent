"""审查客户端、端到端预算及租约独立于主模型，使用HTTP替身验证真实请求参数。"""

from copy import deepcopy
from datetime import timedelta
import json
import os

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentLLMConfig, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder,
    SQLiteAgentStore, StaticAgentClient, TokenBudget, UserMessage,
)
from mahjong_agent_runtime.llm import OPENCODE_GO_BASE_URL
from mahjong_agent_runtime.review_config import (
    build_reply_review_client, reply_review_runtime_config, reply_review_timeout_seconds,
)
from mahjong_agent_runtime.services import AgentRunStateManager
from tests.test_llm_resilience import FakeClock, IncrementalResponse, encoded_response


@pytest.fixture(autouse=True)
def clean_review_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith("MAHJONG_REPLY_REVIEW_LLM_") or key.startswith("MAHJONG_LLM_") or key in {"DEEPSEEK_API_KEY", "MAHJONG_DEEPSEEK_API_KEY"}:
            monkeypatch.delenv(key)


def base_config():
    return AgentLLMConfig(api_key="unit-review-only", model="deepseek-flash", provider="deepseek",
        base_url="https://example.invalid", thinking_mode="enabled", reasoning_effort="high",
        max_tokens=12000, attempt_timeout_seconds=60, retry_attempts=2)


def test_default_review_config_is_independent_and_does_not_mutate_main():
    main = base_config()
    before = deepcopy(main)
    client = build_reply_review_client(main)
    assert main == before and client.config is not main
    assert client.config.model == main.model and client.config.api_key == main.api_key
    assert (client.config.thinking_mode, client.config.reasoning_effort, client.config.max_tokens) == ("enabled", "low", 16384)
    assert client.config.temperature == 0.0 and main.temperature == before.temperature == 0.2
    assert client.config.retry_attempts == 3 and client.config.attempt_timeout_seconds is None
    assert reply_review_timeout_seconds() == 180
    summary = reply_review_runtime_config(client, reply_review_timeout_seconds())
    assert summary["total_timeout_seconds"] == 180 and summary["reasoning_effort"] == "low"
    assert main.api_key not in json.dumps(summary) and main.base_url not in json.dumps(summary)


def test_production_and_real_simulation_use_the_shared_review_factory(tmp_path, monkeypatch):
    from tests.test_agent_app import load_app_module
    from tests.simulation import sim_adapter
    from mahjong_agent_runtime import OpenAICompatibleAgentClient

    main = base_config()
    expected = reply_review_runtime_config(build_reply_review_client(main), reply_review_timeout_seconds())
    app = load_app_module()
    assert reply_review_runtime_config(app.build_reply_self_review_client(main), 180) == expected
    monkeypatch.setattr(sim_adapter, "load_dotenv_defaults", lambda *_: None)
    monkeypatch.setattr(sim_adapter.OpenAICompatibleAgentClient, "from_env", lambda: OpenAICompatibleAgentClient(config=main))
    store = SQLiteAgentStore(tmp_path / "simulation.sqlite3")
    runtime, primary = sim_adapter.build_runtime(store, "real")
    assert primary.config is main and primary.config.reasoning_effort == "high"
    assert runtime.reply_self_review_client is not primary
    assert reply_review_runtime_config(runtime.reply_self_review_client, runtime.reply_self_review_timeout_seconds) == expected
    store._connection.close()


def test_independent_review_overrides_and_missing_configuration(monkeypatch):
    assert build_reply_review_client() is None
    for key, value in {"MODEL":"review-model", "PROVIDER":"deepseek", "API_KEY":"independent-unit-key",
            "BASE_URL":"https://review.invalid/", "THINKING_MODE":"disabled", "REASONING_EFFORT":"max",
            "MAX_COMPLETION_TOKENS":"4096", "TEMPERATURE":"0.4", "TIMEOUT_SECONDS":"210", "ATTEMPT_TIMEOUT_SECONDS":"70",
            "RETRY_ATTEMPTS":"2", "MAX_CONCURRENCY":"1"}.items():
        monkeypatch.setenv("MAHJONG_REPLY_REVIEW_LLM_" + key, value)
    client = build_reply_review_client(base_config())
    assert (client.config.model, client.config.api_key, client.config.base_url) == (
        "review-model", "independent-unit-key", "https://review.invalid")
    assert client.config.max_tokens == 4096 and client.config.attempt_timeout_seconds == 70
    assert client.config.temperature == 0.4
    assert reply_review_timeout_seconds() == 210 and client.config.max_concurrency == 1
    assert reply_review_runtime_config(client, 210)["reasoning_effort"] is None


@pytest.mark.parametrize("prefix,factory_name", [
    ("MAHJONG_LLM_FALLBACK", "fallback_client_from_env"),
    ("MAHJONG_REPLY_REVIEW_LLM", "build_reply_self_review_client"),
    ("MAHJONG_TEXT_GENERATION_LLM", "build_customer_visible_text_generation_client"),
    ("MAHJONG_CONTEXT_SUMMARY_LLM", "build_context_summary_client"),
    ("MAHJONG_WECHATY_INPUT_GATE_LLM", "build_wechaty_input_gate_client"),
])
@pytest.mark.parametrize("mode", ["", "text"])
def test_auxiliary_protocol_and_response_format_inherit_only_for_same_route(monkeypatch, prefix, factory_name, mode):
    """同一路由继承协议和格式，独立连接可以配置自己的参数。"""
    from mahjong_agent_runtime.llm import fallback_client_from_env
    from tests.test_agent_app import load_app_module

    for field in ("API_KEY", "MODEL", "PROVIDER", "BASE_URL", "RESPONSE_FORMAT", "API_PROTOCOL"):
        monkeypatch.delenv(f"{prefix}_{field}", raising=False)
    for key, value in {
        "MAHJONG_LLM_API_KEY": "unit-primary-key",
        "MAHJONG_LLM_MODEL": "unit-primary-model",
        "MAHJONG_LLM_PROVIDER": "openai_compatible",
        "MAHJONG_LLM_BASE_URL": "https://primary.invalid/v1/",
        "MAHJONG_LLM_RESPONSE_FORMAT": mode,
        "MAHJONG_LLM_API_PROTOCOL": "responses",
        f"{prefix}_API_KEY": "unit-auxiliary-key",
        f"{prefix}_MODEL": "unit-auxiliary-model",
    }.items():
        monkeypatch.setenv(key, value)
    if factory_name in {"fallback_client_from_env", "build_context_summary_client"}:
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://primary.invalid/v1/")
    factory = fallback_client_from_env if factory_name == "fallback_client_from_env" else getattr(load_app_module(), factory_name)
    inherited = factory().config
    assert inherited.base_url == "https://primary.invalid/v1" and inherited.response_format == mode
    assert inherited.api_protocol == "responses"

    monkeypatch.setenv(f"{prefix}_PROVIDER", "deepseek")
    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://secondary.invalid/v1/")
    independent = factory().config
    assert independent.provider == "deepseek" and independent.base_url == "https://secondary.invalid/v1"
    assert independent.response_format == "json_object"
    assert independent.api_protocol == "chat_completions"

    monkeypatch.setenv(f"{prefix}_RESPONSE_FORMAT", "")
    monkeypatch.setenv(f"{prefix}_API_PROTOCOL", "responses")
    explicit = factory().config
    assert explicit.response_format == "" and explicit.api_protocol == "responses"


@pytest.mark.parametrize("prefix,factory_name", [
    ("MAHJONG_REPLY_REVIEW_LLM", "build_reply_self_review_client"),
    ("MAHJONG_TEXT_GENERATION_LLM", "build_customer_visible_text_generation_client"),
    ("MAHJONG_CONTEXT_SUMMARY_LLM", "build_context_summary_client"),
    ("MAHJONG_WECHATY_INPUT_GATE_LLM", "build_wechaty_input_gate_client"),
])
def test_auxiliary_api_key_is_scoped_to_primary_provider_and_address(monkeypatch, prefix, factory_name):
    """默认凭证只能发给主模型的供应商和接口，其他连接需要自己的凭证。"""
    from tests.test_agent_app import load_app_module

    for field in ("API_KEY", "MODEL", "PROVIDER", "BASE_URL", "RESPONSE_FORMAT"):
        monkeypatch.delenv(f"{prefix}_{field}", raising=False)
    for key, value in {
        "MAHJONG_LLM_API_KEY": "unit-primary-key",
        "MAHJONG_LLM_MODEL": "unit-primary-model",
        "MAHJONG_LLM_PROVIDER": "openai_compatible",
        "MAHJONG_LLM_BASE_URL": "https://primary.invalid/v1/",
        f"{prefix}_MODEL": "unit-auxiliary-model",
        f"{prefix}_BASE_URL": "https://primary.invalid/v1",
    }.items():
        monkeypatch.setenv(key, value)
    factory = getattr(load_app_module(), factory_name)
    assert factory().config.api_key == "unit-primary-key"

    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://primary.invalid/another/v1")
    with pytest.raises(ValueError, match=f"{prefix}_API_KEY"):
        factory()

    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://primary.invalid/v1")
    monkeypatch.setenv(f"{prefix}_PROVIDER", "deepseek")
    with pytest.raises(ValueError, match=f"{prefix}_API_KEY"):
        factory()

    monkeypatch.delenv(f"{prefix}_BASE_URL")
    monkeypatch.setenv(f"{prefix}_PROVIDER", "opencode_go")
    with pytest.raises(ValueError, match=f"{prefix}_API_KEY"):
        factory()
    monkeypatch.setenv(f"{prefix}_API_KEY", "unit-independent-key")
    independent = factory().config
    assert independent.api_key == "unit-independent-key" and independent.base_url == OPENCODE_GO_BASE_URL

    monkeypatch.delenv("MAHJONG_LLM_MODEL")
    monkeypatch.setenv(f"{prefix}_PROVIDER", "openai_compatible")
    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://independent.invalid/legacy/v1/")
    standalone = factory()
    requests = []
    clock = FakeClock()

    def urlopen(request, **_kwargs):
        requests.append(request)
        return IncrementalResponse(clock, [(0.1, encoded_response())])

    standalone.urlopen, standalone.monotonic_fn = urlopen, clock
    standalone.complete([{"role": "user", "content": "测试"}], trace_id="aux-route", timeout_seconds=2)
    assert requests[0].full_url == "https://independent.invalid/legacy/v1/chat/completions"
    assert requests[0].get_header("Authorization") == "Bearer unit-independent-key"


@pytest.mark.parametrize("field,value", [("THINKING_MODE","invalid"), ("REASONING_EFFORT","medium"),
    ("MAX_COMPLETION_TOKENS","0"), ("MAX_COMPLETION_TOKENS","bad"), ("RETRY_ATTEMPTS","-1"),
    ("ATTEMPT_TIMEOUT_SECONDS","nan"), ("TIMEOUT_SECONDS","0"), ("TIMEOUT_SECONDS","inf")])
def test_invalid_explicit_review_settings_fail_before_http(monkeypatch, field, value):
    monkeypatch.setenv("MAHJONG_REPLY_REVIEW_LLM_" + field, value)
    with pytest.raises(ValueError):
        build_reply_review_client(base_config())
        reply_review_timeout_seconds()


def accepted_review(text):
    return json.dumps({"approved":True, "needs_human":False, "reasoning_summary":"依据充分", "violations":[],
        "item_reviews":[{"item_id":"reply_to_user", "approved":True, "suggested_safe_text":text,
            "reasoning_summary":"依据充分", "violations":[]}]}, ensure_ascii=False)


def test_real_review_processor_sends_low_parameters_and_its_own_timeout_without_changing_main():
    clock = FakeClock()
    requests = []
    review = build_reply_review_client(base_config())
    text = "好的。"
    def urlopen(request, *, timeout):
        requests.append((json.loads(request.data), timeout))
        return IncrementalResponse(clock, [(0.1, encoded_response(accepted_review(text)))])
    review.urlopen, review.monotonic_fn = urlopen, clock
    main = StaticAgentClient([])
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=main, trace_recorder=trace, llm_timeout_seconds=7,
        reply_self_review_enabled=True, reply_self_review_client=review, reply_self_review_timeout_seconds=180)
    message = UserMessage(conversation_id="own", sender_id="customer", sender_name="本人", text="知道了")
    result = runtime.action_processor.review_customer_visible_content(message=message, trace_id="review-transport",
        action=AgentAction(goal="确认收到", objective_status="completed", reasoning_summary="仅确认收到", reply_to_user=text),
        review_items=[{"item_id":"reply_to_user", "recipient_id":"customer", "text":text}],
        context_payload={}, review_scope="reply_to_user", turn_budget=TokenBudget(max_tokens_per_call=64000))
    assert result.result["approved"] is True and main.calls == []
    assert requests[0][1] == pytest.approx(90, abs=.01)
    assert requests[0][0]["thinking"] == {"type":"enabled"}
    assert requests[0][0]["reasoning_effort"] == "low" and requests[0][0]["max_tokens"] == 16384
    assert runtime.llm_timeout_seconds == runtime.action_processor.llm_timeout_seconds == 7
    prompt = next(e for e in trace.get_trace("review-transport") if e.step == "customer_visible_content_review_prompt")
    assert prompt.content["review_runtime_config"]["total_timeout_seconds"] == 180


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_long_review_lease_cannot_be_reclaimed_before_its_budget_finishes(kind, tmp_path):
    store = InMemoryAgentStore() if kind == "memory" else SQLiteAgentStore(tmp_path / "lease.sqlite3")
    runtime = AgentRuntime(llm_client=StaticAgentClient([]), store=store, llm_timeout_seconds=45,
        reply_self_review_enabled=True, reply_self_review_timeout_seconds=180)
    message = UserMessage(conversation_id="own", sender_id="customer", sender_name="本人", text="看看")
    run = runtime.run_state_manager.start(message, trace_id="long-review", run_id="run-review", run_version=0)
    start = run.lease_until - timedelta(seconds=runtime.run_state_manager.lease_seconds)
    recovery = AgentRunStateManager(store, InMemoryTraceRecorder(), worker_id="other-worker")
    assert runtime.run_state_manager.lease_seconds >= 45 + 180 + 30
    assert recovery.recoverable(at=start + timedelta(seconds=224)) == []
    assert recovery.claim(run.run_id, at=start + timedelta(seconds=224)) is None
    assert recovery.claim(run.run_id, at=run.lease_until + timedelta(seconds=1)) is not None
    if kind == "sqlite":
        store._connection.close()


@pytest.mark.parametrize("enabled,review_timeout", [(False,None), (False,180), (True,None)])
def test_unconfigured_review_does_not_expand_runtime_main_budget(enabled, review_timeout):
    runtime = AgentRuntime(llm_client=StaticAgentClient([]), llm_timeout_seconds=45,
        reply_self_review_enabled=enabled, reply_self_review_timeout_seconds=review_timeout)
    assert runtime.llm_timeout_seconds == 45 and runtime.run_state_manager.lease_seconds == 120


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "180"])
def test_runtime_rejects_invalid_explicit_review_deadline(value):
    with pytest.raises(ValueError, match="reply_self_review_timeout_seconds"):
        AgentRuntime(llm_client=StaticAgentClient([]), reply_self_review_timeout_seconds=value)
