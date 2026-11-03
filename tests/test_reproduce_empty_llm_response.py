"""空响应诊断重放只调用模型边界，保留输入与旧证据，不触发业务动作。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading

import pytest

from mahjong_agent_runtime import AgentLLMConfig, OpenAICompatibleAgentClient
from mahjong_agent_runtime.llm import LLMCompletionText, LLMResponseFormatError


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def replay():
    """仅导入诊断脚本，不执行CLI、不读取真实环境配置或调用网络。"""
    path = ROOT / "scripts" / "reproduce_empty_llm_response.py"
    spec = importlib.util.spec_from_file_location("empty_replay_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_trace(tmp_path, *, stage="main", response=""):
    """临时来源包含多行私密输入，用于证明提取和落盘不复制提示词正文。"""
    messages = [{"role": "system", "content": '保持原内容\n反斜线\\和引号"'},
                {"role": "user", "content": "PRIVATE_SOURCE_INPUT_不要复制到结果\n第二行"}]
    names = ("llm_prompt", "llm_response") if stage == "main" else (
        "customer_visible_content_review_prompt", "customer_visible_content_review_response")
    events = [{"step": "context_built", "content": {}},
              {"step": names[0], "content": {"messages": messages, "step_index": 1}},
              {"step": names[1], "content": {"content": response, "step_index": 1}}]
    path = tmp_path / f"original-{stage}.json"
    path.write_text(json.dumps({"trace_id": "original-full-failure", "events": events}, ensure_ascii=False))
    return path, messages


@pytest.mark.parametrize("stage", ["main", "review"])
def test_source_selection_preserves_original_messages_and_trace_bytes(replay, tmp_path, stage):
    """主规划和审查均按显式序号提取，源trace不改写，来源只记录哈希。"""
    path, expected = source_trace(tmp_path, stage=stage, response="  \n")
    before = path.read_bytes()
    messages, source = replay.load_source_prompt(path, stage=stage, event_index=1)
    assert messages == expected
    assert source["messages_sha256"] == replay.message_digest(expected)
    assert source["source_trace_sha256"] == hashlib.sha256(before).hexdigest()
    assert source["source_prompt_event_index"] == 1 and source["source_response_event_index"] == 2
    assert source["source_response_classification"] == "whitespace"
    assert "PRIVATE_SOURCE_INPUT" not in json.dumps(source)
    messages[0]["content"] = "本地修改"
    assert path.read_bytes() == before


@pytest.mark.parametrize("fault", ["wrong_stage", "wrong_index", "negative_index", "nonempty_response", "missing_response", "next_prompt"])
def test_source_selection_rejects_wrong_or_nonempty_turn(replay, tmp_path, fault):
    """不猜测其他回合；选择错误、无原始空响应或跨过下一请求时直接拒绝。"""
    path, _ = source_trace(tmp_path)
    payload = json.loads(path.read_text())
    stage, index = "main", 1
    if fault == "wrong_stage":
        stage = "review"
    elif fault == "wrong_index":
        index = 0
    elif fault == "negative_index":
        index = -1
    elif fault == "nonempty_response":
        payload["events"][-1]["content"]["content"] = "合法非空原响应"
    elif fault == "missing_response":
        payload["events"].pop()
    elif fault == "next_prompt":
        payload["events"].insert(2, deepcopy(payload["events"][1]))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        replay.load_source_prompt(path, stage=stage, event_index=index)


class FakeResponse:
    """仅替换HTTP响应体，仍由生产client提取finish、usage和安全诊断。"""

    def __init__(self, content, index):
        self.status = 200
        self.headers = {"x-request-id": f"unit-request-{index}"}
        self.body = json.dumps({"id": f"unit-response-{index}", "model": "unit-current-model",
            "choices": [{"finish_reason": "length" if content == "" else "stop",
                         "message": {"content": content, "reasoning_content": "PRIVATE_REASONING_MUST_NOT_BE_SAVED"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12,
                      "completion_tokens_details": {"reasoning_tokens": 3}}}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.body


def test_finite_production_calls_keep_current_configuration_and_separate_diagnostics(replay, tmp_path):
    """六次生产client调用覆盖空、白、正文和协议异常，凭证及推理正文均不落盘。"""
    path, expected = source_trace(tmp_path)
    before = path.read_bytes()
    messages, source = replay.load_source_prompt(path, stage="main", event_index=1)
    config = AgentLLMConfig(api_key="UNIT_API_KEY_MUST_NOT_BE_SAVED", model="unit-current-model",
        provider="deepseek", base_url="https://example.invalid", max_tokens=6400,
        thinking_mode="enabled", reasoning_effort="high", retry_attempts=1, max_concurrency=3,
        attempt_timeout_seconds=1)
    calls = []
    lock = threading.Lock()
    contents = ["", " \n", '{"goal":"仅返回动作JSON，禁止执行"}', None, "普通正文", ""]

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        with lock:
            index = len(calls)
            calls.append(payload)
        return FakeResponse(contents[index], index)

    client = OpenAICompatibleAgentClient(config=config, urlopen=urlopen)
    run_dir, summary = replay.reproduce_prompt(client, messages, source, output_root=tmp_path / "runs",
                                              repeat=6, workers=3, timeout_seconds=2)
    assert summary["counts"] == {"empty": 2, "whitespace": 1, "non_empty": 2, "error": 1}
    assert summary["transport_summary"] == {"http_attempt_count": 6,
        "finish_reason_counts": {"length": 2, "stop": 4},
        "usage_totals": {"prompt_tokens": 48, "completion_tokens": 24, "total_tokens": 72,
                         "completion_tokens_details.reasoning_tokens": 18}}
    assert len(calls) == 6
    assert all(call["messages"] == expected and call["model"] == config.model and call["max_tokens"] == 6400
               and call["thinking"] == {"type": "enabled"} and call["reasoning_effort"] == "high" for call in calls)
    records = [json.loads(file.read_text()) for file in sorted(run_dir.glob("attempt_*.json"))]
    assert len(records) == 6 and len({record["diagnostics"]["call_id"] for record in records}) == 6
    assert {record["content"] for record in records if "content" in record} == {"", " \n", contents[2], "普通正文"}
    assert next(record for record in records if record["classification"] == "error")["error_type"] == "LLMResponseFormatError"
    saved = "\n".join(file.read_text() for file in run_dir.glob("*.json"))
    assert not any(secret in saved for secret in (config.api_key, "PRIVATE_SOURCE_INPUT", "PRIVATE_REASONING_MUST_NOT_BE_SAVED"))
    assert path.read_bytes() == before and messages == expected
    assert not list(run_dir.glob("*.sqlite3"))
    assert all(file.stat().st_mode & 0o077 == 0 for file in run_dir.glob("*.json"))


@pytest.mark.parametrize("overrides", [{"repeat": 0}, {"repeat": 13}, {"workers": 0}, {"workers": 4},
                                      {"timeout_seconds": 0}, {"timeout_seconds": float("inf")}])
def test_invalid_limits_never_call_model_or_create_run(replay, tmp_path, overrides):
    """有限次数、并发和超时合同在任何远端调用前验证，不能意外启动无界实验。"""
    path, _ = source_trace(tmp_path)
    messages, source = replay.load_source_prompt(path, stage="main", event_index=1)

    class NeverCalled:
        def complete(self, *args, **kwargs):
            pytest.fail("invalid replay must not call a model")

    with pytest.raises(ValueError):
        replay.reproduce_prompt(NeverCalled(), messages, source, output_root=tmp_path / "runs", **overrides)
    assert not (tmp_path / "runs").exists()


def test_repeated_experiments_and_attempts_never_overwrite_original_evidence(replay, tmp_path):
    """同一源可以显式运行两次，但目录独立且旧实验字节不被覆盖；异常不记录原文。"""
    path, _ = source_trace(tmp_path)
    messages, source = replay.load_source_prompt(path, stage="main", event_index=1)

    class FailingClient:
        config = AgentLLMConfig(api_key="EXCEPTION_SECRET", model="unit", base_url="https://example.invalid")

        def complete(self, *args, **kwargs):
            raise RuntimeError("EXCEPTION_SECRET PRIVATE_ERROR_BODY")

    first, summary = replay.reproduce_prompt(FailingClient(), messages, source, output_root=tmp_path / "runs", repeat=1, workers=1)
    original = {file.name: file.read_bytes() for file in first.glob("*.json")}
    second, _ = replay.reproduce_prompt(FailingClient(), messages, source, output_root=tmp_path / "runs", repeat=1, workers=1)
    assert first != second and summary["counts"]["error"] == 1
    assert original == {file.name: file.read_bytes() for file in first.glob("*.json")}
    assert not any(secret in b"".join(original.values()).decode() for secret in ("EXCEPTION_SECRET", "PRIVATE_ERROR_BODY"))


def test_current_client_input_and_diagnostics_are_isolated_per_call(replay, tmp_path):
    """每次复制请求，提供方即使修改输入也不能污染下一次；结果中的凭证正文应扣留。"""
    path, expected = source_trace(tmp_path)
    messages, source = replay.load_source_prompt(path, stage="main", event_index=1)

    class MutatingClient:
        config = AgentLLMConfig(api_key="KEY_MUST_BE_WITHHELD", model="unit", base_url="https://example.invalid")

        def complete(self, current, **kwargs):
            assert current == expected
            current[0]["content"] = "污染本次副本"
            return LLMCompletionText(self.config.api_key, {"attempts": []})

    run_dir, summary = replay.reproduce_prompt(MutatingClient(), messages, source, output_root=tmp_path / "runs", repeat=2, workers=1)
    assert summary["counts"]["non_empty"] == 2 and messages == expected
    record = json.loads((run_dir / "attempt_01.json").read_text())
    assert record["content"] is None and record["content_withheld"] == "configured_credential_detected"
    assert record["content_sha256"] == hashlib.sha256(b"KEY_MUST_BE_WITHHELD").hexdigest()


@pytest.mark.parametrize("echoed_provider", ["primary", "fallback"])
def test_fallback_response_withholds_any_configured_provider_credential(replay, tmp_path, echoed_provider):
    """真实生产client走回退后，无论正文回显哪家已配置凭证，都只能保存摘要与安全诊断。"""
    path, _ = source_trace(tmp_path)
    messages, source = replay.load_source_prompt(path, stage="main", event_index=1)
    keys = {"primary": "PRIMARY_CONFIGURED_CREDENTIAL", "fallback": "FALLBACK_CONFIGURED_CREDENTIAL"}
    fallback = OpenAICompatibleAgentClient(config=AgentLLMConfig(api_key=keys["fallback"], model="unit-fallback",
        base_url="https://example.invalid", retry_attempts=1),
        urlopen=lambda *args, **kwargs: FakeResponse(keys[echoed_provider], 1))

    def failed_primary(*args, **kwargs):
        raise ConnectionError("offline primary")

    client = OpenAICompatibleAgentClient(config=AgentLLMConfig(api_key=keys["primary"], model="unit-primary",
        base_url="https://example.invalid", retry_attempts=1), urlopen=failed_primary, fallback_client=fallback)
    run_dir, summary = replay.reproduce_prompt(client, messages, source, output_root=tmp_path / "runs", repeat=1, workers=1)
    record = json.loads((run_dir / "attempt_01.json").read_text())
    assert summary["counts"] == {"empty": 0, "whitespace": 0, "non_empty": 1, "error": 0}
    assert record["content"] is None and record["content_withheld"] == "configured_credential_detected"
    assert record["content_sha256"] == hashlib.sha256(keys[echoed_provider].encode()).hexdigest()
    assert record["diagnostics"]["route"] == "fallback"
    assert record["diagnostics"]["fallback"]["requested_model"] == "unit-fallback"
    evidence = "\n".join(file.read_text() for file in run_dir.glob("*.json"))
    assert all(key not in evidence for key in keys.values())
