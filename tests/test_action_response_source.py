"""逐次返回值的来源必须贯穿拒绝、反馈、下一上下文与持久恢复，不改变执行合同。"""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from uuid import UUID
from threading import Barrier

import pytest

from mahjong_agent_runtime import AgentAction, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.budget import TokenBudget
from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_payload
from mahjong_agent_runtime.llm import (
    AgentLLMConfig, LLMCompletionText, OpenAICompatibleAgentClient, compact_model_response_source,
    completion_diagnostics, model_response_source, validate_model_response_source,
)
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.services.run_state_service import AgentRunStateManager


def _id(prefix, number):
    return prefix + UUID(int=number).hex


def _diagnostics(raw="", number=1):
    return {"schema_version": 1, "call_id": _id("llm_call_", number), "route": "primary",
        "provider": "DIAGNOSTIC_PRIVATE_CANARY", "request": {"max_tokens": 16384},
        "reasoning_content": "DIAGNOSTIC_PRIVATE_CANARY", "authorization": "DIAGNOSTIC_PRIVATE_CANARY",
        "attempts": [{"attempt_id": _id("llm_attempt_", number), "attempt_index": 1,
            "http_status": 200, "body_complete": True, "response_json_parsed": True, "choice_count": 1,
            "finish_reason": "stop", "content": {"present": True, "kind": "string", "length": len(raw),
                "empty": raw == "", "whitespace_only": bool(raw) and not raw.strip()}}]}


def _text(raw="", number=1):
    return LLMCompletionText(raw, _diagnostics(raw, number))


def _fallback_diagnostics(returned, number=1, *, queued_failure=False):
    primary = _diagnostics(number=number)
    primary["error_type"] = "TimeoutError"
    if queued_failure:
        primary["attempts"] = []
    else:
        primary["attempts"][0].update(body_complete=False, response_json_parsed=False, error_type="TimeoutError")
    return {"schema_version": 1, "call_id": primary["call_id"], "route": "fallback",
            "primary": primary, "fallback": returned}


def _source(raw):
    return model_response_source(raw, trace_id="source-trace", step_index=1)


def _valid(source, raw, **kwargs):
    return validate_model_response_source(source, raw_response=raw, trace_id="source-trace", step_index=1, **kwargs)


def _waiting():
    return json.dumps(AgentAction(goal="询问当前未知条件", objective_status="waiting_user",
        reasoning_summary="等待客户提供玩法", reply_to_user="想打什么玩法？",
        stop_reason={"can_stop": True, "why": "缺少玩法", "pending_work": [], "depends_on_tool_results": False}).to_dict(), ensure_ascii=False)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    current = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "source.sqlite3")
    yield current
    if isinstance(current, SQLiteAgentStore):
        current._connection.close()


def test_source_has_only_safe_identity_and_hash_without_full_diagnostics():
    raw = _text()
    source = _source(raw)
    assert source == {"schema_version": 1, "trace_id": "source-trace", "step_index": 1,
        "raw_response_sha256": hashlib.sha256(b"").hexdigest(), "call_id": _id("llm_call_", 1),
        "terminal_attempt_id": _id("llm_attempt_", 1), "route": "primary", "response_route": [],
        "http_response_complete": True}
    assert _valid(source, raw) and _valid(source, "", diagnostics=completion_diagnostics(raw), require_http=True)
    assert "CANARY" not in json.dumps(source)


@pytest.mark.parametrize("raw", ["not-json", " \n\t", "答复"])
def test_source_validates_exact_nonempty_text_and_rejects_modified_content(raw):
    response = _text(raw)
    source = _source(response)
    assert _valid(source, raw, diagnostics=completion_diagnostics(response))
    assert not _valid(source, raw + "changed", diagnostics=completion_diagnostics(response))
    assert not validate_model_response_source(source, raw_response=response, trace_id="different", step_index=1)
    assert not validate_model_response_source(source, raw_response=response, trace_id="source-trace", step_index=2)


def test_plain_static_text_has_text_identity_but_no_http_provenance():
    source = _source("not-json")
    assert source["call_id"] is None and source["terminal_attempt_id"] is None
    assert source["route"] is None and source["response_route"] is None and not source["http_response_complete"]
    assert _valid(source, "not-json")
    assert not _valid(source, "not-json", require_http=True)
    assert not _valid(_source(""), "")
    assert not _valid(_source(_text()), "", diagnostics={})


@pytest.mark.parametrize("fault", ["call_id", "attempt_id", "attempt_index", "duplicate_attempt", "http_status",
    "body_complete", "response_json_parsed", "content_length", "content_missing", "content_null", "content_empty_type",
    "terminal_error", "schema_bool", "length_bool"])
@pytest.mark.parametrize("raw", ["", "not-json"])
def test_incomplete_or_inconsistent_diagnostics_never_upgrade_to_http_source(fault, raw):
    diag = _diagnostics(raw)
    attempt = diag["attempts"][0]
    if fault == "call_id": diag["call_id"] = "diagnostic-only-main"
    elif fault == "attempt_id": attempt["attempt_id"] = "diagnostic-only-attempt"
    elif fault == "attempt_index": attempt["attempt_index"] = True
    elif fault == "duplicate_attempt":
        diag["attempts"].append(deepcopy(attempt)); diag["attempts"][-1]["attempt_index"] = 2
    elif fault == "http_status": attempt["http_status"] = 500
    elif fault in {"body_complete", "response_json_parsed"}: attempt[fault] = False
    elif fault == "content_length": attempt["content"]["length"] = len(raw) + 1
    elif fault == "content_missing": attempt["content"]["present"] = False
    elif fault == "content_null": attempt["content"] = {"present": True, "kind": "null", "length": None, "empty": None, "whitespace_only": None}
    elif fault == "content_empty_type": attempt["content"]["empty"] = int(raw == "")
    elif fault == "terminal_error": attempt["error_type"] = "TimeoutError"
    elif fault == "schema_bool": diag["schema_version"] = True
    elif fault == "length_bool": attempt["content"]["length"] = False
    response = LLMCompletionText(raw, diag)
    assert _source(response)["http_response_complete"] is False
    assert not _valid(_source(response), response)


def test_retry_uses_only_last_complete_successful_attempt():
    diag = _diagnostics()
    first = diag["attempts"][0]
    first.update(body_complete=False, response_json_parsed=False, error_type="TimeoutError")
    second = _diagnostics(number=2)["attempts"][0]
    second["attempt_index"] = 2
    diag["attempts"].append(second)
    response = LLMCompletionText("", diag)
    assert _source(response)["terminal_attempt_id"] == _id("llm_attempt_", 2)
    assert _valid(_source(response), response)
    diag["attempts"][-1]["error_type"] = "TimeoutError"
    bad = LLMCompletionText("", diag)
    assert not _valid(_source(bad), bad)


def test_fallback_source_binds_actual_returning_leaf_not_failed_primary():
    leaf = _diagnostics(number=2)
    diag = _fallback_diagnostics(leaf)
    response = LLMCompletionText("", diag)
    source = _source(response)
    assert source["call_id"] == _id("llm_call_", 2) and source["terminal_attempt_id"] == _id("llm_attempt_", 2)
    assert source["route"] == "fallback" and source["response_route"] == ["fallback"]
    assert _valid(source, response)
    diag["fallback"] = _diagnostics("not-empty", number=2)
    mismatch = LLMCompletionText("", diag)
    assert not _valid(_source(mismatch), mismatch)
    diag["fallback"] = {}
    unknown = LLMCompletionText("", diag)
    assert not _valid(_source(unknown), unknown)


@pytest.mark.parametrize("fault", ["no_primary", "outer_id_mismatch", "successful_primary", "primary_schema",
    "primary_route", "leaf_reuses_call_id", "leaf_reuses_attempt_id", "failed_wrapper", "malformed_attempts"])
def test_fallback_requires_exact_failed_primary_provenance_and_distinct_returning_identity(fault):
    diag = _fallback_diagnostics(_diagnostics(number=2))
    if fault == "no_primary": del diag["primary"]
    elif fault == "outer_id_mismatch": diag["call_id"] = _id("llm_call_", 9)
    elif fault == "successful_primary": diag["primary"] = _diagnostics(number=1)
    elif fault == "primary_schema": diag["primary"]["schema_version"] = True
    elif fault == "primary_route": diag["primary"]["route"] = "fallback"
    elif fault == "leaf_reuses_call_id": diag["fallback"]["call_id"] = diag["call_id"]
    elif fault == "leaf_reuses_attempt_id": diag["fallback"]["attempts"][0]["attempt_id"] = diag["primary"]["attempts"][0]["attempt_id"]
    elif fault == "failed_wrapper": diag["error_type"] = "RuntimeError"
    elif fault == "malformed_attempts": diag["primary"]["attempts"] = None
    response = LLMCompletionText("", diag)
    assert not _source(response)["http_response_complete"]
    assert not _valid(_source(response), response)


def test_queued_primary_failure_without_http_attempt_can_have_valid_fallback():
    response = LLMCompletionText("", _fallback_diagnostics(_diagnostics(number=2), queued_failure=True))
    assert _valid(_source(response), response)
    assert _source(response)["call_id"] == _id("llm_call_", 2)


def test_nested_fallback_checks_every_primary_binding_and_identity():
    diag = _fallback_diagnostics(_fallback_diagnostics(_diagnostics(number=3), number=2), number=1)
    response = LLMCompletionText("", diag)
    assert _source(response)["response_route"] == ["fallback", "fallback"]
    assert _source(response)["call_id"] == _id("llm_call_", 3) and _valid(_source(response), response)
    for fault in ("no_inner_failure", "inner_outer_mismatch", "inner_call_reused", "inner_attempt_reused"):
        broken = deepcopy(diag)
        if fault == "no_inner_failure": del broken["fallback"]["primary"]["error_type"]
        elif fault == "inner_outer_mismatch": broken["fallback"]["primary"]["call_id"] = _id("llm_call_", 99)
        elif fault == "inner_call_reused":
            broken["fallback"]["call_id"] = broken["call_id"]
            broken["fallback"]["primary"]["call_id"] = broken["call_id"]
        else:
            broken["fallback"]["primary"]["attempts"][0]["attempt_id"] = broken["primary"]["attempts"][0]["attempt_id"]
        invalid = LLMCompletionText("", broken)
        assert not _valid(_source(invalid), invalid)


def test_source_compaction_is_allowlisted_detached_and_rejects_malformed_identity():
    source = _source(_text())
    polluted = {**source, "raw_response": "PRIVATE_RAW", "reasoning_content": "PRIVATE_REASONING",
                "completion_diagnostics": {"authorization": "PRIVATE_AUTH"}}
    compacted = compact_tool_payload({"response_source": polluted})["response_source"]
    assert compacted == source and "PRIVATE_" not in json.dumps(compacted)
    compacted["response_route"].append("fallback")
    assert source["response_route"] == []
    assert not _valid(polluted, _text())
    for key, value in (("schema_version", True), ("step_index", True), ("route", []), ("call_id", "PRIVATE_AUTH"),
                       ("raw_response_sha256", "PRIVATE_RAW"), ("http_response_complete", 1)):
        assert compact_model_response_source({**source, key: value}) is None


class HTTPResponse:
    status = 200
    headers = {}

    def __init__(self, raw):
        self.body = json.dumps({"model": "test-model", "choices": [{"finish_reason": "stop",
            "message": {"content": raw, "reasoning_content": "PRIVATE_HTTP_REASONING"}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


def _http_client(urlopen):
    return OpenAICompatibleAgentClient(config=AgentLLMConfig(api_key="private-test-auth", model="test-model",
        base_url="https://unreachable.invalid", provider="deepseek", retry_attempts=1, max_concurrency=2), urlopen=urlopen)


def test_real_http_parse_result_has_source_before_action_parsing_without_retry():
    calls = []
    client = _http_client(lambda request, **kwargs: calls.append(request) or HTTPResponse(""))
    raw = client.complete([{"role": "user", "content": "only-test"}], trace_id="source-trace", timeout_seconds=3)
    source = _source(raw)
    assert raw == "" and source["http_response_complete"] and _valid(source, raw)
    assert len(calls) == 1 and client.concurrency_metrics()["http_retries"] == 0
    assert "PRIVATE_HTTP_REASONING" not in json.dumps(source)
    assert "private-test-auth" not in json.dumps(source)


@pytest.mark.parametrize("queue_failure", [False, True])
def test_real_client_fallback_wrapper_produces_valid_returning_leaf_source(queue_failure):
    def primary_failure(*args, **kwargs):
        raise TimeoutError("synthetic primary transport timeout")

    primary = _http_client(primary_failure)
    primary.fallback_client = _http_client(lambda *args, **kwargs: HTTPResponse(""))
    if queue_failure:
        class UnavailableSlot:
            def acquire(self, **kwargs):
                return False
        primary._concurrency_gate = UnavailableSlot()
    raw = primary.complete([{"role": "user", "content": "fallback-test"}], trace_id="source-trace", timeout_seconds=3)
    diagnostics = completion_diagnostics(raw)
    source = _source(raw)
    assert source["route"] == "fallback" and source["response_route"] == ["fallback"]
    assert diagnostics["primary"]["call_id"] == diagnostics["call_id"]
    assert diagnostics["primary"]["error_type"]
    assert source["call_id"] == diagnostics["fallback"]["call_id"] != diagnostics["call_id"]
    assert _valid(source, raw)
    if queue_failure:
        assert diagnostics["primary"]["attempts"] == []


def test_shared_client_concurrent_completions_never_borrow_other_response_identity():
    barrier = Barrier(2)

    def urlopen(request, **kwargs):
        marker = json.loads(request.data)["messages"][0]["content"]
        barrier.wait(timeout=3)
        return HTTPResponse("" if marker == "empty" else "not-json")

    client = _http_client(urlopen)

    def call(marker):
        trace_id = "source-" + marker
        raw = client.complete([{"role": "user", "content": marker}], trace_id=trace_id, timeout_seconds=5)
        source = model_response_source(raw, trace_id=trace_id, step_index=1)
        assert validate_model_response_source(source, raw_response=raw, trace_id=trace_id, step_index=1)
        return raw, source

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(call, ("empty", "nonempty")))
    assert [str(raw) for raw, _ in result] == ["", "not-json"]
    assert len({source["call_id"] for _, source in result}) == 2
    assert len({source["terminal_attempt_id"] for _, source in result}) == 2
    assert result[0][1]["raw_response_sha256"] != result[1][1]["raw_response_sha256"]


class RecordingClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.contexts = []

    def complete(self, messages, **kwargs):
        self.contexts.append(json.loads(messages[-1]["content"]))
        return self.outputs.pop(0)


def test_model_step_preserves_return_object_and_observation_sources_are_independent(store):
    raw = _text()
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(store=store, llm_client=RecordingClient([raw]), trace_recorder=trace)
    message = UserMessage(conversation_id="source-chat", sender_id="customer", sender_name="客户", text="帮我看看")
    step = runtime.action_processor.call_agent_action(message, trace_id="source-trace", built_messages=[{"role": "user", "content": "{}"}],
        step_index=1, budget=TokenBudget(max_tokens_per_call=64000), run_id="source-run", run_version=0)
    assert step.raw_response is raw and step.errors
    feedback = runtime.action_processor.record_action_contract_feedback(message, trace_id="source-trace", raw_response=step.raw_response,
        errors=step.errors, step_index=1)[0]
    expected = _source(raw)
    assert all(event.content["response_source"] == expected for event in trace.events if event.step in {"llm_response", "action_contract_error"})
    assert next(event.content for event in trace.events if event.step == "contract_error_feedback")["result"]["response_source"] == expected
    feedback.result["response_source"]["call_id"] = _id("llm_call_", 99)
    assert all(event.content["response_source"] == expected for event in trace.events if event.step in {"llm_response", "action_contract_error"})
    assert next(event.content for event in trace.events if event.step == "contract_error_feedback")["result"]["response_source"] == expected


def test_consecutive_empty_responses_keep_distinct_exact_sources_through_real_runtime(store):
    trace = InMemoryTraceRecorder()
    client = RecordingClient([_text(number=1), _text(number=2), _text(_waiting(), number=3)])
    runtime = AgentRuntime(store=store, llm_client=client, trace_recorder=trace)
    result = runtime.handle_user_message(UserMessage(conversation_id="source-chat", sender_id="customer", sender_name="客户",
        text="帮我看看", message_id="source-input"), trace_id="source-trace")
    assert result.status == "waiting_user" and len(client.contexts) == 3
    responses = [event.content for event in trace.events if event.step == "llm_response"]
    errors = [event.content for event in trace.events if event.step == "action_contract_error"]
    feedback = [event.content for event in trace.events if event.step == "contract_error_feedback"]
    assert len(errors) == len(feedback) == 2
    for index in range(2):
        source = responses[index]["response_source"]
        assert errors[index]["response_source"] == feedback[index]["result"]["response_source"] == source
        assert validate_model_response_source(source, raw_response="", trace_id="source-trace", step_index=index + 1,
            diagnostics=responses[index]["completion_diagnostics"])
        matching = [item for item in client.contexts[index + 1]["turn_tool_evidence"]
                    if item["name"] == "agent_action_contract" and item["result"]["response_source"] == source]
        assert len(matching) == 1
    assert errors[0]["errors"] == errors[1]["errors"]
    assert errors[0]["response_source"]["raw_response_sha256"] == errors[1]["response_source"]["raw_response_sha256"]
    assert errors[0]["response_source"]["call_id"] != errors[1]["response_source"]["call_id"]
    assert "DIAGNOSTIC_PRIVATE_CANARY" not in json.dumps(client.contexts)
    assert not store.games and not store.invite_drafts and not store.outbound_message_drafts
    turns = [turn.to_dict() for turn in store.recent_turns("source-chat", limit=100)]
    assert "DIAGNOSTIC_PRIVATE_CANARY" not in json.dumps(turns)


def test_source_survives_post_rejection_checkpoint_and_restart(store, monkeypatch):
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(store=store, llm_client=RecordingClient([_text()]), trace_recorder=trace)
    original = AgentRunStateManager.checkpoint

    def checkpoint_then_crash(self, state, **kwargs):
        saved = original(self, state, **kwargs)
        if self is runtime.run_state_manager and kwargs["next_step_index"] == 2:
            raise RuntimeError("crash after rejection checkpoint")
        return saved

    monkeypatch.setattr(AgentRunStateManager, "checkpoint", checkpoint_then_crash)
    with pytest.raises(RuntimeError, match="crash after rejection checkpoint"):
        runtime.handle_user_message(UserMessage(conversation_id="source-chat", sender_id="customer", sender_name="客户",
            text="帮我看看", message_id="recover-source"), trace_id="source-trace")
    expected = next(event.content["response_source"] for event in trace.events if event.step == "llm_response")
    current = SQLiteAgentStore(store.path) if isinstance(store, SQLiteAgentStore) else store
    try:
        state = current.recoverable_agent_runs(at=now(), limit=5)[0]
        assert state.turn_tool_evidence[0]["result"]["response_source"] == expected
        assert "DIAGNOSTIC_PRIVATE_CANARY" not in json.dumps(state.turn_tool_evidence)
        client = RecordingClient([_waiting()])
        restored = AgentRuntime(store=current, llm_client=client, trace_recorder=InMemoryTraceRecorder())
        results = restored.resume_recoverable_runs()
        assert len(results) == 1 and results[0].status == "waiting_user"
        assert client.contexts[0]["turn_tool_evidence"][0]["result"]["response_source"] == expected
        assert "DIAGNOSTIC_PRIVATE_CANARY" not in json.dumps(client.contexts)
        assert not current.games and not current.invite_drafts and not current.outbound_message_drafts
    finally:
        if current is not store:
            current._connection.close()
