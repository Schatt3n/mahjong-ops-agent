"""HTTP 交付边界必须与新输入线性化，已交付的历史不能再被标成待废弃输出。"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, UserMessage


@pytest.fixture(params=["memory", "sqlite"])
def delivery_case(request, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "agent_runtime_app.py"
    spec = importlib.util.spec_from_file_location("api_delivery_app", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "delivery.sqlite3")
    version, _ = store.advance_conversation_version("c", trace_id="input", reason="test_input")
    store.append_assistant_turn("c", "旧回复", "reply", metadata={
        "delivery_status": "pending_operator_send", "conversation_version": version, "recipient_id": "customer",
    })
    runtime = SimpleNamespace(store=store, trace_recorder=InMemoryTraceRecorder())
    runtime.input_receipt_lock = lambda conversation_id: store._lock
    handler = object.__new__(module.AgentRuntimeHandler)
    payload = {"trace_id": "reply", "conversation_id": "c", "final_reply": "旧回复", "status": "waiting_user"}
    yield handler, runtime, payload
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def _new_input(store):
    """模拟新输入事务已推进版本后的待交付输出失效处理。"""
    store.advance_conversation_version("c", trace_id="new-input", reason="new_input")
    store.supersede_pending_outputs("c", sender_id="customer", trace_id="new-input", reason="new_input")


def _reply(store):
    return next(turn for turn in store.recent_turns("c") if turn.trace_id == "reply")


def test_old_unsent_api_reply_is_suppressed(delivery_case):
    handler, runtime, payload = delivery_case
    _new_input(runtime.store)
    writes = []
    handler._json = writes.append
    handler._json_runtime_reply(runtime, payload)
    assert writes[0]["final_reply"] == "" and writes[0]["status"] == "superseded"
    assert writes[0]["reply_suppressed"] is True
    assert payload["final_reply"] == "旧回复"
    assert _reply(runtime.store).content == "旧回复"
    assert _reply(runtime.store).metadata["delivery_status"] == "superseded"


def test_new_input_after_delivery_claim_cannot_retract_delivery(delivery_case):
    handler, runtime, payload = delivery_case
    writes = []

    def write(document):
        assert _reply(runtime.store).metadata["delivery_status"] == "dispatching"
        _new_input(runtime.store)
        writes.append(document)

    handler._json = write
    handler._json_runtime_reply(runtime, payload)
    assert writes[0]["final_reply"] == "旧回复"
    assert _reply(runtime.store).metadata["delivery_status"] == "delivered"
    _new_input(runtime.store)
    assert _reply(runtime.store).metadata["delivery_status"] == "delivered"


def test_failed_http_write_is_not_recorded_as_delivered(delivery_case):
    handler, runtime, payload = delivery_case

    def failed_write(document):
        raise BrokenPipeError("test disconnected client")

    handler._json = failed_write
    with pytest.raises(BrokenPipeError):
        handler._json_runtime_reply(runtime, payload)
    assert _reply(runtime.store).metadata["delivery_status"] == "dispatching"
    assert "api_reply_delivered" not in [event.step for event in runtime.trace_recorder.get_trace("reply")]
    handler._json = lambda document: None
    handler._json_runtime_reply(runtime, payload)
    assert _reply(runtime.store).metadata["delivery_status"] == "delivered"


def test_silent_api_result_has_no_fake_delivery_claim(delivery_case):
    handler, runtime, payload = delivery_case
    payload["final_reply"] = ""
    handler._json = lambda document: None
    handler._json_runtime_reply(runtime, payload)
    assert _reply(runtime.store).metadata["delivery_status"] == "pending_operator_send"


def test_one_shot_casual_reply_uses_the_same_delivery_version(delivery_case):
    handler, runtime, _ = delivery_case
    module = sys.modules[handler.__class__.__module__]
    message = UserMessage(conversation_id="c", sender_id="customer", sender_name="客户", text="谢谢")
    version = runtime.store.conversation_version("c")
    metadata = {"delivery_status": "pending_operator_send", "message_type": "casual_chat"}
    assert module._append_current_casual_reply(runtime, message, "不客气", "casual", metadata, source_version=version) == "不客气"
    turn = next(item for item in runtime.store.recent_turns("c") if item.trace_id == "casual")
    assert turn.metadata["conversation_version"] == version
    assert runtime.store.claim_assistant_reply_delivery("c", trace_id="casual") is True
    _new_input(runtime.store)
    assert module._append_current_casual_reply(runtime, message, "过期回复", "late-casual", metadata, source_version=version) == ""
    assert not any(item.trace_id == "late-casual" for item in runtime.store.recent_turns("c"))


def _batch(runtime, text="帮我组一桌", message_id="batch-input"):
    return runtime.store.upsert_pending_input_fragment(
        UserMessage(conversation_id="c", sender_id="customer", sender_name="客户", text=text, message_id=message_id),
        trace_id=message_id, quiet_deadline=datetime.now().astimezone() + timedelta(seconds=30),
    )[0]


def test_api_queued_ack_has_an_independent_durable_reply(delivery_case, monkeypatch):
    handler, runtime, _ = delivery_case
    module = sys.modules[handler.__class__.__module__]
    monkeypatch.setenv("MAHJONG_API_IMMEDIATE_ACK_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_API_BACKGROUND_PROCESSING_ENABLED", "true")
    monkeypatch.setattr(module, "run_wechaty_input_gate", lambda *args, **kwargs: {
        "action": "process_business", "category": "operational", "should_route": True,
    })
    batch = _batch(runtime)
    routed = module.dispatch_pending_input_batch(runtime, batch, trace_id="origin-trace", quiet_period_elapsed=False,
        trigger="api_message_arrived", audit={"channel": "api"})
    ack = routed["agent_result"]
    assert ack["final_reply"] == "好的，我看看" and ack["trace_id"] != "origin-trace"
    writes = []
    handler._json = writes.append
    handler._json_runtime_reply(runtime, ack)
    assert writes[0]["final_reply"] == ack["final_reply"]
    turn = next(item for item in runtime.store.recent_turns("c") if item.trace_id == ack["trace_id"])
    assert turn.metadata["delivery_status"] == "delivered"
    assert turn.metadata["input_window"] == {"batch_id": batch.batch_id, "batch_version": batch.version, "sender_id": "customer"}


def test_failed_delivery_retry_after_new_input_does_not_send_old_text(delivery_case):
    handler, runtime, payload = delivery_case
    handler._json = lambda document: (_ for _ in ()).throw(BrokenPipeError("not written"))
    with pytest.raises(BrokenPipeError):
        handler._json_runtime_reply(runtime, payload)
    _new_input(runtime.store)
    writes = []
    handler._json = writes.append
    handler._json_runtime_reply(runtime, payload)
    assert writes[0]["final_reply"] == "" and writes[0]["status"] == "superseded"
    assert _reply(runtime.store).metadata["delivery_status"] != "delivered"


def _finished_batch(module, runtime):
    batch = _batch(runtime)
    runtime.store.claim_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version, trace_id="batch-result")
    payload = {"trace_id": "batch-result", "conversation_id": "c", "final_reply": "要我帮你问问吗？", "status": "waiting_user"}
    runtime.store.append_assistant_turn("c", payload["final_reply"], payload["trace_id"], metadata={
        "delivery_status": "pending_operator_send", "conversation_version": runtime.store.conversation_version("c"),
        "input_window": {"batch_id": batch.batch_id, "batch_version": batch.version, "sender_id": "customer"},
    })
    runtime.store.finish_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version,
        status=module.PendingInputBatchStatus.COMPLETED, trace_id="batch-result", decision={"agent_result": payload})
    return batch


def test_background_batch_result_uses_delivery_endpoint(delivery_case, monkeypatch):
    handler, runtime, _ = delivery_case
    module = sys.modules[handler.__class__.__module__]
    batch = _finished_batch(module, runtime)
    monkeypatch.setattr(module, "get_runtime", lambda: runtime)
    handler.path = "/api/input-batches/result"
    handler._read_json = lambda: {"batch_id": batch.batch_id}
    writes = []
    handler._json = lambda document, **kwargs: writes.append((document, kwargs.get("status", 200)))
    handler._dispatch_POST()
    assert writes[0][1] == 200 and writes[0][0]["final_reply"] == "要我帮你问问吗？"
    turn = next(item for item in runtime.store.recent_turns("c") if item.trace_id == "batch-result")
    assert turn.metadata["delivery_status"] == "delivered"


def test_old_background_result_cannot_escape_after_later_input(delivery_case, monkeypatch):
    handler, runtime, _ = delivery_case
    module = sys.modules[handler.__class__.__module__]
    batch = _finished_batch(module, runtime)
    _batch(runtime, text="不要问人了", message_id="later-input")
    monkeypatch.setattr(module, "get_runtime", lambda: runtime)
    handler.path = "/api/input-batches/result"
    handler._read_json = lambda: {"batch_id": batch.batch_id}
    writes = []
    handler._json = lambda document, **kwargs: writes.append((document, kwargs.get("status", 200)))
    handler._dispatch_POST()
    assert writes[0][0].get("final_reply", "") == ""
    assert writes[0][1] == 404 or writes[0][0]["status"] == "superseded"


def test_pending_goal_is_reassessed_by_main_instead_of_single_message_casual_gate(delivery_case, monkeypatch):
    handler, runtime, _ = delivery_case
    module = sys.modules[handler.__class__.__module__]
    initial = UserMessage(conversation_id="c", sender_id="customer", sender_name="客户", text="红中1块，帮我组一桌", message_id="initial-goal")
    runtime.store.accept_user_input(initial, trace_id="initial-goal", message_key="pending-user-goal")
    monkeypatch.setenv("MAHJONG_API_BACKGROUND_PROCESSING_ENABLED", "false")
    monkeypatch.setenv("MAHJONG_API_IMMEDIATE_ACK_ENABLED", "false")
    monkeypatch.setattr(module, "run_wechaty_input_gate", lambda *args, **kwargs: pytest.fail("unfinished goal must reach main"))
    calls = []
    def handle(message, *, trace_id):
        calls.append(message)
        return module.AgentRuntimeResult(trace_id=trace_id, conversation_id="c", final_reply="")
    runtime.handle_user_message = handle
    result = module.route_user_message_with_aggregation(runtime,
        UserMessage(conversation_id="c", sender_id="customer", sender_name="客户", text="谢谢", message_id="supplement"),
        trace_id="supplement", channel="api", audit={"channel": "api"})
    assert len(calls) == 1 and calls[0].text == "谢谢"
    assert result["audit"]["input_gate"]["decision_source"] == "unfinished_goal_state"


def test_new_business_input_releases_old_casual_call_before_another_model_task(delivery_case):
    handler, runtime, _ = delivery_case
    module = sys.modules[handler.__class__.__module__]
    runtime.conversation_lock = runtime.input_receipt_lock
    calls = []

    class OldClient:
        def complete(self, messages, **kwargs):
            calls.append(messages)
            runtime.store.accept_user_input(
                UserMessage(conversation_id="c", sender_id="customer", sender_name="客户", text="改成晚上", message_id="new-goal"),
                trace_id="new-goal", message_key="new-goal",
            )
            return "{}"  # 已过期的输出不应再解析、改写或送去第二次模型审查。

    runtime.llm_client = OldClient()
    result = module.handle_wechaty_casual_chat(
        UserMessage(conversation_id="c", sender_id="customer", sender_name="客户", text="谢谢", message_id="old-casual"),
        trace_id="old-casual", runtime=runtime, gate_decision={"category": "casual_chat"},
    )
    assert result.status == "superseded" and result.final_reply == "" and len(calls) == 1
    assert "wechaty_casual_chat_decision" not in [event.step for event in runtime.trace_recorder.get_trace("old-casual")]
