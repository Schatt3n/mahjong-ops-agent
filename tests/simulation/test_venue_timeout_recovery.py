"""验证同一场营业的超时恢复与原始失败证据的边界。"""

from datetime import date
import json

import pytest

from mahjong_agent_runtime.llm import LLMCompletionText
from mahjong_agent_runtime.tracing import JsonlTraceRecorder
from scripts import agent_runtime_app
from scripts import run_venue_simulation as runner
from tests.simulation.venue_actors import MockActor, ModelActor, SemanticJudge
from tests.simulation.venue_channel import AGENT_ID
from tests.simulation.venue_fixture import VenueFixtureAgent
from tests.simulation.venue_timeout_recovery import verified_runtime_timeout_recoveries


class _RoleClient:
    """一轮角色签到请求超时后，原动作仍须走模型输出解析和正式到店入口。"""

    def __init__(self):
        self.raw = "{}"
        self.first_input = None
        self.timeout_inputs = []
        self.sequence = 0

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.sequence += 1
        event = json.loads(messages[-1]["content"])["events"][-1]
        if event["kind"] == "arrival" and self.first_input is None:
            self.first_input = event["event_id"]
            self.timeout_inputs.append(messages)
            raise TimeoutError("角色模型首次读取超时")
        if event["event_id"] == self.first_input:
            self.timeout_inputs.append(messages)
        sequence = self.sequence
        return LLMCompletionText(self.raw, {
            "schema_version": 1, "route": "primary", "call_id": f"llm_call_{sequence:032x}",
            "attempts": [{"attempt_id": f"llm_attempt_{sequence:032x}", "attempt_index": 1,
                          "http_status": 200, "body_complete": True, "response_json_parsed": True, "choice_count": 1,
                          "content": {"present": True, "kind": "string", "length": len(self.raw),
                                      "empty": False, "whitespace_only": False}}],
        })


class _MockResponseActor(ModelActor):
    """本地传输替身保留正式解析、可见上下文和来源核对。"""

    def decide(self, context, *, trace_id):
        self.client.raw = MockActor(42).decide(context, trace_id=trace_id).raw
        return super().decide(context, trace_id=trace_id)


class _JudgeClient:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.requests.append(messages)
        if self.outcome == "unavailable":
            raise ConnectionError("连接异常，不是读取超时")
        if len(self.requests) == 1 or self.outcome == "timeout":
            raise TimeoutError("裁判读取超时")
        request = json.loads(messages[-1]["content"])
        evidence = request["evidence"]
        refusal = next(row for row in evidence if row.get("social_refusal_candidate"))
        findings = []
        for check in request["checks"]:
            outcome = "fail" if check == "business_facts" and self.outcome == "fail" else "pass"
            cited = refusal if check == "social_refusal" else evidence[0]
            findings.append({"check": check, "verdict": outcome, "evidence_ids": [cited["evidence_id"]],
                             "reason": "受控评审"})
        return json.dumps({"verdict": self.outcome, "findings": findings}, ensure_ascii=False)


@pytest.mark.parametrize(("outcome", "status", "judge_requests"), [
    ("pass", "passed", 2), ("fail", "failed", 2), ("timeout", "inconclusive", 2),
    ("unavailable", "inconclusive", 1),
])
def test_four_hour_session_retries_only_timeout_before_actions_and_judgment(tmp_path, monkeypatch,
                                                                        outcome, status, judge_requests):
    """相同证据只补请求，窗内开局散场，最终判决不被超时豁免。"""
    role_client = _RoleClient()
    judge_client = _JudgeClient(outcome)
    fixture = VenueFixtureAgent(seed=42)
    build_runtime, scenario = runner.build_runtime, runner.VenueScenario
    monkeypatch.setattr(runner, "load_dotenv_defaults", lambda path: None)
    monkeypatch.setattr(runner, "_real_roles", lambda: (_MockResponseActor(role_client), SemanticJudge(judge_client)))
    monkeypatch.setattr(runner, "build_runtime",
                        lambda store, mode, **kw: build_runtime(store, "mock", **(kw | {"offline_client": fixture})))
    monkeypatch.setattr(runner, "VenueScenario", lambda **kw: scenario(**(kw | {"fixture_agent": fixture})))

    def classify_private_text(message, *, trace_id, runtime, quiet_period_elapsed):
        """显式提供有效输入分流，让本用例只注入角色与独立评审超时。"""
        assert quiet_period_elapsed and message.text
        return {"action": "process_business", "should_route": True}

    monkeypatch.setattr(agent_runtime_app, "run_wechaty_input_gate", classify_private_text)
    run_dir = tmp_path / outcome
    report = runner.run_simulation(run_dir, mode="real", model_source="project", seed=42,
                                   scenario_date=date(2026, 11, 25), background_events=0)

    assert report["status"] == status, report
    assert report.get("retry_outcome") == ("recovered" if outcome == "pass" else None)
    assert report["physical_play"]["status"] == "passed"
    assert report["checks"]["four_arrivals"] is True
    assert len(role_client.timeout_inputs) == 2
    assert role_client.timeout_inputs[0] == role_client.timeout_inputs[1]
    assert len(judge_client.requests) == judge_requests
    if judge_requests == 2:
        assert judge_client.requests[0] == judge_client.requests[1]
    assert report["model_timeout_retries"][0]["model_stage"] == "role"
    attempts = [json.loads(line) for line in run_dir.joinpath("model_timeout_attempts.jsonl").read_text().splitlines()]
    assert [entry["phase"] for entry in attempts] == ["original", "retry"] * judge_requests
    assert attempts[0]["result"]["action"] is None and attempts[1]["result"]["action"] == {"action": "arrive"}
    events = [json.loads(line) for line in run_dir.joinpath("events.jsonl").read_text().splitlines()]
    assert len([entry for entry in events if entry["event_id"] == role_client.first_input]) == 1


def test_full_lifecycle_reports_runtime_timeout_with_original_delivery(tmp_path, monkeypatch):
    """丢失首次主模型响应后，原来话的实际送达回执和开散场都必须存在。"""
    class OnceTimedOutPlanner(VenueFixtureAgent):
        def __init__(self, *, seed):
            super().__init__(seed=seed)
            self.timed_out = False

        def complete(self, messages, *, trace_id, timeout_seconds):
            message = self._context_payload(messages).get("current_message") or {}
            phase, _ = self.turns.get(message.get("message_id"), ("", {}))
            if phase == "need" and not self.timed_out:
                self.timed_out = True
                raise TimeoutError("组局来话首次模型读取超时")
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(runner, "VenueFixtureAgent", OnceTimedOutPlanner)
    run_dir = tmp_path / "planner-timeout"
    report = runner.run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0)

    assert report["status"] == "passed" and report["retry_outcome"] == "recovered"
    assert report["checks"]["actual_four_hour_finish"] and report["checks"]["four_real_confirmations"]
    recovery, = report["model_timeout_retries"]
    assert recovery["model_stage"] == "agent" and recovery["reason"] == "planner_timeout"
    chat = [json.loads(line) for line in run_dir.joinpath("chat.jsonl").read_text().splitlines()]
    first = next(row for row in chat if row["message_id"] == recovery["message_id"])
    final = next(row for row in chat if row["message_id"] == recovery["reply_message_id"])
    assert final["direction"] == "outbound" and final["recipient_id"] == first["sender_id"]
    assert final["trace_id"] == first["trace_id"] == recovery["trace_id"]


@pytest.mark.parametrize("fault", [
    None, "wrong_run", "wrong_version", "wrong_source", "wrong_recipient", "wrong_conversation",
    "not_a_timeout", "no_final", "wait_only", "old_final", "duplicate_final",
])
def test_runtime_recovery_requires_same_attempt_and_delivered_business_reply(tmp_path, fault):
    """只回复稍等、错绑其他来话、重复外发或只有内部终态时不得作为恢复证据。"""
    trace_path = tmp_path / "agent_trace.log"
    traces = JsonlTraceRecorder(trace_path)
    trace_id = "trace-a"
    identity = {"message_id": "source-a", "run_id": "run-a", "run_version": 1,
                "reason": "runtime_exception" if fault == "not_a_timeout" else "reply_review_timeout"}
    reply = "稍等一下。" if fault == "wait_only" else "好，给你留上了。"
    final = {"reply": reply, "objective_status": "completed"}
    if fault == "old_final":
        traces.record(trace_id, "final_output", final)
    traces.record(trace_id, "agent_runtime_attention", identity | {"retrying": True})
    if fault not in {"old_final", "no_final"}:
        traces.record(trace_id, "final_output", final)
    traces.record(trace_id, "agent_runtime_attention_resolved", identity | {
        "runtime_status": "completed",
        **({"run_id": "different"} if fault == "wrong_run" else {}),
        **({"run_version": 2} if fault == "wrong_version" else {}),
    })
    inbound = {"conversation_id": "chat-a", "kind": "private", "sender_id": "customer-a", "recipient_id": AGENT_ID,
               "direction": "inbound", "sequence": 1, "message_id": "source-a", "trace_id": trace_id}
    outbound = {"conversation_id": "chat-b" if fault == "wrong_conversation" else "chat-a",
                "kind": "private", "sender_id": AGENT_ID,
                "recipient_id": "customer-b" if fault == "wrong_recipient" else "customer-a",
                "direction": "outbound", "sequence": 2, "message_id": "reply-a", "trace_id": trace_id,
                "source_message_id": "source-b" if fault == "wrong_source" else "source-a", "text": reply}
    messages = [inbound, outbound]
    if fault == "duplicate_final":
        messages.append(outbound | {"sequence": 3, "message_id": "reply-duplicate"})

    recovered = verified_runtime_timeout_recoveries(trace_path, messages)
    assert recovered == ([] if fault else [identity | {"trace_id": trace_id, "model_stage": "agent",
                                                       "reply_message_id": "reply-a", "transport_recovered": True}])


@pytest.mark.parametrize("fault", [None, "wrong_private_conversation", "wrong_recipient", "wrong_room", "wrong_version"])
def test_group_topic_recovery_can_use_only_original_senders_authorized_private_route(tmp_path, fault):
    """群转私回复必须是分类成功后的授权独立会话，文本相同也不能给另一位顾客。"""
    trace_path = tmp_path / "agent_trace.log"
    traces = JsonlTraceRecorder(trace_path)
    trace_id = "trace-group"
    identity = {"source_message_id": "group-original", "room_id": "room-a", "batch_id": "batch-a", "batch_version": 1}
    reply = "晚一点那桌还有位，你想来吗？"
    traces.record(trace_id, "group_topic_retry_queued", identity | {"technical_failure": "read_timeout"})
    traces.record(trace_id, "final_output", {"reply": reply, "objective_status": "waiting_user"})
    traces.record(trace_id, "group_topic_retry_finished", identity | {
        "status": "completed", "reason": "group_topic_retry_routed", "route_action": "private_switch",
        "private_conversation_id": "chat-a", **({"batch_version": 2} if fault == "wrong_version" else {}),
    })
    inbound = {"conversation_id": "group-a", "room_id": "room-a", "kind": "group", "sender_id": "customer-a",
               "recipient_id": "room-a", "direction": "inbound", "sequence": 1, "message_id": "group-original",
               "trace_id": trace_id}
    outbound = {"conversation_id": "chat-b" if fault == "wrong_private_conversation" else "chat-a",
                "room_id": "room-b" if fault == "wrong_room" else "room-a", "kind": "private", "sender_id": AGENT_ID,
                "recipient_id": "customer-b" if fault == "wrong_recipient" else "customer-a", "direction": "outbound",
                "sequence": 2, "message_id": "private-reply", "trace_id": trace_id,
                "source_message_id": "group-original", "text": reply}

    recovered = verified_runtime_timeout_recoveries(trace_path, [inbound, outbound])
    assert recovered == ([] if fault else [{"model_stage": "group_topic", "reason": "group_topic_read_timeout",
        "trace_id": trace_id, "message_id": "group-original", "batch_id": "batch-a", "batch_version": 1,
        "reply_message_id": "private-reply", "transport_recovered": True}])
