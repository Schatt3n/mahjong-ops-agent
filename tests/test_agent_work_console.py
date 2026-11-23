"""验证运行期间的进度、合批关联、发送证据和控制台脱敏。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import (
    AgentLLMConfig, CustomerProfile, Game, InMemoryAgentStore, JsonlTraceRecorder,
    OpenAICompatibleAgentClient,
)
from mahjong_agent_runtime.group_chat import ChannelIdentity
from mahjong_agent_runtime.models import GameParticipant
from scripts.agent_work_console import agent_work_snapshot, delivery_observation


def test_recent_inbound_follows_background_run_before_and_after_reply(tmp_path):
    trace = JsonlTraceRecorder(tmp_path / "trace.log")
    store = InMemoryAgentStore()
    runtime = SimpleNamespace(trace_recorder=trace, store=store)
    store.upsert_channel_observation({
        "channel": "wechaty", "source_message_id": "m1", "trace_id": "origin",
        "sender_name": "测试联系人", "text": "今晚还有位置吗？",
        "route_status": "queued", "route_mode": "background_agent",
        "payload": {"route_result": {"input_batch": {"batch_id": "batch1"}}},
    })
    trace.record("origin", "input_batch_background_queued", {"batch": {"batch_id": "batch1"}})
    trace.record("origin", "customer_visible_immediate_ack", {"reply": "我看一下。"})
    trace.record("origin", "wechaty_bridge_delivery", {
        "kind": "ack", "delivery_id": "ack-attempt", "outcome": "confirmed", "message_id": "ack1",
    })
    trace.record("other", "input_background_dispatch_due", {"batch_id": "different"})
    trace.record("other", "final_output", {"reply": "这是另一个人的回复"})
    trace.record("worker", "input_background_dispatch_due", {"batch_id": "batch1"})
    trace.record("worker", "llm_http_attempt_started", {"model": "test-model", "attempt": 1})
    active = agent_work_snapshot(runtime, selected="wechaty:m1")["detail"]
    assert active["phase"] == "等待模型 API 返回"
    assert active["call_started_at"]
    assert active["trace_ids"] == ["origin", "worker"]
    assert active["deliveries"]["ack"]["status"] == "confirmed"
    assert active["reply"] == ""
    assert "另一个人" not in json.dumps(active, ensure_ascii=False)

    trace.record("worker", "llm_http_retry_scheduled", {"model": "test-model", "attempt": 1, "error_type": "TimeoutError"})
    retrying = agent_work_snapshot(runtime)["detail"]
    assert retrying["phase"] == "模型请求失败，等待重试"
    trace.record("worker", "llm_response", {"content": "隐藏模型原文", "elapsed_ms": 1030})
    trace.record("worker", "action_proposed", {
        "goal": "查询空位", "reasoning_summary": "当前客户想了解当晚空位",
        "tool_calls": [{"name": "search_current_games", "arguments": {"private": "隐藏工具参数"}}],
    })
    trace.record("worker", "tool_result", {"name": "search_current_games", "allowed": True,
        "result": {"remaining_seats": 1, "private_notes": "隐藏内部备注"}})
    trace.record("worker", "final_output", {"reply": "还剩一个位置。"})
    trace.record("worker", "delayed_reply_delivery_started", {"delivery_id": "reply-attempt"})
    submitted = agent_work_snapshot(runtime)["detail"]
    assert submitted["reply"] == "还剩一个位置。"
    assert submitted["deliveries"]["reply"]["status"] == "sending"
    assert submitted["call_started_at"] is None
    serialized = json.dumps(submitted, ensure_ascii=False)
    assert "查询空位" in serialized and "remaining_seats=1" in serialized and "隐藏" not in serialized

    trace.record("worker", "delayed_reply_delivered", {"delivery_id": "reply-attempt", "sent": True, "response": {"message_id": "wx-reply-2"}})
    confirmed = agent_work_snapshot(runtime)["detail"]
    assert confirmed["phase"] == "微信通道已确认"
    assert confirmed["deliveries"]["reply"]["message_id"] == "wx-reply-2"
    assert len(confirmed["delivery_attempts"]) == 2


def test_multiple_sends_in_one_trace_keep_business_reply_and_notification_separate(tmp_path):
    trace = JsonlTraceRecorder(tmp_path / "trace.log")
    store = InMemoryAgentStore()
    store.upsert_channel_observation({
        "channel": "wechaty", "source_message_id": "group-message", "trace_id": "group-run",
        "room_topic": "测试群", "is_room": True, "text": "还有位吗？",
    })
    trace.record("group-run", "final_output", {"reply": "业务答复"})
    trace.record("group-run", "managed_group_outbound_delivered", {
        "delivery_id": "reply-1", "platform_message_id": "wx-1", "text": "业务答复", "target_type": "room",
        "metadata": {"message_type": "agent_reply", "source_message_id": "group-message"},
    })
    trace.record("group-run", "system_trigger_received", {})
    trace.record("group-run", "final_output", {"reply": "另一位顾客的通知"})
    trace.record("group-run", "system_trigger_completed", {})
    trace.record("group-run", "managed_group_outbound_started", {"delivery_id": "notice-1", "target_type": "contact"})
    trace.record("group-run", "managed_group_outbound_suppressed", {
        "delivery_id": "notice-1", "reason": "managed_room_outbound_disabled", "target_type": "contact",
    })
    detail = agent_work_snapshot(SimpleNamespace(store=store, trace_recorder=trace))["detail"]
    assert detail["reply"] == "业务答复"
    assert detail["phase"] == "微信通道已确认"
    assert detail["deliveries"]["reply"]["message_id"] == "wx-1"
    assert [(item["delivery_id"], item["kind"], item["status"]) for item in detail["delivery_attempts"]] == [
        ("reply-1", "reply", "confirmed"), ("notice-1", "notification", "blocked"),
    ]


def test_group_timeline_separates_routing_query_and_review(tmp_path):
    trace = JsonlTraceRecorder(tmp_path / "trace.log")
    store = InMemoryAgentStore()
    store.upsert_channel_observation({"channel": "wechaty", "source_message_id": "group-input", "trace_id": "group-run",
        "room_topic": "测试群", "is_room": True, "route_mode": "managed_group", "text": "还有位吗？"})
    for step, payload in [
        ("llm_http_retry_scheduled", {"attempt": 1, "error_type": "URLError"}),
        ("group_semantic_routing_completed", {"action": "agent_loop"}),
        ("llm_request_started", {}), ("llm_http_attempt_started", {"attempt": 1}),
        ("tool_result", {"name": "search_current_games", "allowed": True, "result": {"matches": [], "alternatives": []}}),
        ("llm_request_started", {}), ("llm_http_attempt_started", {"attempt": 1}),
        ("customer_visible_content_review_prompt", {}), ("llm_http_attempt_started", {"attempt": 1}),
    ]:
        trace.record("group-run", step, payload)
    detail = agent_work_snapshot(SimpleNamespace(store=store, trace_recorder=trace))["detail"]
    displayed = {item["label"]: item["detail"] for item in detail["timeline"]}
    assert "群消息分流 · 模型请求失败，等待重试" in displayed
    assert "群消息分流完成" in displayed
    assert "主模型决策 · 等待模型 API 返回" in displayed
    assert displayed["查局完成"] == "匹配 0 局 · 备选 0 局"
    assert "处理查局结果 · 等待模型 API 返回" in displayed
    assert detail["phase"] == "回复审查 · 等待模型 API 返回"


def test_group_timeline_follows_contract_retry_request_trace(tmp_path):
    """合同纠错的真实请求与首次请求同属当前群消息的时间线。"""
    trace = JsonlTraceRecorder(tmp_path / "trace.log")
    store = InMemoryAgentStore()
    store.upsert_channel_observation({"channel": "wechaty", "source_message_id": "group-input", "trace_id": "group-run",
        "is_room": True, "route_mode": "managed_group", "text": "还有位吗？"})
    for attempt, request_trace in enumerate(("group-run", "group-run_contract_retry_1")):
        trace.record("group-run", "group_semantic_routing_prompt", {
            "messages": [], "step_index": attempt, "request_trace_id": request_trace})
        trace.record(request_trace, "llm_http_attempt_started", {"attempt": 1, "model": "fixture"})
    detail = agent_work_snapshot(SimpleNamespace(store=store, trace_recorder=trace))["detail"]
    assert sum("等待模型 API 返回" in event["label"] for event in detail["timeline"]) == 2


def test_silent_group_registration_finishes_model_wait_and_links_persisted_game(tmp_path):
    trace = JsonlTraceRecorder(tmp_path / "trace.log")
    store = InMemoryAgentStore()
    runtime = SimpleNamespace(store=store, trace_recorder=trace)
    store.upsert_channel_observation({
        "channel": "wechaty", "source_message_id": "group-input", "trace_id": "group-run",
        "room_topic": "测试群", "is_room": True, "text": "明天晚上1人无烟",
    })
    trace.record("group-run", "llm_http_attempt_started", {"model": "fixture", "attempt": 1})
    assert agent_work_snapshot(runtime)["detail"]["call_started_at"]
    game = Game(game_id="game_registered", conversation_id="internal-chat", organizer_id="private-customer",
        organizer_name="私人姓名", requirement={"smoke_preference": "no_smoking", "recruitment_mode": "board_only"})
    store.games[game.game_id] = game
    trace.record("group-run", "wechaty_managed_group_message_processed",
                 {"result": {"action": "register_self_recruitment", "game_id": game.game_id, "reply": ""}})
    snapshot = agent_work_snapshot(runtime)
    assert snapshot["detail"]["phase"] == "已静默登记"
    assert snapshot["detail"]["call_started_at"] is None
    assert snapshot["detail"]["reply"] == ""
    assert snapshot["detail"]["game_id"] == snapshot["games"][0]["game_id"] == game.game_id
    assert snapshot["games"][0]["status"] == "forming"
    assert snapshot["games"][0]["source"] == "群内自招"
    assert "private-customer" not in str(snapshot["games"])


def test_game_board_displays_known_companions_without_turning_unregistered_seats_into_people(tmp_path):
    store = InMemoryAgentStore()
    trace = JsonlTraceRecorder(tmp_path / "trace.log")
    store.upsert_customer(CustomerProfile(
        customer_id="host-test-id", display_name="内部备注", public_name="桌主小岚",
        private_remark="内部备注", notes="私有记录", preferred_games=["测试偏好"],
    ))
    store.upsert_channel_identity(ChannelIdentity(
        channel="wechaty", external_user_id="wxid_test_companion", customer_id="companion-test-id",
        public_name="同行小文", private_conversation_id="private-test-conversation",
    ))
    store.games["fixture-roster"] = Game(
        game_id="fixture-roster", conversation_id="fixture-conversation",
        organizer_id="host-test-id", organizer_name="内部备注", requirement={},
        participants=[
            GameParticipant("host-test-id", "内部备注", status="joined", seat_count=3,
                            known_member_ids=["host-test-id", "companion-test-id"], anonymous_seat_count=1),
            GameParticipant("confirmed-test-id", "已确认的客人", status="confirmed"),
            GameParticipant("declined-test-id", "已退出的客人", status="declined"),
        ],
    )
    board = agent_work_snapshot(SimpleNamespace(store=store, trace_recorder=trace))["games"][0]
    roster = board["roster"]
    assert roster["organizer"] == {"name": "桌主小岚", "is_seated": True}
    group, confirmed, declined = roster["participants"]
    assert group["status"] == "joined" and group["seats"] == 3
    assert group["named_companions"] == [{"name": "同行小文"}]
    assert group["anonymous_seats"] == 1
    assert confirmed["status"] == "confirmed" and confirmed["seats"] == 1
    assert declined["status"] == "declined" and declined["seats"] == 0
    assert sum(row["seats"] for row in roster["participants"]) == board["claimed_seats"] == 4
    payload = json.dumps(roster, ensure_ascii=False)
    assert all(secret not in payload for secret in (
        "wxid_test_companion", "host-test-id", "内部备注", "私有记录", "测试偏好", "private-test-conversation",
    ))


def test_http_observer_sees_attempt_before_transport_and_retry_before_next_attempt():
    observed = []

    def transport(request, *, timeout):
        assert observed[-1][1] == "llm_http_attempt_started"
        raise TimeoutError("sensitive upstream detail")

    config = AgentLLMConfig(api_key="private-fixture-key", model="fixture-model",
                           base_url="https://example.test", retry_attempts=2, retry_base_delay_seconds=0)
    client = OpenAICompatibleAgentClient(config, urlopen=transport, request_observer=lambda *args: observed.append(args))
    with pytest.raises(RuntimeError):
        client.complete([{"role": "user", "content": "private-body"}], trace_id="test-trace", timeout_seconds=5)
    assert [event[1] for event in observed] == [
        "llm_http_attempt_started", "llm_http_retry_scheduled", "llm_http_attempt_started",
    ]
    payloads = [event[2] for event in observed]
    assert [event["attempt"] for event in payloads] == [1, 1, 2]
    assert payloads[1]["error_type"] == "TimeoutError"
    assert len({event["call_id"] for event in payloads}) == 1
    assert "private" not in str(payloads) and "sensitive" not in str(payloads)


def test_bridge_cannot_claim_confirmed_send_without_platform_id():
    with pytest.raises(ValueError, match="receipt"):
        delivery_observation({"trace_id": "origin", "kind": "reply", "delivery_id": "attempt-1", "outcome": "confirmed"})
    _, data = delivery_observation({"trace_id": "origin", "kind": "reply", "delivery_id": "attempt-1", "outcome": "blocked",
        "reason": "send_channel_disabled", "arbitrary": "ignored"})
    assert data == {"kind": "reply", "delivery_id": "attempt-1", "outcome": "blocked", "reason": "send_channel_disabled", "message_id": ""}


def test_selected_model_io_keeps_full_prompt_response_and_redacts_credentials(tmp_path):
    """调试正文必须对应选中 Trace，列表与默认摘要不携带模型私有上下文。"""
    trace = JsonlTraceRecorder(tmp_path / "trace.log")
    store = InMemoryAgentStore()
    runtime = SimpleNamespace(store=store, trace_recorder=trace)
    store.upsert_channel_observation({"channel": "wechaty", "source_message_id": "input",
        "trace_id": "selected-run", "text": "测试输入"})
    instruction = "系统提示词：" + "完整上下文" * 500
    trace.record("selected-run", "llm_prompt", {"step_index": 2, "messages": [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps({"query": "测试输入", "api_key": "private-fixture-key",
                                                  "nested": {"refresh_token": "private-fixture-refresh"}})},
        {"role": "user", "content": "password=private-fixture-password"},
        {"role": "user", "content": "cookie=sid=private-fixture-cookie"},
    ]})
    trace.record("selected-run", "llm_http_attempt_started", {"model": "fixture", "attempt": 1})
    trace.record("other-run", "llm_response", {"content": "别的会话不能出现"})
    trace.record("selected-run", "llm_response", {"content": '{"reply":"模型原始输出","token":"private-fixture-token"}',
        "step_index": 2, "elapsed_ms": 47, "completion_diagnostics": {"call_id": "call-fixture"}})
    default = agent_work_snapshot(runtime)
    assert instruction not in json.dumps(default, ensure_ascii=False)
    snapshot = agent_work_snapshot(runtime, selected="wechaty:input", include_model_io=True)
    bodies = [event["model_io"] for event in snapshot["detail"]["timeline"] if event.get("model_io")]
    assert [item["kind"] for item in bodies] == ["request", "response"]
    assert instruction in json.loads(bodies[0]["body"])[0]["content"]
    assert "模型原始输出" in bodies[1]["body"]
    assert bodies[1]["elapsed_ms"] == 47 and bodies[1]["call_id"] == "call-fixture"
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert "private-fixture" not in serialized and "别的会话" not in serialized
    assert "[已隐藏凭证]" in serialized
    assert instruction not in json.dumps(snapshot["records"], ensure_ascii=False)


@pytest.mark.parametrize("prefix", ["wechaty_input_gate", "wechaty_casual_chat",
    "customer_visible_text_generation", "customer_visible_content_review"])
def test_model_io_supports_auxiliary_stages_and_empty_response(tmp_path, prefix):
    """分流、生成和审查的输入输出按原始阶段展示，空返回也能排查。"""
    store = InMemoryAgentStore()
    trace = JsonlTraceRecorder(tmp_path / "trace.log")
    store.upsert_channel_observation({"channel": "wechaty", "source_message_id": "m",
        "trace_id": "run", "text": "测试消息"})
    trace.record("run", prefix + "_prompt", {"messages": [{"role": "system", "content": "仅供本次调用"}]})
    trace.record("run", prefix + "_response", {"content": "", "elapsed_ms": 12})
    snapshot = agent_work_snapshot(SimpleNamespace(store=store, trace_recorder=trace), include_model_io=True)
    io = [event["model_io"] for event in snapshot["detail"]["timeline"] if event.get("model_io")]
    assert len(io) == 2 and io[0]["stage"] == io[1]["stage"]
    assert io[1]["body"] == "" and io[1]["elapsed_ms"] == 12
