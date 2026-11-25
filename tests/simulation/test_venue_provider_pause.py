"""用本地供应商替身核对角色传输中断、同波决策留存和真实渠道续跑。"""

from collections import Counter
from contextlib import closing
from datetime import date, datetime, time
import json
import sqlite3
from threading import Lock
from urllib.error import URLError

import pytest

from mahjong_agent_runtime.llm import LLMCompletionText
from mahjong_agent_runtime.models import DEFAULT_TZ
from scripts import run_venue_simulation as runner
from tests.simulation.venue_actors import ModelActor, SemanticJudge, generate_personas
from tests.simulation.venue_channel import AGENT_ID
from tests.simulation.venue_judging import AGENT_CHECK, FIRST_CONTACT_CHECK, ROLE_CHECK
from tests.simulation.venue_role_journal import VenueRoleJournal
from tests.simulation.venue_world import VenueWorld


_DATE = date(2026, 11, 25)
_SEED = 42


class _RoleClient:
    """按身份及事件类型注入故障，模型动作仍通过正式解析合同。"""

    def __init__(self, persona_id, failing_kind):
        self.persona_id, self.failing_kind = persona_id, failing_kind
        self.online = False
        self.calls = Counter()
        self.failed_event_ids = []
        self._lock = Lock()

    def complete(self, messages, *, trace_id, timeout_seconds):
        """同一个事件请求从日志中可核对，成功替身模拟可验证的完整 HTTP 响应。"""
        request = json.loads(messages[-1]["content"])
        event = request["events"][-1]
        event_id = event["event_id"]
        with self._lock:
            self.calls[event_id] += 1
            sequence = sum(self.calls.values())
            if not self.online and request["persona"]["persona_id"] == self.persona_id and event["kind"] == self.failing_kind:
                self.failed_event_ids.append(event_id)
                raise URLError("local provider outage fixture")
        action = {"action": "wait"}
        if event["kind"] in {"activity", "delivered"}:
            action = {"action": "send", "kind": "private", "recipient_id": AGENT_ID,
                      "text": "老板，今晚还有位吗？" if event["kind"] == "activity" else "那明晚呢？"}
        raw = json.dumps(action, ensure_ascii=False)
        return LLMCompletionText(raw, {
            "schema_version": 1, "route": "primary", "call_id": f"llm_call_{sequence:032x}",
            "attempts": [{"attempt_id": f"llm_attempt_{sequence:032x}", "attempt_index": 1,
                          "http_status": 200, "body_complete": True, "response_json_parsed": True, "choice_count": 1,
                          "content": {"present": True, "kind": "string", "length": len(raw),
                                      "empty": False, "whitespace_only": False}}],
        })


class _JudgeClient:
    """给续跑后的合同检查提供本地有效判决，记录暂停时有没有请求评审。"""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, *, trace_id, timeout_seconds):
        """依据待判项目返回该项目允许引用的已归档证据。"""
        self.calls += 1
        request = json.loads(messages[-1]["content"])
        evidence_kinds = {
            ROLE_CHECK: {"customer_message", "customer_action"},
            AGENT_CHECK: {"agent_reply"},
            FIRST_CONTACT_CHECK: {"customer_message"},
        }
        findings = [{
            "check": check, "verdict": "pass", "reason": "本地测试判决",
            "evidence_ids": [item["evidence_id"] for item in request["evidence"] if item["kind"] in evidence_kinds[check]],
        } for check in request["checks"]]
        return json.dumps({"verdict": "pass", "findings": findings}, ensure_ascii=False)


def _archived_message_ids(run_dir):
    """直接读取渠道事实源，确保没有依赖可能还未导出的报告。"""
    with closing(sqlite3.connect(run_dir / "chat.sqlite3")) as db:
        return [row[0] for row in db.execute("SELECT message_id FROM venue_messages ORDER BY sequence")]


@pytest.mark.parametrize(("failing_kind", "already_delivered"), [("activity", False), ("delivered", True)])
def test_rolling_provider_pause_keeps_channel_archive_and_reuses_persisted_peer_decision(
    tmp_path, monkeypatch, failing_kind, already_delivered,
):
    """失败前可有已交付消息，同一波无投递，只补请求缺失的角色。"""
    start_at = datetime.combine(_DATE, time(19), tzinfo=DEFAULT_TZ)
    target = VenueWorld(_SEED, start_at).personas(generate_personas(_SEED))[0]
    role_client = _RoleClient(target.persona_id, failing_kind)
    judge_client = _JudgeClient()
    build = runner.build_runtime
    monkeypatch.setattr(runner, "load_dotenv_defaults", lambda path: None)
    monkeypatch.setattr(runner, "_real_roles", lambda: (ModelActor(role_client), SemanticJudge(judge_client)))
    monkeypatch.setattr(runner, "build_runtime", lambda store, mode, **kwargs: build(store, "mock", **kwargs))
    run_dir = tmp_path / "provider-pause"
    first = runner.run_simulation(
        run_dir, mode="real", model_source="project", seed=_SEED, scenario_date=_DATE, background_events=0,
        scenario_name="rolling-roles", active_users=2, workers=2, dialogue_turns=2, judge_users=2,
    )

    assert first["status"] == "paused_transport", first
    pause_path = run_dir / "provider_pause.json"
    checkpoint = json.loads(pause_path.read_text(encoding="utf-8"))
    assert checkpoint == first["provider_pause"]
    assert not (run_dir / "report.json").exists()
    assert judge_client.calls == 0
    failed_id, = role_client.failed_event_ids
    journal = VenueRoleJournal(run_dir / "role_events.sqlite3")
    failed = journal.get(failed_id)
    assert failed.status == "pending" and failed.decision is None and failed.result is None
    peers = [event for event in journal.events() if event.due_at == failed.due_at and event.event_id != failed_id
             and event.status == "pending" and event.decision is not None]
    peer, = peers
    channel_before = _archived_message_ids(run_dir)
    assert bool(channel_before) == already_delivered
    assert set(checkpoint["pending_event_ids"]) == {failed_id, peer.event_id}
    assert checkpoint["archived_messages"] == len(channel_before)
    assert checkpoint["saved_peer_decisions"] == 1
    assert not {f"customer:{failed_id}", f"customer:{peer.event_id}"} & set(channel_before)
    assert [attempt["event_id"] for attempt in journal.transport_attempts()] == [failed_id]
    if already_delivered:
        # 第二次断线仍停在原事件，保留两次尝试且不消耗同波已经成功的决定。
        retry = runner.resume_rolling_simulation(run_dir)
        assert retry["status"] == "paused_transport"
        assert retry["provider_pause"]["cumulative_failed_requests"] == 2
        assert journal.get(peer.event_id).decision == peer.decision
        assert role_client.calls[peer.event_id] == 1
        assert _archived_message_ids(run_dir) == channel_before and judge_client.calls == 0
        assert [attempt["event_id"] for attempt in journal.transport_attempts()] == [failed_id, failed_id]
    ready_before = {event.event_id for event in journal.events() if event.status == "finished"}
    calls_before = role_client.calls.copy()
    roster = (run_dir / "personas.json").read_bytes()

    role_client.online = True
    final = runner.resume_rolling_simulation(run_dir)

    assert final["status"] == "passed", final
    assert final["role_transport_recovery"] == {"failed_requests": calls_before[failed_id], "affected_events": 1}
    assert (run_dir / "personas.json").read_bytes() == roster
    assert json.loads((run_dir / "report.json").read_text(encoding="utf-8"))["status"] == "passed"
    assert journal.get(peer.event_id).decision == peer.decision
    for event_id in ready_before | {peer.event_id}:
        assert journal.get(event_id).status == "finished"
        assert role_client.calls[event_id] == calls_before[event_id]
    assert journal.get(failed_id).status == "finished"
    assert role_client.calls[failed_id] == calls_before[failed_id] + 1
    channel_after = _archived_message_ids(run_dir)
    assert channel_after[:len(channel_before)] == channel_before
    assert len(channel_after) == len(set(channel_after))
    assert {f"customer:{failed_id}", f"customer:{peer.event_id}"} <= set(channel_after)
