"""验证独立补判只请求原传输失败样本，不修改原账本或覆盖旧判决。"""

from hashlib import sha256
import json
import sys

import pytest

from scripts import replay_venue_judge
from scripts.replay_venue_judge import retry_unavailable
from tests.simulation.venue_actors import SemanticJudge
from tests.simulation.venue_judging import ROLLING_REVIEW_INSTRUCTIONS, ROLE_CHECK, review_dialogue


class _JudgeClient:
    """离线判定替身保留请求编号与引用，并按测试需要模拟掉线。"""

    def __init__(self, outcome="pass"):
        self.outcome = outcome
        self.requests = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        """同样走正式判决格式与证据引用检查。"""
        payload = json.loads(messages[-1]["content"])
        self.requests.append((payload, trace_id))
        assert messages[0]["content"].endswith(ROLLING_REVIEW_INSTRUCTIONS)
        if self.outcome == "transport_error":
            raise TimeoutError("offline fixture")
        if self.outcome == "invalid":
            return "判定格式损坏"
        findings = []
        for check in payload["checks"]:
            kind = "customer_message" if check == ROLE_CHECK else "agent_reply"
            evidence = next(item["evidence_id"] for item in payload["evidence"] if item["kind"] == kind)
            findings.append({"check": check, "verdict": self.outcome, "evidence_ids": [evidence], "reason": "离线判决"})
        return json.dumps({"verdict": self.outcome, "findings": findings}, ensure_ascii=False)


def _write_rows(path, rows):
    """按原运行格式冻结独立的动作账本及渠道账本。"""
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def frozen_archive(tmp_path):
    """冻结两条传输失败、已有有效反例及其他不可补判状态。"""
    run = tmp_path / "recorded-rolling-run"
    run.mkdir()
    turns, chat, samples = [], [], []
    for customer, outcome in (("a", "transport_error"), ("b", "fail"), ("c", "invalid"), ("d", "transport_error")):
        stamp = "2026-11-25T19:00:00+08:00"
        turn = {"actor_id": customer, "message_id": f"in:{customer}", "event_id": f"event:{customer}",
                "virtual_at": stamp, "runtime_status": "completed", "trace_id": f"agent:{customer}",
                "decision": {"source": "model", "status": "valid", "action": {"action": "send"},
                             "trace_id": f"role:{customer}", "context": {"events": []}}}
        incoming = {"kind": "private", "direction": "inbound", "sender_id": customer, "recipient_id": "simulation-agent",
                    "conversation_id": f"dm:{customer}", "message_id": turn["message_id"], "sent_at": stamp, "text": "今晚有位吗"}
        reply = {**incoming, "direction": "outbound", "sender_id": "simulation-agent", "recipient_id": customer,
                 "message_id": f"out:{customer}", "source_message_id": turn["message_id"], "text": "今晚没有合适的局"}
        turns.append(turn)
        chat.extend((incoming, reply))
        samples.append(review_dialogue(SemanticJudge(_JudgeClient(outcome)), customer_id=customer, turns=[turn],
            records=[incoming, reply], role_only=True, delivered_history=True, additional_instructions=ROLLING_REVIEW_INSTRUCTIONS))
    _write_rows(run / "events.jsonl", turns)
    _write_rows(run / "chat.jsonl", chat)
    (run / "report.json").write_text(json.dumps({"mode": "real", "scenario": "rolling-roles", "status": "inconclusive",
        "semantic_samples": samples}, ensure_ascii=False), encoding="utf-8")
    for name in ("agent.sqlite3", "chat.sqlite3"):
        (run / name).write_bytes(b"database access is forbidden for this replay fixture")
    return run


def test_retry_keeps_original_scope_and_records_only_transport_rescores(frozen_archive, tmp_path):
    """已判有效失败和无效格式保持原状，只把原传输失败编号提交本地替身。"""
    source = {path.name: path.read_bytes() for path in frozen_archive.iterdir()}
    output = tmp_path / "retry.json"
    client = _JudgeClient()
    report = retry_unavailable(frozen_archive, output, judge_factory=lambda _: SemanticJudge(client))

    assert report["status"] == "reassessed" and report["original_run_status"] == "inconclusive"
    assert [(request["case_id"], trace) for request, trace in client.requests] == [
        ("dialogue:a", "judge-retry:a"), ("dialogue:d", "judge-retry:d")]
    assert report["retry_verdicts"] == {"pass": 2}
    assert report["retry_samples"][0]["retry"]["findings"][0]["evidence_ids"] == ["in:a"]
    assert report["untouched_samples"] == [{"case_id": "dialogue:b", "status": "valid", "verdict": "fail"},
                                            {"case_id": "dialogue:c", "status": "invalid", "verdict": "inconclusive"}]
    assert report["source_sha256"] == {name: sha256(source[name]).hexdigest()
                                        for name in ("report.json", "events.jsonl", "chat.jsonl")}
    assert {path.name: path.read_bytes() for path in frozen_archive.iterdir()} == source
    assert json.loads(output.read_text(encoding="utf-8")) == report

    with pytest.raises(FileExistsError):
        retry_unavailable(frozen_archive, output, judge_factory=lambda _: pytest.fail("重复路径不能请求评审"))
    with pytest.raises(ValueError, match="原运行归档之外"):
        retry_unavailable(frozen_archive, frozen_archive / "report.json", judge_factory=lambda _: pytest.fail("归档必须只读"))


def test_retry_refuses_any_projection_drift_before_requesting_even_unchanged_cases(frozen_archive, tmp_path):
    """第二条候选的聊天漂移时，第一条相同证据也须等待同批证据核对。"""
    chat_path = frozen_archive / "chat.jsonl"
    chat = [json.loads(line) for line in chat_path.read_text(encoding="utf-8").splitlines()]
    chat[-1]["text"] = "篡改的原剧本"
    _write_rows(chat_path, chat)
    output = tmp_path / "retry.json"

    with pytest.raises(ValueError, match="判定证据与原报告不同：dialogue:d"):
        retry_unavailable(frozen_archive, output, judge_factory=lambda _: pytest.fail("有漂移时不应构造客户端"))
    assert not output.exists()


def test_retry_cli_uses_a_sibling_artifact_without_a_customer_override(frozen_archive, monkeypatch, capsys):
    """自动补判入口使用原名册，在默认路径保留单独的判决记录。"""
    client = _JudgeClient()
    monkeypatch.setattr(replay_venue_judge, "_configured_retry_judge", lambda report, **options: SemanticJudge(client))
    monkeypatch.setattr(sys, "argv", ["replay_venue_judge.py", str(frozen_archive), "--retry-unavailable"])

    assert replay_venue_judge.main() == 0
    output = frozen_archive.with_name(f"{frozen_archive.name}.judge-retry.json")
    assert json.loads(capsys.readouterr().out)["output"] == str(output)
    assert json.loads(output.read_text(encoding="utf-8"))["retry_verdicts"] == {"pass": 2}


def test_verify_only_checks_candidates_without_initializing_model(frozen_archive, tmp_path):
    """额度不可用时仍可核对证据，预检不创建补判报告。"""
    output = tmp_path / "retry.json"
    report = retry_unavailable(frozen_archive, output, verify_only=True,
                               judge_factory=lambda _: pytest.fail("预检不能初始化客户端"))
    assert report["status"] == "verified_only" and report["candidate_samples"] == 2
    assert report["retry_samples"] == [] and not output.exists()


def test_retry_keeps_a_completed_review_if_later_request_interrupts(frozen_archive, tmp_path):
    """发生硬中断时，已完成一次的独立判决可查，后续样本不会冒充完成。"""
    class InterruptedJudgeClient(_JudgeClient):
        def complete(self, messages, *, trace_id, timeout_seconds):
            if self.requests:
                raise KeyboardInterrupt
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)

    output = tmp_path / "retry.json"
    with pytest.raises(KeyboardInterrupt):
        retry_unavailable(frozen_archive, output, judge_factory=lambda _: SemanticJudge(InterruptedJudgeClient()))
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "reassessing" and report["candidate_samples"] == 2
    assert [sample["original"]["case_id"] for sample in report["retry_samples"]] == ["dialogue:a"]


@pytest.mark.parametrize("override", [{"mode": "mock"}, {"scenario": "lifecycle"}, {"status": "paused_transport"}])
def test_retry_rejects_an_archive_outside_terminal_real_rolling_scope(frozen_archive, tmp_path, override):
    """未结束的运行、其他场景和模拟报告都不提交判定。"""
    path = frozen_archive / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(report | override, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="仅支持已结束的真实滚动营业运行"):
        retry_unavailable(frozen_archive, tmp_path / "retry.json", judge_factory=lambda _: pytest.fail("不能用于其他运行"))
