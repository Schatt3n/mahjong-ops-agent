"""验证顾客和店家分别有可回查的评测证据，并保留模型故障。"""

import json

import pytest

from .venue_actors import SemanticJudge
from .venue_judging import AGENT_CHECK, ROLE_CHECK, review_dialogue


def _chat(message_id, *, reply_to=None, recipient=None, conversation="dm-a", text=None):
    return {
        "message_id": message_id, "direction": "outbound" if reply_to else "inbound", "kind": "private",
        "conversation_id": conversation, "source_message_id": reply_to, "text": text or message_id,
        "sender_id": "simulation-agent" if reply_to else "a",
        "recipient_id": recipient or ("a" if reply_to else "simulation-agent"),
    }


def _turn(index, *, cause=None):
    return {
        "event_id": f"turn-{index}", "actor_id": "a", "message_id": f"u{index}", "caused_by": cause,
        "trace_id": f"agent-trace-{index}", "runtime_status": "completed",
        "decision": {
            "trace_id": f"role-trace-{index}", "source": "model", "status": "valid", "action": {"action": "send"},
            "diagnostics": {"private-provider-detail": "noise" * 5000},
            "context": {"history": [{"text": "private-other-message"}], "events": [{"details": {"case_intent": "只问有无空桌"}}],
                        "persona": {"speaking_style": "简短口语", "persona_id": "private-actor-id", "preferred_game": "杭州麻将"}},
        },
        "tool_results": [{"name": "search_current_games", "called": True, "allowed": True,
                          "result": {"matches": [], "alternatives": [], "instruction": "discarded"}}],
        "outputs": [{"message_id": "unarchived-output", "text": "still-private"}],
    }


class _JudgeClient:
    def __init__(self, *, role_evidence=("E02",), agent_evidence=("E07", "E08"), agent_verdict="pass", timeout=False):
        self.role_evidence = role_evidence
        self.agent_evidence = agent_evidence
        self.agent_verdict = agent_verdict
        self.timeout = timeout
        self.messages = ()

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.messages = messages
        if self.timeout:
            raise TimeoutError("request detail should remain private")
        return json.dumps({"verdict": self.agent_verdict, "findings": [
            {"check": ROLE_CHECK, "verdict": "pass", "evidence_ids": self.role_evidence, "reason": "问法和偏好符合。"},
            {"check": AGENT_CHECK, "verdict": self.agent_verdict, "evidence_ids": self.agent_evidence, "reason": "核对客户消息与占座结果。"},
        ]}, ensure_ascii=False)


def test_compact_review_preserves_real_turns_individuality_and_verified_outcome():
    """只有实际聊天与有效工具结果进入判定请求，角色私有事实只供角色项参考。"""
    turns = [_turn(1), _turn(2, cause="a1")]
    records = [
        _chat("u1", text="老板杭州麻将有吗"), _chat("a1", reply_to="u1", text="想打几点的？"),
        _chat("foreign", reply_to="u1", recipient="b", conversation="dm-b", text="other-person-private"),
        _chat("u2", text="七点左右"), _chat("a2", reply_to="u2", text="这个时间目前没合适的。"),
    ]
    client = _JudgeClient()
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=turns, records=records,
                             outcome={"kind": "no_match", "status": "passed", "source_message_ids": ["u1", "u2"]})
    assert result["status"] == "valid" and result["verdict"] == "pass"
    policy, payload = client.messages
    assert "店家永远不知道" in policy["content"]
    assert result["projection"]["evidence_bytes"] < result["projection"]["source_turn_bytes"] / 20
    request = {row["evidence_id"]: row for row in json.loads(payload["content"])["evidence"]}
    assert list(request) == [f"E{index:02d}" for index in range(1, 9)]
    assert request["E05"]["caused_by"] == "E03" and request["E05"]["input"] == "E06"
    assert request["E07"]["answers"] == "E06" and request["E08"]["responds_to"] == "E06"
    assert result["evidence_aliases"]["E07"] == "a2"
    assert result["findings"][0]["evidence_ids"] == ("u1",)
    assert result["findings"][1]["evidence_ids"] == ("a2", "agent-trace-2")
    assert json.loads(result["raw"])["findings"][1]["evidence_ids"] == ["E07", "E08"]
    rows = {row["evidence_id"]: row for row in result["evidence"]}
    assert list(rows) == ["role-trace-1", "u1", "a1", "agent-trace-1", "role-trace-2", "u2", "a2", "agent-trace-2"]
    assert rows["a1"]["text"] == "想打几点的？"
    assert rows["role-trace-2"]["caused_by"] == "a1"
    assert rows["role-trace-1"]["private_profile"] == {"speaking_style": "简短口语", "preferred_game": "杭州麻将"}
    assert "private_profile" not in rows["role-trace-2"]
    assert rows["role-trace-1"]["situation"] == {"case_intent": "只问有无空桌"}
    assert rows["agent-trace-2"]["final_outcome"] == {"kind": "no_match", "status": "passed"}
    assert rows["agent-trace-2"]["tools"][0]["result"]["matches"]["count"] == 0
    for omitted in ("private-provider-detail", "unarchived-output", "other-person-private", "private-actor-id", "private-other-message"):
        assert omitted not in payload["content"]


@pytest.mark.parametrize("role_evidence,agent_evidence,has_reply", [
    (("E02",), ("E03", "E01"), True),
    (("E03",), ("E03",), True),
    (("E02",), ("E04",), True),
    (("E02",), ("E03",), False),
    (("E02",), ("E99",), True),
])
def test_wrong_scope_or_missing_delivered_reply_cannot_pass(role_evidence, agent_evidence, has_reply):
    """引用存在也要属于正确角色，店家通过必须引用真正送达的答复。"""
    records = [_chat("u1")]
    if has_reply:
        records.append(_chat("a1", reply_to="u1"))
    client = _JudgeClient(role_evidence=role_evidence, agent_evidence=agent_evidence)
    report = review_dialogue(SemanticJudge(client), customer_id="a", turns=[_turn(1)], records=records)
    assert report["status"] == "invalid" and report["verdict"] == "inconclusive"
    assert report["findings"] == () and report["raw"]


def test_wait_is_a_valid_role_action_and_transport_is_preserved():
    """顾客沉默等候可以被评估，供应商超时仍没有语义结论。"""
    wait = _turn(2, cause="a1")
    wait.pop("message_id")
    wait["decision"]["action"] = {"action": "wait"}
    records = [_chat("u1"), _chat("a1", reply_to="u1", text="稍后有空通知你")]
    client = _JudgeClient(role_evidence=("E05",), agent_evidence=("E03",))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[_turn(1), wait], records=records)
    assert result["status"] == "valid" and result["verdict"] == "pass"
    timeout = review_dialogue(SemanticJudge(_JudgeClient(timeout=True)), customer_id="a", turns=[_turn(1)], records=records)
    assert (timeout["status"], timeout["verdict"], timeout["error"]) == ("transport_error", "inconclusive", "TimeoutError")


def test_provider_failure_is_not_a_valid_role_action():
    """顾客模型失败仍有可诊断的轨迹，轨迹不能替代角色作出的动作。"""
    failed = _turn(2, cause="a1")
    failed.pop("message_id")
    failed["decision"].update(action=None, status="transport_error")
    role = _turn(1)
    records = [_chat("u1"), _chat("a1", reply_to="u1")]
    client = _JudgeClient(role_evidence=("E05",), agent_evidence=("E03",))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[role, failed], records=records)
    assert (result["status"], result["verdict"]) == ("invalid", "inconclusive")


def test_short_aliases_keep_seat_mutation_and_semantic_rejection_auditable():
    """真实写入座位与房间预订有不同证据，引用映射仍保留有效失败。"""
    turn = _turn(1)
    turn["tool_results"].append({"name": "join_game", "called": True, "allowed": True,
                                 "result": {"recorded_status": "confirmed", "game": {"game_id": "g1", "remaining_seats": 0}}})
    records = [_chat("u1", text="包间已经订好了吗？我现在就定？"), _chat("a1", reply_to="u1", text="给你定下了。")]
    client = _JudgeClient(agent_evidence=("E02", "E03", "E04"), agent_verdict="fail")
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn], records=records)
    assert (result["status"], result["verdict"]) == ("valid", "fail")
    assert result["findings"][1]["evidence_ids"] == ("u1", "a1", "agent-trace-1")
    execution = result["evidence"][-1]["tools"]
    assert execution[-1]["result"] == {"recorded_status": "confirmed", "game": {"game_id": "g1", "remaining_seats": 0}}
    assert all(tool["name"] != "reserve_room" for tool in execution)
    assert "区分席位确认、空房查询和房间预订" in client.messages[0]["content"]
