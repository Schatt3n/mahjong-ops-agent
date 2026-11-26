"""验证顾客和店家分别有可回查的评测证据，并保留模型故障。"""

import json
from pathlib import Path

import pytest

from . import venue_judging
from .test_venue_actors import _client as _http_client
from .venue_actors import ActorContext, ModelActor, SemanticJudge, generate_personas
from .venue_judging import (
    AGENT_CHECK, DURATION_CHECK, FIRST_CONTACT_CHECK, ROLE_CHECK, ROLLING_CHECK_IDS, judge_with_short_evidence_ids,
    judge_reply_with_verified_seats, prepare_dialogue_review, review_dialogue,
)


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
    def __init__(self, *, role_evidence=("E02",), agent_evidence=("E07", "E08"), role_verdict="pass", agent_verdict="pass",
                 extra_findings=(), timeout=False):
        self.role_evidence = role_evidence
        self.agent_evidence = agent_evidence
        self.role_verdict = role_verdict
        self.agent_verdict = agent_verdict
        self.extra_findings = extra_findings
        self.timeout = timeout
        self.messages = ()

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.messages = messages
        if self.timeout:
            raise TimeoutError("request detail should remain private")
        checks = json.loads(messages[-1]["content"])["checks"]
        role, agent = (ROLLING_CHECK_IDS[ROLE_CHECK], ROLLING_CHECK_IDS[AGENT_CHECK]) if (
            ROLLING_CHECK_IDS[ROLE_CHECK] in checks) else (ROLE_CHECK, AGENT_CHECK)
        findings = [
            {"check": role, "verdict": self.role_verdict, "evidence_ids": self.role_evidence, "reason": "问法和偏好符合。"},
            {"check": agent, "verdict": self.agent_verdict, "evidence_ids": self.agent_evidence, "reason": "核对客户消息与占座结果。"},
            *self.extra_findings,
        ]
        verdicts = {finding["verdict"] for finding in findings}
        overall = next(verdict for verdict in ("fail", "inconclusive", "pass") if verdict in verdicts)
        return json.dumps({"verdict": overall, "findings": findings}, ensure_ascii=False)


@pytest.mark.parametrize("citations", [("E01", "E02"), ("delivery:venue-event-0026",), ("E03",)])
def test_lifecycle_citation_labels_distinguish_finish_from_customer_delivery(citations):
    """评审可引用真实散场行和另一条送达行，不能把散场编号改造为送达编号。"""
    rows = [
        {"event_id": "venue-event-0026", "kind": "finish", "room_inventory_write": {"operation": "play_end"}},
        {"event_id": "delivery:venue-event-0027", "kind": "slot_query", "caused_by": "owner:room-query",
         "customer_message": {"text": "现在有空房吗？"}, "outputs": [{"text": "现在有空房。"}]},
    ]

    class JudgeClient:
        def complete(self, messages, **_):
            evidence = json.loads(messages[-1]["content"])["evidence"]
            assert [(row["evidence_id"], row["kind"]) for row in evidence] == [("E01", "finish"), ("E02", "slot_query")]
            assert evidence[1]["caused_by"] == "owner:room-query"
            assert "现场行只证明现场事实" in messages[0]["content"]
            return json.dumps({"verdict": "pass", "findings": [{"check": "business_facts", "verdict": "pass",
                               "evidence_ids": citations, "reason": "核对空房答复及真实散场"}]}, ensure_ascii=False)

    result = judge_with_short_evidence_ids(SemanticJudge(JudgeClient()), "lifecycle", rows, trace_id="independent",
                                           checks=("business_facts",), instructions="核对交易以外的现场事实。")
    assert [row["evidence_id"] for row in result.evidence] == ["venue-event-0026", "delivery:venue-event-0027"]
    assert [row["evidence_alias"] for row in result.evidence] == ["E01", "E02"]
    if len(citations) == 2:
        assert (result.status, result.verdict) == ("valid", "pass")
        assert result.findings[0].evidence_ids == ("venue-event-0026", "delivery:venue-event-0027")
    else:
        assert (result.status, result.verdict, result.error) == (
            "invalid", "inconclusive", "finding cites unavailable evidence",
        )
        assert result.findings == ()
    assert tuple(json.loads(result.raw)["findings"][0]["evidence_ids"]) == citations


@pytest.mark.parametrize(("reply", "independent_verdict", "expected_checks", "fact_verdict", "cited", "overall"), [
    ("七点杭麻一缺三", "pass", ("time",), "pass", ["E01", "E02", "E03"], "pass"),
    ("七点杭麻三缺一", "pass", ("time",), "fail", ["E01", "E02", "E03"], "fail"),
    ("七点杭麻一缺三", "fail", ("time",), "pass", ["E01", "E02", "E03"], "fail"),
    ("七点杭麻一缺三", "inconclusive", ("time",), "pass", ["E01", "E02", "E03"], "inconclusive"),
    ("七点杭麻一缺三", "pass", ("time",), "pass", ["E01", "E02"], "inconclusive"),
    ("七点杭麻三缺一吗？", "pass", ("time", "seats"), None, ["E01", "E02", "E03"], "pass"),
    ("七点那桌可以拼。", "pass", ("time", "seats"), None, ["E01", "E02", "E03"], "pass"),
])
def test_verified_seat_count_compares_authoritative_rows_before_independent_review(
    reply, independent_verdict, expected_checks, fact_verdict, cited, overall,
):
    """可证明的数字是独立事实判项，未核实的说法仍进入模型且其他项的失败继续生效。"""
    class JudgeClient:
        def complete(self, messages, **_):
            payload = json.loads(messages[-1]["content"])
            assert tuple(payload["checks"]) == expected_checks
            return json.dumps({"verdict": independent_verdict, "findings": [
                {"check": check, "verdict": independent_verdict, "evidence_ids": cited, "reason": "核对原始资料"}
                for check in payload["checks"]
            ]}, ensure_ascii=False)

    rows = [{"message_id": "actual-reply", "text": reply, "citation_required": True},
            {"event_id": "read-result", "citation_required": True, "verified_seat_candidates": [
                {"game_id": "game-night", "game_type": "hangzhou_mahjong", "planned_start_at": "2026-11-25T19:00:00+08:00",
                 "claimed_seats": 1, "remaining_seats": 3, "seats_total": 4},
            ]}, {"message_id": "earlier-customer", "text": "早一点晚一点都可以。", "citation_required": True}]
    result = judge_reply_with_verified_seats(SemanticJudge(JudgeClient()), "proof", rows, trace_id="proof",
        checks=("time", "seats"), seat_check="seats", delivered_reply_id="actual-reply", seat_source_id="read-result",
        instructions="按查局和原话核对。")
    assert result["verdict"] == overall
    assert result["seat_fact_check"]["verdict"] == fact_verdict
    assert result["seat_fact_check"]["evidence_ids"] == ["actual-reply", "read-result"]
    assert result["independent_judgment"]["verdict"] == independent_verdict
    assert [finding["check"] for finding in result["independent_judgment"]["findings"]] == list(expected_checks)


def test_compact_review_preserves_real_turns_individuality_and_verified_outcome():
    """只有实际聊天与有效工具结果进入判定请求，角色私有事实只供角色项参考。"""
    turns = [_turn(1), _turn(2, cause="a1")]
    records = [
        _chat("u1", text="老板杭州麻将有吗"), _chat("a1", reply_to="u1", text="想打几点的？"),
        _chat("foreign", reply_to="u1", recipient="b", conversation="dm-b", text="other-person-private"),
        _chat("u2", text="七点左右"), _chat("a2", reply_to="u2", text="这个时间目前没合适的。"),
    ]
    client = _JudgeClient(role_evidence=("E02", "E06"))
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
    assert result["findings"][0]["evidence_ids"] == ("u1", "u2")
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


def test_rolling_wire_checks_explain_own_subjects_without_allowing_public_delivery_as_shop_proof():
    """固定短编号和逐项链接只指导引用，群看板仍不能替代店家的本人回复。"""
    turn = _turn(1, cause="notice")
    notice = {"message_id": "notice", "direction": "outbound", "kind": "group",
              "recipient_id": "group", "sender_id": "simulation-agent", "text": "今晚七点有一桌",
              "sent_at": "2026-11-25T18:20:00+08:00"}
    turn["decision"]["context"]["history"] = [notice]
    records = [notice, _chat("u1"), _chat("a1", reply_to="u1")]
    arguments = {"customer_id": "a", "turns": [turn], "records": records, "delivered_history": True,
                 "rolling_check_ids": True}
    good = _JudgeClient(role_evidence=("E03", "E01"), agent_evidence=("E04", "E05"))
    result = review_dialogue(SemanticJudge(good), **arguments)
    assert result["status"] == "valid"
    assert result["projection"]["rolling_check_ids"] is True
    assert [finding["check"] for finding in result["findings"]] == [ROLE_CHECK, AGENT_CHECK]
    assert json.loads(result["raw"])["findings"][0]["check"] == "role_behavior"
    prompt, payload = good.messages
    assert json.loads(payload["content"])["checks"] == ["role_behavior", "shop_behavior"]
    rules = json.loads(prompt["content"].splitlines()[-1])
    assert rules["role_behavior"]["allowed_evidence_ids"] == ["E01", "E02", "E03", "E04"]
    assert rules["role_behavior"]["required_for_pass"] == ["E03"]
    assert rules["shop_behavior"]["allowed_evidence_ids"] == ["E03", "E04", "E05"]
    assert rules["shop_behavior"]["subject_evidence_ids"] == ["E04"]
    assert "max_duration_hours 只表示上限" in prompt["content"]
    assert "仅凭它与开场的差值不能推断已经约定" in prompt["content"]
    assert "顾客项判重复时引用两条实际来话和适用的中间回复" in prompt["content"]
    assert "店家选错查局时间、漏答或误写属于店家项" in prompt["content"]

    wrong_citation = _JudgeClient(role_evidence=("E03",), agent_evidence=("E04", "E01"), agent_verdict="fail")
    assert review_dialogue(SemanticJudge(wrong_citation), **arguments)["error"] == (
        "judgment does not cite its review subject")
    unsolicited_duration = _JudgeClient(role_evidence=("E03",), agent_evidence=("E04",), extra_findings=({
        "check": "确认现成局席位时", "verdict": "fail", "evidence_ids": ["E05"], "reason": "误添时长项。",
    },))
    assert review_dialogue(SemanticJudge(unsolicited_duration), **arguments)["error"] == (
        "findings must cover every requested check exactly once")


@pytest.mark.parametrize(("first_failure", "second_verdict"), [
    ("customer pass must cite every archived action", "fail"),
    ("overall verdict contradicts findings", "pass"),
    ("customer pass must cite every archived action", "invalid"),
    (None, "fail"),
    ("exhausted", "invalid"),
])
def test_rolling_retry_keeps_evidence_boundary_and_actual_attempts(monkeypatch, first_failure, second_verdict):
    """完整返回的无效评审可独立重审一次，保留原文和来源，不能替换有效失败或扩时。"""
    wait = _turn(2, cause="a1")
    wait.pop("message_id")
    wait["decision"]["action"] = {"action": "wait"}
    turns = [_turn(1), wait]
    records = [_chat("u1"), _chat("a1", reply_to="u1")]
    prepared = prepare_dialogue_review(customer_id="a", turns=turns, records=records, delivered_history=True)
    labels = {source: alias for alias, source in prepared.aliases.items()}
    role_all = [labels[source] for source in ("u1", "role-trace-2")]

    def response(overall, *, role="pass", include_wait=True):
        return json.dumps({"verdict": overall, "findings": [
            {"check": "role_behavior", "verdict": role, "evidence_ids": role_all if include_wait else role_all[:1],
             "reason": "检查实发消息及等待。"},
            {"check": "shop_behavior", "verdict": "pass", "evidence_ids": [labels["a1"]],
             "reason": "检查本人收到的店家回执。"},
        ]}, ensure_ascii=False)

    if first_failure == "overall verdict contradicts findings":
        first = response("fail")
    elif first_failure is None:
        first = response("fail", role="fail")
    else:
        first = response("pass", include_wait=False)
    second = response("fail", role="fail") if second_verdict == "fail" else response(
        "pass", include_wait=second_verdict != "invalid")
    captured = []
    clock = iter((0.0, 8.0 if first_failure == "exhausted" else 3.0))
    monkeypatch.setattr(venue_judging, "monotonic", lambda: next(clock))
    judge = SemanticJudge(_http_client((first, second), captured), timeout_seconds=8)
    result = venue_judging.judge_prepared_dialogue(judge, prepared, trace_id="judge:a", rolling_check_ids=True,
                                                   retry_invalid=True)

    assert (result["status"], result["verdict"]) == (
        ("invalid", "inconclusive") if second_verdict == "invalid" else ("valid", second_verdict))
    assert result["response_source"]["http_response_complete"] is True
    if first_failure in (None, "exhausted"):
        assert len(captured) == 1 and "prior_attempts" not in result
        return
    assert len(captured) == 2
    prior, = result["prior_attempts"]
    assert (prior["status"], prior["verdict"], prior["error"], prior["raw"]) == ("invalid", "inconclusive", first_failure, first)
    assert prior["response_source"]["step_index"] == 1 and result["response_source"]["step_index"] == 2
    assert prior["diagnostics"]["timeout_budget_seconds"] == 8
    assert result["diagnostics"]["timeout_budget_seconds"] == 5
    assert captured[0]["messages"][1] == captured[1]["messages"][1]
    assert first_failure in captured[1]["messages"][0]["content"] and first not in captured[1]["messages"][0]["content"]


@pytest.mark.parametrize("include_quotes", [False, True])
def test_rolling_duration_subjects_link_confirmed_join_to_prior_delivered_terms(include_quotes):
    """列出正确席位链接，但有链接仍须引用回执原话，晚到的回复不能作为依据。"""
    opening, join, later = _turn(1), _turn(2, cause="a1"), _turn(3, cause="a2")
    join["tool_results"].append({"name": "join_game", "called": True, "allowed": True,
                                 "result": {"recorded_status": "confirmed", "game": {"game_id": "g1"}}})
    inbound = "那桌给我留一个，我最多打三小时。"
    terms = "这桌打三个钟头，七点开始。"
    records = [_chat("u1", text="这桌打到几点"), _chat("a1", reply_to="u1", text=terms),
               _chat("u2", text=inbound), _chat("a2", reply_to="u2", text="给你留好了。"),
               _chat("u3", text="下次还有吗"), _chat("a3", reply_to="u3", text="再看。")]
    reason = f"顾客「{inbound}」；店家提前交代「{terms}」" if include_quotes else "此前已明确桌子的时长。"
    client = _JudgeClient(role_evidence=("E02", "E06", "E10"), agent_evidence=("E07",),
                          extra_findings=({"check": "duration_at_join", "verdict": "pass",
                                           "evidence_ids": ("E03", "E06", "E08"), "reason": reason},))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[opening, join, later],
                             records=records, rolling_check_ids=True)
    rules = json.loads(client.messages[0]["content"].splitlines()[-1])
    assert rules["duration_at_join"]["confirmed_joins"] == [{
        "execution": "E08", "customer": "E06", "candidate_replies": ["E03", "E07"],
    }]
    if include_quotes:
        assert result["status"] == "valid" and result["findings"][-1]["check"] == DURATION_CHECK
        assert result["findings"][-1]["evidence_ids"] == ("a1", "u2", "agent-trace-2")
    else:
        assert result["error"] == "duration review needs linked join and delivered proof"


@pytest.mark.parametrize(("before", "previous", "inbound", "confirmation", "decision", "explanation"), [
    ("这桌还有位吗？", "这桌还差三位，你要算一个吗？", "算我一个吧，七点半左右到。",
     "好，位置给你留好了。", "pass", "本人无条件报名。"),
    ("请问这桌打到几点？", "这桌打多久还没定。", "好，先给我留个位，我到时再问你打多久。",
     "好，位置给你留好了。", "pass", "本人明确将时长询问留到以后。"),
    ("请问这桌打到几点？", "这桌打多久还没定。", "我最多打三小时，合适才能留。",
     "这桌打四小时，位置给你留好了。", "fail", "店家把默认四小时说成约定，条件也与三小时上限冲突。"),
], ids=("unconditional", "deferred", "hard-limit"))
def test_rolling_duration_distinguishes_a_condition_from_a_normal_confirmation(
    before, previous, inbound, confirmation, decision, explanation,
):
    """源于滚动档 0102、0235 同一桌的两次实际留位，保留已送达文字和时长硬限制对照。"""
    opening, join = _turn(1), _turn(2, cause="a1")
    join["tool_results"].append({"name": "join_game", "called": True, "allowed": True,
        "result": {"recorded_status": "confirmed", "game": {"game_id": "g1", "requirement": {
            "planned_start_at": "2026-11-26T19:30:00+08:00", "planned_end_at": "2026-11-26T23:30:00+08:00",
        }}}})
    rows = [_chat("u1", text=before),
            _chat("a1", reply_to="u1", text=previous), _chat("u2", text=inbound),
            _chat("a2", reply_to="u2", text=confirmation)]
    prepared = prepare_dialogue_review(customer_id="a", turns=[opening, join], records=rows)
    alias = {key: label for label, key in prepared.aliases.items()}
    client = _JudgeClient(role_evidence=(alias["u1"], alias["u2"]), agent_evidence=(alias["a2"],), extra_findings=({
        "check": "duration_at_join", "verdict": decision, "reason": f"「{inbound}」「{confirmation}」{explanation}",
        "evidence_ids": (alias["u2"], alias["a2"], alias["agent-trace-2"]),
    },))
    result = venue_judging.judge_prepared_dialogue(SemanticJudge(client), prepared, trace_id="duration-conditionality",
                                                  rolling_check_ids=True)
    assert (result["status"], result["verdict"]) == ("valid", decision)
    assert result["findings"][-1]["evidence_ids"] == ("u2", "a2", "agent-trace-2")
    prompt = client.messages[0]["content"]
    assert "如果时长无关或明确延后，引用正常确认回复" in prompt
    assert "未主动报时长不能作为失败理由" in prompt
    assert "没有重述旧的硬限制并不等于撤回" in prompt
    assert "缺少该证据或报错时长不能判通过" in prompt
    assert "planned_end_at 是计划排期" in prompt


def test_wait_is_a_valid_role_action_and_transport_is_preserved():
    """顾客沉默等候可以被评估，供应商超时仍没有语义结论。"""
    wait = _turn(2, cause="a1")
    wait.pop("message_id")
    wait["decision"]["action"] = {"action": "wait"}
    records = [_chat("u1"), _chat("a1", reply_to="u1", text="稍后有空通知你")]
    client = _JudgeClient(role_evidence=("E02", "E05"), agent_evidence=("E03",))
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
    client = _JudgeClient(agent_evidence=("E02", "E03", "E04"), agent_verdict="fail", extra_findings=({
        "check": DURATION_CHECK, "verdict": "pass", "evidence_ids": ("E02", "E03", "E04"),
        "reason": "「包间已经订好了吗？我现在就定？」「给你定下了。」未涉及时长。",
    },))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn], records=records)
    assert (result["status"], result["verdict"]) == ("valid", "fail")
    assert result["findings"][1]["evidence_ids"] == ("u1", "a1", "agent-trace-1")
    execution = result["evidence"][-1]["tools"]
    assert execution[-1]["result"] == {"recorded_status": "confirmed", "game": {"game_id": "g1", "remaining_seats": 0}}
    assert all(tool["name"] != "reserve_room" for tool in execution)
    assert "区分席位确认、空房查询和房间预订" in client.messages[0]["content"]


def test_customer_pass_cannot_skip_a_followup():
    """已有首轮证据仍不足以判整段通过，逐点重问可以用对应的入站消息单独定位。"""
    turns = [_turn(1), _turn(2, cause="a1"), _turn(3, cause="a2")]
    turns[0]["decision"]["context"]["events"][0]["details"].update(
        earliest_start_time="19:00", latest_start_time="21:00",
    )
    records = [
        _chat("u1", text="老板七点有川麻吗？"), _chat("a1", reply_to="u1", text="没有，要我帮你组吗？"),
        _chat("u2", text="那八点有吗？"), _chat("a2", reply_to="u2", text="没有。"),
        _chat("u3", text="那九点呢？"), _chat("a3", reply_to="u3", text="没有。"),
    ]
    incomplete = _JudgeClient(role_evidence=("E02", "E06"), agent_evidence=("E11",))
    result = review_dialogue(SemanticJudge(incomplete), customer_id="a", turns=turns, records=records)
    assert (result["status"], result["verdict"], result["error"]) == (
        "invalid", "inconclusive", "customer pass must cite every archived action",
    )
    detector = _JudgeClient(role_evidence=("E02", "E06", "E10"), agent_evidence=("E11",), role_verdict="fail")
    result = review_dialogue(SemanticJudge(detector), customer_id="a", turns=turns, records=records)
    assert (result["status"], result["verdict"]) == ("valid", "fail")
    assert result["findings"][0]["evidence_ids"] == ("u1", "u2", "u3")
    assert result["evidence"][0]["situation"]["latest_start_time"] == "21:00"


def test_first_contact_actor_sees_own_availability_without_the_generated_name():
    """首轮角色可只问关键条件，名册全名和未公开的牌桌条件不出现在生成输入中。"""
    class CapturingActor:
        def complete(self, messages, *, trace_id, timeout_seconds):
            self.messages = messages
            return '{"action":"send","text":"七到九点有川麻没"}'

    person = generate_personas(42, count=1)[0]
    client = CapturingActor()
    decision = ModelActor(client).decide(ActorContext(person, "2026-11-25T18:25:00+08:00", events=({
        "kind": "session_open", "issuer": "venue", "details": {
            "case_intent": "第一次找不熟的老板问局", "earliest_start_time": "19:00", "latest_start_time": "21:00",
            "game_type": "四川麻将", "duration_hours": 3, "table_duration_hours": 5,
        },
    },)), trace_id="first-contact")

    policy, payload = client.messages
    context = json.loads(payload["content"])
    assert "name" not in context["persona"] and person.name not in payload["content"]
    assert context["events"][0]["details"] == {
        "case_intent": "第一次找不熟的老板问局", "earliest_start_time": "19:00", "latest_start_time": "21:00",
        "game_type": "四川麻将", "duration_hours": 3,
    }
    assert "普通开口不用报自己是谁" in policy["content"]
    assert "只有需要区分几桌或提出新的限制才重提条件" in policy["content"]
    assert decision.action.text == "七到九点有川麻没"


def test_duration_judgment_keeps_private_preferences_table_truth_and_delivered_reply_separate():
    """后台固定三小时不能证明顾客提前知道，也不能替代店家送达的时长确认。"""
    opening, response = _turn(1), _turn(2, cause="a1")
    opening["decision"]["context"]["events"][0]["details"].update(
        case_intent="第一次找不熟的老板问局", duration_hours=3,
    )
    table = {"game_id": "g1", "requirement": {"duration_kind": "fixed", "duration_hours": 3,
             "lifecycle_expires_at": "2026-11-25T22:00:00+08:00"}}
    opening["tool_results"][0]["result"]["matches"] = [{"game": table}]
    response["tool_results"] = [{"name": "join_game", "called": True, "allowed": True,
                                 "result": {"game": table, "recorded_status": "confirmed"}}]
    records = [
        _chat("u1", text="老板，我叫张某某。今晚想打杭麻，一块底，无烟，一个人，有合适的吗？"),
        _chat("a1", reply_to="u1", text="七点杭麻，一块无烟，三缺一。"),
        _chat("u2", text="合适，我来，打三个钟头左右。"),
        _chat("a2", reply_to="u2", text="好，已经留好位置了。"),
    ]
    client = _JudgeClient(role_evidence=("E02",), agent_evidence=("E03", "E06", "E07", "E08"),
                          role_verdict="fail", agent_verdict="fail", extra_findings=(
        {"check": FIRST_CONTACT_CHECK, "verdict": "fail", "evidence_ids": ("E01", "E02"), "reason": "无必要主动报全名。"},
        {"check": DURATION_CHECK, "verdict": "fail", "evidence_ids": ("E03", "E06", "E07", "E08"),
         "reason": "没有向顾客交代牌桌时长。"},
    ))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[opening, response], records=records,
                             first_contact=True)

    assert (result["status"], result["verdict"]) == ("valid", "fail")
    policy, payload = client.messages
    request = json.loads(payload["content"])
    assert request["checks"] == [ROLE_CHECK, AGENT_CHECK, FIRST_CONTACT_CHECK, DURATION_CHECK]
    evidence = {row["evidence_id"]: row for row in request["evidence"]}
    assert evidence["E01"]["situation"]["duration_hours"] == 3
    assert evidence["E01"]["situation"]["knows_shopkeeper"] is False
    assert evidence["E04"]["tools"][0]["result"]["matches"]["options"][0]["terms"] == {
        "duration_kind": "fixed", "duration_hours": 3,
    }
    assert evidence["E07"]["text"] == "好，已经留好位置了。"
    assert "已知熟人、转介绍" in policy["content"]
    assert "勿按字数、字段数或固定句式判定" in policy["content"]
    assert "顾客知道桌子时长，必须能指向该轮之前实际看到的文字" in policy["content"]
    assert "不要用自动安排的过期时间替代约定时长" in policy["content"]


def test_style_pass_cannot_omit_repeat_readback_after_an_opening():
    """陌生顾客的独立口吻验收覆盖首轮和继续报名的来话。"""
    first, second = _turn(1), _turn(2, cause="a1")
    first["decision"]["context"]["events"][0]["details"]["knows_shopkeeper"] = False
    messages = [
        _chat("u1", text="今晚有无烟的吗"), _chat("a1", reply_to="u1", text="有一桌七点到十点，一块无烟。"),
        _chat("u2", text="一块无烟、七点到十点正合适，给我留位。"), _chat("a2", reply_to="u2", text="好。"),
    ]
    client = _JudgeClient(role_evidence=("E02", "E06"), extra_findings=({
        "check": FIRST_CONTACT_CHECK, "verdict": "pass", "evidence_ids": ("E02",), "reason": "只检查了开场。",
    },))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[first, second], records=messages)
    assert (result["status"], result["error"]) == ("invalid", "customer pass must cite every archived action")


def test_known_contact_can_use_a_relevant_form_of_address():
    """现实已知关系允许为当前邀约对号，陌生顾客的专项评审不应套给熟人。"""
    turn = _turn(1)
    turn["decision"]["context"]["events"][0]["details"]["knows_shopkeeper"] = True
    messages = [_chat("u1", text="我小李，刚说帮你叫的那个今晚来不了哈"),
                _chat("a1", reply_to="u1", text="行，我再问问。")]
    client = _JudgeClient(agent_evidence=("E03",))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn], records=messages)
    assert result["verdict"] == "pass"
    assert json.loads(client.messages[-1]["content"])["checks"] == [ROLE_CHECK, AGENT_CHECK]


@pytest.mark.parametrize("disclosed, expected", [(False, "invalid"), (True, "valid")])
def test_duration_pass_needs_matching_verbatim_delivered_quote(disclosed, expected):
    """通过的一句话必须来自账本；后台数据与虚构的说明都不能代替这句回复。"""
    turn = _turn(1)
    turn["tool_results"].append({"name": "join_game", "called": True, "allowed": True,
                                 "result": {"recorded_status": "confirmed", "game": {"game_id": "g1"}}})
    stated = "这桌打三个钟头，给你留好了。"
    messages = [_chat("u1", text="我来，能打三个钟头。"), _chat("a1", reply_to="u1", text=stated if disclosed else "留好了。")]
    client = _JudgeClient(agent_evidence=("E02", "E03", "E04"), extra_findings=({
        "check": DURATION_CHECK, "verdict": "pass", "evidence_ids": ("E02", "E03", "E04"),
        "reason": f"实际来话「我来，能打三个钟头。」；回复「{stated}」确认时长。",
    },))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn], records=messages)
    assert result["status"] == expected

def test_real_corpus_fragments_keep_every_original_bubble_and_do_not_expose_other_threads():
    """真人微信可把钟点和问题拆在连续气泡里，评审仍须逐条对齐同一入站。"""
    path = (Path(__file__).resolve().parents[2] / "eval" / "golden"
            / "real_owner_chat_timeline_20260627_20260719_owner_b.json")
    source = json.loads(path.read_text(encoding="utf-8"))
    case = next(item for item in source["episodes"] if item["id"] == "intermediary_role_correction")
    opening = [row for row in source["messages"] if case["start_turn"] <= row["turn"] < case["start_turn"] + 4]
    assert [row["role"] for row in opening] == ["customer", "customer", "customer", "boss"]
    ids = ("f1", "f2", "u1")
    chat = [_chat(key, text=row["text"]) for key, row in zip(ids, opening)]
    chat.insert(1, _chat("cross", conversation="unrelated", text="此对话不可见"))
    chat.extend((_chat("late", text="聚合锚点之后的新来话"), _chat("a1", reply_to="u1", text=opening[-1]["text"])))
    turn = _turn(1)
    turn["message_ids"] = ["cross", "late", "f2", "u1", "f1", "f1"]
    client = _JudgeClient(role_evidence=("E02", "E03", "E04"), agent_evidence=("E05", "E06"))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn], records=chat)
    assert result["verdict"] == "pass"
    prompt, payload = client.messages
    evidence = json.loads(payload["content"])["evidence"]
    received = [row for row in evidence if row["kind"] == "customer_message"]
    assert [row["text"] for row in received] == [row["text"] for row in opening[:3]]
    assert {row["input_batch"] for row in received} == {"E04"}
    assert "聚合锚点之后的新来话" not in payload["content"] and "此对话不可见" not in payload["content"]
    assert "短答、合理改口和自然收尾可以通过" in prompt["content"]
    omitted = _JudgeClient(role_evidence=("E03", "E04"), agent_evidence=("E05",))
    bad = review_dialogue(SemanticJudge(omitted), customer_id="a", turns=[turn], records=chat)
    assert bad["error"] == "customer pass must cite every archived action"


def test_reply_judge_only_uses_time_range_bounds_and_cannot_cite_private_world_as_owner_evidence():
    """世界变化仅供顾客行为验收，实际查询区间仍能按同一回复复核。"""
    turn = _turn(1)
    event = turn["decision"]["context"]["events"][0]
    event.update(kind="traffic", at="2026-11-25T18:21:00+08:00")
    event["details"].update(delay_minutes=10, location="near_store", another_person="foreign")
    bounds = {"start_at": "2026-11-25T19:00:00+08:00", "end_at": "2026-11-25T21:00:00+08:00"}
    turn["tool_results"][0]["result"]["requirement"] = {"start_time_range": {**bounds, "private_field": "not-allowed"}}
    client = _JudgeClient(role_evidence=("E01", "E02"), agent_evidence=("E01", "E03", "E04"))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn],
                             records=[_chat("u1"), _chat("a1", reply_to="u1")],
                             additional_instructions="专门核对这条改期。")
    assert result["error"] == "judgment does not cite its review subject"
    prompt, payload = client.messages
    role, _, _, execution = json.loads(payload["content"])["evidence"]
    assert role["private_world"] == {"kind": "traffic", "at": event["at"], "delay_minutes": 10,
                                     "location": "near_store"}
    assert execution["tools"][0]["result"]["query"] == {"start_time_range": bounds}
    assert "foreign" not in payload["content"] and "not-allowed" not in payload["content"]
    assert "专门核对这条改期。" in prompt["content"]


@pytest.mark.parametrize(("customer", "shop", "role_verdict", "agent_verdict"), [
    ("群里一缺那桌是川麻？另外七点后有没一块红中？", "你说群里1号那桌吗？七点后暂时没有一块的红中。", "fail", "pass"),
    ("群里1号是川麻吧？另有七点后的一块红中吗？", "等我问店员。", "pass", "fail"),
    ("群里那桌是川麻？", "你问的是1号川麻，还是后面发的红中？", "pass", "pass"),
])
def test_archive_and_judge_distinguish_customer_clarity_from_shop_handling(customer, shop, role_verdict, agent_verdict):
    """顾客的指代和店家的适当澄清分别引用来话、看板与实际送达。"""
    board = {**_chat("outbound:board:before", text="当前缺人局：\n1、人齐开 川麻 2～3元 无烟 173"),
             "direction": "outbound", "kind": "group", "sender_id": "simulation-agent", "recipient_id": "room-a",
             "sent_at": "2026-11-25T18:20:00+08:00"}
    later = {**board, "message_id": "outbound:board:after", "text": "本轮尚未见到的第二版"}
    unseen = {**board, "message_id": "outbound:board:unseen", "text": "未送达给这位顾客"}
    turn = _turn(1)
    turn["decision"]["context"]["history"] = [dict(board), dict(later)]
    inbound = _chat("u1", text=customer)
    records = [board, unseen, inbound, _chat("a1", reply_to="u1", text=shop), later]
    client = _JudgeClient(role_evidence=("E01", "E03"), agent_evidence=("E03", "E04", "E05"),
                          role_verdict=role_verdict, agent_verdict=agent_verdict)
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn], records=records)
    assert result["status"] == "valid"
    assert {finding["check"]: finding["verdict"] for finding in result["findings"]} == {
        ROLE_CHECK: role_verdict, AGENT_CHECK: agent_verdict,
    }
    system, payload = client.messages
    assert "收件人视角是否好懂" in system["content"]
    assert "针对性追问" in system["content"] and "不能代替可回答的部分" in system["content"]
    assert "不能归成顾客问法" in system["content"]
    assert all(text not in payload["content"] for text in (later["text"], unseen["text"]))
    evidence = json.loads(payload["content"])["evidence"]
    assert evidence[0]["kind"] == "group_delivery" and evidence[0]["text"] == board["text"]
    assert evidence[2]["kind"] == "customer_message" and evidence[2]["channel_kind"] == "private"
    assert result["findings"][0]["evidence_ids"] == (board["message_id"], "u1")
    assert result["findings"][1]["evidence_ids"] == ("u1", "a1", "agent-trace-1")


def test_judge_receives_creation_duration_limit_without_inventing_fixed_duration():
    """创建条件中的三小时上限可审计，未知的约定时长依然未知。"""
    turn = _turn(1)
    turn["tool_results"].append({"name": "create_game", "called": True, "allowed": True,
        "result": {"game": {"game_id": "g1", "status": "forming", "requirement": {"max_duration_hours": 3}}}})
    client = _JudgeClient(agent_evidence=("E03", "E04"))
    review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn],
                    records=[_chat("u1"), _chat("a1", reply_to="u1")])
    execution = json.loads(client.messages[-1]["content"])["evidence"][-1]
    assert execution["tools"][-1]["result"]["game"]["terms"] == {"max_duration_hours": 3}


def test_judge_keeps_same_turn_table_seats_and_terms_from_read_receipt():
    """真实查局回执支持人齐开及缺三位，群转私聊的可见与引用范围不扩大。"""
    board = _chat("outbound:board:visible", text="当前缺人局：\n1、人齐开 杭麻 1～3元 烟况待定 173") | {
        "direction": "outbound", "kind": "group", "sender_id": "simulation-agent", "recipient_id": "room-a",
        "sent_at": "2026-11-26T11:07:30+08:00",
    }
    turn = _turn(1, cause=board["message_id"])
    turn["decision"]["context"]["history"] = [dict(board)]
    receipt = turn["tool_results"][0]["result"]
    receipt["requirement"] = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking"}
    receipt["matches"] = [
        {"game": {"game_id": "g1", "status": "forming", "planned_start_at": None, "planned_end_at": None,
                  "remaining_seats": 3,
                  "requirement": {"game_type": "hangzhou_mahjong", "start_time_kind": "asap_when_full",
                                  "stake_range": {"min": 1, "max": 3, "internal_label": "private-field"}},
                  "seat_summary": {"claimed_seats": 1, "remaining_seats": 3, "seats_total": 4,
                                   "participant_ids": ["private-participant"]},
                  "participants": ["private-game-member"]},
         "match_kind": "exact"},
        {"game": {"game_id": "g2", "remaining_seats": 2}},
    ]
    raw_before = json.dumps(turn, sort_keys=True)
    records = [board, _chat("u1", text="老板 杭麻这桌还有位吗 一底的 我中午能到"),
               _chat("a1", reply_to="u1", text="杭麻1～3块，现在1个人、还缺3位，人齐就开，烟况还没定。")]
    client = _JudgeClient(role_evidence=("E01", "E03"), agent_evidence=("E03", "E04", "E05"))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[turn], records=records,
                             rolling_check_ids=True)
    assert (result["status"], result["verdict"], result["projection"]["version"]) == ("valid", "pass", 9)
    assert json.dumps(turn, sort_keys=True) == raw_before
    system, payload = client.messages
    assert "不要从查询参数反推某张结果桌的条件" in system["content"]
    assert all(secret not in payload["content"] for secret in ("private-field", "private-participant", "private-game-member"))
    evidence = json.loads(payload["content"])["evidence"]
    assert evidence[0]["kind"] == "group_delivery" and evidence[0]["text"] == board["text"]
    lookup = evidence[-1]["tools"][0]["result"]
    assert lookup["query"]["smoke_preference"] == "no_smoking"
    assert lookup["matches"]["options"] == [
        {"game_id": "g1", "status": "forming", "remaining_seats": 3,
         "seat_summary": {"claimed_seats": 1, "remaining_seats": 3, "seats_total": 4},
         "terms": {"game_type": "hangzhou_mahjong", "stake_range": {"min": 1, "max": 3},
                   "start_time_kind": "asap_when_full"},
         "match_kind": "exact"},
        {"game_id": "g2", "remaining_seats": 2},
    ]
    assert result["evidence"][-1]["tools"][0]["result"] == lookup
    wrong_scope = _JudgeClient(role_evidence=("E01", "E03"), agent_evidence=("E01", "E04", "E05"))
    denied = review_dialogue(SemanticJudge(wrong_scope), customer_id="a", turns=[turn], records=records)
    assert (denied["status"], denied["error"]) == ("invalid", "judgment does not cite its review subject")


@pytest.mark.parametrize("owner, actual_owner", [("a", True), ("other", False)])
def test_silent_customer_reviews_only_personal_operator_invitation(owner, actual_owner):
    """本人的实际邀约可触发合理沉默；另一个客户收到的话不可进入顾客证据。"""
    invitation = {"message_id": "invite", "direction": "owner", "kind": "private", "text": "今晚还有空位",
                  "sender_id": "simulation-owner", "recipient_id": owner, "conversation_id": "dm-a",
                  "sent_at": "2026-11-25T18:20:00+08:00"}
    wait = _turn(1, cause="invite")
    wait.pop("message_id")
    wait["virtual_at"] = "2026-11-25T18:21:00+08:00"
    wait["decision"]["action"] = {"action": "wait"}

    class RoleJudge:
        def complete(self, messages, *, trace_id, timeout_seconds):
            self.payload = json.loads(messages[-1]["content"])
            action = self.payload["evidence"][-1]
            return json.dumps({"verdict": "pass", "findings": [{
                "check": ROLE_CHECK, "verdict": "pass", "reason": "可以不回。", "evidence_ids": [action["evidence_id"]],
            }]}, ensure_ascii=False)

    client = RoleJudge()
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[wait], records=[invitation], role_only=True)
    assert result["verdict"] == "pass"
    assert client.payload["checks"] == [ROLE_CHECK]
    roles = client.payload["evidence"]
    assert len(roles) == (2 if actual_owner else 1)
    assert roles[-1]["caused_by"] == ("E01" if actual_owner else None)
    if actual_owner:
        assert roles[0]["kind"] == "owner_invitation" and roles[0]["text"] == invitation["text"]

def test_role_only_flag_cannot_skip_review_of_an_actual_shop_reply():
    """静默对照专门免除不存在的店家回复，不能跳过已经发生的执行。"""
    client = _JudgeClient(agent_evidence=("E03",))
    result = review_dialogue(SemanticJudge(client), customer_id="a", turns=[_turn(1)], role_only=True,
                             records=[_chat("u1"), _chat("a1", reply_to="u1")])
    assert result["verdict"] == "pass"
    assert json.loads(client.messages[-1]["content"])["checks"] == [ROLE_CHECK, AGENT_CHECK]


def _peer_case():
    """用实际出入方向构造搭子来话，保留可供错误归因的假店家轨迹。"""
    incoming = {**_chat("peer-in", recipient="a", conversation="peer-a-b"), "direction": "peer",
                "sender_id": "b", "sent_at": "2026-11-25T18:20:00+08:00", "text": "你今晚来不来？"}
    outgoing = {**_chat("u1", recipient="b", conversation="peer-a-b"), "direction": "peer",
                "sent_at": "2026-11-25T18:21:00+08:00", "text": "我七点半以后有空。"}
    turn = _turn(1, cause="peer-in")
    turn["virtual_at"] = outgoing["sent_at"]
    turn["decision"]["context"]["history"] = [dict(incoming)]
    return turn, incoming, outgoing


@pytest.mark.parametrize("case", ["expired", "after_inquiry", "same_tick", "unseen", "future_board", "mismatched_text"])
def test_rolling_board_expiry_uses_original_receipt_and_pre_inquiry_change(case):
    """失效证明仅绑定实际看过的旧看板及早于来话的该桌变化。"""
    board_id = "outbound:board:historical"
    published_text = "2、人齐开 杭麻 1～3元 无烟 173"
    board = {"message_id": board_id, "sender_id": "simulation-agent", "recipient_id": "public-room",
             "direction": "outbound", "kind": "group", "text": published_text,
             "sent_at": "2026-11-26T19:00:00+08:00" if case == "future_board" else "2026-11-26T11:37:30+08:00"}
    incoming = dict(_chat("u1", text="看板上那桌无烟杭麻还缺人吗？"), sent_at="2026-11-26T18:35:00+08:00")
    turn = _turn(1)
    turn["decision"]["context"]["history"] = [] if case == "unseen" else [dict(board)]
    if case == "mismatched_text":
        turn["decision"]["context"]["history"][0]["text"] = "另一桌"
    hour = {"after_inquiry": "18:40", "same_tick": "18:35"}.get(case, "17:05")
    expired = {"item_no": 2, "published_text": published_text, "cancelled_at": f"2026-11-26T{hour}:00+08:00"}
    prepared = prepare_dialogue_review(customer_id="a", turns=[turn], records=[board, incoming],
        delivered_history=True, board_expirations={board_id: [expired]})
    evidence = next(row for row in prepared.request if row["kind"] == "agent_execution")
    if case == "expired":
        assert prepared.aliases[evidence["board_receipt"]] == board_id
        assert evidence["prior_board_expirations"] == [expired]
        assert prepared.projected.allowed[AGENT_CHECK] >= {"agent-trace-1"}
    else:
        assert "prior_board_expirations" not in evidence


@pytest.mark.parametrize("mismatch", [None, "foreign_recipient", "unseen", "wrong_sender", "future"])
def test_peer_projection_uses_exact_own_cause_without_faking_shop_execution(mismatch):
    """只给评审实际看到的因果来话，搭子回复只作为角色动作。"""
    turn, incoming, outgoing = _peer_case()
    history = turn["decision"]["context"]["history"]
    if mismatch == "foreign_recipient":
        incoming["recipient_id"] = history[0]["recipient_id"] = "another-customer"
    elif mismatch == "unseen":
        history.clear()
    elif mismatch == "wrong_sender":
        history[0]["sender_id"] = "another-customer"
    elif mismatch == "future":
        incoming["sent_at"] = history[0]["sent_at"] = "2026-11-25T18:22:00+08:00"
    foreign = {**incoming, "message_id": "foreign-peer", "recipient_id": "someone-else", "text": "foreign-peer-secret"}
    history.append(foreign)
    prepared = prepare_dialogue_review(customer_id="a", turns=[turn], records=[foreign, incoming, outgoing],
                                       role_only=True, delivered_history=True)
    assert prepared.checks == (ROLE_CHECK,)
    rows = {row["evidence_id"]: row for row in prepared.archive}
    assert set(rows) == {"role-trace-1", "u1"} | ({"peer-in"} if mismatch is None else set())
    assert rows["role-trace-1"]["caused_by"] == ("peer-in" if mismatch is None else None)
    assert rows["role-trace-1"]["input"] == "u1" and rows["u1"]["kind"] == "peer_message"
    assert rows["u1"]["text"] == outgoing["text"]
    assert prepared.projected.anchors[ROLE_CHECK] == {"u1"}
    assert prepared.projected.allowed[AGENT_CHECK] == set()
    if mismatch is None:
        assert rows["peer-in"]["kind"] == "peer_delivery" and rows["peer-in"]["text"] == incoming["text"]


def test_peer_evidence_cannot_support_shop_claim_or_be_skipped_in_a_role_pass():
    """同一顾客后来联系店家时，独立判决仍须引用各路自己的已归档证据。"""
    turn, incoming, outgoing = _peer_case()
    records = [incoming, outgoing, _chat("u2"), _chat("a2", reply_to="u2")]
    turns = [turn, _turn(2)]
    prepared = prepare_dialogue_review(customer_id="a", turns=turns, records=records,
                                       role_only=True, delivered_history=True)
    aliases = {original: alias for alias, original in prepared.aliases.items()}
    kwargs = {"customer_id": "a", "turns": turns, "records": records, "role_only": True, "delivered_history": True}
    good = _JudgeClient(role_evidence=(aliases["u1"], aliases["u2"]), agent_evidence=(aliases["a2"],))
    assert review_dialogue(SemanticJudge(good), **kwargs)["status"] == "valid"
    assert "顾客项通过须逐条引用实际发送的 peer_message" in good.messages[0]["content"]

    shop_overreach = _JudgeClient(role_evidence=(aliases["u1"], aliases["u2"]),
                                  agent_evidence=(aliases["a2"], aliases["peer-in"]))
    assert review_dialogue(SemanticJudge(shop_overreach), **kwargs)["error"] == "judgment does not cite its review subject"
    incomplete_role = _JudgeClient(role_evidence=(aliases["u2"],), agent_evidence=(aliases["a2"],))
    assert review_dialogue(SemanticJudge(incomplete_role), **kwargs)["error"] == "customer pass must cite every archived action"
