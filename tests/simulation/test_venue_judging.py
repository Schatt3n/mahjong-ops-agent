"""验证顾客和店家分别有可回查的评测证据，并保留模型故障。"""

import json
from pathlib import Path

import pytest

from .venue_actors import ActorContext, ModelActor, SemanticJudge, generate_personas
from .venue_judging import AGENT_CHECK, DURATION_CHECK, FIRST_CONTACT_CHECK, ROLE_CHECK, review_dialogue


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
        findings = [
            {"check": ROLE_CHECK, "verdict": self.role_verdict, "evidence_ids": self.role_evidence, "reason": "问法和偏好符合。"},
            {"check": AGENT_CHECK, "verdict": self.agent_verdict, "evidence_ids": self.agent_evidence, "reason": "核对客户消息与占座结果。"},
            *self.extra_findings,
        ]
        verdicts = {finding["verdict"] for finding in findings}
        overall = next(verdict for verdict in ("fail", "inconclusive", "pass") if verdict in verdicts)
        return json.dumps({"verdict": overall, "findings": findings}, ensure_ascii=False)


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
