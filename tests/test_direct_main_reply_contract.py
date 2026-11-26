"""核对主回复、上下文和审查合同同向；离线测试不冒充真实模型语义通过。"""

from datetime import datetime
import json
from pathlib import Path

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentContextBuilder, AgentRuntime, CustomerProfile, InMemoryAgentStore, InMemoryTraceRecorder,
    SQLiteAgentStore, StaticAgentClient, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.customer_visible_contract import (
    PREFERRED_OPERATION_ACK_PHRASES, customer_visible_table_start_violations,
)
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_result_dict
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract
from mahjong_agent_runtime.models import using_clock
from tests.simulation.venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from tests.simulation.venue_fixture import _call, _reply


ROOT = Path(__file__).resolve().parents[1]


def main_context():
    """可靠画像与客户本次条件一同送给主模型，最终话术无须依赖另一次润色获取事实。"""
    store = InMemoryAgentStore()
    store.upsert_customer(CustomerProfile(customer_id="customer", display_name="客户",
        preferred_games=["hangzhou_mahjong"], preferred_stakes=["0.5"], smoke_preference="no_smoke"))
    message = UserMessage(conversation_id="direct-reply", sender_id="customer", sender_name="客户",
                          text="帮我找六点半无烟的", message_id="direct-input")
    payload = AgentContextBuilder(store, ToolGateway(store)).build(message, trace_id="direct-reply").payload
    return message, payload


def test_main_context_assigns_final_wording_and_state_to_the_agent_with_all_facts():
    message, payload = main_context()
    assert payload["current_message"]["text"] == message.text
    assert payload["sender_profile"]["preferred_stakes"] == ["0.5"]
    assert "最终客户可见回复" in payload["output_contract"]["field_types"]["reply_to_user"]
    guidance = payload["planning_contract"]["tool_selection_hints"]["final_customer_reply"]
    assert "已明确且未变化" in guidance and "不依赖后续润色" in guidance
    assert "waiting_user" in payload["output_contract"]["stop_reason_contract"]["waiting_decision"]


@pytest.mark.parametrize("text", ["七点三缺一，你打吗？", "七点三缺一，0.5无烟杭麻，你打吗？"])
def test_real_option_review_distinguishes_omission_and_minor_echo_from_fact_errors(text):
    """省略和偶尔复述都不改变真实局事实；判定器仍需检查不同条件和真正的新问题。"""
    message, context = main_context()
    action = AgentAction(goal="介绍真实现成局并等客户决定", objective_status="waiting_user", reasoning_summary="",
                         reply_to_user=text, objective_state={"known_facts": {"stake": "0.5", "smoke_preference": "no_smoke"}})
    context["active_game_visible_summaries"] = [{"user_visible_summary": "七点三缺一", "stake": "0.5", "smoke_preference": "no_smoke"}]
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="reply_to_user", review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": text}])
    fidelity = payload["semantic_fidelity_contract"]
    assert fidelity["allow_known_user_echo_omission"] is True
    rules = "\n".join(fidelity["rules"])
    assert "包括真实局选项" in rules and "已明确且未变化" in rules
    assert "偶尔复述" in rules and "风格瑕疵" in rules
    # 核验实际待答协议，避免把某个示例问句写成必须保留的审查口令。
    assert "waiting_user" in rules and "semantic_contradiction" in rules
    assert payload["action_boundary"]["objective_status"] == "waiting_user"
    assert payload["action_boundary"]["turn_state_semantics"] == {
        "scope": "current_conversation_turn",
        "current_user_answer_requirement": "required",
        "business_goal_completion_implied": False,
        "mismatch_violation": "semantic_contradiction",
    }
    assert payload["review_items"][0]["text"] == text


def test_recipient_boundaries_remain_distinct_from_known_condition_omission():
    """不能把发起人已知条件当成另一候选人也知道，更不能借省略授权披露关系。"""
    message, context = main_context()
    action = AgentAction(goal="候选邀约", objective_status="completed", reasoning_summary="", reply_to_user="")
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="tool_message", review_items=[{"item_id": "candidate", "recipient_id": "other", "text": "七点0.5无烟，打吗？"}])
    assert payload["semantic_fidelity_contract"]["allow_known_user_echo_omission"] is False
    rules = "\n".join(payload["semantic_fidelity_contract"]["rules"])
    assert "实际收件人" in rules
    assert "对他尚未知" in rules


def test_on_site_arrival_guidance_separates_reported_arrival_seat_and_staffed_handoff():
    """到店来话、已留座和只读房态会进入审查，三者都不能支持现场接应。"""
    _, context = main_context()
    message = UserMessage(conversation_id="direct-on-site", sender_id="customer", sender_name="客户",
                          text="老板，我到了，七点那桌杭麻底注1无烟的，在哪桌坐？", message_id="on-site-input")
    context["sender_active_game_memberships"] = [{
        "game_id": "table-at-seven", "participant_status": "confirmed", "participation_already_recorded": True,
    }]
    context["turn_tool_evidence"] = [{
        "name": "check_room_availability", "called": True, "allowed": True, "error": None,
        "result": {"available_room_ids": ["东一包间"], "configured": True},
    }]
    text = "七点那桌给你留着呢，在东一包间，我带你过去。"
    action = AgentAction(goal="答复客户到店问位置", objective_status="completed", reasoning_summary="",
                         reply_to_user=text)
    review = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="reply_to_user", review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": text}])

    assert review["review_items"][0]["text"] == text
    assert review["sender_active_game_memberships"] == context["sender_active_game_memberships"]
    assert review["turn_tool_evidence"][0]["result"]["available_room_ids"] == ["东一包间"]
    prompts = ROOT / "src/mahjong_agent_runtime/prompts"
    main = (prompts / "agent_runtime_system.md").read_text(encoding="utf-8")
    reviewer = (prompts / "agent_runtime_reply_self_review.md").read_text(encoding="utf-8")
    main_rule = next(line for line in main.splitlines() if "顾客到了问坐哪桌" in line)
    review_rule = next(line for line in reviewer.splitlines() if "审查到店交接和问桌号" in line)
    assert all(phrase in main for phrase in ("实际签到须由可信门店事件证明", "不要为此重报座"))
    assert all(phrase in main_rule for phrase in ("实际分配且允许披露", "不能证明分配", "不虚构前台", "有责任人的现场交接"))
    assert all(phrase in review_rule for phrase in ("同一局已分配", "有责任人的真实交接", "semantic_contradiction", "确认已留席"))


def test_review_uses_same_tables_structured_seats_over_stored_free_text():
    """已保存的局摘要也可能写错，保留原始证据并明确以最新实际缺口审查。"""
    message, context = main_context()
    seat_summary = {"claimed_seats": 1, "remaining_seats": 3, "seats_total": 4}
    stale_summary = "杭麻无烟，19:30 三缺一"
    game = {"game_id": "table-1930", "planned_start_at": "2026-11-26T19:30:00+08:00",
            "seat_summary": seat_summary, "requirement": {"user_visible_summary": stale_summary}}
    context["turn_tool_evidence"] = [{"name": "create_game", "called": True, "allowed": True,
                                     "error": None, "result": {"game": game}}]
    context["active_game_visible_summaries"] = [{"game_id": game["game_id"], "seat_summary": seat_summary,
                                                "user_visible_summary": stale_summary}]
    text = "19:30三缺一。"
    action = AgentAction(goal="告诉顾客同桌目前缺几人", objective_status="completed",
                         reasoning_summary="", reply_to_user=text)
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="reply_to_user", review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": text}])
    assert payload["turn_tool_evidence"][0]["result"]["game"]["seat_summary"]["remaining_seats"] == 3
    assert payload["active_game_visible_summaries"][0]["user_visible_summary"] == stale_summary
    assert payload["semantic_fidelity_contract"]["seat_count_evidence"] == [{
        "game_id": "table-1930", "planned_start_at": "2026-11-26T19:30:00+08:00", **seat_summary,
        "four_seat_shorthand": "173 / 一缺三", "remaining_seats_display": "还缺3人",
    }]
    rule = next(rule for rule in payload["semantic_fidelity_contract"]["rules"] if "173/一缺三" in rule)
    assert "最新成功结果的 seat_summary.remaining_seats" in rule
    assert "claimed_seats 是已占位数" in rule
    assert "结构化席位优先于自由填写" in rule
    assert "semantic_contradiction" in rule


def test_room_label_is_projected_only_to_a_current_private_participant(tmp_path):
    """同局房间来自店员已落库分配，顾客能看到桌名和可用时刻，其他会话或群内看不到。"""
    from datetime import timedelta

    assigned_at = datetime.fromisoformat("2026-11-26T18:55:00+08:00")
    start = assigned_at + timedelta(minutes=5)
    store = SQLiteAgentStore(tmp_path / "assigned.sqlite3")
    with using_clock(lambda: assigned_at):
        store.configure_rooms(["staff-east-internal-id"])
        game, _ = store.create_game(conversation_id="host-chat", organizer_id="host", organizer_name="牌友",
            requirement={"game_type": "hangzhou_mahjong", "start_time_kind": "scheduled",
                         "planned_start_at": start.isoformat(), "duration_hours": 4},
            known_players=[], trace_id="seed-ready")
        for person in ("customer", "a", "b"):
            store.join_game(game_id=game.game_id, customer_id=person, display_name=person, trace_id="join-ready")
        store.assign_game_room(game.game_id, room_id="staff-east-internal-id", public_room_label="东一桌",
                               at=assigned_at, trace_id="owner-confirmed")
        builder = AgentContextBuilder(store, ToolGateway(store))
        inbound = UserMessage("guest-chat", "customer", "顾客", "老板我到了，坐哪桌？", sent_at=assigned_at)
        built = builder.build(inbound, trace_id="responder").payload
        expected = {"public_label": "东一桌", "available_from": start.isoformat(), "access_window_started": False}
        visible, = built["active_game_visible_summaries"]
        assert visible["game_id"] == game.game_id and visible["assigned_room"] == expected
        assert "staff-east-internal-id" not in json.dumps(built, ensure_ascii=False)

        for message in (UserMessage("host-chat", "outsider", "顾客", inbound.text, sent_at=assigned_at),
                        UserMessage("room-chat", "customer", "顾客", inbound.text,
                                    sent_at=assigned_at, metadata={"source": "group", "is_room": True})):
            payload = builder.build(message, trace_id="not-authorized").payload
            assert "东一桌" not in json.dumps(payload, ensure_ascii=False)

    assert store.require_game(game.game_id).actual_started_at is None
    assert not store.require_game(game.game_id).actual_attendee_ids


def test_review_projects_distinct_tables_and_updates_only_the_same_game_id():
    """邻近桌、同刻另一桌与失败工具都不能冲销这张桌最新的实际空位。"""
    message, context = main_context()

    def game(id_, start, remaining):
        return {"game_id": id_, "planned_start_at": start,
                "seat_summary": {"claimed_seats": 4 - remaining, "remaining_seats": remaining, "seats_total": 4}}

    earlier = "2026-11-26T19:30:00+08:00"
    later = "2026-11-26T20:05:00+08:00"
    first = game("table-1", earlier, 3)
    same_time = game("table-2", earlier, 1)
    nearby = game("table-3", later, 2)
    context["active_game_visible_summaries"] = [game("table-1", earlier, 1), game("table-4", earlier, 3)]
    context["turn_tool_evidence"] = [
        {"name": "search_current_games", "called": True, "allowed": True, "error": None,
         "result": {"matches": [{"game": first}, {"game": same_time}],
                    "nearby_options": [{"game": nearby, "requires_timing_confirmation": True}],
                    "requirement": game("query-is-not-a-table", earlier, 1)}},
        {"name": "search_customers", "called": True, "allowed": True, "error": None,
         "result": {"continuation": {"authoritative_facts": {"game_id": "table-1", "seat_summary": {
             "claimed_seats": 2, "remaining_seats": 2, "seats_total": 4,
         }}}}},
        {"name": "join_game", "called": True, "allowed": False, "error": "join denied",
         "result": {"game": game("table-1", earlier, 1)}},
    ]
    text = "七点半一桌还差俩，另一桌三缺一；八点零五也还差俩。你打哪桌？"
    action = AgentAction(goal="给顾客比较真实的牌局选项", objective_status="waiting_user",
                         reasoning_summary="", reply_to_user=text)
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="reply_to_user", review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": text}])

    actual = {row["game_id"]: row for row in payload["semantic_fidelity_contract"]["seat_count_evidence"]}
    assert payload["semantic_fidelity_contract"]["game_availability_evidence"]["latest_search"][
        "options_requiring_timing_confirmation"
    ] == [{"game_id": "table-3", "planned_start_at": later}]
    assert {key: (row["planned_start_at"], row["remaining_seats"], row["four_seat_shorthand"])
            for key, row in actual.items()} == {
        "table-1": (earlier, 2, "272 / 二缺二"),
        "table-2": (earlier, 1, "371 / 三缺一"),
        "table-3": (later, 2, "272 / 二缺二"),
        "table-4": (earlier, 3, "173 / 一缺三"),
    }


@pytest.mark.parametrize(("inbound", "sent_at", "opening", "terms", "wrong", "corrected"), [
    (
        "老板好，17:30 那桌 2 到 3 的底还在凑人吗，我 18:30 才能到，能搭一桌的话算我一个，"
        "最多打三个半小时，行的话跟我说一声",
        "2026-11-26T11:39:30+08:00", "17:30",
        {"stake_range": {"min": 2, "max": 3}},
        "17:30那桌还在凑，底2到3、三缺一；你18:30才到、最多打三个半小时，"
        "这桌能不能等到那时候我这边还没答复，先帮你登记问一下。",
        "17:30那桌还在凑，底2到3、一缺三。18:30能不能到时再加入，桌上还没定。",
    ),
    (
        "老板，今晚七点有红中麻将吗？",
        "2026-11-26T18:00:00+08:00", "19:00",
        {"game_type": "red_center_mahjong", "stake": "1", "smoke_preference": "no_smoking"},
        "七点有一桌红中，1块无烟，三缺一，你打吗？",
        "七点有一桌红中，1块无烟，一缺三，你打吗？",
    ),
], ids=["archived-delayed-table", "archived-timeout-recovery"])
def test_live_seats_override_erroneous_review_approval_before_customer_delivery(
    tmp_path, inbound, sent_at, opening, terms, wrong, corrected,
):
    """复现两次错误的审查放行，查局、修正回灌和顾客收件均走正式执行链。"""
    at = datetime.fromisoformat(sent_at)
    opening_at = datetime.fromisoformat(f"{at.date().isoformat()}T{opening}:00+08:00")
    requirement = {**terms, "start_time_kind": "scheduled", "planned_start_at": opening_at.isoformat()}
    lookup = _call("search_current_games", {"requirement": requirement})
    model = StaticAgentClient([lookup, _reply(wrong, status="waiting_user"), _reply(corrected, status="waiting_user")])
    reviewer = StaticAgentClient([
        json.dumps({"approved": True, "needs_human": False, "violations": [], "reasoning_summary": "座位数量正确",
                    "item_reviews": [{"item_id": "reply_to_user", "approved": True, "violations": [],
                        "reasoning_summary": "座位数量正确", "suggested_safe_text": text, "follow_up_commitments": []}]},
                   ensure_ascii=False)
        for text in (wrong, corrected)
    ])
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    traces = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=model, store=store, trace_recorder=traces,
                           reply_self_review_enabled=True, reply_self_review_client=reviewer)
    channel = SandboxChannel(runtime, room_id="seat-evidence", identities={"host": "牌友", "guest": "顾客"},
                             archive_path=tmp_path / "chat.sqlite3")
    message = MessageEnvelope("archived-seat-claim", "guest", inbound, at, "private", AGENT_ID)

    with using_clock(lambda: at):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id("host"),
            organizer_id="host", organizer_name="牌友", requirement={**requirement, "duration_hours": 4},
            known_players=[{"customer_id": "host", "display_name": "牌友", "seat_count": 1}], trace_id="seed")
        result = channel.send(message)

    history = traces.get_trace(result["trace_id"])
    lookup_result, = [event.content for event in history
                      if event.step == "tool_result" and event.content.get("name") == "search_current_games"]
    match, = lookup_result["result"]["matches"]
    assert match["game"]["game_id"] == game.game_id and match["game"]["seat_summary"]["remaining_seats"] == 3
    rejected, approved = [event.content for event in history if event.step == "customer_visible_content_review_result"]
    assert rejected["raw_approved"] is True and rejected["approved"] is False
    rule = "customer_visible_contract:seat_count.current_table_mismatch"
    assert rule in rejected["item_reviews"][0]["violations"]
    assert approved["approved"] is True and not approved["item_reviews"][0]["violations"]
    retry = json.loads(model.calls[2]["messages"][-1]["content"])
    feedback, = [item["result"] for item in retry["turn_tool_evidence"]
                 if item["name"] == "customer_visible_content_review"]
    assert rule in feedback["item_reviews"][0]["violations"]
    assert rejected["item_reviews"][0]["suggested_safe_text"] == wrong
    assert feedback["item_reviews"][0]["suggested_safe_text"] == ""
    receipt, = result["archive_outputs"]
    assert receipt["text"] == corrected and receipt["recipient_id"] == "guest"
    assert receipt["source_message_id"] == message.message_id and receipt["trace_id"] == result["trace_id"]
    assert [row["text"] for row in channel.events_for("guest")] == [inbound, corrected]
    assert store.require_game(game.game_id).to_dict() == game.to_dict()


@pytest.mark.parametrize("review_enabled", [True, False])
def test_expired_planned_opening_cannot_be_delivered_as_actual_start(tmp_path, review_enabled):
    """真实查询和私聊收件复现原局仍在凑人，错误的模型批准或关闭审查均不放行。"""
    at = datetime.fromisoformat("2026-11-25T19:35:00+08:00")
    opening_at = "2026-11-25T19:30:00+08:00"
    inbound = "老板，那桌杭麻1到3档、烟都行的，现在还在凑人吗？我最早八点过一点能到，能留位吗？"
    wrong = "有一桌19:30开的，1到3档、烟都行，现在就一个人，还差三位。不过你八点过才到，人家已经开打了，要不我先帮你问问桌上能不能等你到？"
    corrected = "原定19:30那桌杭麻还在凑，差三位。你八点过才到，这桌能不能等还不确定，要我先帮你问问吗？"
    terms = {"game_type": "hangzhou_mahjong", "stake_range": {"min": 1, "max": 3}, "smoke_preference": "any"}
    lookup = _call("search_current_games", {"requirement": {"game_type": "hangzhou_mahjong",
        "stake_range": {"min": 1, "max": 3}, "start_time_kind": "asap_when_full"}, "lookup_scope": "specific_table"})
    model = StaticAgentClient([lookup, _reply(wrong, status="waiting_user"), _reply(corrected, status="waiting_user")])
    reviewer = StaticAgentClient([
        json.dumps({"approved": True, "needs_human": False, "violations": [], "reasoning_summary": "同意",
                    "item_reviews": [{"item_id": "reply_to_user", "approved": True, "violations": [],
                        "reasoning_summary": "同意", "suggested_safe_text": text, "follow_up_commitments": []}]},
                   ensure_ascii=False)
        for text in (wrong, corrected)
    ])
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    traces = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=model, store=store, trace_recorder=traces,
                           reply_self_review_enabled=review_enabled, reply_self_review_client=reviewer)
    channel = SandboxChannel(runtime, room_id="archived-v4-opening", identities={"host": "牌友", "guest": "顾客"},
                             archive_path=tmp_path / "chat.sqlite3")
    message = MessageEnvelope("archived-v4-opening", "guest", inbound, at, "private", AGENT_ID)

    with using_clock(lambda: at):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id("host"), organizer_id="host",
            organizer_name="牌友", requirement={**terms, "start_time_kind": "scheduled", "planned_start_at": opening_at,
                "max_duration_hours": 2.5}, known_players=[{"customer_id": "host", "seat_count": 1}], trace_id="seed")
        result = channel.send(message)

    history = traces.get_trace(result["trace_id"])
    lookup_result, = [event.content for event in history
                      if event.step == "tool_result" and event.content.get("name") == "search_current_games"]
    observed, = lookup_result["result"]["matches"]
    assert observed["game"]["status"] == "forming" and observed["game"]["planned_start_at"] == opening_at
    assert observed["game"]["seat_summary"]["remaining_seats"] == 3
    reviews = [event.content for event in history if event.step == "customer_visible_content_review_result"]
    assert [review["approved"] for review in reviews] == ([True, True] if review_enabled else [])
    if review_enabled:
        review_payload = json.loads(reviewer.calls[0]["messages"][-1]["content"])
        evidence, = review_payload["review_items"][0]["seat_claim_evidence"]
        assert evidence["status"] == "forming" and evidence["actual_started_at"] is None
    retry = json.loads(model.calls[2]["messages"][-1]["content"])
    rule = "customer_visible_contract:game_start.current_table_not_started"
    feedback, = [item["result"] for item in retry["turn_tool_evidence"]
                 if rule in item.get("result", {}).get("violations", [])]
    assert feedback["seat_claim_evidence"] == [{"game_id": game.game_id, "status": "forming",
        "actual_started_at": None, "planned_start_at": opening_at, "game_type": "hangzhou_mahjong",
        "remaining_seats": 3, "seats_total": 4}]
    receipt, = result["archive_outputs"]
    assert receipt["text"] == corrected and receipt["recipient_id"] == "guest"
    assert receipt["source_message_id"] == message.message_id and receipt["trace_id"] == result["trace_id"]
    assert [row["text"] for row in channel.events_for("guest")] == [inbound, corrected]
    assert store.require_game(game.game_id).to_dict() == game.to_dict()


def test_start_state_guard_preserves_questions_uncertainty_and_other_table_scope():
    """到店后的不确定推测、别桌和没有实时来源的局况不由本规则代判。"""
    fact = {"game_id": "table", "status": "forming", "actual_started_at": None,
            "planned_start_at": "2026-11-25T19:30:00+08:00"}
    assert customer_visible_table_start_violations("那桌已经开打了。", [fact])
    assert customer_visible_table_start_violations("19:30那桌已经开打了。", [fact])
    for text in ("这桌已经开打了吗？", "19:30那桌还在凑，没实际开打。",
                 "19:30那桌，你是说，人家已经开打了，还是还在凑？",
                 "19:30那桌还在凑，到时候人家可能已经开打了。",
                 "19:30那桌还在凑，等你到的时候人家已经开打了。",
                 "19:30那桌还在凑，隔壁那桌已经开打了。", "20:30那桌，他说那桌已经开打了。",
                 "20:30那桌已经开打了。",
                 "19:30那桌，听说，人家已经开打了。",
                 "19:30那桌，牌友问‘那桌已经开打了’是不是真的。"):
        assert not customer_visible_table_start_violations(text, [fact]), text
    assert not customer_visible_table_start_violations("那桌已经开打了。", [{
        **fact, "status": "ready", "actual_started_at": "2026-11-25T19:34:00+08:00"}])
    assert not customer_visible_table_start_violations("那桌已经开打了。", [
        {key: value for key, value in fact.items() if key != "actual_started_at"}])
    assert not customer_visible_table_start_violations("那桌已经开打了。", [fact, {**fact, "game_id": "other"}])


def test_actual_start_survives_search_transport_without_private_attendee_data():
    """压缩查询只保留计划和实际两个时间证据，屏蔽到店名单及订房身份。"""
    actual = "2026-11-25T19:34:00+08:00"
    planned = "2026-11-25T19:30:00+08:00"
    raw = {"name": "search_current_games", "called": True, "allowed": True, "result": {"matches": [{"game": {
        "game_id": "table", "status": "ready", "planned_start_at": planned, "actual_started_at": actual,
        "actual_attendee_ids": ["private-attendee"], "play_reservation_id": "private-reservation",
    }}]}}
    projected = compact_tool_result_dict(raw)["result"]["matches"][0]["game"]
    assert projected == {"game_id": "table", "status": "ready", "planned_start_at": planned,
                         "actual_started_at": actual, "requirement": {}}


@pytest.mark.parametrize(("agreed", "review_enabled"), [(False, True), (False, False), (True, True)])
def test_live_0151_duration_fit_requires_same_table_agreement(tmp_path, agreed, review_enabled):
    """原句下时长不实被重写，已约相同时间及明确待定说法可以真实送达。"""
    at = datetime.fromisoformat("2026-11-26T09:39:30+08:00")
    at_table = "2026-11-26T18:30:00+08:00"
    until = "2026-11-26T23:30:00+08:00"
    inbound = "老板 那桌人齐开的杭麻1块底还有位吗 我六点半到能打到十一点半 烟都行"
    wrong = ("那桌现在1个人，还差3个，人齐就开。你六点半到、打到十一点半这个时长没问题，"
             "不过这桌是人齐开，不保证会等你到六点半，也没有固定开局时间——"
             "你到时候来能不能正好赶上我得看当时凑没凑齐。要来的话尽早过来呗？")
    corrected = "那桌现在一人还差三位，人齐就开。你六点半到、能打到十一点半，这桌还没约多久，到时也得看还有没有位。"
    confirmed = "那桌现在一个，还差三位。你六点半到、打到十一点半这个时长没问题，已经约了五个钟头。"
    terms = {"game_type": "hangzhou_mahjong", "stake": 1, "start_time_kind": "asap_when_full"}
    if agreed:
        inbound = inbound.replace("人齐开的", "六点半开的")
        terms.update(start_time_kind="scheduled", planned_start_at=at_table)
    first = confirmed if agreed else wrong
    delivered = confirmed if agreed else corrected
    replies = [_reply(first, status="completed")]
    if not agreed:
        replies.append(_reply(corrected, status="completed"))
    main = StaticAgentClient([_call("search_current_games", {"lookup_scope": "specific_table", "requirement": terms}), *replies])
    reviewer = StaticAgentClient([json.dumps({
        "approved": True, "needs_human": False, "violations": [], "reasoning_summary": "仍有空位",
        "item_reviews": [{"item_id": "reply_to_user", "approved": True, "violations": [],
                          "suggested_safe_text": text, "follow_up_commitments": []}],
    }, ensure_ascii=False) for text in ([first] if agreed else [first, corrected])])
    store = SQLiteAgentStore(tmp_path / "table-duration.sqlite3")
    traces = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=main, store=store, trace_recorder=traces,
                           reply_self_review_enabled=review_enabled, reply_self_review_client=reviewer)
    channel = SandboxChannel(runtime, room_id="duration-fit", identities={"host": "牌友", "guest": "顾客"},
                             archive_path=tmp_path / "chat.sqlite3")
    message = MessageEnvelope("archived-0151-duration", "guest", inbound, at, "private", AGENT_ID)
    with using_clock(lambda: at):
        game, _ = store.create_game(conversation_id=channel.private_conversation_id("host"),
            organizer_id="host", organizer_name="牌友", requirement=terms | ({"duration_hours": 5} if agreed else {}),
            known_players=[], trace_id="seed")
        result = channel.send(message)

    history = traces.get_trace(result["trace_id"])
    query, = [event.content["result"] for event in history if event.step == "tool_result"
                and event.content.get("name") == "search_current_games"]
    option, = query["matches"]
    assert option["game"]["game_id"] == game.game_id and option["game"]["seat_summary"]["remaining_seats"] == 3
    expected_evidence = {"game_id": game.game_id, "agreed_duration_hours": 5 if agreed else None,
                         "agreed_end_at": until if agreed else None, "planned_start_at": at_table if agreed else None}
    if review_enabled:
        payload = json.loads(reviewer.calls[0]["messages"][-1]["content"])
        review_item, = payload["review_items"]
        assert review_item["table_duration_fit_evidence"] == expected_evidence
    else:
        assert not reviewer.calls
    rule = "customer_visible_contract:table_duration.unverified_fit"
    rejections = [event.content["result"] for event in history if event.step == "tool_result"
                  and event.content.get("name") == "customer_visible_content_review"
                  and rule in event.content.get("result", {}).get("violations", [])]
    if agreed:
        assert not rejections
    else:
        rejected, = rejections
        assert rejected["table_duration_fit_evidence"] == expected_evidence
        replanned = json.loads(main.calls[2]["messages"][-1]["content"])
        feedback, = [item["result"] for item in replanned["turn_tool_evidence"]
                     if rule in item.get("result", {}).get("violations", [])]
        assert feedback["table_duration_fit_evidence"] == expected_evidence
    receipt, = result["archive_outputs"]
    assert receipt["text"] == delivered and receipt["source_message_id"] == message.message_id
    assert [row["text"] for row in channel.events_for("guest")] == [inbound, delivered]
    assert store.require_game(game.game_id).to_dict() == game.to_dict()
    store._connection.close()


def test_review_treats_only_candidate_search_and_customer_cutoff_as_read_only():
    """零候选是查询结果，顾客指定时刻本身不证明已开始联系或已安排跟进。"""
    _, context = main_context()
    message = UserMessage(conversation_id="direct-ask-only", sender_id="customer", sender_name="客户",
                          text="帮我问问有没有人愿意凑一桌，我等到8点半没信就算了", message_id="ask-only")
    context["turn_tool_evidence"] = [{"name": "search_customers", "called": True, "allowed": True,
                                     "error": None, "result": {"candidates": []}}]
    text = "暂时没问着人，到8点半没信就算了。"
    action = AgentAction(goal="只查现有意向", objective_status="completed", reasoning_summary="", reply_to_user=text)
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="reply_to_user", review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": text}])
    assert payload["turn_tool_evidence"][0]["result"] == {"candidates": []}
    assert payload["follow_up_commitment_evidence"] == []
    rule = next(rule for rule in payload["semantic_fidelity_contract"]["rules"] if rule.startswith("search_customers"))
    assert "零结果也不能说没问着人" in rule
    assert "截止不等于店家已安排" in rule


def test_main_and_review_prompts_share_direct_reply_and_nonfixed_confirmation_rules():
    main = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text()
    review = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_reply_self_review.md").read_text()
    assert "不依赖后续润色" in main
    assert "你打吗" in main and "你打吗" in review
    assert "已明确且未变化" in main and "已明确且未变化" in review
    assert "偶尔复述" in review
    assert "无具体局选项的服务提议可省去" not in review


def test_future_flexible_opening_handoff_preserves_delegation_and_does_not_place_early(tmp_path):
    """有未来时段的人齐开暂不落库，实际送达澄清，明确约定后再建；普通人齐开继续可用。"""
    at = datetime.fromisoformat("2026-11-26T09:37:00+08:00")
    fixed_at = "2026-11-26T13:00:00+08:00"
    window = {"start_at": "2026-11-26T12:00:00+08:00", "end_at": "2026-11-26T18:00:00+08:00"}
    terms = {"game_type": "hangzhou_mahjong", "stake": 1}
    first = "老板 那就人齐了开吧 午后都行 一点底杭麻"
    handoff = "午后暂时没找到能搭的现成桌。人齐开可能提前，你愿意定个午后的开场时间吗？"
    confirmation = "那下午一点开吧，帮我组一桌。"
    receipt = "好，下午一点这桌记上了，还差三位。"
    now_request = "老板帮我组一桌一块底杭麻，人齐就开。"
    now_receipt = "好，给你记上了，人齐开，还差三位。"

    def create_args(customer, name, requirement):
        return {"organizer_id": customer, "organizer_name": name, "known_players": [], "requirement": requirement}

    class ShopClient(StaticAgentClient):
        """普通人齐开沿用真实回执的局号完成当下招募查询。"""

        def complete(self, messages, **options):
            context = json.loads(messages[-1]["content"])
            if context["current_message"]["text"] == now_request:
                tools = context["turn_tool_evidence"]
                if tools and tools[-1]["name"] == "create_game" and tools[-1]["allowed"]:
                    game_id = tools[-1]["result"]["game"]["game_id"]
                    self.outputs.insert(0, _call("search_customers", {"game_id": game_id}))
            return super().complete(messages, **options)

    main = ShopClient([
        _call("search_current_games", {"lookup_scope": "personal", "requirement": terms | {"start_time_range": window}}),
        _call("create_game", create_args("guest", "顾客", terms | {"start_time_kind": "asap_when_full",
              "start_time_range": window, "earliest_start_at": window["start_at"]})),
        _reply(handoff, status="waiting_user"),
        _call("create_game", create_args("guest", "顾客", terms | {"start_time_kind": "scheduled",
              "start_time": "13:00", "planned_start_at": fixed_at})),
        _reply(receipt),
        _call("create_game", create_args("walkin", "另一顾客", terms | {"start_time_kind": "asap_when_full"})),
        _reply(now_receipt),
    ])
    availability = StaticAgentClient([json.dumps({
        "clock_role": "other", "lookup_scope": "personal", "opening_window": window,
        "opening_dayparts": {"date": "2026-11-26", "parts": ["afternoon"], "source_ref": 1},
        "accepted_stake_range": {"min": 1, "max": 1}, "known_cap_score": None,
        "has_fixed_opening_authority": False, "opening_customer_ref": None,
        "current_ref": 1, "supporting_customer_refs": [1],
    }), json.dumps({
        "clock_role": "opening", "lookup_scope": "personal",
        "opening_window": {"start_at": fixed_at, "end_at": fixed_at},
        "accepted_stake_range": {"min": 1, "max": 1}, "known_cap_score": None,
        "has_fixed_opening_authority": True, "opening_customer_ref": 3,
        "current_ref": 3, "supporting_customer_refs": [1, 3],
    })])
    authority = StaticAgentClient([json.dumps({"scope": "create", "customer_ref": ref, "shop_offer_ref": None})
                                   for ref in (3, 1)])
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    traces = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=main, store=store, trace_recorder=traces,
                           table_availability_client=availability, creation_authorization_client=authority)
    channel = SandboxChannel(runtime, room_id="afternoon-window", identities={"guest": "顾客", "walkin": "另一顾客"},
                             archive_path=tmp_path / "chat.sqlite3")
    with using_clock(lambda: at):
        before = channel.send(MessageEnvelope("afternoon-inbound", "guest", first, at, "private", AGENT_ID))
        assert before["final_reply"] == handoff
        outbound, = before["archive_outputs"]
        assert outbound["source_message_id"] == "afternoon-inbound" and outbound["text"] == handoff
        assert not store.games and not store.invite_drafts and not store.room_reservations and not authority.calls
        before_trace = traces.get_trace(before["trace_id"])
        verified, = [event.content["decision"] for event in before_trace if event.step == "table_availability_authorization_result"]
        assert verified["named_future_opening"] == {"date": "2026-11-26", "parts": ["afternoon"]}
        assert verified["opening_window"] == window and not verified["has_fixed_opening_authority"]
        business_tools = [event.content for event in before_trace if event.step == "tool_result"]
        read, denied = [tool for tool in business_tools if tool["name"] in {"search_current_games", "create_game"}]
        assert read["called"] and read["allowed"] and read["result"]["matches"] == []
        assert read["result"]["requirement"]["start_time_range"] == window
        assert not denied["called"] and "no enforceable earliest start" in denied["error"]
        assert json.loads(main.calls[2]["messages"][-1]["content"])["current_message"]["text"] == first

    next_at = at.replace(minute=38)
    with using_clock(lambda: next_at):
        after = channel.send(MessageEnvelope("confirmed-inbound", "guest", confirmation, next_at, "private", AGENT_ID))
        assert after["final_reply"] == receipt
        after_trace = traces.get_trace(after["trace_id"])
        created, = [event.content for event in after_trace
                    if event.step == "tool_result" and event.content.get("name") == "create_game"]
        assert created["called"] and created["allowed"]
        scheduled, = store.games.values()
        assert scheduled.planned_start_at.isoformat() == fixed_at
        assert scheduled.organizer_id == "guest"

        unbounded = channel.send(MessageEnvelope("unbounded-inbound", "walkin", now_request, next_at, "private", AGENT_ID))
        assert unbounded["final_reply"] == now_receipt
        flexible = next(game for game in store.games.values() if game.organizer_id == "walkin")
        assert flexible.planned_start_at is None and flexible.requirement["start_time_kind"] == "asap_when_full"
        assert flexible.requirement["latest_start_at"] == "2026-11-26T13:38:00+08:00"
    assert len(availability.calls) == 2 and len(authority.calls) == 2
    assert not store.room_reservations and not store.invite_drafts
    store._connection.close()


def test_main_prompt_grounds_organizer_receipt_in_table_state_and_delivery():
    """待组桌和待审批邀约只能支持实际进展，回执不得扩大成已联系或未来通知。"""
    prompts = ROOT / "src/mahjong_agent_runtime/prompts"
    prompt = (prompts / "agent_runtime_system.md").read_text(encoding="utf-8")
    copywriting = (prompts / "customer_visible_text_generation.md").read_text(encoding="utf-8")
    guidance = prompt.split("对画像高置信默认，这些槽位可以用于查局和决策", 1)[1].split(
        "回复客户时，要把内部槽位翻译", 1
    )[0]
    assert "牌局仍待招募" in guidance and "确认已登记" in guidance
    assert "最新席位结果" in guidance
    assert "实际送达证据" in guidance
    assert "主动通知发起人须有对应的后续安排" in guidance
    assert all(example not in guidance for example in ("我帮你问问", "我帮你看看", "有消息跟你说"))
    assert all("问问" not in phrase and "消息" not in phrase for phrase in PREFERRED_OPERATION_ACK_PHRASES)
    assert "保留实际进展" in copywriting


def test_current_room_query_contract_uses_backend_default_instead_of_forcing_a_time_question():
    _, payload = main_context()
    guidance = payload["planning_contract"]["tool_selection_hints"]["game_lookup_vs_room_availability"]
    assert "check_room_availability({})" in guidance and "4小时" in guidance
    assert "不代表客户预约" in guidance and "部分明确" in guidance
    assert "duration_minutes" in guidance and "explicit_duration" in guidance


def test_main_prompt_distinguishes_room_inquiry_from_booking_authorization():
    """只问房态时不能占房，拒绝公开组局机会也不凭空创建状态。"""
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_system.md").read_text(encoding="utf-8")
    assert "只问某时段能不能约或有没有空房时" in prompt
    assert "即使人数和时长都已说清楚，也先不占房" in prompt
    assert "顾客明确委托预订或接受店家明确的订房提议" in prompt
    assert "本人没有待答邀约或已占席位" in prompt


def test_planner_and_reviewer_distinguish_public_opt_out_from_actual_withdrawal():
    """公开牌桌的选择与取消本人席位分开，另找和提醒继续核对各自请求。"""
    prompts = ROOT / "src/mahjong_agent_runtime/prompts"
    main = (prompts / "agent_runtime_system.md").read_text(encoding="utf-8")
    review = (prompts / "agent_runtime_reply_self_review.md").read_text(encoding="utf-8")
    assert "不告诉顾客需要自行联系别人退席" in main
    assert "不同桌的本人邀约或席位也不能作为依据" in main
    assert "同条消息还要找别的局或请求后续通知时，分别按实际请求处理" in main
    assert "核对本人已占的席位" in review
    assert "要求顾客自己找人退席，或要求主模型补记退出，以 `semantic_contradiction` 拒绝" in review
    assert "另找牌局或将来有某类局时通知" in review

    assert "先查现成局或复用仍有效的查询" in main
    assert "用户另外表达现在想找局时，应推进查局请求，即使没有用问句" in main
    assert "同一桌即使仍出现在查询结果，也不能再次作为推荐或邀约" in review
    assert "不能用「要不要帮你找」重问已经明确的请求" in review
    assert "同时问能否将来通知也不替代当前查局" in review


@pytest.mark.parametrize("has_reserved_seat", (False, True))
def test_review_uses_own_seat_evidence_only_for_current_private_recipient(has_reserved_seat):
    """核对本人已占席或没有占席，其他渠道不读取这份私有投影。"""
    message, context = main_context()
    seat = {"game_id": "other-table", "participant_status": "confirmed", "participation_already_recorded": True}
    context["sender_active_game_memberships"] = [
        {**seat, "seat_count": 1, "write_instruction": "internal planner guidance"}
    ] if has_reserved_seat else []
    action = AgentAction(goal="按本人实际状态处理这次不参加", objective_status="completed",
                         reasoning_summary="", reply_to_user="好，那这桌就不去了。")
    item = {"item_id": "reply_to_user", "recipient_id": message.sender_id, "text": action.reply_to_user}
    private = build_reply_self_review_payload(message=message, action=action, context_payload=context,
        review_scope="reply_to_user", review_items=[item])
    assert private["sender_active_game_memberships"] == ([seat] if has_reserved_seat else [])
    for review_scope, review_item in (
        ("reply_to_user", {**item, "channel": "group"}),
        ("tool_message", {**item, "recipient_id": "another-customer"}),
    ):
        other = build_reply_self_review_payload(message=message, action=action, context_payload=context,
            review_scope=review_scope, review_items=[review_item])
        assert "sender_active_game_memberships" not in other


def test_review_output_example_keeps_required_empty_follow_up_assessment():
    """无后续承诺的普通查房也会送入私聊审查，成功示例必须展示应填的空数组。"""
    prompt = (ROOT / "src/mahjong_agent_runtime/prompts/agent_runtime_reply_self_review.md").read_text(encoding="utf-8")
    example = prompt.split("输出必须是 JSON object", 1)[1]
    payload, _ = json.JSONDecoder().raw_decode(example[example.index("{"):])
    assert payload["item_reviews"][0]["follow_up_commitments"] == []


def test_search_feedback_does_not_reintroduce_a_fixed_closing_or_treat_accurate_echo_as_fact_error():
    """工具回灌合同与主提示同向，不能在下一步又把允许的话术标成事实错误。"""
    from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract

    contract = current_game_search_reply_contract({"stake": "0.5"}, [])
    assert "no fixed closing" in contract["customer_visible_rule"]
    assert "style issue" in contract["customer_visible_rule"]
    assert "七点三缺一，0.5无烟杭麻，打吗" in contract["style_only_echo_examples"]
    assert "七点三缺一，0.5无烟杭麻，打吗" not in contract["bad_reply_examples"]


def test_specific_table_inquiry_does_not_filter_it_out_with_requested_changes():
    """指定桌的定位条件与顾客希望改变的条件应分别进入规划和独立审查。"""
    prompts = ROOT / "src/mahjong_agent_runtime/prompts"
    main = (prompts / "agent_runtime_system.md").read_text(encoding="utf-8")
    review = (prompts / "agent_runtime_reply_self_review.md").read_text(encoding="utf-8")
    assert all(key in main for key in ("named_table_lookup", "scope.call_id", "role=condition"))
    lookup = game_query_strategy_contract()["named_table_lookup"]
    assert "role=current" in lookup and "scope.call_id" in lookup
    assert "role=condition" in lookup and "不当作定位过滤" in lookup
    assert "先定位原桌、再据实说明差异或待定项" in review
    assert "不能据此断定所指桌不存在" in review
