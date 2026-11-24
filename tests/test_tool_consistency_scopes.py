"""跨工具条件按所属对象比较，不能把目标桌况复制成新局人数。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest

from mahjong_agent_runtime import (
    AgentAction,
    AgentContextBuilder,
    AgentRuntime,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    StaticAgentClient,
    ToolCall,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.group_chat.models import ChannelIdentity
from mahjong_agent_runtime.services import ToolExecutionService
from mahjong_agent_runtime.task_context import TaskContextManager
from mahjong_agent_runtime.tool_consistency import validate_explicit_task_fact_consistency


@pytest.fixture(params=["memory", "sqlite"])
def chain(request, tmp_path):
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "scopes.sqlite3")
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)
    service = ToolExecutionService(store=store, tool_gateway=gateway, trace_recorder=trace)
    message = UserMessage(
        conversation_id="scope-chat", sender_id="customer", sender_name="客户",
        text="我自己来，想找三缺一，没有就帮我组", message_id="scope-message",
    )
    store.append_user_turn(message, "scope-trace")
    yield store, service, message
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


def execute(chain, call, *, facts=None, previous=None, step=1, context_payload=None):
    store, service, message = chain
    assertions = [
        {"field": field, "value": value, "role": "current", "source_message_id": message.message_id}
        for field, value in (facts or {}).items()
    ]
    processed = service.execute_tool_calls(
        AgentAction(
            goal="推进客户组局", objective_status="needs_tool", reasoning_summary="验证工具作用域",
            tool_calls=[call], objective_state={"fact_assertions": assertions},
        ),
        message=message, trace_id="scope-trace", previous_step_tool_results=previous or [],
        step_index=step, run_id="scope-run", run_version=store.conversation_version(message.conversation_id),
        context_payload=context_payload,
    )
    return processed.tool_results[0]


def create_call(requirement, *, seats=1):
    return ToolCall(
        name="create_game",
        arguments={
            "requirement": requirement, "organizer_id": "customer", "organizer_name": "客户",
            "requesting_party": {"contact_id": "customer", "contact_name": "客户", "seat_count": seats},
        },
    )


def group_delegate(chain, *, party_size=None, text=None):
    """模拟群入口已持久化的本人来源，再由正式上下文构建器生成信任边界。"""
    store, service, _ = chain
    identity = ChannelIdentity(
        channel="console", external_user_id="member", customer_id="customer", public_name="客户",
        private_conversation_id="scope-group-private", can_private_message=True, is_friend=True,
    )
    store.upsert_channel_identity(identity)
    message = UserMessage(
        conversation_id=identity.private_conversation_id, sender_id=identity.customer_id, sender_name=identity.public_name,
        text=text or ("老板，帮我找个国标1块无烟的。" if party_size is None else "老板，帮我找个国标1块无烟的，我这边两个人。"),
        message_id="scope-group-source", metadata={
            "source": "group_private_continuation",
            "group_topic_source": {"channel": identity.channel, "room_id": "scope-room",
                                   "external_user_id": identity.external_user_id},
        },
    )
    constraints = {} if party_size is None else {"party_size": party_size}
    store.record_group_topic_message(
        channel=identity.channel, room_id="scope-room", customer_id=identity.customer_id, message_id=message.message_id,
        text=message.text, sent_at=message.sent_at, intent="delegate", public_summary={},
        classification={"intent": "delegate", "goal_relation": "new", "topic_action": "new", "constraints": constraints},
    )
    return store, service, message


def build_delegate_context(chain):
    """复用真实任务管理和来源投影，固定模型提案只验证写入守卫。"""
    store, service, message = chain
    TaskContextManager(store).prepare(message, trace_id="scope-trace")
    store.append_user_turn(message, "scope-trace")
    return AgentContextBuilder(store, service.tool_gateway).build(message, trace_id="scope-trace").payload


@pytest.mark.parametrize("party_fact", [None, "party_size", "requesting_party.seat_count"])
def test_unknown_group_delegate_party_allows_lookup_but_rejects_creation(chain, party_fact):
    delegated = group_delegate(chain)
    store, service, _ = delegated
    context = build_delegate_context(delegated)
    assert context["group_topic_context"]["own_condition_sources"][0]["constraints"] == {}
    requirement = {"game_type": "national_standard_mahjong", "stake": "1", "smoke_preference": "no_smoking"}
    read = execute(delegated, ToolCall("search_current_games", {"requirement": requirement}), context_payload=context)
    assert read.called and read.allowed and read.error is None
    assert read.result["matches"] == []
    assert "requesting_party" not in read.result["requirement"]

    proposed = create_call(requirement)
    if party_fact:
        proposed.arguments["requesting_party"]["source"] = "organizer_default"
        proposed.arguments["known_players"] = [{"customer_id": "customer", "display_name": "客户", "seat_count": 1}]
    else:
        proposed.arguments.pop("requesting_party")
    transitions = list(store.transitions)
    result = execute(delegated, proposed, facts={party_fact: 1} if party_fact else None, previous=[read], step=2,
                     context_payload=context)
    assert not result.called and not result.allowed and "confirmed party size" in result.error
    assert result.result["validation_error"]["code"] == "explicit_task_fact_consistency_rejected"
    assert not result.state_transitions and store.transitions == transitions
    assert not store.games and not store.invite_drafts and not store.outbound_message_drafts and not store.scheduled_tasks
    assert [
        event.content["call"]["name"]
        for event in service.trace_recorder.get_trace("scope-trace")
        if event.step == "tool_called"
    ] == ["search_current_games"]


def test_group_delegate_creation_requires_actual_confirmed_seats(chain):
    delegated = group_delegate(chain, party_size=2)
    store, _, _ = delegated
    context = build_delegate_context(delegated)
    assert context["group_topic_context"]["own_condition_sources"][0]["constraints"]["party_size"] == 2
    requirement = {"game_type": "national_standard_mahjong", "stake": "1", "known_player_count": 2}
    incorrect = execute(delegated, create_call(requirement), context_payload=context)
    assert not incorrect.called and not incorrect.allowed and not store.games
    correct = execute(delegated, create_call(requirement, seats=2), step=2, context_payload=context)
    assert correct.called and correct.allowed and correct.error is None
    game = store.require_game(correct.result["game"]["game_id"])
    assert [(party.contact_id, party.seat_count) for party in game.parties] == [("customer", 2)]
    assert game.remaining_seats() == 2


@pytest.mark.parametrize("party_fact", ["known_player_count", "requesting_party.seat_count"])
def test_group_delegate_can_use_confirmed_source_when_router_omits_count(chain, party_fact):
    """路由漏提人数时，允许主模型绑定本人原话中的已知人数，写入仍核对实际座位。"""
    delegated = group_delegate(chain, text="请帮我找明天国标1块无烟的，我一个人。")
    store, _, _ = delegated
    context = build_delegate_context(delegated)
    assert context["group_topic_context"]["own_condition_sources"][0]["constraints"] == {}
    requirement = {"game_type": "national_standard_mahjong", "stake": "1"}
    mismatch = execute(delegated, create_call(requirement, seats=2), facts={party_fact: 1}, context_payload=context)
    assert not mismatch.called and not store.games
    result = execute(delegated, create_call(requirement), facts={party_fact: 1}, step=2, context_payload=context)
    assert result.called and result.allowed and result.error is None
    assert store.require_game(result.result["game"]["game_id"]).seat_summary()["claimed_seats"] == 1


def test_group_delegate_with_dotted_seat_fact_creates_without_repeated_model_calls(chain):
    """群路由漏提人数时，完整运行时第一次建局就核验本人的人数原话。"""
    store, service, message = group_delegate(chain, text="老板，帮我找明晚九点半川麻1块无烟的，我一个人。")
    start = (message.sent_at + timedelta(days=1)).replace(hour=21, minute=30, second=0, microsecond=0)
    requirement = {"game_type": "sichuan_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                   "start_time_kind": "scheduled", "start_time": "21:30", "planned_start_at": start.isoformat()}
    create = replace(create_call(requirement), call_id="first-create", depends_on=[], reason="现成局未匹配，按同一人数继续承接")
    search = ToolCall("search_current_games", {"requirement": {**requirement, "requesting_party": create.arguments["requesting_party"]}},
                      call_id="initial-search", depends_on=[], reason="先查现成局")
    assertion = {"field": "requesting_party.seat_count", "value": 1, "role": "current", "source_message_id": message.message_id}
    actions = [AgentAction(goal="按原话帮客户找局", objective_status="needs_tool", reasoning_summary=call.reason, tool_calls=[call],
                           objective_state={"fact_assertions": [assertion]},
                           stop_reason={"can_stop": False, "why": call.reason, "pending_work": [call.name],
                                        "depends_on_tool_results": True}) for call in (search, create)]
    actions.append(AgentAction(goal="按原话帮客户找局", objective_status="completed", reasoning_summary="已承接找局",
                               reply_to_user="好，帮你找人。", stop_reason={"can_stop": True, "why": "已安排待组",
                                                                      "pending_work": [], "depends_on_tool_results": True}))
    llm = StaticAgentClient([json.dumps(action.to_dict(), ensure_ascii=False) for action in actions])
    trace = service.trace_recorder
    runtime = AgentRuntime(llm_client=llm, store=store, tool_gateway=service.tool_gateway, trace_recorder=trace)

    result = runtime.handle_user_message(message, trace_id="dotted-seat-first-create")
    events = trace.get_trace(result.trace_id)
    assert result.status == "completed", [event.content for event in events if event.step.endswith("error")]
    assert next(event.content["group_topic_context"]["own_condition_sources"][0]["constraints"]
                for event in events if event.step == "context_built") == {}
    assert not any(event.step in {"action_contract_error", "tool_explicit_fact_consistency_error"} for event in events)
    assert [tool.name for tool in result.tool_results] == ["search_current_games", "create_game"]
    assert all(tool.called and tool.allowed and not tool.error for tool in result.tool_results)
    game = store.require_game(result.tool_results[-1].result["game"]["game_id"])
    assert game.seat_summary()["claimed_seats"] == 1


@pytest.mark.parametrize("uncertain", ["我一个人？", "有人说：我一个人。", "有人说“我想打，我一个人，还能找吗”。"])
def test_nonasserted_count_cannot_support_group_game_creation(chain, uncertain):
    """疑问或转述中的数字不能佐证模型误标的当前己方人数。"""
    delegated = group_delegate(chain, text=f"请帮我找明天国标1块无烟的。{uncertain}")
    store, _, _ = delegated
    context = build_delegate_context(delegated)
    proposed = create_call({"game_type": "national_standard_mahjong", "stake": "1"})
    result = execute(delegated, proposed, facts={"requesting_party.seat_count": 1}, context_payload=context)
    assert not result.called and not result.allowed and not store.games


def test_private_followup_can_confirm_party_for_same_group_delegate_task(chain):
    delegated = group_delegate(chain)
    store, service, source = delegated
    original = build_delegate_context(delegated)
    task_id = original["task_context_window"]["task_context_id"]
    requirement = {"game_type": "national_standard_mahjong", "stake": "1"}
    unresolved = replace(source, text="条件还是一样", message_id="scope-private-unresolved",
                         sent_at=source.sent_at + timedelta(seconds=1), metadata={})
    unresolved_chain = (store, service, unresolved)
    context = build_delegate_context(unresolved_chain)
    assert context["group_topic_context"] is None
    assert context["task_context_window"]["task_context_id"] == task_id
    premature = execute(unresolved_chain, create_call(requirement), context_payload=context)
    assert not premature.called and not premature.allowed and "confirmed party size" in premature.error
    assert not store.games

    confirmed = replace(unresolved, text="我这边两个人", message_id="scope-private-party",
                        sent_at=unresolved.sent_at + timedelta(seconds=1))
    confirmed_chain = (store, service, confirmed)
    context = build_delegate_context(confirmed_chain)
    assert context["group_topic_context"] is None
    assert context["task_context_window"]["task_context_id"] == task_id
    result = execute(confirmed_chain, create_call(requirement, seats=2), facts={"party_size": 2}, step=2,
                     context_payload=context)
    assert result.called and result.allowed and result.error is None
    game = store.require_game(result.result["game"]["game_id"])
    assert game.conversation_id == source.conversation_id
    assert [(party.contact_id, party.seat_count) for party in game.parties] == [("customer", 2)]
    assert game.remaining_seats() == 2


@pytest.mark.parametrize("stake", [{"stake": "1"}, {"stake_range": {"min": 1, "max": 256}}])
@pytest.mark.parametrize("party_size", [1, 2])
def test_target_three_present_query_can_create_a_different_new_game(chain, stake, party_size):
    shared = {"game_type": "hangzhou_mahjong", "start_time_kind": "asap_when_full", **stake}
    facts = {"known_player_count": party_size, "needed_seats": 4 - party_size, **shared}
    query = execute(chain, ToolCall(name="search_current_games", arguments={"requirement": {
        **shared, "known_player_count": 3, "needed_seats": 1,
        "requesting_party": {"contact_id": "customer", "seat_count": party_size},
    }}), facts=facts)
    assert query.allowed and query.called
    assert query.result["matches"] == []
    assert query.result["requirement"]["known_player_count"] == 3
    assert query.result["requirement"]["needed_seats"] == 1

    created = execute(chain, create_call({**shared, "known_player_count": party_size, "needed_seats": 4 - party_size},
                                        seats=party_size), facts=facts, previous=[query], step=2)
    assert created.allowed and created.called and created.error is None
    game_id = created.result["game"]["game_id"]
    store = chain[0]
    if isinstance(store, SQLiteAgentStore):
        # 必须验证落库后的事实，不能只核对工具返回的内存对象。
        recovered = SQLiteAgentStore(store.path)
        try:
            game = recovered.require_game(game_id)
        finally:
            recovered._connection.close()
    else:
        game = store.require_game(game_id)
    assert game.requirement["known_player_count"] == party_size
    assert game.requirement["needed_seats"] == 4 - party_size
    assert game.remaining_seats() == 4 - party_size
    assert [(party.contact_id, party.seat_count) for party in game.parties] == [("customer", party_size)]


@pytest.mark.parametrize("tool_name", ["search_current_games", "search_customers"])
def test_query_table_snapshot_does_not_redefine_current_customer_facts(tool_name):
    call = ToolCall(name=tool_name, arguments={"requirement": {
        "known_player_count": 3, "needed_seats": 1, "seat_format": "371",
    }})
    facts = {"known_player_count": 1, "needed_seats": 3, "seat_format": "173"}
    assert validate_explicit_task_fact_consistency(call, {"explicit_task_facts": {"facts": facts}})[0] is None


def test_absent_requesting_party_is_not_compared_as_an_explicit_single_person():
    call = ToolCall(name="search_current_games", arguments={"requirement": {
        "known_player_count": 3, "needed_seats": 1,
    }})
    error, reference = validate_explicit_task_fact_consistency(
        call, {"explicit_task_facts": {"facts": {"known_player_count": 2, "needed_seats": 2}}},
    )
    assert error is None
    assert reference["known_player_count"] == 2


@pytest.mark.parametrize("size_field", ["seat_count", "party_size", "requesting_party"])
@pytest.mark.parametrize("fact_field", ["known_player_count", "party_size", "seat_count", "requesting_party", "requesting_party.seat_count"])
def test_search_requesting_party_still_must_match_explicit_customer_count(chain, size_field, fact_field):
    requirement = {"game_type": "hangzhou_mahjong", "known_player_count": 3, "needed_seats": 1}
    requirement[size_field] = {"contact_id": "customer", "seat_count": 1} if size_field == "requesting_party" else 1
    fact = {"seat_count": 2} if fact_field == "requesting_party" else 2
    result = execute(chain, ToolCall(name="search_current_games", arguments={"requirement": requirement}),
                     facts={fact_field: fact})
    assert not result.allowed and not result.called
    assert "requesting_party.seat_count" in result.error
    assert chain[0].games == {}


def test_conflicting_flat_and_dotted_seat_assertions_cannot_share_scope(chain):
    """等价字段使用同一个事实键，冲突需要模型核对，反馈仍带本人来源。"""
    result = execute(chain, create_call({"game_type": "hangzhou_mahjong", "stake": "1"}),
                     facts={"seat_count": 1, "requesting_party.seat_count": 2})
    assert not result.called and not result.allowed and not chain[0].games
    assert "conflicting current facts for seat_count" in result.error
    assert result.result["reference_fact_sources"] == {
        "seat_count": {"source_message_id": chain[2].message_id, "role": "current"},
    }


@pytest.mark.parametrize("change", ["declared_count", "requesting_party", "known_players"])
def test_create_cannot_hide_a_customer_count_change_in_another_argument(chain, change):
    call = create_call({"known_player_count": 2, "needed_seats": 2}, seats=2)
    if change == "declared_count":
        call.arguments["requirement"] = {"known_player_count": 1, "needed_seats": 3}
    elif change == "requesting_party":
        call.arguments["requesting_party"]["seat_count"] = 1
    else:
        # 已有 known_players 的显式人数优先于 requesting_party；与真实写入路径保持一致。
        call.arguments["known_players"] = [{"customer_id": "customer", "display_name": "客户", "seat_count": 1}]
    result = execute(chain, call, facts={"known_player_count": 2, "needed_seats": 2})
    assert not result.allowed and not result.called
    assert "explicit current-task facts" in result.error
    assert chain[0].games == {}


def test_separately_named_companion_is_part_of_the_confirmed_initial_count(chain):
    call = create_call({"known_player_count": 2, "needed_seats": 2}, seats=1)
    call.arguments["known_players"] = [{"customer_id": "friend", "display_name": "同行朋友", "seat_count": 1}]
    result = execute(chain, call, facts={"known_player_count": 2, "needed_seats": 2})
    assert result.allowed and result.called
    assert result.result["game"]["requirement"]["known_player_count"] == 2
    assert result.result["game"]["remaining_seats"] == 2


def test_explicit_own_party_size_is_checked_separately_from_game_population(chain):
    call = create_call({"known_player_count": 3, "needed_seats": 1}, seats=1)
    call.arguments["known_players"] = [{"customer_id": "friend", "display_name": "同行朋友", "seat_count": 2}]
    result = execute(chain, call, facts={"known_player_count": 3, "party_size": 2})
    assert not result.allowed and not result.called
    assert "requesting_party.seat_count" in result.error
    assert chain[0].games == {}


def test_candidate_search_keeps_persisted_game_population_over_the_old_customer_snapshot(chain):
    store = chain[0]
    created = execute(chain, create_call({"game_type": "hangzhou_mahjong", "known_player_count": 1, "needed_seats": 3}),
                      facts={"known_player_count": 1, "needed_seats": 3})
    game_id = created.result["game"]["game_id"]
    store.record_candidate_reply(
        game_id=game_id, customer_id="friend", display_name="同行朋友", status="accepted", seat_count=1,
        trace_id="friend-joined",
    )
    searched = execute(chain, ToolCall(name="search_customers", arguments={
        "game_id": game_id, "requirement": {"known_player_count": 1, "needed_seats": 3},
    }), facts={"known_player_count": 1, "needed_seats": 3}, previous=[created], step=2)
    assert searched.allowed and searched.called
    assert searched.result["requirement"]["known_player_count"] == 2
    assert searched.result["requirement"]["needed_seats"] == 2
    assert store.require_game(game_id).requirement["known_player_count"] == 2


@pytest.mark.parametrize("field,value", [
    ("stake", "2"), ("start_time_kind", "scheduled"), ("smoke_preference", "smoke"),
    ("game_type", "red_center_mahjong"), ("cap_score", 128),
    ("duration_hours", 6), ("stake_range", {"min": 1, "max": 2}), ("start_time", "18:00"),
])
def test_same_customer_constraints_still_cannot_drift_between_query_and_creation(chain, field, value):
    requirement = {
        "game_type": "hangzhou_mahjong", "stake": "1", "cap_score": 256,
        "start_time_kind": "asap_when_full", "smoke_preference": "no_smoke", "duration_hours": 4,
        "known_player_count": 3, "needed_seats": 1,
    }
    if field == "stake_range":
        requirement.pop("stake")
        requirement["stake_range"] = {"min": 1, "max": 3}
    if field == "start_time":
        requirement.update(start_time_kind="scheduled", start_time="16:00")
    query = execute(chain, ToolCall(name="search_current_games", arguments={"requirement": requirement}))
    assert query.allowed and query.called
    new_requirement = {**deepcopy(requirement), "known_player_count": 1, "needed_seats": 3, field: value}
    result = execute(chain, create_call(new_requirement), previous=[query], step=2)
    assert not result.allowed and not result.called
    assert "previous search_current_games" in result.error
    assert field in result.error
    assert chain[0].games == {}


@pytest.mark.parametrize("invalid", ["party_schema", "capacity", "ownership"])
def test_scope_separation_does_not_weaken_gateway_or_store_guards(chain, invalid):
    call = create_call({"game_type": "hangzhou_mahjong"})
    if invalid == "party_schema":
        call.arguments["requesting_party"]["seat_count"] = 5
        error = "seat_count must be <= 4"
    elif invalid == "capacity":
        call.arguments["requesting_party"]["seat_count"] = 3
        call.arguments["known_players"] = [{"customer_id": "friend", "display_name": "同行朋友", "seat_count": 2}]
        error = "exceed table capacity"
    else:
        call.arguments["organizer_id"] = "other-customer"
        error = "authenticated sender_id"
    result = execute(chain, call)
    assert not result.allowed and not result.called
    assert error in result.error
    assert chain[0].games == {}
