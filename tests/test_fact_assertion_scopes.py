"""多业务对象的事实绑定不能被全局字典覆盖，也不能借范围声明绕过事实或权限。"""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta

import pytest

from mahjong_agent_runtime import AgentAction, InMemoryAgentStore, SQLiteAgentStore, InMemoryTraceRecorder, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.services import ToolExecutionService


@pytest.fixture(params=["memory", "sqlite"])
def chain(request, tmp_path):
    store = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "facts.sqlite3")
    start = now() + timedelta(hours=1)
    requirement = {"game_type": "hangzhou_mahjong", "stake": "0.5", "smoke_preference": "no_smoking",
                   "planned_start_at": start.isoformat(), "duration_hours": 4}
    game, _ = store.create_game(conversation_id="private_a", organizer_id="A", organizer_name="A", requirement=requirement,
        known_players=[{"customer_id": a, "display_name": a} for a in "ABC"], trace_id="fixture")
    message = UserMessage(conversation_id="private_a", sender_id="A", sender_name="A", text="保留两个时间备选，早场ABC，晚场ABE。", message_id="source")
    store.append_user_turn(message, "source")
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store=store, tool_gateway=ToolGateway(store, trace_recorder=trace), trace_recorder=trace)
    yield store, game, message, service
    if request.param == "sqlite":
        store._connection.close()


def assertion(value, scope=None, *, source="source", field="planned_start_at"):
    result = {"field": field, "value": value, "source_message_id": source, "role": "current"}
    if scope is not None:
        result["scope"] = scope
    return result


def calls(game):
    early = game.planned_start_at + timedelta(minutes=30)
    late = game.planned_start_at + timedelta(hours=1)
    return [ToolCall("update_game_requirement", {"game_id": game.game_id, "requirement_patch": {"planned_start_at": early.isoformat()}, "reason": "早场已同意"}, call_id="early"),
            ToolCall("create_game", {"organizer_id": "A", "organizer_name": "A", "alternative_to_game_id": game.game_id,
                "requirement": {"game_type": "hangzhou_mahjong", "stake": "0.5", "smoke_preference": "no_smoking", "duration_hours": 4, "planned_start_at": late.isoformat()},
                "known_players": [{"customer_id": a, "display_name": a} for a in "ABE"]}, call_id="late", depends_on=["early"])]


def execute(chain, tool_calls, facts):
    store, game, message, service = chain
    return service.execute_tool_calls(AgentAction(goal="并行待组方案", objective_status="needs_tool", reasoning_summary="来源事实分方案核对",
        objective_state={"fact_assertions": facts}, tool_calls=tool_calls), message=message, trace_id="scoped_facts",
        previous_step_tool_results=[], step_index=1, run_id="run", run_version=store.conversation_version(message.conversation_id)).tool_results


@pytest.mark.parametrize("bounds,allowed", [({"min": 0.5, "max": 0.5}, True), ({"min": 0.25, "max": 1}, False)])
def test_scoped_fixed_stake_accepts_only_equal_range_through_real_gateway(chain, bounds, allowed):
    store, game, _, _ = chain
    before = deepcopy(store.require_game(game.game_id).to_dict())
    call = ToolCall("search_customers", {"game_id": game.game_id, "requirement": {"stake_range": bounds}, "limit": 10}, call_id="find")
    results = execute(chain, [call], [assertion("0.5", {"call_id": "find"}, field="stake")])
    result = results[0]
    assert result.called is allowed and result.allowed is allowed
    if allowed:
        assert not result.error and result.result["requirement"]["stake"] == "0.5"
        assert "stake_range" not in result.result["requirement"]
    else:
        assert "explicit fixed stake" in result.error
    assert store.require_game(game.game_id).to_dict() == before


@pytest.mark.parametrize("scope_kind", ["call_id", "game_id"])
def test_two_option_times_bind_to_their_own_targets_and_really_persist(chain, scope_kind):
    store, game, _, _ = chain
    tool_calls = calls(game)
    early = tool_calls[0].arguments["requirement_patch"]["planned_start_at"]
    late = tool_calls[1].arguments["requirement"]["planned_start_at"]
    early_scope = {"call_id": "early"} if scope_kind == "call_id" else {"game_id": game.game_id}
    facts = [assertion(early, early_scope), assertion(late, {"call_id": "late"}), assertion("0.5", field="stake")]
    result = execute(chain, tool_calls, facts)
    assert len(result) == 2 and all(r.called and r.allowed and not r.error for r in result)
    games = store.games
    assert len(games) == 2
    assert games[game.game_id].planned_start_at.isoformat() == early
    alternative = next(g for g in games.values() if g.game_id != game.game_id)
    assert alternative.planned_start_at.isoformat() == late and alternative.option_root_game_id == game.game_id
    assert {p.customer_id for p in alternative.participants} == set("ABE")


def test_swapped_option_values_still_rejected_before_writes(chain):
    store, game, _, _ = chain
    tool_calls = calls(game)
    early = tool_calls[0].arguments["requirement_patch"]["planned_start_at"]
    late = tool_calls[1].arguments["requirement"]["planned_start_at"]
    tool_calls[0].arguments["requirement_patch"]["planned_start_at"] = late
    tool_calls[1].arguments["requirement"]["planned_start_at"] = early
    before = deepcopy(store.require_game(game.game_id).to_dict())
    result = execute(chain, tool_calls, [assertion(early, {"call_id": "early"}), assertion(late, {"call_id": "late"})])
    assert not any(r.called for r in result) and any("explicit current-task facts" in (r.error or "") for r in result)
    assert len(store.games) == 1 and store.require_game(game.game_id).to_dict() == before


@pytest.mark.parametrize("reverse", [False, True])
def test_unscoped_conflicting_values_never_choose_the_last_or_match_call_arguments(chain, reverse):
    store, game, _, _ = chain
    call = calls(game)[0]
    facts = [assertion(call.arguments["requirement_patch"]["planned_start_at"]), assertion((game.planned_start_at + timedelta(hours=1)).isoformat())]
    if reverse:
        facts.reverse()
    before = deepcopy(store.require_game(game.game_id).to_dict())
    result = execute(chain, [call], facts)[0]
    assert not result.called and "conflicting current facts" in result.error
    assert store.require_game(game.game_id).to_dict() == before


@pytest.mark.parametrize("scope", ["late", {"tool": "late"}, {"call_id": "missing"}, {"game_id": ""}])
def test_invalid_scope_cannot_silently_disable_fact_guard(chain, scope):
    store, game, _, _ = chain
    call = calls(game)[0]
    before = deepcopy(store.require_game(game.game_id).to_dict())
    result = execute(chain, [call], [assertion(call.arguments["requirement_patch"]["planned_start_at"], scope)])[0]
    assert not result.called and "invalid fact scope" in result.error
    assert store.require_game(game.game_id).to_dict() == before


@pytest.mark.parametrize("reverse", [False, True])
def test_scoped_fact_cannot_override_a_conflicting_global_fact(chain, reverse):
    store, game, _, _ = chain
    call = calls(game)[0]
    facts = [assertion(game.planned_start_at.isoformat()), assertion(call.arguments["requirement_patch"]["planned_start_at"], {"call_id": "early"})]
    if reverse:
        facts.reverse()
    result = execute(chain, [call], facts)[0]
    assert not result.called and "conflicting current facts" in result.error
    assert store.require_game(game.game_id).planned_start_at == game.planned_start_at


@pytest.mark.parametrize("origin", ["other_sender", "other_conversation", "missing"])
def test_scope_cannot_make_an_untrusted_source_authoritative(chain, origin):
    store, game, message, _ = chain
    if origin != "missing":
        store.append_user_turn(UserMessage(conversation_id="foreign" if origin == "other_conversation" else message.conversation_id,
            sender_id="B" if origin == "other_sender" else "A", sender_name="他人", text="另一个时间", message_id="untrusted"), "foreign")
    call = calls(game)[0]
    good = game.planned_start_at.isoformat()
    facts = [assertion(good, {"call_id": "early"}), assertion(call.arguments["requirement_patch"]["planned_start_at"], {"call_id": "early"}, source="untrusted")]
    result = execute(chain, [call], facts)[0]
    assert not result.called and "explicit current-task facts" in result.error
    assert store.require_game(game.game_id).planned_start_at == game.planned_start_at


def test_identical_duplicate_assertions_remain_valid(chain):
    call = calls(chain[1])[0]
    fact = assertion(call.arguments["requirement_patch"]["planned_start_at"], {"call_id": "early"})
    result = execute(chain, [call], [fact, deepcopy(fact)])[0]
    assert result.called and result.allowed and not result.error


def test_rejection_feedback_preserves_only_the_target_scope_source(chain):
    _, game, _, _ = chain
    tool_calls = calls(game)
    early = tool_calls[0].arguments['requirement_patch']['planned_start_at']
    late = tool_calls[1].arguments['requirement']['planned_start_at']
    tool_calls[0].arguments['requirement_patch']['planned_start_at'] = late
    result = execute(chain, tool_calls, [assertion(early, {'call_id':'early'}), assertion(late, {'call_id':'late'})])[0]
    assert not result.called
    assert result.result['reference_fact_sources'] == {
        'planned_start_at': {'source_message_id':'source','role':'current','scope':{'call_id':'early'}}}


def test_conflicting_fact_feedback_keeps_all_distinct_message_sources(chain):
    store, game, message, _ = chain
    store.append_user_turn(UserMessage(conversation_id=message.conversation_id,sender_id=message.sender_id,
        sender_name=message.sender_name,text='另一个方案时间',message_id='second-source'),'second')
    call = calls(game)[0]
    facts = [assertion(game.planned_start_at.isoformat()),
        assertion(call.arguments['requirement_patch']['planned_start_at'], source='second-source')]
    result = execute(chain, [call], facts)[0]
    assert not result.called and 'conflicting current facts' in result.error
    sources = result.result['reference_fact_sources']['planned_start_at']['assertions']
    assert {s['source_message_id'] for s in sources} == {'source','second-source'}
    assert len(sources) == 2
