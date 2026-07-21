from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

from mahjong_agent_runtime import (
    ContextSummaryManager,
    ContextSummaryPolicy,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.context import AgentContextBuilder
from mahjong_agent_runtime.models import (
    ConversationRole,
    ConversationTurn,
    Game,
    GameParticipant,
    InviteStatus,
    now,
)
from mahjong_agent_runtime.summary_evaluation import ContextSummaryQualityEvaluator
from mahjong_agent_runtime.summary_evaluation import DecisionConsistencyReport


@dataclass(slots=True)
class PayloadFunctionClient:
    responder: Callable[[dict[str, Any]], dict[str, Any]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        payload = json.loads(messages[-1]["content"])
        self.calls.append(
            {
                "messages": messages,
                "payload": payload,
                "trace_id": trace_id,
                "timeout_seconds": timeout_seconds,
            }
        )
        return json.dumps(self.responder(payload), ensure_ascii=False)


@dataclass(slots=True)
class ContextSensitiveDecisionClient:
    scenario: str
    calls: list[dict[str, Any]] = field(default_factory=list)

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        payload = json.loads(messages[-1]["content"])
        self.calls.append({"payload": payload, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        if self.scenario == "failed_attempts":
            action = self._failed_attempts(payload)
        elif self.scenario == "constraint":
            action = self._constraint(payload)
        elif self.scenario == "continuation":
            action = self._continuation(payload)
        elif self.scenario == "tool_dependency":
            action = self._tool_dependency(payload)
        else:  # pragma: no cover - test fixture misuse
            raise AssertionError(f"unknown scenario: {self.scenario}")
        return json.dumps(action, ensure_ascii=False)

    @staticmethod
    def _failed_attempts(payload: dict[str, Any]) -> dict[str, Any]:
        history = _history_text(payload)
        facts = _checkpoint_facts(payload)
        completed_steps = {str(item) for item in facts.get("completed_steps") or []}
        created = "create_game 成功" in history or "create_game" in completed_steps
        if created:
            return _reply_action("已建了新局，正在邀人。")
        return _tool_action(
            "search_current_games",
            {"requirement": {"stake": "0.5", "smoke_preference": "no_smoking"}},
        )

    @staticmethod
    def _constraint(payload: dict[str, Any]) -> dict[str, Any]:
        history = _history_text(payload)
        facts = _checkpoint_facts(payload)
        constraints = "\n".join(str(item) for item in facts.get("temporary_constraints") or [])
        excludes_old_wang = "不叫老王" in history or "老王" in constraints
        customer_id = "liu" if excludes_old_wang else "old_wang"
        display_name = "刘哥" if excludes_old_wang else "老王"
        game_id = _active_game_id(payload) or str(facts.get("active_game_id") or "")
        return _tool_action(
            "create_invite_drafts",
            {
                "game_id": game_id,
                "invitations": [
                    {
                        "customer_id": customer_id,
                        "display_name": display_name,
                        "message_text": "今晚七点0.5无烟，打吗？",
                    }
                ],
            },
        )

    @staticmethod
    def _continuation(payload: dict[str, Any]) -> dict[str, Any]:
        history = _history_text(payload)
        facts = _checkpoint_facts(payload)
        progress = dict(facts.get("candidate_progress") or {})
        has_completed = (
            "张哥 已确认" in history
            and "李哥 已确认" in history
            and "王姐 等回复" in history
        ) or (
            progress.get("confirmed_customer_ids") == ["zhang", "li"]
            and progress.get("pending_customer_ids") == ["wang_jie"]
        )
        next_customer_id = "candidate_four" if has_completed else "zhang"
        next_name = "赵哥" if has_completed else "张哥"
        game_id = _active_game_id(payload) or str(facts.get("active_game_id") or "")
        return _tool_action(
            "create_invite_drafts",
            {
                "game_id": game_id,
                "invitations": [
                    {
                        "customer_id": next_customer_id,
                        "display_name": next_name,
                        "message_text": "今晚七点0.5无烟，打吗？",
                    }
                ],
            },
        )

    @staticmethod
    def _tool_dependency(payload: dict[str, Any]) -> dict[str, Any]:
        history = _history_text(payload)
        facts = _checkpoint_facts(payload)
        game_id = "game_456" if "game_456" in history else str(facts.get("active_game_id") or "")
        if not game_id:
            return _tool_action("search_current_games", {"requirement": {"stake": "0.5"}})
        return _tool_action(
            "update_game_requirement",
            {
                "game_id": game_id,
                "requirement_patch": {"start_time": "19:00"},
                "reason": "用户明确把开始时间改到19:00",
            },
        )


def test_summary_quality_retains_failed_attempts_and_does_not_repeat_search() -> None:
    store = InMemoryAgentStore()
    conversation_id = "summary_quality_failed_attempts"
    game = _seed_game(store, conversation_id=conversation_id)
    turns = [
        ("user", "帮我约个0.5无烟的"),
        ("assistant", "我先看看现有局。"),
        ("tool", "search_current_games 无匹配 第1次"),
        ("assistant", "再确认一下附近时间。"),
        ("tool", "search_current_games 无匹配 第2次"),
        ("assistant", "最后扩大时间范围看一次。"),
        ("tool", "search_current_games 无匹配 第3次"),
        ("assistant", "没有现成局，开始新建。"),
        ("tool", f"create_game 成功 game_id={game.game_id}"),
        ("assistant", "新局已建，正在邀人。"),
    ]
    _append_turns(store, conversation_id, turns)

    def summarize(payload: dict[str, Any]) -> dict[str, Any]:
        history = _summary_history_text(payload)
        assert history.count("search_current_games 无匹配") == 3
        assert payload["active_games"][0]["game_id"] == game.game_id
        _assert_decision_critical_hints(payload)
        return _summary_output(
            summary="0.5无烟局已经创建，当前正在邀人，不要再重复搜索现有局。",
            facts={
                "current_objective": "为用户组0.5无烟局",
                "active_game_id": game.game_id,
                "failed_attempts": [
                    {"tool": "search_current_games", "outcome": "no_match", "count": 3}
                ],
                "completed_steps": ["search_current_games", "create_game"],
                "pending_work": ["等待并继续邀约候选人"],
            },
        )

    report = _evaluate(
        store,
        conversation_id=conversation_id,
        current_text="那现在什么情况？",
        decision_scenario="failed_attempts",
        summary_responder=summarize,
    )

    assert report.consistent is True
    assert report.tool_calls_consistent is True
    assert report.before.tool_names == []
    assert report.after.tool_names == []
    assert report.after.reply_to_user == "已建了新局，正在邀人。"
    assert report.checkpoint is not None
    assert report.checkpoint.facts["failed_attempts"][0]["count"] == 3
    assert report.compressed_context_audit["included_turn_count"] == 0


def test_summary_quality_retains_temporary_exclusion_constraint() -> None:
    store = InMemoryAgentStore()
    conversation_id = "summary_quality_constraint"
    game = _seed_game(store, conversation_id=conversation_id)
    turns = [
        ("user", "今晚帮我组个0.5无烟局"),
        ("user", "这次不叫老王，上次他放鸽子了"),
        ("assistant", "好。"),
        ("tool", f"create_game 成功 game_id={game.game_id}"),
    ] + [("assistant", f"组局流程记录 {index}") for index in range(8)]
    _append_turns(store, conversation_id, turns)

    def summarize(payload: dict[str, Any]) -> dict[str, Any]:
        assert "这次不叫老王" in _summary_history_text(payload)
        _assert_decision_critical_hints(payload)
        return _summary_output(
            summary="继续为当前局邀人，本次临时排除老王。",
            facts={
                "current_objective": "继续邀请候选人",
                "active_game_id": game.game_id,
                "temporary_constraints": ["本次不邀请老王：上次放鸽子"],
                "completed_steps": ["create_game"],
                "pending_work": ["邀请其他候选人"],
            },
        )

    report = _evaluate(
        store,
        conversation_id=conversation_id,
        current_text="继续邀人吧",
        decision_scenario="constraint",
        summary_responder=summarize,
    )

    assert report.consistent is True
    invitation = report.after.tool_calls[0]["arguments"]["invitations"][0]
    assert invitation["customer_id"] == "liu"
    assert invitation["customer_id"] != "old_wang"
    assert "老王" in report.checkpoint.facts["temporary_constraints"][0]


def test_summary_quality_detects_when_constraint_is_lost() -> None:
    """Negative control: the evaluator must fail when compression changes a decision."""

    store = InMemoryAgentStore()
    conversation_id = "summary_quality_constraint_lost"
    game = _seed_game(store, conversation_id=conversation_id)
    _append_turns(
        store,
        conversation_id,
        [
            ("user", "今晚帮我组个0.5无烟局"),
            ("user", "这次不叫老王，上次他放鸽子了"),
            ("tool", f"create_game 成功 game_id={game.game_id}"),
        ]
        + [("assistant", f"组局流程记录 {index}") for index in range(9)],
    )

    def summarize_without_constraint(payload: dict[str, Any]) -> dict[str, Any]:
        assert "这次不叫老王" in _summary_history_text(payload)
        return _summary_output(
            summary="继续为当前局邀人。",
            facts={
                "current_objective": "继续邀请候选人",
                "active_game_id": game.game_id,
                "completed_steps": ["create_game"],
                "pending_work": ["邀请候选人"],
            },
        )

    report = _evaluate(
        store,
        conversation_id=conversation_id,
        current_text="继续邀人吧",
        decision_scenario="constraint",
        summary_responder=summarize_without_constraint,
    )

    assert report.consistent is False
    assert report.tool_calls_consistent is False
    assert report.before.tool_calls[0]["arguments"]["invitations"][0]["customer_id"] == "liu"
    assert report.after is not None
    assert report.after.tool_calls[0]["arguments"]["invitations"][0]["customer_id"] == "old_wang"
    assert any("tool_calls changed" in item for item in report.differences)


def test_summary_quality_continues_candidate_pipeline_without_duplicate_invites() -> None:
    store = InMemoryAgentStore()
    conversation_id = "summary_quality_continuation"
    game = _seed_game(store, conversation_id=conversation_id)
    drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[
            {"customer_id": "zhang", "display_name": "张哥", "message_text": "打吗？"},
            {"customer_id": "li", "display_name": "李哥", "message_text": "打吗？"},
            {"customer_id": "wang_jie", "display_name": "王姐", "message_text": "打吗？"},
        ],
        trace_id="trace_summary_quality_invites",
    )
    drafts[0].status = InviteStatus.CONFIRMED
    drafts[1].status = InviteStatus.CONFIRMED
    drafts[2].status = InviteStatus.SENT
    turns = [
        ("user", "继续帮我找够人"),
        ("tool", "候选队列 zhang,li,wang_jie,candidate_four,candidate_five"),
        ("tool", "张哥 已确认"),
        ("tool", "李哥 已确认"),
        ("tool", "王姐 等回复"),
    ] + [("assistant", f"候选流程记录 {index}") for index in range(7)]
    _append_turns(store, conversation_id, turns)

    def summarize(payload: dict[str, Any]) -> dict[str, Any]:
        history = _summary_history_text(payload)
        assert "张哥 已确认" in history
        assert "李哥 已确认" in history
        assert "王姐 等回复" in history
        statuses = {item["customer_id"]: item["status"] for item in payload["invite_drafts"]}
        assert statuses == {
            "zhang": "confirmed",
            "li": "confirmed",
            "wang_jie": "sent",
        }
        _assert_decision_critical_hints(payload)
        return _summary_output(
            summary="已确认张哥和李哥，王姐仍在等回复；继续从赵哥开始邀约，不要重复邀请前三人。",
            facts={
                "current_objective": "完成剩余候选人邀约",
                "active_game_id": game.game_id,
                "candidate_progress": {
                    "confirmed_customer_ids": ["zhang", "li"],
                    "pending_customer_ids": ["wang_jie"],
                    "remaining_customer_ids": ["candidate_four", "candidate_five"],
                },
                "completed_steps": ["invite:zhang", "invite:li", "invite:wang_jie"],
                "pending_work": ["从candidate_four继续邀约", "等待wang_jie回复"],
            },
        )

    report = _evaluate(
        store,
        conversation_id=conversation_id,
        current_text="继续约人吧",
        decision_scenario="continuation",
        summary_responder=summarize,
    )

    assert report.consistent is True
    invited = report.after.tool_calls[0]["arguments"]["invitations"]
    assert [item["customer_id"] for item in invited] == ["candidate_four"]
    assert report.checkpoint.facts["candidate_progress"]["pending_customer_ids"] == ["wang_jie"]


def test_summary_quality_retains_tool_result_entity_for_follow_up_mutation() -> None:
    store = InMemoryAgentStore()
    conversation_id = "summary_quality_tool_dependency"
    game = Game(
        game_id="game_456",
        conversation_id=conversation_id,
        organizer_id="user_a",
        organizer_name="用户A",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "start_time": "18:30",
            "known_player_count": 1,
            "needed_seats": 3,
        },
        participants=[GameParticipant(customer_id="user_a", display_name="用户A")],
        planned_start_at=now() + timedelta(hours=3),
        expires_at=now() + timedelta(hours=7),
    )
    store.games[game.game_id] = game
    turns = [
        ("user", "今晚帮我建一个18:30的局"),
        ("assistant", "我先建局。"),
        ("tool", "create_game 成功 game_id=game_456 start_time=18:30"),
    ] + [("assistant", f"建局后的流程记录 {index}") for index in range(5)]
    _append_turns(store, conversation_id, turns)

    def summarize(payload: dict[str, Any]) -> dict[str, Any]:
        assert "game_456" in _summary_history_text(payload)
        assert payload["active_games"][0]["game_id"] == "game_456"
        _assert_decision_critical_hints(payload)
        return _summary_output(
            summary="game_456已创建，原定18:30开始，后续修改必须继续操作该局。",
            facts={
                "current_objective": "维护已创建局",
                "active_game_id": "game_456",
                "confirmed_facts": {"start_time": "18:30"},
                "completed_steps": ["create_game"],
                "pending_work": [],
            },
        )

    report = _evaluate(
        store,
        conversation_id=conversation_id,
        current_text="帮我把时间改到19:00",
        decision_scenario="tool_dependency",
        summary_responder=summarize,
    )

    assert report.consistent is True
    assert report.after.tool_names == ["update_game_requirement"]
    arguments = report.after.tool_calls[0]["arguments"]
    assert arguments["game_id"] == "game_456"
    assert arguments["requirement_patch"]["start_time"] == "19:00"


def _evaluate(
    store: InMemoryAgentStore,
    *,
    conversation_id: str,
    current_text: str,
    decision_scenario: str,
    summary_responder: Callable[[dict[str, Any]], dict[str, Any]],
) -> DecisionConsistencyReport:
    trace = InMemoryTraceRecorder()
    summary_manager = ContextSummaryManager(
        store=store,
        llm_client=PayloadFunctionClient(summary_responder),
        trace_recorder=trace,
        policy=ContextSummaryPolicy(max_summary_input_tokens=20_000),
    )
    evaluator = ContextSummaryQualityEvaluator(
        context_builder=AgentContextBuilder(store, ToolGateway(store)),
        summary_manager=summary_manager,
        decision_client=ContextSensitiveDecisionClient(decision_scenario),
        trace_recorder=trace,
    )
    return evaluator.evaluate(
        message=UserMessage(
            conversation_id=conversation_id,
            sender_id="user_a",
            sender_name="用户A",
            text=current_text,
            message_id=f"{conversation_id}_current",
        ),
        trace_id=f"trace_{conversation_id}",
    )


def _seed_game(store: InMemoryAgentStore, *, conversation_id: str) -> Game:
    game, _ = store.create_game(
        conversation_id=conversation_id,
        organizer_id="user_a",
        organizer_name="用户A",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoking",
            "start_time_kind": "asap_when_full",
            "known_player_count": 1,
            "needed_seats": 3,
        },
        known_players=[{"customer_id": "user_a", "display_name": "用户A"}],
        trace_id=f"trace_seed_{conversation_id}",
    )
    return game


def _append_turns(
    store: InMemoryAgentStore,
    conversation_id: str,
    turns: list[tuple[str, str]],
) -> None:
    for index, (role, content) in enumerate(turns, start=1):
        trace_id = f"trace_{conversation_id}_seed_{index}"
        if role == ConversationRole.USER.value:
            store.append_turn(
                conversation_id,
                ConversationTurn(
                    role=ConversationRole.USER,
                    content=content,
                    trace_id=trace_id,
                    sender_id="user_a",
                    sender_name="用户A",
                ),
            )
        elif role == ConversationRole.ASSISTANT.value:
            store.append_assistant_turn(conversation_id, content, trace_id)
        elif role == ConversationRole.TOOL.value:
            store.append_tool_turn(conversation_id, content, trace_id)
        else:  # pragma: no cover - test fixture misuse
            raise AssertionError(f"unknown role: {role}")


def _summary_output(*, summary: str, facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": summary,
        "facts": facts,
        "open_questions": [],
        "confidence": 0.98,
    }


def _reply_action(reply: str) -> dict[str, Any]:
    return {
        "goal": "根据当前上下文继续任务",
        "objective_status": "completed",
        "reasoning_summary": "已有足够上下文，可直接回答。",
        "reply_to_user": reply,
        "tool_calls": [],
        "needs_human": False,
        "stop_reason": {
            "can_stop": True,
            "why": "本轮已完成。",
            "pending_work": [],
            "depends_on_tool_results": True,
        },
        "badcase": None,
    }


def _tool_action(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": "根据当前上下文继续任务",
        "objective_status": "needs_tool",
        "reasoning_summary": "需要继续执行尚未完成的动作。",
        "reply_to_user": "",
        "tool_calls": [
            {
                "name": name,
                "arguments": arguments,
                "reason": "继续当前任务且避免重复已完成动作。",
            }
        ],
        "needs_human": False,
        "stop_reason": {
            "can_stop": False,
            "why": "仍需工具执行。",
            "pending_work": [name],
            "depends_on_tool_results": True,
        },
        "badcase": None,
    }


def _history_text(payload: dict[str, Any]) -> str:
    return "\n".join(str(item.get("content") or "") for item in payload.get("recent_conversation") or [])


def _summary_history_text(payload: dict[str, Any]) -> str:
    return "\n".join(str(item.get("content") or "") for item in payload.get("recent_conversation") or [])


def _checkpoint_facts(payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint = payload.get("conversation_checkpoint")
    return dict(checkpoint.get("facts") or {}) if isinstance(checkpoint, dict) else {}


def _active_game_id(payload: dict[str, Any]) -> str:
    games = payload.get("active_games") or []
    return str(games[0].get("game_id") or "") if games else ""


def _assert_decision_critical_hints(payload: dict[str, Any]) -> None:
    hints = payload["summary_contract"]["decision_critical_fact_hints"]
    assert "failed_attempts" in hints
    assert "temporary_constraints" in hints
    assert "candidate_progress" in hints
    assert "completed_steps" in hints
    assert "pending_work" in hints
