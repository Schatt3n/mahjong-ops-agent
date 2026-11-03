"""验证普通私聊建局的原子看板责任、授权边界与发布后继合并。"""

from __future__ import annotations

from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import json

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentRuntime, CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, HookEvent, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.group_chat import BoardEngine, GroupBoardTrigger, GroupRoomPolicy
from mahjong_agent_runtime.models import ScheduledTaskStatus


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "board.sqlite3")
    yield value
    if isinstance(value, SQLiteAgentStore):
        value._connection.close()


def _policy(store, room_id="board-a", **kwargs):
    store.upsert_group_room_policy(GroupRoomPolicy(room_id=room_id, **kwargs))


def _create(store, *, conversation_id="wechaty:contact:ordinary", owner="owner"):
    return store.create_game(
        conversation_id=conversation_id, organizer_id=owner, organizer_name="常客",
        requirement={"game_type": "national_standard_mahjong", "stake": "1", "cap_score": 256,
                     "smoke_preference": "any", "start_time_kind": "asap_when_full"},
        known_players=[{"customer_id": owner, "display_name": "常客", "seat_count": 1}],
        trace_id="trace-create",
    )


def _tasks(store):
    return [item for item in store.scheduled_tasks.values() if item.task_type == "publish_group_board"]


def test_list_group_room_policies_is_complete_ordered_and_detached(store):
    assert store.list_group_room_policies() == []
    _policy(store, "z-board", managed=False)
    _policy(store, "a-board", board_enabled=False, outbound_enabled=False)
    before = {room: store.get_group_room_policy(room).to_dict() for room in ("a-board", "z-board")}
    policies = store.list_group_room_policies()
    assert [item.room_id for item in policies] == ["a-board", "z-board"]
    policies[0].board_enabled = True
    policies.clear()
    assert {room: store.get_group_room_policy(room).to_dict() for room in before} == before


def _create_event(game, transition, *, deduplicated=False):
    return HookEvent(name="after_tool_execute", trace_id="trace-replay", payload={
        "call": {"name": "create_game", "arguments": {"organizer_id": game.organizer_id}},
        "result": {"name": "create_game", "called": True, "allowed": True,
                   "deduplicated": deduplicated, "error": None,
                   "result": {"game": {"game_id": game.game_id}},
                   "state_transitions": [transition.to_dict()]},
        "conversation_id": game.conversation_id, "sender_id": game.organizer_id,
    })


def test_ordinary_private_game_atomically_enters_configured_boards_without_friend_or_switch(store):
    _policy(store, "board-a", outbound_enabled=True)
    _policy(store, "board-b", outbound_enabled=False)
    _policy(store, "observer", managed=False)
    _policy(store, "no-board", board_enabled=False)
    game, _ = _create(store)

    assert {item.room_id for item in store.game_conversation_links(game_id=game.game_id)} == {"board-a", "board-b"}
    assert {item.aggregate_id for item in _tasks(store)} == {"board-a", "board-b"}
    assert all(item.payload["publication_id"] for item in _tasks(store))
    assert {game.game_id} == {item.game_id for item in store.get_board_eligible_games("board-b")}
    status = store.game_board_arrangement(game.game_id)
    assert {key: status[key] for key in ("status", "linked_room_count", "pending_task_count",
                                       "outbound_authorized_room_count", "publication_sent")} == {
        "status": "scheduled", "linked_room_count": 2, "pending_task_count": 2,
        "outbound_authorized_room_count": 1, "publication_sent": False,
    }
    assert set(status) == {"status", "linked_room_count", "pending_task_count",
                           "outbound_authorized_room_count", "publication_sent", "instruction"}


@pytest.mark.parametrize("conversation", ["group:board-a:customer:owner", "wechaty:room:board-a:post:message"])
def test_group_native_game_stays_in_origin_board(store, conversation):
    _policy(store, "board-a")
    _policy(store, "unrelated")
    game, _ = _create(store, conversation_id=conversation)
    assert {item.room_id for item in store.game_conversation_links(game_id=game.game_id)} == {"board-a"}
    assert {item.aggregate_id for item in _tasks(store)} == {"board-a"}
    assert store.get_board_eligible_games("unrelated") == []


def test_unknown_group_origin_does_not_fall_back_to_all_boards(store):
    _policy(store, "unrelated")
    game, _ = _create(store, conversation_id="group:not-configured:customer:owner")
    assert store.game_conversation_links(game_id=game.game_id) == []
    assert _tasks(store) == []
    assert store.game_board_arrangement(game.game_id)["status"] == "unconfigured"


def test_no_configured_board_is_explicitly_unconfigured(store):
    game, _ = _create(store)
    assert store.game_board_arrangement(game.game_id)["status"] == "unconfigured"
    assert _tasks(store) == []


def test_create_failure_rolls_back_game_links_tasks_and_transitions(store, monkeypatch):
    _policy(store, "board-a")
    _policy(store, "board-b")
    name = "_ensure_group_board_publish_task_in_transaction" if isinstance(store, SQLiteAgentStore) else "ensure_group_board_publish_task"
    original = getattr(type(store), name)

    def fail_second(self, **kwargs):
        if kwargs["room_id"] == "board-b":
            raise RuntimeError("injected task persistence failure")
        return original(self, **kwargs)

    monkeypatch.setattr(type(store), name, fail_second)
    with pytest.raises(RuntimeError, match="injected task persistence failure"):
        _create(store)
    assert store.games == {}
    assert store.game_conversation_links() == []
    assert _tasks(store) == []
    assert store.transitions == []


def test_processing_publication_has_one_merged_successor_with_distinct_identity(store):
    _policy(store)
    _create(store)
    first = _tasks(store)[0]
    claimed = store.claim_scheduled_task(first.task_id, at=first.due_at)
    assert claimed is not None
    successor, _ = store.ensure_group_board_publish_task(
        room_id="board-a", due_at=first.due_at + timedelta(seconds=30), trace_id="event-1",
    )
    identity = successor.payload["publication_id"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda index: store.ensure_group_board_publish_task(
            room_id="board-a", due_at=first.due_at + timedelta(seconds=index), trace_id=f"event-{index}",
        )[0].task_id, range(1, 21)))
    assert set(results) == {successor.task_id}
    assert len(_tasks(store)) == 2
    pending = [item for item in _tasks(store) if item.status == ScheduledTaskStatus.PENDING]
    assert len(pending) == 1 and pending[0].task_id == successor.task_id
    assert pending[0].payload["publication_id"] == identity != first.payload["publication_id"]
    assert store.scheduled_tasks[first.task_id].status == ScheduledTaskStatus.PROCESSING
    assert successor.task_id != first.task_id


def test_replay_keeps_current_responsibility_and_repairs_missing_task(store):
    _policy(store)
    game, transition = _create(store)
    trigger = GroupBoardTrigger(store=store)
    before = {item.task_id: item.to_dict() for item in _tasks(store)}
    trigger(_create_event(game, transition, deduplicated=True))
    assert {item.task_id: item.to_dict() for item in _tasks(store)} == before
    if isinstance(store, SQLiteAgentStore):
        with store._write_transaction():
            store._connection.execute("DELETE FROM runtime_scheduled_agent_tasks")
    else:
        store.scheduled_tasks.clear()
    trigger(_create_event(game, transition, deduplicated=True))
    assert len(_tasks(store)) == 1
    assert store.game_board_arrangement(game.game_id)["status"] == "scheduled"


def test_new_link_repair_schedules_successor_when_current_board_already_captured_old_games(store):
    """固定旧板已渲染的交错；新补群归属必须有后继，不能被任意processing覆盖。"""
    first_game, _ = _create(store, conversation_id="private-first", owner="first-owner")
    repaired_game, _ = _create(store, conversation_id="private-later", owner="later-owner")
    assert _tasks(store) == []
    _policy(store)
    store.ensure_game_board_arrangement(first_game.game_id, trace_id="initial-link")
    first_task = _tasks(store)[0]
    claimed = store.claim_scheduled_task(first_task.task_id, at=first_task.due_at)
    assert claimed is not None
    old_board_captured, release_old_board = Event(), Event()

    class PausedFirstMessenger(FakeMessenger):
        """只在内存模拟旧发送等待，既不调用审查模型，也不访问真实微信。"""

        def send_group_message(self, room_id, text, *, metadata=None):
            if not self.messages:
                old_board_captured.set()
                assert release_old_board.wait(5)
            return super().send_group_message(room_id, text, metadata=metadata)

    messenger = PausedFirstMessenger()
    engine = BoardEngine(store=store, messenger=messenger)
    with ThreadPoolExecutor(max_workers=1) as pool:
        old_publish = pool.submit(engine.publish, "board-a", trace_id="old-board",
                                  publication_id=first_task.payload["publication_id"])
        try:
            assert old_board_captured.wait(5)
            repaired = store.ensure_game_board_arrangement(repaired_game.game_id, trace_id="dedup-repair", refresh=False)
            pending = [task for task in _tasks(store) if task.status == ScheduledTaskStatus.PENDING]
            assert len(pending) == 1
            successor = pending[0]
            assert successor.task_id != first_task.task_id
            assert successor.payload["publication_id"] != first_task.payload["publication_id"]
            assert repaired["status"] == "scheduled" and repaired["pending_task_count"] == 2
            # 再次重放不再产生dirty归属，仍复用已保留的同一后继。
            before = {task.task_id: task.to_dict() for task in _tasks(store)}
            store.ensure_game_board_arrangement(repaired_game.game_id, trace_id="repeat-repair", refresh=False)
            assert {task.task_id: task.to_dict() for task in _tasks(store)} == before
        finally:
            release_old_board.set()
        old_snapshot = old_publish.result(timeout=5)
    assert [item.game_id for item in old_snapshot.items] == [first_game.game_id]
    store.complete_scheduled_task(first_task.task_id, trace_id="old-complete", at=first_task.due_at,
        expected_attempts=claimed.attempts, expected_lease_until=claimed.lease_until)
    restored = SQLiteAgentStore(store.path) if isinstance(store, SQLiteAgentStore) else store
    try:
        pending = [task for task in _tasks(restored) if task.status == ScheduledTaskStatus.PENDING]
        assert len(pending) == 1 and pending[0].task_id == successor.task_id
        next_claim = restored.claim_scheduled_task(successor.task_id, at=successor.due_at)
        assert next_claim is not None
        next_snapshot = BoardEngine(store=restored, messenger=messenger).publish("board-a", trace_id="next-board",
            publication_id=successor.payload["publication_id"])
        assert {item.game_id for item in next_snapshot.items} == {first_game.game_id, repaired_game.game_id}
        assert next_snapshot.snapshot_id != old_snapshot.snapshot_id
        assert len(messenger.messages) == 2
    finally:
        if restored is not store:
            restored._connection.close()


def test_new_link_repair_merges_into_board_that_has_not_started(store):
    """已有pending尚未取板时直接合并，新归属不会制造多份同群待发任务。"""
    first_game, _ = _create(store, conversation_id="private-first", owner="first-owner")
    repaired_game, _ = _create(store, conversation_id="private-later", owner="later-owner")
    _policy(store)
    store.ensure_game_board_arrangement(first_game.game_id, trace_id="initial-link")
    first_task = _tasks(store)[0]
    store.ensure_game_board_arrangement(repaired_game.game_id, trace_id="repair", refresh=False)
    assert len(_tasks(store)) == 1 and _tasks(store)[0].task_id == first_task.task_id
    assert {game.game_id for game in store.get_board_eligible_games("board-a")} == {first_game.game_id, repaired_game.game_id}


def test_failed_new_link_repair_rolls_back_link_and_preserves_existing_publication(store, monkeypatch):
    """后继写入失败时新关联也回滚，不能留下已登记但无刷新责任的局。"""
    first_game, _ = _create(store, conversation_id="private-first", owner="first-owner")
    repaired_game, _ = _create(store, conversation_id="private-later", owner="later-owner")
    _policy(store)
    store.ensure_game_board_arrangement(first_game.game_id, trace_id="initial-link")
    first_task = _tasks(store)[0]
    assert store.claim_scheduled_task(first_task.task_id, at=first_task.due_at) is not None
    before = {task.task_id: task.to_dict() for task in _tasks(store)}
    links = [item.to_dict() for item in store.game_conversation_links()]

    def cannot_enqueue(*args, **kwargs):
        """只注入持久化失败，原Game、原租约及其他客户事实必须保留。"""
        raise RuntimeError("injected successor persistence failure")

    name = "_ensure_group_board_publish_task_in_transaction" if isinstance(store, SQLiteAgentStore) else "ensure_group_board_publish_task"
    monkeypatch.setattr(type(store), name, cannot_enqueue)
    with pytest.raises(RuntimeError, match="successor persistence failure"):
        store.ensure_game_board_arrangement(repaired_game.game_id, trace_id="repair", refresh=False)
    assert [item.to_dict() for item in store.game_conversation_links()] == links
    assert {task.task_id: task.to_dict() for task in _tasks(store)} == before
    assert repaired_game.game_id in store.games


def test_tool_result_exposes_only_safe_arrangement_status(store):
    _policy(store, "private-group-id")
    result = ToolGateway(store).execute(ToolCall(name="create_game", arguments={
        "organizer_id": "owner", "organizer_name": "常客",
        "requirement": {"game_type": "national_standard_mahjong", "stake": "1",
                        "start_time_kind": "asap_when_full"},
        "known_players": [],
    }, reason="受托建局"), trace_id="trace-tool", conversation_id="private-chat", sender_id="owner", sender_name="常客", step_index=1)
    assert result.called and result.allowed and not result.error
    status = result.result["board_arrangement"]
    assert status["status"] == "scheduled" and status["pending_task_count"] == 1
    assert status["publication_sent"] is False
    assert "private-group-id" not in str(status)


class FakeMessenger:
    def __init__(self):
        self.messages = []

    def send_group_message(self, room_id, text, *, metadata=None):
        self.messages.append((room_id, text))
        return f"sent-{len(self.messages)}"


def test_successful_publication_replay_does_not_schedule_duplicate_generation(store):
    _policy(store)
    game, transition = _create(store)
    first = _tasks(store)[0]
    claimed = store.claim_scheduled_task(first.task_id, at=first.due_at)
    messenger = FakeMessenger()
    snapshot = BoardEngine(store=store, messenger=messenger).publish("board-a", trace_id="trace-publish")
    assert snapshot is not None and [item.game_id for item in snapshot.items] == [game.game_id]
    completed, _ = store.complete_scheduled_task(
        first.task_id, trace_id="trace-complete", at=first.due_at,
        expected_attempts=claimed.attempts, expected_lease_until=claimed.lease_until,
    )
    assert completed.status == ScheduledTaskStatus.COMPLETED
    before = {item.task_id: item.to_dict() for item in _tasks(store)}
    GroupBoardTrigger(store=store)(_create_event(game, transition, deduplicated=True))
    assert {item.task_id: item.to_dict() for item in _tasks(store)} == before
    assert len(messenger.messages) == 1
    assert store.game_board_arrangement(game.game_id)["publication_sent"] is True


def test_completed_publication_new_event_uses_new_task_and_publication_identity(store):
    _policy(store)
    _create(store)
    first = _tasks(store)[0]
    claimed = store.claim_scheduled_task(first.task_id, at=first.due_at)
    store.complete_scheduled_task(
        first.task_id, trace_id="trace-complete", at=first.due_at,
        expected_attempts=claimed.attempts, expected_lease_until=claimed.lease_until,
    )
    next_task, _ = store.ensure_group_board_publish_task(
        room_id="board-a", due_at=first.due_at, trace_id="new-change",
    )
    assert next_task.task_id != first.task_id
    assert next_task.payload["publication_id"] != first.payload["publication_id"]
    assert store.scheduled_tasks[first.task_id].status == ScheduledTaskStatus.COMPLETED


@pytest.mark.parametrize("has_candidate", [False, True])
def test_runtime_main_and_production_review_receive_safe_board_arrangement(store, has_candidate):
    _policy(store, "NEVER_EXPOSE_PRIVATE_BOARD_ID")
    if has_candidate:
        store.upsert_customer(CustomerProfile(customer_id="candidate", display_name="候选客户",
            preferred_games=["national_standard_mahjong"], preferred_stakes=["1"], smoke_preference="any"))
    main_contexts, review_contexts = [], []

    class MainClient:
        def complete(self, messages, **kwargs):
            context = json.loads(messages[-1]["content"])
            main_contexts.append(context)
            step = len(main_contexts)
            if step == 1:
                calls = [ToolCall(name="create_game", arguments={
                    "organizer_id": "owner", "organizer_name": "常客",
                    "requirement": {"game_type": "national_standard_mahjong", "stake": "1", "cap_score": 256,
                                    "smoke_preference": "any", "start_time_kind": "asap_when_full"},
                    "known_players": [{"customer_id": "owner", "display_name": "常客", "seat_count": 1}],
                }, reason="按明确委托真实建局", call_id="create-1", depends_on=[])]
            elif step == 2:
                calls = [ToolCall(name="search_customers", arguments={
                    "game_id": context["active_games"][0]["game_id"], "exclude_customer_ids": ["owner"],
                }, reason="按新局真实条件找候选人", call_id="search-2", depends_on=[])]
            else:
                calls = []
            action = AgentAction(goal="按委托组局", objective_status="needs_tool" if calls else "completed",
                reasoning_summary="使用真实工具结果和已安排的看板责任", tool_calls=calls,
                reply_to_user="好，这桌记下了。" if not calls else "", stop_reason={
                    "can_stop": not calls, "why": "继续执行" if calls else "本轮结束，看板任务已持久化",
                    "pending_work": ["真实工具"] if calls else [], "depends_on_tool_results": True,
                })
            return json.dumps(action.to_dict(), ensure_ascii=False)

    class ReviewClient:
        def complete(self, messages, **kwargs):
            context = json.loads(messages[-1]["content"])
            review_contexts.append(context)
            return json.dumps({"approved": True, "needs_human": False, "violations": [],
                "reasoning_summary": "回复仅确认真实登记，未声称已发布或联系别人",
                "item_reviews": [{"item_id": item["item_id"], "approved": True, "violations": [],
                    "suggested_safe_text": item["text"], "reasoning_summary": "与真实登记相符"}
                    for item in context["review_items"]]}, ensure_ascii=False)

    runtime = AgentRuntime(store=store, llm_client=MainClient(),
        reply_self_review_enabled=True, reply_self_review_client=ReviewClient())
    result = runtime.handle_user_message(UserMessage(conversation_id="ordinary-private", sender_id="owner",
        sender_name="常客", text="我一个人，国标1块256封顶，烟不限，人齐开，帮我组一桌", message_id="private-request"))
    assert result.status == "completed" and len(main_contexts) == 3 and len(review_contexts) == 1
    expected = next(item.result["board_arrangement"] for item in result.tool_results if item.name == "create_game")
    assert expected["status"] == "scheduled" and expected["publication_sent"] is False
    for context in [main_contexts[1], main_contexts[2], review_contexts[0]]:
        evidence = next(item for item in context["turn_tool_evidence"] if item["name"] == "create_game")
        assert evidence["result"]["board_arrangement"] == expected
    assert "NEVER_EXPOSE_PRIVATE_BOARD_ID" not in json.dumps(main_contexts + review_contexts)


@pytest.mark.parametrize("configured_board", [False, True])
@pytest.mark.parametrize("has_candidate", [False, True])
def test_candidate_continuation_uses_real_board_responsibility_without_forcing_private_outreach(
    store, configured_board, has_candidate,
):
    if configured_board:
        _policy(store, "PRIVATE_BOARD_IDENTIFIER", outbound_enabled=False)
    if has_candidate:
        store.upsert_customer(CustomerProfile(customer_id="candidate", display_name="候选客户",
            preferred_games=["national_standard_mahjong"], preferred_stakes=["1"], smoke_preference="any"))
    game, _ = _create(store)
    before_tasks = {item.task_id: item.to_dict() for item in _tasks(store)}
    result = ToolGateway(store).execute(ToolCall(name="search_customers", arguments={
        "game_id": game.game_id, "exclude_customer_ids": ["owner"],
    }), trace_id="trace-search", conversation_id=game.conversation_id,
        sender_id="owner", sender_name="常客", step_index=2)
    assert result.called and result.allowed and not result.error
    assert bool(result.result["candidates"]) is has_candidate
    continuation = result.result["continuation"]
    assert continuation["can_stop"] is (configured_board or not has_candidate)
    board = continuation["authoritative_facts"]["board_arrangement"]
    assert board == store.game_board_arrangement(game.game_id)
    assert "PRIVATE_BOARD_IDENTIFIER" not in json.dumps(board)
    assert {item.task_id: item.to_dict() for item in _tasks(store)} == before_tasks
    assert not store.invite_drafts and not store.outbound_message_drafts
    if configured_board:
        assert board["status"] == "scheduled" and board["publication_sent"] is False
        assert continuation["pending_capabilities"] == []
        assert "Scheduling is not publication" in continuation["instruction"]
        assert "automatic private notification" in continuation["instruction"]
        if has_candidate:
            assert "Private invitation drafts are optional, not required" in continuation["instruction"]
    elif has_candidate:
        assert continuation["pending_capabilities"] == ["prepare_candidate_outreach"]
        assert continuation["suggested_tools"] == ["create_invite_drafts"]
    else:
        assert board["status"] == "unconfigured"
        assert "do not claim that a group refresh or publication was arranged" in continuation["instruction"]


def test_past_publication_without_current_refresh_responsibility_does_not_release_private_continuation(store):
    _policy(store)
    game, _ = _create(store)
    task = _tasks(store)[0]
    claimed = store.claim_scheduled_task(task.task_id, at=task.due_at)
    assert BoardEngine(store=store, messenger=FakeMessenger()).publish("board-a", trace_id="published") is not None
    store.complete_scheduled_task(task.task_id, trace_id="complete", at=task.due_at,
        expected_attempts=claimed.attempts, expected_lease_until=claimed.lease_until)
    # 模拟公开条件已变而刷新责任尚未补齐，旧成功快照不能当作本轮已更新的证据。
    store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake": "2"},
        reason="测试公开条件改变后的旧快照边界", trace_id="changed")
    store.upsert_customer(CustomerProfile(customer_id="candidate", display_name="候选客户",
        preferred_games=["national_standard_mahjong"], preferred_stakes=["2"], smoke_preference="any"))
    result = ToolGateway(store).execute(ToolCall(name="search_customers", arguments={
        "game_id": game.game_id, "exclude_customer_ids": ["owner"],
    }), trace_id="trace-search", conversation_id=game.conversation_id,
        sender_id="owner", sender_name="常客", step_index=3)
    assert result.called and result.allowed and not result.error and result.result["candidates"]
    continuation = result.result["continuation"]
    board = continuation["authoritative_facts"]["board_arrangement"]
    assert board["publication_sent"] is True and board["pending_task_count"] == 0
    assert continuation["can_stop"] is False
    assert continuation["pending_capabilities"] == ["prepare_candidate_outreach"]
