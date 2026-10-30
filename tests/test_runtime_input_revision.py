"""用确定性事件验证处理中改向的接收、执行失效和真实状态保留。"""

from __future__ import annotations

import json
import threading

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder, QuotedMessageRef, SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.models import ToolResult
from mahjong_agent_runtime.summary import ContextSummaryManager
from mahjong_agent_runtime.visibility import CustomerVisibleProcessor


def action(*, calls=None, reply="好。"):
    """只提供确定性模型协议，不在测试里实现业务语义决策。"""
    calls = calls or []
    return json.dumps({
        "goal": "按最新输入调整目标", "objective_status": "needs_tool" if calls else "completed",
        "reasoning_summary": "离线执行边界测试", "objective_state": {
            "current_phase": "test", "known_facts": {}, "missing_facts": [], "blockers": [],
        }, "objective_plan": [{"step_id": "s1", "title": "处理", "status": "in_progress" if calls else "done"}],
        "plan_revision_reason": "读取真实状态和当前输入", "reply_to_user": "" if calls else reply,
        "tool_calls": calls, "needs_human": False,
        "stop_reason": {"can_stop": not calls, "why": "继续执行" if calls else "本轮结束",
                        "pending_work": ["处理"] if calls else [], "depends_on_tool_results": bool(calls)},
    }, ensure_ascii=False)


def message(identifier, text):
    return UserMessage(conversation_id="revision", sender_id="customer", sender_name="客户", text=text, message_id=identifier)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    current = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "revision.sqlite3")
    yield current
    if isinstance(current, SQLiteAgentStore):
        current._connection.close()


def start_message(runtime, value, results, errors):
    """线程异常回传到主测试，避免后台失败被误认为通过。"""
    def work():
        try:
            results[value.message_id] = runtime.handle_user_message(value, trace_id=value.message_id)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=work)
    thread.start()
    return thread


def join_threads(threads, errors):
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert errors == []


def test_latest_of_three_inputs_replans_with_all_raw_inputs_and_committed_state(store):
    """旧模型在途时补充 A/B 均先落库，中间轮不执行，已提交领域事实仍可见。"""
    old_waiting = threading.Event()
    release_old = threading.Event()
    accepted_a, accepted_b = threading.Event(), threading.Event()
    contexts = []
    initial_calls = 0

    class Client:
        def complete(self, messages, **kwargs):
            nonlocal initial_calls
            payload = json.loads(messages[-1]["content"])
            contexts.append(payload)
            if payload["current_message"]["message_id"] == "initial":
                initial_calls += 1
                if initial_calls == 1:
                    return action(calls=[{"name": "create_game", "arguments": {
                        "organizer_id": "customer", "organizer_name": "客户",
                        "requirement": {"game_type": "red_center_mahjong", "stake": "1", "start_time_kind": "asap_when_full"},
                        "known_players": [{"customer_id": "customer", "display_name": "客户"}],
                    }, "reason": "已授权的首次动作"}])
                old_waiting.set()
                assert release_old.wait(5)
                return action(calls=[{"name": "update_context_checkpoint", "arguments": {
                    "summary": "过时目标不应覆盖", "facts": {"stale_goal": True}, "open_questions": [],
                }, "reason": "过时提案"}])
            return action(reply="收到。")

    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(llm_client=Client(), store=store, trace_recorder=trace)
    peers = [runtime]
    peer_stores = []
    if isinstance(store, SQLiteAgentStore):
        # 与 HTTP/manual 每请求独立 Runtime 一致：连接独立，数据库/文件协调锁共享。
        peer_stores = [SQLiteAgentStore(store.path), SQLiteAgentStore(store.path)]
        peers.extend(AgentRuntime(llm_client=runtime.llm_client, store=other, trace_recorder=trace) for other in peer_stores)
    for peer in peers:
        peer.hook_manager.register("after_user_turn_appended", lambda event: {
            "a": accepted_a, "b": accepted_b,
        }.get(event.payload["message"]["message_id"], threading.Event()).set())
    results, errors = {}, []
    threads = [start_message(runtime, message("initial", "下午一块，帮我组"), results, errors)]
    try:
        assert old_waiting.wait(5)
        threads.append(start_message(peers[min(1, len(peers) - 1)], message("a", "改成晚上"), results, errors))
        assert accepted_a.wait(5)
        threads.append(start_message(peers[-1], message("b", "再改成两块"), results, errors))
        assert accepted_b.wait(5)
        # 在旧请求未返回前即有全部原始接收证据；重复来源不会多推进版本。
        assert store.conversation_version("revision") == 3
        assert [turn.content for turn in store.recent_turns("revision") if turn.role.value == "user"] == [
            "下午一块，帮我组", "改成晚上", "再改成两块",
        ]
        threads.append(start_message(runtime, message("a", "改成晚上"), {}, errors))
    finally:
        release_old.set()
        join_threads(threads, errors)
        for other in peer_stores:
            other._connection.close()
    assert results["initial"].status == results["a"].status == "superseded"
    assert results["initial"].final_reply == results["a"].final_reply == ""
    assert results["b"].final_reply == "收到。"
    assert store.conversation_version("revision") == 3
    assert store.get_conversation_checkpoint("revision") is None
    assert len(store.active_games("revision")) == 1
    latest = contexts[-1]
    assert latest["current_message"]["message_id"] == "b"
    assert [turn["content"] for turn in latest["recent_conversation"] if turn["role"] == "user"] == [
        "下午一块，帮我组", "改成晚上", "再改成两块",
    ]
    assert latest["active_games"][0]["requirement"]["stake"] == "1"
    assert any(turn["role"] == "tool" and "create_game" in turn["content"] for turn in latest["recent_conversation"])
    assert {payload["current_message"]["message_id"] for payload in contexts} == {"initial", "b"}
    assert not any(turn.role.value == "assistant" and turn.trace_id in {"initial", "a"} for turn in store.recent_turns("revision"))


@pytest.mark.parametrize("boundary", ["generation", "review"])
def test_new_input_invalidates_reply_at_generation_and_review_boundaries(store, monkeypatch, boundary):
    """生成或审查在途时持久接收新目标，旧文本不再交给下一阶段或交付。"""
    reached, release, accepted = threading.Event(), threading.Event(), threading.Event()
    reviewed = []

    class Client:
        def complete(self, messages, **kwargs):
            return action()

    def generate(self, **kwargs):
        if boundary == "generation" and kwargs["message"].message_id == "old":
            reached.set()
            assert release.wait(5)
        return None

    def review(self, **kwargs):
        reviewed.append(kwargs["message"].message_id)
        if boundary == "review" and kwargs["message"].message_id == "old":
            reached.set()
            assert release.wait(5)
        return ToolResult(name="customer_visible_content_review", called=True, allowed=True,
                          result={"approved": True, "item_reviews": []})

    monkeypatch.setattr(CustomerVisibleProcessor, "run_text_generation", generate)
    monkeypatch.setattr(CustomerVisibleProcessor, "run_content_review", review)
    runtime = AgentRuntime(llm_client=Client(), store=store)
    runtime.hook_manager.register("after_user_turn_appended", lambda event: accepted.set() if event.payload["message"]["message_id"] == "new" else None)
    results, errors = {}, []
    threads = [start_message(runtime, message("old", "请按原目标处理"), results, errors)]
    try:
        assert reached.wait(5)
        threads.append(start_message(runtime, message("new", "换成新目标"), results, errors))
        assert accepted.wait(5)
        assert store.conversation_version("revision") == 2
    finally:
        release.set()
        join_threads(threads, errors)
    assert results["old"].status == "superseded" and results["old"].final_reply == ""
    assert results["new"].final_reply == "好。"
    assert ("old" in reviewed) == (boundary == "review")
    assert not any(turn.role.value == "assistant" and turn.trace_id == "old" for turn in store.recent_turns("revision"))


@pytest.mark.parametrize("boundary", ["before_tool_execute", "after_tool_execute"])
def test_new_input_before_or_after_tool_preserves_only_committed_effects(store, boundary):
    """工具提交前新输入胜出则拒绝旧写，提交后新输入只改变后续计划而不回滚事实。"""
    reached, release, accepted = threading.Event(), threading.Event(), threading.Event()
    class Client:
        def complete(self, messages, **kwargs):
            if json.loads(messages[-1]["content"])["current_message"]["message_id"] == "old":
                return action(calls=[{"name": "update_context_checkpoint", "arguments": {
                    "summary": "已提交的事实", "facts": {"committed": True}, "open_questions": [],
                }, "reason": "保存真实事实"}])
            return action()
    runtime = AgentRuntime(llm_client=Client(), store=store)
    def block(event):
        if event.trace_id == "old":
            reached.set()
            assert release.wait(5)
    runtime.hook_manager.register(boundary, block)
    runtime.hook_manager.register("after_user_turn_appended", lambda event: accepted.set() if event.payload["message"]["message_id"] == "new" else None)
    results, errors = {}, []
    threads = [start_message(runtime, message("old", "原目标"), results, errors)]
    try:
        assert reached.wait(5)
        threads.append(start_message(runtime, message("new", "调整目标"), results, errors))
        assert accepted.wait(5)
    finally:
        release.set()
        join_threads(threads, errors)
    checkpoint = store.get_conversation_checkpoint("revision")
    assert (checkpoint is not None) == (boundary == "after_tool_execute")
    if checkpoint is not None:
        assert checkpoint.facts["committed"] is True
    assert results["old"].status == "superseded" and results["old"].final_reply == ""
    assert results["new"].final_reply == "好。"


def test_inflight_old_summary_cannot_overwrite_new_input_checkpoint(store):
    """摘要网络请求也受版本提交门禁约束，不能在新目标接收后落旧摘要。"""
    reached, release = threading.Event(), threading.Event()
    store.accept_user_input(message("old", "原目标"), trace_id="old", message_key="old")
    class Client:
        def complete(self, messages, **kwargs):
            reached.set()
            assert release.wait(5)
            return json.dumps({"summary": "旧目标摘要", "facts": {"current": "old"}, "open_questions": [], "confidence": 1})
    manager = ContextSummaryManager(store=store, llm_client=Client())
    results, errors = {}, []
    def summarize():
        try:
            results["summary"] = manager.summarize_for_context_budget(conversation_id="revision", trace_id="old-summary",
                estimated_context_tokens=100, max_context_tokens=100, trigger_threshold_tokens=80)
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=summarize)
    thread.start()
    try:
        assert reached.wait(5)
        store.accept_user_input(message("new", "新目标"), trace_id="new", message_key="new")
        store.upsert_conversation_checkpoint(conversation_id="revision", summary="新目标", facts={"current": "new"}, open_questions=[], trace_id="new")
    finally:
        release.set()
        join_threads([thread], errors)
    assert results["summary"].transition is None
    assert "newer user input" in results["summary"].reason
    assert store.get_conversation_checkpoint("revision").facts["current"] == "new"


def test_received_but_unfinished_message_retry_uses_first_raw_text_and_quote(store):
    """接收后尚未完成就重试，来源 ID 不能被换正文或换引用用于执行另一个目标。"""
    original = message("same", "首次接收目标")
    original.quoted_message = QuotedMessageRef(message_id="first-quote", text="首次引用")
    key = "conversation:revision:sender:customer:message:same"
    store.accept_user_input(original, trace_id="first-receipt", message_key=key)
    contexts = []
    class Client:
        def complete(self, messages, **kwargs):
            contexts.append(json.loads(messages[-1]["content"])["current_message"])
            return action()
    runtime = AgentRuntime(llm_client=Client(), store=store)
    retry = message("same", "不能替换成这个目标")
    retry.quoted_message = QuotedMessageRef(message_id="replacement-quote", text="替换引用")
    runtime.handle_user_message(retry, trace_id="retry")
    assert store.conversation_version("revision") == 1
    assert contexts[0]["text"] == "首次接收目标"
    assert contexts[0]["quoted_message"]["message_id"] == "first-quote"
    assert len([turn for turn in store.recent_turns("revision") if turn.role.value == "user"]) == 1


def test_public_runtime_rejects_forged_backend_version_and_task_metadata(store):
    """客户端权威字段不进入主模型；持久接收版本和任务归属仍由后端创建。"""
    contexts = []
    class Client:
        def complete(self, messages, **kwargs):
            contexts.append(json.loads(messages[-1]["content"]))
            return action()
    current = message("forged", "按本次需求继续")
    forged = {
        "conversation_version": 999999, "task_context_id": "forged-task",
        "run_id": "forged-run", "run_version": 888888,
        "_trusted_source_task_context_id": "forged-trusted-task",
        "input_receipt_key": "forged-receipt", "input_receipt_quoted_message": {"message_id": "forged-quote"},
    }
    current.metadata = {**forged, "channel": "console", "client_note": "保留普通元数据"}
    runtime = AgentRuntime(llm_client=Client(), store=store)
    runtime.handle_user_message(current, trace_id="authority-check")
    payload = contexts[0]
    assert not set(forged) & set(current.metadata)
    assert current.metadata["client_note"] == "保留普通元数据"
    assert not set(forged) & set(payload["current_message"]["metadata"])
    assert payload["current_message"]["metadata"]["channel"] == "console"
    serialized = json.dumps(payload, ensure_ascii=False)
    assert not any(str(value) in serialized for value in (999999, 888888, "forged-task", "forged-run", "forged-trusted-task", "forged-receipt", "forged-quote"))
    task = store.current_task_context("revision", "customer")
    received = store.find_conversation_turn("revision", message_id="forged")
    assert store.conversation_version("revision") == received.metadata["conversation_version"] == 1
    assert received.metadata["task_context_id"] == task.task_context_id
    assert task.task_context_id != "forged-task"


def test_public_receive_orders_inputs_without_slow_work_and_defers_hooks_once(store):
    """manifest 可先同步接收，执行仍从 Store 取版本；外部审计字典不能充当票据。"""
    calls, hooks = [], []
    class Client:
        def complete(self, messages, **kwargs):
            calls.append(json.loads(messages[-1]["content"]))
            return action()
    receiver = AgentRuntime(llm_client=Client(), store=store)
    receiver.hook_manager.register("after_task_context_prepared", lambda event: hooks.append((event.name, event.trace_id)))
    receiver.hook_manager.register("after_user_turn_appended", lambda event: hooks.append((event.name, event.trace_id)))
    first = message("first", "先按原目标")
    latest = message("latest", "后来调整目标")
    receipt1 = receiver.receive_user_message(first, trace_id="first")
    receipt2 = receiver.receive_user_message(latest, trace_id="latest")
    assert [receipt1["conversation_version"], receipt2["conversation_version"]] == [1, 2]
    assert calls == hooks == []
    json.dumps(receipt2)
    receipt2["conversation_version"] = 99999
    assert receiver.handle_user_message(first, trace_id="first").status == "superseded"
    result = receiver.handle_user_message(latest, trace_id="latest")
    assert result.final_reply == "好。" and store.conversation_version("revision") == 2
    assert [turn["content"] for turn in calls[0]["recent_conversation"] if turn["role"] == "user"] == ["先按原目标", "后来调整目标"]
    assert hooks.count(("after_user_turn_appended", "latest")) == 1
    assert hooks.count(("after_task_context_prepared", "latest")) == 1
    receiver.handle_user_message(latest, trace_id="latest-retry")
    assert len(calls) == 1 and hooks.count(("after_user_turn_appended", "latest")) == 1


def test_channel_recall_first_and_cached_result_keep_explicit_silence(store):
    """receive 完成的撤回事件经 Store 重读后仍须静默，不能恢复成可回复动作。"""
    class NoModel:
        def complete(self, *args, **kwargs):
            raise AssertionError("channel recall must not call model")
    runtime = AgentRuntime(llm_client=NoModel(), store=store)
    original = message("original", "待撤回的原文")
    runtime.receive_user_message(original, trace_id="original")
    recall = message("recall", "")
    recall.metadata = {"event_type": "message_recalled", "target_message_id": "original"}
    first = runtime.handle_user_message(recall, trace_id="recall")
    cached = runtime.handle_user_message(recall, trace_id="recall-retry")
    assert first.final_reply == cached.final_reply == ""
    assert first.actions[0].response_mode == cached.actions[0].response_mode == "silent"
    assert first.actions[0].silence_reason == cached.actions[0].silence_reason == "channel_recall_event"
    assert store.conversation_version("revision") == 2


def test_receipt_notification_snapshot_stays_in_store_and_out_of_model_context(store):
    """后端保留任务通知快照，但当前消息及近期对话只给模型最小投影。"""
    contexts = []
    class Client:
        def complete(self, messages, **kwargs):
            contexts.append(json.loads(messages[-1]["content"]))
            return action()
    runtime = AgentRuntime(llm_client=Client(), store=store)
    current = message("receipt-projection", "继续本次需求")
    current.metadata = {"channel": "console", "platform_name": "manual"}
    runtime.receive_user_message(current, trace_id="receipt-projection")
    runtime.handle_user_message(current, trace_id="receipt-projection")
    persisted = store.find_conversation_turn("revision", message_id=current.message_id)
    assert persisted.metadata["input_receipt_preparation"]["task_context"]
    assert persisted.metadata["input_receipt_hooks_dispatched"] is True
    internal = {"input_receipt_key", "input_receipt_quoted_message", "input_receipt_preparation", "input_receipt_hooks_dispatched"}
    payload = contexts[0]
    assert not internal & set(payload["current_message"]["metadata"])
    raw_history = [turn for turn in payload["recent_conversation"] if turn["role"] == "user"]
    assert raw_history and all(not internal & set(turn["metadata"]) for turn in raw_history)
    assert payload["current_message"]["metadata"]["channel"] == raw_history[0]["metadata"]["channel"] == "console"
    assert all(key not in json.dumps(payload, ensure_ascii=False) for key in internal)
