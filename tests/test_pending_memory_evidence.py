"""待审长期画像只能绑定当前顾客的真实原话。"""

from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.input_aggregation import aggregate_pending_input_batch
from mahjong_agent_runtime.models import DEFAULT_TZ, CustomerProfile


NOW = datetime(2026, 11, 25, 18, tzinfo=DEFAULT_TZ)
RAW = "老板，跟你说下，我以后都只打无烟的，推荐时帮我留意一下。今晚我有事不过去了。"
QUOTE = "我以后都只打无烟的"
PARAPHRASE = "客户说以后都只打无烟的，推荐时帮留意"


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    return InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "memory.sqlite3")


def inbound(text=RAW, *, sender="customer", message_id="source", when=NOW, **metadata):
    return UserMessage("chat", sender, sender, text, sent_at=when, message_id=message_id, metadata=metadata)


def intent(quote=QUOTE, **extra):
    return {"customer_id": "customer", "memory_type": "preference", "field": "smoke_preference",
            "value": "no_smoking", "evidence": quote, "confidence": 1, "scope": "long_term", **extra}


def write(store, *, pending, task=None, source_id="source"):
    arguments = {"pending_long_term_memories": pending}
    if task:
        arguments["task_memories"] = [task]
    return ToolGateway(store).execute(ToolCall("record_user_memory", arguments), trace_id="current", step_index=1,
        conversation_id="chat", sender_id="customer", sender_name="客户", source_message_id=source_id)


def assert_rejected(result, index=0):
    assert not result.called and not result.allowed and result.state_transitions == []
    assert result.result["validation_error"]["code"] == "pending_memory_evidence_not_verbatim"
    assert result.result["validation_error"]["item_index"] == index


def test_pending_rejects_paraphrase_atomically_then_accepts_exact_quote(store):
    """混合提案先检查全批，修正原话后只写一次，后端来源覆盖模型填写的元数据。"""
    store.upsert_customer(CustomerProfile("customer", "同名"))
    store.append_user_turn(inbound(), "current")
    task_only = {**intent(), "scope": "current_task"}
    correct = intent(metadata={"source_message_id": "fake", "task_context_id": "fake"})
    assert_rejected(write(store, pending=[correct, intent(PARAPHRASE)], task=task_only), index=1)
    assert store.pending_memory_candidates == {} and store.task_memories == {}
    accepted = write(store, pending=[correct], task=task_only)
    retry = write(store, pending=[correct], task=task_only)
    assert accepted.called and accepted.allowed and accepted.error is None and retry.deduplicated
    candidate, = store.pending_memory_candidates.values()
    assert candidate.evidence == QUOTE and candidate.metadata["source_message_id"] == "source"
    assert "task_context_id" not in candidate.metadata
    assert candidate.status == "pending_review" and len(store.task_memories) == 1
    assert store.customers["customer"].smoke_preference is None


def test_pending_reference_resolves_only_to_earlier_unwithdrawn_same_customer_task(store):
    """相同文字或时间不能授权引用他人、无关任务、撤回消息和当前轮之后的消息。"""
    earlier = NOW - timedelta(minutes=3)
    store.append_user_turn(inbound(message_id="unbound", when=earlier), "past")
    store.activate_task_context(conversation_id="chat", customer_id="customer", trace_id="setup",
        activity_at=earlier, started_at=earlier, reason="record", force_new=True, archive_previous=False)
    store.append_user_turn(inbound(message_id="my-previous", when=earlier), "past")
    store.append_user_turn(inbound(sender="another", message_id="theirs", when=earlier), "past")
    store.append_user_turn(inbound(message_id="withdrawn", when=earlier, context_invalidated=True), "past")
    store.append_user_turn(inbound("还是照刚才说的哈", message_id="source"), "current")
    store.append_user_turn(inbound(message_id="future", when=NOW + timedelta(minutes=1)), "future")
    for invalid_id in (None, "theirs", "unbound", "withdrawn", "future", "unknown"):
        kwargs = {"evidence_message_id": invalid_id} if invalid_id is not None else {}
        assert_rejected(write(store, pending=[intent(**kwargs)]))
    result = write(store, pending=[intent(evidence_message_id="my-previous")])
    assert result.allowed and result.result["pending_long_term_memories"][0]["metadata"]["source_message_id"] == "my-previous"


def test_pending_evidence_uses_true_aggregation_fragment_not_reconstructed_message(store):
    """原样来自实际同一片段时才持久化，来源指向微信原消息。"""
    deadline = NOW + timedelta(seconds=30)
    store.upsert_pending_input_fragment(inbound("老板", message_id="first"),
        trace_id="part1", quiet_deadline=deadline)
    batch, _, _ = store.upsert_pending_input_fragment(inbound(QUOTE, message_id="second"),
        trace_id="part2", quiet_deadline=deadline)
    combined = aggregate_pending_input_batch(batch, quiet_period_elapsed=True, trigger="test")
    store.append_user_turn(combined, "current")
    assert_rejected(write(store, pending=[intent(combined.text)], source_id=combined.message_id))
    result = write(store, pending=[intent()], source_id=combined.message_id)
    assert result.allowed and result.result["pending_long_term_memories"][0]["metadata"]["source_message_id"] == "second"


def test_task_only_evidence_semantics_stay_unchanged_without_source_turn(store):
    """当前任务可以保留结构化解释，长期候选没有原话不得写入。"""
    assert_rejected(write(store, pending=[intent()]))
    result = write(store, pending=[], task=intent(PARAPHRASE, scope="current_task"))
    assert result.allowed and result.result["task_memories"][0]["evidence"] == PARAPHRASE
    assert not store.pending_memory_candidates
