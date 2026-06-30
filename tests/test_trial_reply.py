from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.models import CandidateRecommendation, GameRequest
from mahjong_agent.trial_reply import TrialReplyDraftAdapter, TrialReplyDraftCallbacks, TrialReplyDraftInput


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 1, 16, 0, tzinfo=TZ)


def test_trial_reply_draft_adapter_generates_reply_and_updates_memory() -> None:
    suggested_calls: list[dict] = []
    memory_calls: list[dict] = []
    game = GameRequest(
        id="game_1",
        organizer_id="zhang",
        organizer_name="张哥",
        channel_id="boss_trial",
    )
    recommendations = [CandidateRecommendation(customer_id="ran", display_name="冉姐", score=100)]
    outbox = [{"id": "out_1", "customer_id": "ran"}]
    pool_matches = [{"game_id": "pool_1"}]
    tool_results = {"send_message": {"called": True, "outbox": outbox}}

    def suggested_reply(**kwargs):
        suggested_calls.append(kwargs)
        return {"text": "好的，我帮你问问。", "source": "llm", "status": "待审批"}

    def update_sender_memory_after_reply(**kwargs) -> None:
        memory_calls.append(kwargs)

    adapter = TrialReplyDraftAdapter(
        TrialReplyDraftCallbacks(
            suggested_reply=suggested_reply,
            update_sender_memory_after_reply=update_sender_memory_after_reply,
        )
    )

    result = adapter.draft(
        TrialReplyDraftInput(
            conversation_id="boss_trial",
            sender_id="zhang",
            sender_name="张哥",
            source_text="帮我组一桌",
            effective_text="帮我组一桌",
            trace_id="trace_reply",
            game=game,
            workflow_followup_context={"previous_system_suggested_reply": "要组一个吗？"},
            missing_fields=[],
            decision_reply="收到",
            parsed={"intent_action": "find_players"},
            recommendations=recommendations,
            outbox=outbox,
            pool_matches=pool_matches,
            tool_results=tool_results,
            now=NOW,
        )
    )

    assert result.suggested_reply == {"text": "好的，我帮你问问。", "source": "llm", "status": "待审批"}
    assert suggested_calls[0]["tool_results"] is tool_results
    assert suggested_calls[0]["pool_matches"] is pool_matches
    assert suggested_calls[0]["recommendations"] is recommendations
    assert suggested_calls[0]["outbox"] is outbox
    assert suggested_calls[0]["game"] is game
    assert memory_calls == [
        {
            "conversation_id": "boss_trial",
            "sender_id": "zhang",
            "trace_id": "trace_reply",
            "suggested_reply": result.suggested_reply,
            "parsed": {"intent_action": "find_players"},
            "tool_results": tool_results,
            "pool_matches": pool_matches,
            "now": NOW,
        }
    ]
