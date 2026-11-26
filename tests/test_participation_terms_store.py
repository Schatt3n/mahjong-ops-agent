"""核对 Memory 与 SQLite 不会在审核后改期时用旧桌况占席。"""

from datetime import datetime

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.domains.game_participants import ParticipationTermsChangedError, participation_table_terms
from mahjong_agent_runtime.models import InviteStatus, using_clock


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("method", ["join_game", "record_candidate_reply"])
@pytest.mark.parametrize("duration_added", [False, True])
def test_participation_transaction_rechecks_current_table_before_side_effects(tmp_path, backend, method, duration_added):
    """旧开场或空授权不会改变席位、邀约、任务和转移记录；有效桌况仍正常占席。"""
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "participation.sqlite3")
    clock = datetime.fromisoformat("2026-11-25T19:37:00+08:00")
    with using_clock(lambda: clock):
        game, _ = store.create_game(
            conversation_id="host-thread", organizer_id="host", organizer_name="发起人", known_players=[],
            requirement={"game_type": "hangzhou_mahjong", "stake": "1", "planned_start_at": "2026-11-25T20:45:00+08:00"},
            trace_id="fixture",
        )
        drafts, _ = store.create_invite_drafts(
            game_id=game.game_id,
            invitations=[{"customer_id": "guest", "display_name": "顾客", "message_text": "这桌打吗？"}],
            trace_id="fixture",
        )
        reviewed_terms = participation_table_terms(game)
        patch = {"planned_start_at": "2026-11-25T21:00:00+08:00"}
        if duration_added:
            patch["duration_hours"] = 2
        store.update_game_requirement(game_id=game.game_id,
            requirement_patch=patch,
            reason="发起人此后改期", trace_id="organizer-change")
        transition_count = len(store.transitions)

        submit = getattr(store, method)
        arguments = {"game_id": game.game_id, "customer_id": "guest", "display_name": "顾客", "seat_count": 1,
                     "trace_id": "seat-write", "schedule_full_table_notice": True}
        if method == "record_candidate_reply":
            arguments["status"] = "accepted"

        for invalid_terms in ((reviewed_terms,) if duration_added else ({}, reviewed_terms)):
            with pytest.raises(ParticipationTermsChangedError, match="未占席"):
                submit(**arguments, expected_participation_terms=invalid_terms)
            assert not any(seat.customer_id == "guest" for seat in store.require_game(game.game_id).participants)
            assert store.invite_drafts[drafts[0].draft_id].status == InviteStatus.PENDING_APPROVAL
            assert len(store.transitions) == transition_count

        current = store.require_game(game.game_id)
        assert current.planned_start_at.isoformat() == "2026-11-25T21:00:00+08:00"
        expected = {} if duration_added else participation_table_terms(current)
        updated, _ = submit(**arguments, expected_participation_terms=expected)
        assert any(seat.customer_id == "guest" and seat.status == "confirmed" for seat in updated.participants)
