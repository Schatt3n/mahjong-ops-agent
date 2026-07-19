from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "run_concurrency_eval.py"
    spec = importlib.util.spec_from_file_location("run_concurrency_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deterministic_concurrency_suite_preserves_business_invariants(tmp_path) -> None:
    module = load_module()

    results = module.run_deterministic_suite(tmp_path, operations=12, workers=6)

    assert len(results) == 8
    assert all(result.passed for result in results), [result.to_dict() for result in results if not result.passed]
    by_name = {result.name: result for result in results}
    assert by_name["duplicate_message_idempotency"].checks[1]["actual"] == 1
    assert by_name["parallel_conversation_isolation"].metrics["model_max_concurrency"] >= 2
    assert by_name["last_seat_race"].checks[0]["actual"] == 1
    shared_race = by_name["shared_participant_first_ready_wins_race"]
    assert shared_race.checks[1]["actual"] == 1
    assert shared_race.checks[2]["actual"] == 1
    assert shared_race.checks[4]["actual"] == 11
    evidence = shared_race.metrics["evidence"]
    assert evidence["test_kind"] == "deterministic_sqlite_concurrency"
    assert evidence["production_entrypoint"] == "SQLiteAgentStore.record_candidate_reply"
    assert len(evidence["fixture"]["initial_games"]) == 12
    assert len(evidence["simulated_candidate_replies"]) == 12
    assert evidence["outcome"]["winner"]["status"] == "ready"
    assert len(evidence["outcome"]["losers"]) == 11
    assert evidence["outcome"]["released_shared_participation_count"] == 11
    assert evidence["state_transitions"]
    assert by_name["room_inventory_race"].checks[0]["actual"] == 1
    assert by_name["duplicate_invite_race"].checks[0]["actual"] == 1
