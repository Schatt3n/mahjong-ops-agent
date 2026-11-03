"""查询口径修订保留旧证据，并只改变可选征询与回复状态一致性验收。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_product_contracts as policy


def test_active_policy_preserves_original_fixture_and_every_other_hard_obligation():
    raw = policy.ACTIVE_REVISION_FIXTURE_PATH.read_bytes()
    original = json.loads(raw)
    before = deepcopy(original)
    revision = json.loads(policy.ACTIVE_REVISION_POLICY_PATH.read_text())
    assert hashlib.sha256(raw).hexdigest() == revision["source_fixture_sha256"]
    current = policy.apply_active_revision_policy(original)
    assert original == before
    assert revision["source"]["not_original_human_transcript"] is True
    assert revision["source"]["user_quotes"][0] == "两种都允许，状态与实际回复一致（推荐）"
    assert current["messages"] == original["messages"]
    assert current["fixture"] == original["fixture"]
    assert current["expected"]["latest_requirement"] == original["expected"]["latest_requirement"]
    assert current["expected"]["forbidden_meaning"] == original["expected"]["forbidden_meaning"]
    old_checks = set(original["expected"]["hard_checks"])
    new_checks = set(current["expected"]["hard_checks"])
    assert old_checks - new_checks == {"latest_waits_for_customer_choice"}
    assert new_checks - old_checks == {"latest_reply_and_status_are_consistent"}
    assert current["expected"]["required_meaning"][-1]["id"] == "reply_state_coherence"
    assert "并征询是否接受" in original["expected"]["required_meaning"][0]["description"]
    assert current["product_contract_revision_ids"] == [revision["revision_id"]]


@pytest.mark.parametrize("fault", ["input", "fixture", "old_expectation", "json_type", "already_applied"])
def test_changed_or_already_revised_fixture_is_rejected(fault):
    original = json.loads(policy.ACTIVE_REVISION_FIXTURE_PATH.read_bytes())
    if fault == "input":
        original["messages"][1]["text"] = "另一个问题"
    elif fault == "fixture":
        original["fixture"]["existing_game"]["stake"] = "1"
    elif fault == "old_expectation":
        original["expected"]["forbidden_meaning"] = []
    elif fault == "json_type":
        original["schema_version"] = True
    else:
        original = policy.apply_active_revision_policy(original)
    with pytest.raises(ValueError, match="drifted"):
        policy.apply_active_revision_policy(original)


def test_policy_asset_edit_requires_explicit_review(tmp_path, monkeypatch):
    original = json.loads(policy.ACTIVE_REVISION_FIXTURE_PATH.read_bytes())
    revision = json.loads(policy.ACTIVE_REVISION_POLICY_PATH.read_text())
    revision["current_expectation"]["forbidden_meaning"] = []
    changed = tmp_path / "changed.jsonl"
    changed.write_text(json.dumps(revision, ensure_ascii=False) + "\n")
    monkeypatch.setattr(policy, "ACTIVE_REVISION_POLICY_PATH", changed)
    with pytest.raises(ValueError, match="without review"):
        policy.apply_active_revision_policy(original)
