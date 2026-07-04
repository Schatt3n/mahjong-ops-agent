from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_badcase_regression_coverage.py"


def load_checker_module():
    spec = importlib.util.spec_from_file_location("check_badcase_regression_coverage", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_badcases_have_verified_regression_refs() -> None:
    module = load_checker_module()

    fixed, open_count, errors = module.audit_badcases()

    assert fixed > 0
    assert open_count == 0
    assert errors == []


def test_live_eval_regression_refs_are_verified() -> None:
    module = load_checker_module()

    ids_by_type = module.dataset_ids()

    assert "live_eval" in ids_by_type
    assert "profile_default_matched_game" in ids_by_type["live_eval"]
    assert "accept_existing_offer_marks_game_ready" in ids_by_type["live_eval"]
    assert module.validate_refs(
        {
            "regression_refs": [
                {"type": "live_eval", "id": "profile_default_matched_game"},
            ]
        },
        ids_by_type,
    ) == []


def test_open_badcase_fails_audit(tmp_path: Path, monkeypatch) -> None:
    module = load_checker_module()
    badcase_path = tmp_path / "badcases.jsonl"
    badcase_path.write_text(
        '{"id":"badcase_open_001","triage_status":"new","kind":"badcase"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "BADCASE_PATHS", [badcase_path])

    fixed, open_count, errors = module.audit_badcases()

    assert fixed == 0
    assert open_count == 1
    assert errors == [
        "badcases.jsonl:badcase_open_001: badcase is not closed; triage_status/status=new"
    ]
