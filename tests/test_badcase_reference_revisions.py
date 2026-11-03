"""测试引用重构只追加元数据；摘要、旧引用和新保障不匹配时必须阻断覆盖门禁。"""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference_fixture(tmp_path, monkeypatch):
    """临时源资产与真实可解析函数只验证引用迁移，不调用模型或修改仓库历史。"""
    spec = importlib.util.spec_from_file_location("badcase_reference_checker", ROOT / "scripts/check_badcase_regression_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "eval/badcases/badcases.jsonl"
    source.parent.mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_flow.py").write_text("def test_direct_reply_review():\n    pass\n\ndef test_stable_fact_boundary():\n    pass\n")
    live = tmp_path / "live.py"
    live.write_text("")
    previous = [{"type": "pytest", "id": "tests/test_flow.py::test_removed_generation"},
                {"type": "pytest", "id": "tests/test_flow.py::test_stable_fact_boundary"}]
    original = {"id": "badcase-source", "kind": "badcase", "triage_status": "fixed",
                "expected": {"fact_boundary": "不得捏造公开事实"}, "regression_refs": previous}
    source.write_text(json.dumps(original, ensure_ascii=False) + "\n")
    new = {"type": "pytest", "id": "tests/test_flow.py::test_direct_reply_review"}
    revision = {"schema_version": 1, "kind": "badcase_regression_reference_revision", "id": "reference-refactor-1",
        "source": {"kind": "code_refactor", "reason": "主回复直接生成后仍独立审查，引用因测试函数重命名迁移。"},
        "source_path": "eval/badcases/badcases.jsonl", "source_record_id": original["id"],
        "source_record_sha256": module.source_record_digest(original), "previous_regression_refs": deepcopy(previous),
        "replacements": [{"old_ref": previous[0], "new_refs": [new], "coverage_intent": "新主回复仍受审查；保留原事实边界回归。"}],
        "current_regression_refs": [new, previous[1]]}
    asset = source.with_name("regression_reference_revisions.jsonl")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "BADCASE_PATHS", [source])
    monkeypatch.setattr(module, "REFERENCE_REVISION_PATH", asset)
    monkeypatch.setattr(module, "DATASET_PATHS", {})
    monkeypatch.setattr(module, "LIVE_EVAL_SCRIPT", live)
    return module, source, asset, original, revision


def write_revision(asset, revision):
    """故障只写临时修订元数据，不修正被测原始记录来迁就摘要。"""
    asset.write_text(json.dumps(revision, ensure_ascii=False) + "\n")


def test_reference_revision_preserves_original_bytes_and_every_unreplaced_obligation(reference_fixture):
    module, source, asset, original, revision = reference_fixture
    before = source.read_bytes()
    write_revision(asset, revision)
    assert module.regression_reference_revisions()[original["id"]] == revision["current_regression_refs"]
    assert module.audit_badcases() == (1, 0, [])
    assert source.read_bytes() == before
    assert json.loads(source.read_text()) == original


@pytest.mark.parametrize("fault", ["digest", "source_record", "source_path", "original_changed", "old_refs", "old_ref",
    "new_function", "new_type", "drop_stable_ref", "extra_current_ref", "coverage_intent", "provenance", "schema", "empty_replacements", "duplicate_new_ref"])
def test_changed_source_hash_or_reference_contract_blocks_coverage(reference_fixture, fault):
    module, source, asset, original, revision = reference_fixture
    if fault == "digest":
        revision["source_record_sha256"] = "0" * 64
    elif fault == "source_record":
        revision["source_record_id"] = "another-record"
    elif fault == "source_path":
        revision["source_path"] = "eval/badcases/another.jsonl"
    elif fault == "original_changed":
        original["expected"]["fact_boundary"] = "放宽事实边界"
        source.write_text(json.dumps(original, ensure_ascii=False) + "\n")
    elif fault == "old_refs":
        revision["previous_regression_refs"] = revision["previous_regression_refs"][::-1]
    elif fault == "old_ref":
        revision["replacements"][0]["old_ref"] = {"type": "pytest", "id": "tests/test_flow.py::not_the_old_ref"}
    elif fault == "new_function":
        revision["replacements"][0]["new_refs"][0]["id"] = "tests/test_flow.py::not_defined"
    elif fault == "new_type":
        revision["replacements"][0]["new_refs"][0]["type"] = "self_reported_pass"
    elif fault == "drop_stable_ref":
        revision["current_regression_refs"] = revision["current_regression_refs"][:1]
    elif fault == "extra_current_ref":
        revision["current_regression_refs"].append({"type": "pytest", "id": "tests/test_flow.py::unlisted"})
    elif fault == "coverage_intent":
        revision["replacements"][0].pop("coverage_intent")
    elif fault == "provenance":
        revision["source"]["kind"] = "direct_user_instruction"
    elif fault == "schema":
        revision["schema_version"] = True
    elif fault == "empty_replacements":
        revision["replacements"] = []
    elif fault == "duplicate_new_ref":
        revision["replacements"][0]["new_refs"] *= 2
    write_revision(asset, revision)
    with pytest.raises(ValueError):
        module.regression_reference_revisions()
    assert module.audit_badcases()[2]


@pytest.mark.parametrize("fault", ["revision_id", "target_id", "original_id", "replacement"])
def test_duplicate_revision_sources_or_replacements_are_rejected(reference_fixture, fault):
    module, source, asset, original, revision = reference_fixture
    write_revision(asset, revision)
    if fault in {"revision_id", "target_id"}:
        duplicate = deepcopy(revision)
        if fault == "target_id":
            duplicate["id"] = "a-second-id-cannot-revise-the-same-source"
        with asset.open("a") as output:
            output.write(json.dumps(duplicate) + "\n")
    elif fault == "original_id":
        with source.open("a") as output:
            output.write(json.dumps(original) + "\n")
    else:
        revision["replacements"] *= 2
        write_revision(asset, revision)
    with pytest.raises(ValueError):
        module.regression_reference_revisions()


def test_missing_revision_cannot_hide_removed_original_test(reference_fixture):
    module, _, asset, _, _ = reference_fixture
    asset.write_text("")
    assert any("pytest function not found" in error for error in module.audit_badcases()[2])


def test_reference_revision_cannot_close_an_open_badcase(reference_fixture):
    module, source, asset, original, revision = reference_fixture
    original["triage_status"] = "new"
    source.write_text(json.dumps(original) + "\n")
    revision["source_record_sha256"] = module.source_record_digest(original)
    write_revision(asset, revision)
    fixed, opened, errors = module.audit_badcases()
    assert fixed == 0 and opened == 1 and any("badcase is not closed" in error for error in errors)
