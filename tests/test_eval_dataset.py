from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_scenario_eval.py"
BOSS_TRIAL_SCRIPT = ROOT / "scripts" / "run_boss_trial_app.py"
BOSS_TRIAL_GOLDEN_PATH = ROOT / "eval" / "boss_trial_golden.jsonl"
LLM_ENV_KEYS = [
    "MAHJONG_LLM_API_KEY",
    "MAHJONG_LLM_PROVIDER",
    "MAHJONG_LLM_MODEL",
    "MAHJONG_LLM_BASE_URL",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
]


def load_eval_module():
    spec = importlib.util.spec_from_file_location("run_scenario_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_boss_trial_module():
    spec = importlib.util.spec_from_file_location("run_boss_trial_app_eval", BOSS_TRIAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BOSS_TRIAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def without_llm_env():
    saved = {key: os.environ.get(key) for key in LLM_ENV_KEYS}
    for key in LLM_ENV_KEYS:
        os.environ.pop(key, None)
    return saved


def restore_env(saved) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_golden_dataset_loads_and_has_unique_ids() -> None:
    module = load_eval_module()

    scenarios = module.load_scenarios(module.GOLDEN_PATH)
    ids = [scenario.id for scenario in scenarios]

    assert len(scenarios) == 11
    assert len(ids) == len(set(ids))
    assert module.count_checks(scenarios) == 11


def test_boss_trial_golden_dataset_scores_reply_style() -> None:
    records = read_jsonl(BOSS_TRIAL_GOLDEN_PATH)
    ids = [str(record.get("id")) for record in records]

    assert records
    assert len(ids) == len(set(ids))

    saved_env = without_llm_env()
    try:
        module = load_boss_trial_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            for record in records:
                store = module.TrialStore(Path(temp_dir) / f"{record['id']}.db")
                service = module.BossTrialService(store)
                for setup_input in record.get("setup_inputs", []):
                    service.analyze(dict(setup_input))
                result = service.analyze(dict(record["input"]))
                expected = record["expected"]
                suggested = result["suggested_reply"]["text"]

                if expected.get("parsed_user_intent"):
                    assert result["parsed"]["user_intent"] == expected["parsed_user_intent"], record["id"]
                if expected.get("suggested_reply_exact"):
                    assert suggested == expected["suggested_reply_exact"], record["id"]
                for fragment in expected.get("suggested_reply_contains", []):
                    assert fragment in suggested, f"{record['id']} should contain {fragment!r}: {suggested}"
                for fragment in expected.get("forbidden_in_suggested_reply", []):
                    assert fragment not in suggested, f"{record['id']} should not contain {fragment!r}: {suggested}"
                if expected.get("pool_match_count_min") is not None:
                    assert len(result.get("pool_matches") or []) >= int(expected["pool_match_count_min"]), record["id"]
                if expected.get("outbox_count") is not None:
                    assert len(result.get("outbox") or []) == int(expected["outbox_count"]), record["id"]
                if expected.get("state_game_count") is not None:
                    assert len((result.get("state") or {}).get("games") or []) == int(expected["state_game_count"]), record["id"]
                if "parsed_current_player_count" in expected:
                    assert result["parsed"].get("current_player_count") == expected["parsed_current_player_count"], record["id"]
                if "parsed_missing_count" in expected:
                    assert result["parsed"].get("missing_count") == expected["parsed_missing_count"], record["id"]
                if expected.get("missing_fields") is not None:
                    assert result.get("missing_fields") == expected["missing_fields"], record["id"]
                if expected.get("invite_text_contains") or expected.get("forbidden_in_invite_text"):
                    assert result["outbox"], record["id"]
                    for item in result["outbox"]:
                        invite_text = item["message_text"]
                        for fragment in expected.get("invite_text_contains", []):
                            assert fragment in invite_text, f"{record['id']} invite should contain {fragment!r}: {invite_text}"
                        for fragment in expected.get("forbidden_in_invite_text", []):
                            assert fragment not in invite_text, f"{record['id']} invite should not contain {fragment!r}: {invite_text}"
                if expected.get("candidate_gender_sequence"):
                    sequence = [item.get("gender") for item in result.get("outbox", [])[: len(expected["candidate_gender_sequence"])]]
                    assert sequence == expected["candidate_gender_sequence"], record["id"]
                if expected.get("candidate_preference_genders"):
                    preference = result["parsed"].get("candidate_composition_preference") or {}
                    assert preference.get("preferred_candidate_genders") == expected["candidate_preference_genders"], record["id"]
    finally:
        restore_env(saved_env)
