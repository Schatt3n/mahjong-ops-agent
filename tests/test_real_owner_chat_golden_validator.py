from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_real_owner_chat_golden.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_real_owner_chat_golden", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_owner_chat_golden_dataset_validator_passes_current_dataset() -> None:
    module = load_validator_module()

    records = module.read_jsonl(module.DEFAULT_DATASET_PATH)
    errors = module.validate_dataset(records)

    assert errors == []


def test_real_owner_chat_golden_validator_rejects_broken_turn_references() -> None:
    module = load_validator_module()
    transcript = {
        "kind": "real_owner_chat_golden",
        "id": "broken_transcript",
        "source": {"source_image_files": ["one.png"]},
        "messages": [
            {
                "turn": 1,
                "role": "customer",
                "text": "帮我约个6.30无烟的",
                "source_image": "one.png",
            }
        ],
        "business_facts": [{"id": "broken_fact", "evidence_turns": [2], "fact": "bad"}],
        "eval_cases": [{"id": "broken_case", "context_turns": [1, 3], "input_turn": 4}],
    }

    errors = module.validate_dataset([transcript])

    assert "broken_fact" in "\n".join(errors)
    assert "invalid context_turns [3]" in "\n".join(errors)
    assert "invalid input_turn 4" in "\n".join(errors)


def test_real_owner_chat_golden_validator_requires_supplement_contract() -> None:
    module = load_validator_module()
    transcript = {
        "kind": "real_owner_chat_golden",
        "id": "parent",
        "source": {"source_image_files": ["one.png"]},
        "messages": [{"turn": 1, "role": "customer", "text": "x", "source_image": "one.png"}],
        "business_facts": [{"id": "fact", "evidence_turns": [1], "fact": "ok"}],
        "eval_cases": [],
    }
    supplement = {
        "kind": "real_owner_chat_eval_supplement",
        "id": "supplement",
        "parent_id": "parent",
        "hidden_context": [{"after_turn": 99, "event": "bad"}],
        "customer_profile_assumptions": {"profile_facts": ["只有一条不完整画像"]},
        "business_facts": [{"id": "profile_defaults_fill_missing_slots"}],
        "eval_cases": [{"id": "initial_request_uses_profile_defaults_and_searches_pool", "hidden_context_refs": ["missing"]}],
    }

    errors = module.validate_dataset([transcript, supplement])
    text = "\n".join(errors)

    assert "missing required business facts" in text
    assert "missing required eval cases" in text
    assert "after_turn not found" in text
    assert "missing profile assumption" in text
    assert "hidden_context_ref not found" in text
