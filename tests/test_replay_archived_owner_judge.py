"""离线验证独立诊断不能覆盖原运行、隐瞒业务失败或改判有效结论。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mahjong_agent_runtime import AgentLLMConfig
from scripts import replay_archived_owner_judge as replay
from run_concurrency_eval import completed_action_json
from tests.semantic_http_stub import semantic_http_completion


def _verdict(ids, verdict="clear"):
    return json.dumps({"reason": "离线受控回执", "criteria": [
        {"id": criterion, "verdict": verdict, "reason": "固定测试。"} for criterion in ids
    ]}, ensure_ascii=False)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    scene = next(item for item in replay.owner.live_eval_scenarios() if item.scenario_id == "configured_room_inventory_query")
    config = AgentLLMConfig(api_key="unit-only", provider="deepseek", model="test-model",
                            base_url="https://example.invalid", retry_attempts=1)
    calls = []

    class Judge:
        def __init__(self, *, config):
            self.config = config

        def complete(self, messages, *, trace_id, timeout_seconds):
            calls.append((messages, trace_id))
            return semantic_http_completion("{", config=self.config, trace_id=trace_id)

    class Main:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return next(outputs)

    base = json.loads(completed_action_json(""))
    base.update(objective_status="needs_tool",
                tool_calls=[{"name": "check_room_availability", "arguments": {}, "reason": "根据来话读取真实房态。"}],
                stop_reason={"can_stop": False, "why": "还需查询房间", "pending_work": ["查看房态"],
                             "depends_on_tool_results": True})
    outputs = iter([json.dumps(base, ensure_ascii=False), completed_action_json("没满，现在8间都空着。")])
    main = Main()
    main.config = config
    monkeypatch.setattr(replay.judge, "OpenAICompatibleAgentClient", Judge)
    args = SimpleNamespace(db_path=tmp_path / "owner.sqlite3", max_steps=5, max_calls_per_turn=5,
                           max_tokens_per_call=64000, timeout_seconds=20, skip_review=True, skip_text_generation=True)
    case = replay.owner.run_scenario(main, args, scene, iteration=2)
    assert case["status"] == "failed" and len(calls) == 2
    source = tmp_path / "run.json"
    source.write_text(json.dumps({"live": {"reports": [case]}}, ensure_ascii=False), encoding="utf-8")
    return source, case, config, calls


def test_new_diagnostic_only_judges_frozen_source_and_preserves_bytes(archive, tmp_path, monkeypatch):
    source, case, config, calls = archive
    raw_path, db_path = Path(case["raw_trace_path"]), Path(case["db_path"])
    archived = {path: path.read_bytes() for path in (source, raw_path, db_path)}

    class Judge:
        def __init__(self, *, config):
            self.config = config

        def complete(self, messages, *, trace_id, timeout_seconds):
            calls.append((messages, trace_id))
            request = json.loads(messages[-1]["content"])
            original = case["semantic_assessment"]
            assert request["evidence"] == original["judge_evidence"]
            assert request["expectation"] == original["judge_expectation"]
            ids = [entry["id"] for entry in request["expectation"]["checks"]]
            return semantic_http_completion(_verdict(ids), config=self.config, trace_id=trace_id)

    monkeypatch.setattr(replay.judge, "OpenAICompatibleAgentClient", Judge)
    monkeypatch.setattr(replay.OpenAICompatibleAgentClient, "from_env", lambda: SimpleNamespace(config=config))
    monkeypatch.setattr(replay.owner, "run_scenario", lambda *a, **k: pytest.fail("replayed main Agent"))
    output = tmp_path / "diagnostic.json"
    argv = ["--source-report", str(source), "--scenario", case["scenario_id"], "--iteration", "2"]
    assert replay.main([*argv, "--verify-only"]) == 0
    assert len(calls) == 2
    assert replay.main([*argv, "--output", str(output)]) == 0
    diagnostic = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == 3 and calls[-1][1].startswith(case["trace_id"] + "_semantic_diagnostic_")
    assert diagnostic["status"] == "passed" and diagnostic["original_case_status"] == "failed"
    assert diagnostic["main_agent_executed"] is False and diagnostic["full_gate"] is False
    assert {path: path.read_bytes() for path in archived} == archived
    with pytest.raises(SystemExit) as rejected:
        replay.main([*argv, "--output", str(output)])
    assert rejected.value.code == 2 and len(calls) == 3


@pytest.mark.parametrize("tamper", ["reply", "hard_check", "valid_negative", "frozen_evidence"])
def test_archived_case_requires_matching_input_proof_and_no_existing_valid_verdict(archive, tamper):
    source, case, _, _ = archive
    assessment = case["semantic_assessment"]
    if tamper == "reply":
        case["final_reply"] = "今天满了。"
    elif tamper == "hard_check":
        case["checks"] = [check for check in case["checks"] if check["name"] != "should_call_check_room_availability"]
    elif tamper == "valid_negative":
        ids = [entry["id"] for entry in assessment["judge_expectation"]["checks"]]
        assessment["attempts"][0]["raw_response"] = _verdict(ids, verdict="violated")
    else:
        assessment["evidence"]["reply"] = "今天满了。"
        assessment["judge_evidence"] = replay.judge.project_semantic_judge_evidence(assessment["evidence"])
    source.write_text(json.dumps({"reports": [case]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        replay.load_archived_case(source, case["scenario_id"], 2)


def test_legacy_canonical_ids_are_valid_verdicts_even_after_wire_change():
    ids = ["source.required", "source.forbidden"]
    assert replay._valid_archived_verdict(_verdict(ids, verdict="violated"), ids, legacy=True)
    assert not replay._valid_archived_verdict(_verdict(["source.required", "other.rule"]), ids, legacy=True)
