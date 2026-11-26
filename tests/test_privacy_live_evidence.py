"""隐私运行器的失败原因和原始轨迹必须能回溯，复跑不能覆盖旧证据。"""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import AgentAction, AgentLLMConfig, OpenAICompatibleAgentClient


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_privacy_isolation_live_eval.py"
    spec = importlib.util.spec_from_file_location("privacy_evidence_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provider_failure_keeps_complete_original_trace_across_replays(module, tmp_path):
    args = SimpleNamespace(db_path=tmp_path / "privacy.sqlite3", max_tokens_per_call=64000,
        max_calls_per_turn=8, max_steps=4, timeout_seconds=1)
    probe = module.load_privacy_probes(module.DEFAULT_CASES_PATH)[0]

    class FailingClient:
        def __init__(self, marker):
            self.marker = marker

        def complete(self, *args, **kwargs):
            raise RuntimeError(self.marker)

    # 真实入口分别配置主模型与审查模型；本例只验证主调用失败的原始证据，
    # 显式注入审查替身，禁止从本机配置创建真实客户端或绕过独立审查配置。
    args.review_client = FailingClient("unexpected_review_call")
    first = module.run_case(FailingClient("provider_failure_first"), args, probe)
    original = Path(first["raw_trace_path"]).read_bytes()
    second = module.run_case(FailingClient("provider_failure_second"), args, probe)
    assert first["status"] == second["status"] == "failed"
    assert first["raw_trace_path"] != second["raw_trace_path"]
    assert Path(first["raw_trace_path"]).read_bytes() == original
    for report, marker in ((first, "provider_failure_first"), (second, "provider_failure_second")):
        raw = json.loads(Path(report["raw_trace_path"]).read_text())
        errors = [event for event in raw["events"] if event["step"] == "llm_error"]
        assert errors and marker in json.dumps(errors)
        assert marker in json.dumps(report["decision_trace"])
        assert report["checks"]["no_model_or_contract_error"] is False
        assert "unexpected_review_call" not in json.dumps(raw)


class PrivacyClient:
    """在真实运行器和审查入口后注入确定性回复，其余数据和权限路径保持真实。"""

    def __init__(self):
        self.main_calls = 0
        self.failing_transport = OpenAICompatibleAgentClient(config=AgentLLMConfig(api_key="test-only", model="unit",
            base_url="https://example.invalid", retry_attempts=1), urlopen=self.timeout)

    @staticmethod
    def timeout(*args, **kwargs):
        raise TimeoutError("test upstream response wait")

    def complete(self, messages, **kwargs):
        request = json.loads(messages[-1]["content"])
        if "review_items" in request:
            return json.dumps({"approved": True, "needs_human": False, "reasoning_summary": "审查替身",
                "violations": [], "item_reviews": [{"item_id": item["item_id"], "approved": True,
                    "suggested_safe_text": item["text"], "reasoning_summary": "验证对外证据", "violations": []}
                    for item in request["review_items"]]})
        if "source_text" in request:
            return json.dumps({"text": request["source_text"]})
        self.main_calls += 1
        if self.main_calls == 1:
            return self.failing_transport.complete(messages, **kwargs)
        action = AgentAction(goal="保护客户隐私", objective_status="completed", reasoning_summary="不透露其他客户信息",
                             reply_to_user="别人的私事我不方便说。",
                             stop_reason={"can_stop": True, "why": "已答复", "pending_work": [], "depends_on_tool_results": False})
        return json.dumps(action.to_dict(), ensure_ascii=False)


@pytest.mark.parametrize("leak_observed", [False, True])
def test_privacy_transport_retry_uses_fresh_db_and_never_hides_observed_leak(module, monkeypatch, tmp_path, leak_observed):
    import eval_retry

    probe = module.load_privacy_probes(module.DEFAULT_CASES_PATH)[0]
    client = PrivacyClient()
    args = SimpleNamespace(db_path=tmp_path / "privacy.sqlite3", report_path=tmp_path / "report.json",
                           max_tokens_per_call=64000, max_calls_per_turn=8, max_steps=4, timeout_seconds=2,
                           review_client=client)
    scope = tmp_path / "scope"
    monkeypatch.setenv(eval_retry.SCOPE_ENV, str(scope))
    monkeypatch.setenv(eval_retry.WINDOW_ENV, "0")
    monkeypatch.setattr(eval_retry, "sleep", lambda _seconds: None)
    if leak_observed:
        monkeypatch.setattr(module, "extract_new_customer_visible_texts", lambda *_: [f"B_PRIVATE_CANARY_{probe.case_id}_91D7"])

    result = module.run_case_attempt(client, args, probe)
    attempts = [json.loads(path.read_text()) for path in scope.rglob("eval_result.json")]
    assert len(attempts) == client.main_calls == (1 if leak_observed else 2)
    if leak_observed:
        assert result["status"] == "failed" and "eval_inconclusive" not in result
        assert result["checks"]["no_customer_visible_leak"] is False
    else:
        failed = next(item for item in attempts if item["status"] == "blocked")
        assert failed["eval_inconclusive"]["category"] == "unattributed_timeout"
        assert result["status"] == "passed" and result["run_id"] == failed["run_id"] == tmp_path.name
        assert Path(failed["db_path"]).is_file() and Path(result["db_path"]).is_file()
        assert failed["db_path"] != result["db_path"] and failed["raw_trace_path"] != result["raw_trace_path"]
        original = json.loads(Path(failed["raw_trace_path"]).read_text())
        assert any(event["step"] == "llm_error" for event in original["events"])
