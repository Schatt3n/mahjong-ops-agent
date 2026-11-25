#!/usr/bin/env python3
"""用同一份沙盒会话归档重放 OpenCode Go 评测并汇报请求规模与传输证据。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mahjong_agent_runtime.llm import llm_conversation_scope  # noqa: E402
from scripts.run_venue_simulation import _client, _model_execution, _select_local_model, load_dotenv_defaults  # noqa: E402
from tests.simulation.venue_actors import SemanticJudge  # noqa: E402
from tests.simulation.venue_judging import (  # noqa: E402
    ROLLING_REVIEW_INSTRUCTIONS, judge_prepared_dialogue, prepare_dialogue_review, review_dialogue,
)
from tests.simulation.venue_models import opencode_auxiliary  # noqa: E402


def _read_rows(path: Path) -> list[dict]:
    """读取已归档的逐行事件或聊天账本。"""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(result: dict) -> dict:
    """投影无原文、无凭证的对照结果，可直接作为仓库的排查记录。"""
    diagnostics = result["diagnostics"]
    attempts = []
    for row in diagnostics.get("attempts", []):
        usage = row.get("usage") or {}
        attempts.append({
            "http_status": row.get("http_status"), "elapsed_ms": row.get("http_elapsed_ms"),
            "timeout_seconds": row.get("timeout_seconds"), "error_type": row.get("error_type"),
            "incomplete_reason": row.get("incomplete_reason"), "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        })
    size = (result.get("projection") or {}).get("evidence_bytes")
    if size is None:
        size = len(json.dumps(result["evidence"], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return {"status": result["status"], "verdict": result["verdict"], "error": result.get("error"),
            "serialized_evidence_bytes": size, "elapsed_ms": diagnostics.get("total_elapsed_ms"),
            "reasoning_effort": (diagnostics.get("request") or {}).get("reasoning_effort"), "attempts": attempts}


def _canonical_json(value: object) -> str:
    """忽略报告持久化时数组的元组类型，保留证据顺序及所有字段。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _configured_retry_judge(report: dict, *, timeout: float) -> SemanticJudge:
    """恢复同一模型、协议和推理设置，避免将更换判定器误作原判定器恢复。"""
    original = (report.get("model_execution") or {}).get("judge") or {}
    if not original.get("provider") or not original.get("model"):
        raise ValueError("原报告缺少可核对的判定器配置")
    if original["provider"] == "opencode_go":
        config = _select_local_model("opencode-go", None, workers=1)
        client = opencode_auxiliary(config, max_tokens=2600, workers=1)
    else:
        load_dotenv_defaults(ROOT / ".env")
        client = _client("MAHJONG_VENUE_JUDGE")
    current = _model_execution(client, timeout_seconds=timeout)
    fields = ("model", "provider", "protocol", "max_output_tokens", "reasoning_effort_on_wire", "thinking_mode_on_wire")
    if changed := [field for field in fields if original.get(field) != current.get(field)]:
        raise ValueError(f"当前判定器与原报告配置不同：{', '.join(changed)}")
    return SemanticJudge(client, timeout_seconds=timeout)


def retry_unavailable(run_dir: Path, output: Path, *, judge_factory: Callable[[dict], SemanticJudge],
                      verify_only: bool = False) -> dict:
    """补判传输失败的原样本，只读归档并在调用前核对所有重建的证据。"""
    run_dir, output = run_dir.resolve(), output.resolve()
    if not verify_only and output.is_relative_to(run_dir):
        raise ValueError("补判报告必须写在原运行归档之外")
    if not verify_only and output.exists():
        raise FileExistsError(f"补判报告已存在：{output}")

    sources = {name: (run_dir / name).read_bytes() for name in ("report.json", "events.jsonl", "chat.jsonl")}
    report = json.loads(sources["report.json"])
    if (report.get("mode"), report.get("scenario")) != ("real", "rolling-roles") or report.get("status") not in {
        "passed", "failed", "inconclusive",
    }:
        raise ValueError("仅支持已结束的真实滚动营业运行")
    saved = report.get("semantic_samples")
    if not isinstance(saved, list) or any(not isinstance(sample, dict) for sample in saved):
        raise ValueError("原报告缺少判定样本")
    ids = [sample.get("case_id") for sample in saved]
    if any(not isinstance(case_id, str) or not case_id.startswith("dialogue:") or case_id == "dialogue:" for case_id in ids):
        raise ValueError("原报告含无效顾客判定编号")
    if len(ids) != len(set(ids)):
        raise ValueError("原报告含重复判定编号")

    turns = defaultdict(list)
    for line in sources["events.jsonl"].splitlines():
        if line.strip():
            row = json.loads(line)
            turns[row.get("actor_id")].append(row)
    records = [json.loads(line) for line in sources["chat.jsonl"].splitlines() if line.strip()]
    prepared = []
    untouched = []
    for case_id, sample in zip(ids, saved, strict=True):
        original = {key: sample.get(key) for key in ("case_id", "status", "verdict")}
        if original["status"] != "transport_error":
            untouched.append(original)
            continue
        customer_id = case_id.removeprefix("dialogue:")
        if not turns.get(customer_id) or not sample.get("evidence"):
            raise ValueError(f"原判定缺少可核对的证据：{case_id}")
        review = prepare_dialogue_review(customer_id=customer_id, turns=turns[customer_id], records=records,
                                        role_only=True, delivered_history=True)
        if _canonical_json(review.archive) != _canonical_json(sample["evidence"]):
            raise ValueError(f"判定证据与原报告不同：{case_id}")
        prepared.append((customer_id, original, review))

    artifact = {
        "status": "verified_only" if verify_only else "reassessing" if prepared else "no_transport_errors",
        "run": run_dir.name, "original_run_status": report["status"],
        "source_sha256": {name: sha256(content).hexdigest() for name, content in sources.items()},
        "candidate_samples": len(prepared),
        "untouched_samples": untouched,
        "retry_samples": [],
        "retry_verdicts": {},
    }
    if verify_only:
        return artifact
    judge = judge_factory(report) if prepared else None
    # 先通过独占创建保留进度。中断时已经完成的每份判决仍能独立核对。
    with output.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for index, (customer_id, original, review) in enumerate(prepared, start=1):
        with llm_conversation_scope(f"venue-judge-retry:{run_dir.name}:{customer_id}"):
            result = judge_prepared_dialogue(judge, review, trace_id=f"judge-retry:{customer_id}",
                additional_instructions=ROLLING_REVIEW_INSTRUCTIONS)
        replay = _summary(result) | {
            "trace_id": result["trace_id"], "response_source": result.get("response_source"),
            "findings": [{**finding, "evidence_ids": list(finding["evidence_ids"])} for finding in result["findings"]],
        }
        artifact["retry_samples"].append({"original": original, "retry": replay})
        verdicts = Counter(row["retry"]["verdict"] for row in artifact["retry_samples"])
        artifact["retry_verdicts"] = dict(verdicts)
        if index == len(prepared):
            artifact["status"] = "retry_incomplete" if verdicts["inconclusive"] else "reassessed"
        _persist_retry_progress(output, artifact)
    return artifact


def _persist_retry_progress(output: Path, artifact: dict) -> None:
    """完整替换独占保留的报告版本，中断不会把已写判决截成半份 JSON。"""
    with NamedTemporaryFile(dir=output.parent, mode="w", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(artifact, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> int:
    """从只读本地材料构造对照组，沿用已配置的 OpenCode Go 凭证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--customer")
    parser.add_argument("--layout", choices=("compact", "full", "both"), default="compact")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--retry-unavailable", action="store_true", help="按原归档补判真实滚动演练的传输失败样本")
    parser.add_argument("--output", type=Path, help="补判证据报告路径，默认写到运行目录旁")
    parser.add_argument("--verify-only", action="store_true", help="只核对待补判证据，不请求模型或保存补判")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 180:
        parser.error("--timeout 必须在 1 到 180 秒之间")
    if args.retry_unavailable:
        if args.customer or args.layout != "compact":
            parser.error("--retry-unavailable 使用原报告的评审名册及紧凑证据")
        if args.verify_only and args.output:
            parser.error("--verify-only 只打印核对结果，不写输出文件")
        run_dir = args.run_dir.resolve()
        output = args.output or run_dir.with_name(f"{run_dir.name}.judge-retry.json")
        try:
            result = retry_unavailable(run_dir, output, verify_only=args.verify_only,
                                       judge_factory=lambda report: _configured_retry_judge(report, timeout=args.timeout))
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            parser.error(str(exc))
        status = {"status": result["status"], "candidate_samples": result["candidate_samples"],
                  "retry_verdicts": result["retry_verdicts"]}
        if not args.verify_only:
            status["output"] = str(output)
        print(json.dumps(status, ensure_ascii=False))
        return int(result["status"] == "retry_incomplete")
    if args.output or args.verify_only or not args.customer:
        parser.error("手动判定必须提供 --customer，--output 和 --verify-only 仅用于 --retry-unavailable")
    turns = [row for row in _read_rows(args.run_dir / "events.jsonl") if row.get("actor_id") == args.customer]
    if not turns:
        parser.error("没有该顾客的已归档对话")
    records = _read_rows(args.run_dir / "chat.jsonl")
    report_path = args.run_dir / "report.json"
    outcome = None
    historical = False
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        historical = report.get("scenario") in {"historical-group", "historical-private"}
        outcome = next((row for row in report.get("acceptance", {}).get("outcomes", ())
                        if row.get("customer_id") == args.customer), None)
    config = _select_local_model("opencode-go", None, workers=1)
    judge = SemanticJudge(opencode_auxiliary(config, max_tokens=2600, workers=1), timeout_seconds=args.timeout)
    with llm_conversation_scope(f"venue-judge-replay:{args.run_dir.name}:{args.customer}"):
        if args.layout in {"full", "both"}:
            result = judge.judge(f"dialogue:{args.customer}", turns, trace_id=f"judge-full:{args.customer}",
                checks=("顾客发言符合个人说话方式、已知信息及意向",
                        "续轮真实回应前文，Agent 没有重复追问已知条件或误报结果")).to_dict()
            print(json.dumps({"layout": "full", **_summary(result)}, ensure_ascii=False), flush=True)
        if args.layout in {"compact", "both"}:
            result = review_dialogue(judge, customer_id=args.customer, turns=turns, records=records, outcome=outcome,
                role_only=historical, delivered_history=historical)
            print(json.dumps({"layout": "compact", **_summary(result)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
