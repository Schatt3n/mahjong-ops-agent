"""汇总聊天模拟结果，并提供受白名单限制的只读观测与控制数据。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = ROOT / "runtime_data" / "periodic_chat_simulation"
DEFAULT_ENV_FILE = ROOT / ".env.simulation.local"
STATIC_PAGE_PATH = ROOT / "scripts" / "static" / "simulation_observability.html"
_CONTROL_LOCK = threading.Lock()


def simulation_dashboard_html() -> str:
    return STATIC_PAGE_PATH.read_text(encoding="utf-8")


def simulation_observability_payload(
    *,
    runtime_dir: Path | None = None,
    selected_run_id: str | None = None,
    max_runs: int = 30,
) -> dict[str, Any]:
    """整合调度器状态、已完成的报告和当前实时流。"""

    base = (runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve()
    state = _load_json(base / "state.json")
    control = _load_json(base / "control.json", {"enabled": True})
    run_summaries = _list_run_summaries(base, max_runs=max_runs)
    selected_id = _safe_run_id(selected_run_id)
    if not selected_id:
        selected_id = str(state.get("current_run_id") or state.get("last_run_id") or "")
    if not selected_id and run_summaries:
        selected_id = str(run_summaries[0]["run_id"])

    selected = _load_run(base, selected_id) if selected_id else _empty_run()
    generator_env = _read_env_metadata(DEFAULT_ENV_FILE)
    pid = int(state.get("pid") or 0)
    scheduler = {
        **state,
        "enabled": bool(control.get("enabled", True)),
        "run_requested": bool(control.get("run_request_id"))
        and control.get("run_request_id") != state.get("handled_run_request_id"),
        "process_alive": _process_alive(pid),
        "runtime_dir": str(base),
        "event_log_path": str(base / "events.jsonl"),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scheduler": scheduler,
        "generator": {
            "configured": bool(generator_env.get("MAHJONG_SIM_GENERATOR_API_KEY")),
            "mode": generator_env.get("MAHJONG_PERIODIC_SIM_MESSAGE_MODE", "rule"),
            "model": generator_env.get("MAHJONG_SIM_GENERATOR_MODEL", "glm-4.7-flash"),
            "base_url": generator_env.get(
                "MAHJONG_SIM_GENERATOR_BASE_URL",
                "https://open.bigmodel.cn/api/paas/v4",
            ),
            "api_key_exposed": False,
        },
        "runs": run_summaries,
        "selected_run": selected,
        "scheduler_events": _tail_jsonl(base / "events.jsonl", 80),
    }


def update_simulation_control(
    action: str,
    *,
    runtime_dir: Path | None = None,
) -> dict[str, Any]:
    """更新simulationcontrol，同时维护相关状态不变量。

"""

    normalized = str(action or "").strip().lower()
    if normalized not in {"pause", "resume", "run_now", "start"}:
        raise ValueError("unsupported simulation control action")
    base = (runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    if normalized == "start":
        simulator_env = _read_env_metadata(DEFAULT_ENV_FILE)
        llm_mode = simulator_env.get("MAHJONG_PERIODIC_SIM_LLM_MODE", "real")
        message_mode = simulator_env.get("MAHJONG_PERIODIC_SIM_MESSAGE_MODE", "model" if llm_mode == "real" else "rule")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_periodic_chat_simulator.py"),
                "start",
                "--runtime-dir",
                str(base),
                "--env-file",
                str(DEFAULT_ENV_FILE),
                "--message-mode",
                message_mode,
                "--mode",
                llm_mode,
                "--workers",
                "1",
                "--rate",
                "1",
                "--request-timeout",
                "180",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "action": normalized,
            "return_code": completed.returncode,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }

    with _CONTROL_LOCK:
        path = base / "control.json"
        control = _load_json(path, {"enabled": True})
        if normalized == "pause":
            control["enabled"] = False
        elif normalized == "resume":
            control["enabled"] = True
        else:
            control["enabled"] = True
            control["run_request_id"] = uuid.uuid4().hex
        control["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(path, control)
    return {"ok": True, "action": normalized, "control": control}


def _list_run_summaries(runtime_dir: Path, *, max_runs: int) -> list[dict[str, Any]]:
    runs_dir = runtime_dir / "runs"
    if not runs_dir.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for run_dir in sorted(
        (item for item in runs_dir.iterdir() if item.is_dir()),
        key=lambda item: item.name,
        reverse=True,
    )[: max(1, min(int(max_runs), 100))]:
        report = _load_json(run_dir / "report.json")
        live_turns = _tail_jsonl(run_dir / "live_events.jsonl", 10000)
        transcript = report.get("transcript") if isinstance(report.get("transcript"), list) else live_turns
        summaries.append(
            {
                "run_id": run_dir.name,
                "status": str(report.get("status") or ("running" if live_turns else "starting")),
                "quality_status": str(report.get("quality_status") or "unknown"),
                "started_at": report.get("started_at"),
                "finished_at": report.get("finished_at"),
                "total_messages": int(report.get("total_messages") or len(transcript or [])),
                "group_messages": int(report.get("group_messages") or _count_channel(transcript, "group")),
                "private_messages": int(report.get("private_messages") or _count_channel(transcript, "private")),
                "agent_response_latency_ms": report.get("agent_response_latency_ms") or {},
                "llm_mode": report.get("llm_mode"),
                "message_generation_mode": report.get("message_generation_mode") or _generation_mode(transcript),
                "message_generation_model": report.get("message_generation_model") or _generation_model(transcript),
            }
        )
    return summaries


def _load_run(runtime_dir: Path, run_id: str) -> dict[str, Any]:
    safe_id = _safe_run_id(run_id)
    if not safe_id:
        return _empty_run()
    run_dir = runtime_dir / "runs" / safe_id
    report = _load_json(run_dir / "report.json")
    live_events = _tail_jsonl(run_dir / "live_events.jsonl", 10000)
    transcript = report.get("transcript") if isinstance(report.get("transcript"), list) else live_events
    transcript = [item for item in transcript or [] if isinstance(item, dict)]
    conversations = _group_conversations(transcript)
    reviews = _tail_jsonl(run_dir / "reviews.jsonl", 10000)
    latest_reviews = {str(item.get("sequence")): item for item in reviews}
    return {
        "run_id": safe_id,
        "status": str(report.get("status") or ("running" if live_events else "starting")),
        "quality_status": str(report.get("quality_status") or "unknown"),
        "report": report,
        "report_path": str(run_dir / "report.json"),
        "live_event_path": str(run_dir / "live_events.jsonl"),
        "transcript": transcript,
        "conversations": conversations,
        "topics": _group_topics(transcript),
        "participants": _group_participants(transcript),
        "reviews": latest_reviews,
        "review_summary": {
            "reviewed_turns": len(latest_reviews),
            "incorrect_turns": sum(item.get("verdict") == "incorrect" for item in latest_reviews.values()),
            "uncertain_turns": sum(item.get("verdict") == "uncertain" for item in latest_reviews.values()),
            "business_evaluation": "not_evaluated",
            "topic_evaluation": "not_evaluated",
        },
        "turn_count": len(transcript),
    }


def _group_topics(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按模拟器剧本线程组织审阅视图，明确该标签不是模型预测结果。"""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for turn in transcript:
        key = (str(turn.get("conversation_id") or "unknown"), str(turn.get("thread_id") or ""))
        item = grouped.setdefault(key, {
            "conversation_id": key[0], "thread_id": key[1], "label_source": "simulator",
            "participants": {}, "turns": [], "title": str((turn.get("user") or {}).get("text") or "未命名话题"),
        })
        user = turn.get("user") or {}
        item["participants"][str(user.get("customer_id") or "unknown")] = str(user.get("display_name") or "未命名用户")
        item["turns"].append(turn)
    result = []
    for index, item in enumerate(grouped.values(), 1):
        item["view_id"] = f"topic_{index}"
        item["participants"] = [{"customer_id": key, "display_name": name} for key, name in item["participants"].items()]
        item["turn_count"] = len(item["turns"])
        result.append(item)
    return result


def _group_participants(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按稳定模拟用户汇总跨群聊和私聊的输入，不改变实际会话边界。"""

    grouped: dict[str, dict[str, Any]] = {}
    for turn in transcript:
        user = turn.get("user") or {}
        customer_id = str(user.get("customer_id") or "unknown")
        item = grouped.setdefault(customer_id, {"customer_id": customer_id, "display_name": str(user.get("display_name") or customer_id), "turns": []})
        item["turns"].append(turn)
    return [{**item, "turn_count": len(item["turns"])} for item in grouped.values()]


def save_simulation_review(
    run_id: str, sequence: int, verdict: str, note: str = "", *, runtime_dir: Path | None = None,
) -> dict[str, Any]:
    """以追加证据保存人工逐轮判断，禁止覆盖报告、编造回合或改写业务状态。"""

    safe_id = _safe_run_id(run_id)
    if not safe_id or verdict not in {"acceptable", "incorrect", "uncertain"}:
        raise ValueError("invalid run id or review verdict")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not isinstance(note, str) or len(note) > 2000:
        raise ValueError("invalid sequence or note")
    base = (runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve()
    selected = _load_run(base, safe_id)
    if not any(turn.get("sequence") == sequence for turn in selected["transcript"]):
        raise ValueError("review turn does not exist")
    review = {"sequence": sequence, "verdict": verdict, "note": note.strip(), "reviewed_at": datetime.now(timezone.utc).isoformat()}
    with _CONTROL_LOCK:
        with (base / "runs" / safe_id / "reviews.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(review, ensure_ascii=False) + "\n")
    return {"ok": True, "review": review}


def simulation_transcript_export(
    run_id: str, *, format: str = "markdown", runtime_dir: Path | None = None,
) -> tuple[bytes, str, str]:
    """导出整轮对话与逐步执行证据；数据源严格限于既有隔离模拟记录。"""

    from mahjong_agent_runtime.agent_state import redact_runtime_payload

    safe_id = _safe_run_id(run_id)
    if not safe_id or format not in {"markdown", "jsonl"}:
        raise ValueError("invalid run id or export format")
    selected = _load_run((runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve(), safe_id)
    turns = selected["transcript"]
    if not turns:
        raise ValueError("run has no transcript")
    turns = redact_runtime_payload(turns)
    report = selected["report"]
    run_metadata = {
        "run_id": safe_id,
        "llm_mode": report.get("llm_mode"),
        "main_agent_model": report.get("main_agent_model"),
        "message_generation_mode": report.get("message_generation_mode"),
        "message_generation_model": report.get("message_generation_model"),
        "configuration": report.get("configuration") or {},
        "main_agent_runtime_config": report.get("main_agent_runtime_config") or {},
        "quality_status": selected["quality_status"],
        "quality_issues": report.get("quality_issues") or [],
        "unresolved_agent_output_count": report.get("unresolved_agent_output_count"),
        "business_evaluation": "not_evaluated",
        "topic_label_source": "simulator",
        "execution_scope": "simulation_adapter_direct_agent_runtime",
    }
    if format == "jsonl":
        rows = [{**turn, "simulation_run": run_metadata, "human_review": selected["reviews"].get(str(turn.get("sequence")))} for turn in turns]
        data = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
        return data.encode("utf-8"), "application/x-ndjson; charset=utf-8", f"{safe_id}.jsonl"
    lines = [f"# 模拟聊天室多轮对话：{safe_id}", "", f"- 主 Agent 模式：{report.get('llm_mode') or '未记录'}",
             f"- 主 Agent 模型：{report.get('main_agent_model') or '未记录'}",
             f"- 主 Agent 单次上下文预算：{(report.get('main_agent_runtime_config') or {}).get('max_tokens_per_call', '未记录')}",
             f"- 客户消息模式：{report.get('message_generation_mode') or '未记录'} / {report.get('message_generation_model') or '未调用生成模型'}",
             f"- 工程运行质量：{selected['quality_status']}", f"- 回合数：{len(turns)}", "- 业务完成与话题识别：待人工验收。剧本话题不是模型预测。",
             f"- Agent 未完成收尾回合数：{report.get('unresolved_agent_output_count', '未记录')}",
             "- 执行入口：模拟适配器直接调用主 Agent Runtime，未覆盖生产群聊聚合入口和微信通道。",
             "- 随机种子控制虚拟用户与调度，不保证真实模型逐字复现。", "",
             "## 模拟设置", _markdown_block(json.dumps(run_metadata["configuration"], ensure_ascii=False, indent=2)),
             "## 完整对话时间线", ""]
    for turn in turns:
        user, agent = turn.get("user") or {}, turn.get("agent") or {}
        generation = user.get("generation") or {}
        lines.extend([f"## 第 {turn.get('sequence')} 轮 · {user.get('display_name') or user.get('customer_id')}", "",
                      f"- 会话：{turn.get('conversation_id')}", f"- 剧本话题：{turn.get('thread_id') or '未记录'}",
                      f"- 消息来源：{generation.get('source') or 'rule'} / {generation.get('model') or '无模型'}",
                      f"- Agent 目标状态：{agent.get('objective_status') or '未记录'}",
                      f"- trace：{agent.get('trace_id') or '未记录'}", "", "### 客户原话", _markdown_block(str(user.get('text') or '')),
                      "### Agent 实际回复", _markdown_block(str(agent.get('reply') or '（无回复）')), ""])
    lines.extend(["## 逐轮执行证据", ""])
    for turn in turns:
        generation = (turn.get("user") or {}).get("generation") or {}
        agent = turn.get("agent") or {}
        lines.extend([f"### 第 {turn.get('sequence')} 轮证据", _markdown_block(json.dumps({
                          "generation_context": generation.get("context") or {}, "actions": agent.get("actions") or [],
                          "simulation_observation": turn.get("simulation_observation") or {},
                          "tool_results": turn.get("tool_results") or [], "state_transitions": turn.get("state_transitions") or [],
                          "status_code": turn.get("status_code"), "error": turn.get("error"),
                          "human_review": selected["reviews"].get(str(turn.get("sequence"))),
                      }, ensure_ascii=False, indent=2)), ""])
    return "\n".join(lines).encode("utf-8"), "text/markdown; charset=utf-8", f"{safe_id}.md"


def _markdown_block(value: str) -> str:
    """使用足够长的围栏保留原始消息，防止消息里的 Markdown 改变报告结构。"""

    import re

    fence = "`" * max(3, 1 + max((len(item) for item in re.findall(r"`+", value)), default=0))
    return f"{fence}\n{value}\n{fence}\n"


def _group_conversations(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for turn in transcript:
        conversation_id = str(turn.get("conversation_id") or "unknown")
        user = turn.get("user") if isinstance(turn.get("user"), dict) else {}
        record = grouped.setdefault(
            conversation_id,
            {
                "conversation_id": conversation_id,
                "channel": str(turn.get("channel") or "private"),
                "participants": {},
                "turns": [],
                "latest_observed_at": "",
            },
        )
        customer_id = str(user.get("customer_id") or "unknown")
        record["participants"][customer_id] = str(user.get("display_name") or customer_id)
        record["turns"].append(turn)
        record["latest_observed_at"] = str(turn.get("observed_at") or record["latest_observed_at"])
    result: list[dict[str, Any]] = []
    for record in grouped.values():
        record["participants"] = [
            {"customer_id": customer_id, "display_name": display_name}
            for customer_id, display_name in record["participants"].items()
        ]
        record["turn_count"] = len(record["turns"])
        result.append(record)
    return sorted(
        result,
        key=lambda item: (
            str(item.get("latest_observed_at") or ""),
            int((item.get("turns") or [{}])[-1].get("sequence") or 0),
        ),
        reverse=True,
    )


def _read_env_metadata(path: Path) -> dict[str, str]:
    allowed = {
        "MAHJONG_PERIODIC_SIM_LLM_MODE",
        "MAHJONG_SIM_GENERATOR_API_KEY",
        "MAHJONG_SIM_GENERATOR_MODEL",
        "MAHJONG_SIM_GENERATOR_BASE_URL",
        "MAHJONG_PERIODIC_SIM_MESSAGE_MODE",
    }
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in allowed:
            values[key] = value.strip().strip("'").strip('"')
    for key in allowed:
        if os.getenv(key):
            values[key] = str(os.getenv(key))
    return values


def _safe_run_id(run_id: str | None) -> str:
    value = str(run_id or "").strip()
    return value if value and value.replace("_", "").isalnum() else ""


def _count_channel(transcript: Any, channel: str) -> int:
    return sum(
        isinstance(item, dict) and str(item.get("channel") or "") == channel
        for item in transcript or []
    )


def _generation_mode(transcript: Any) -> str:
    sources = {
        str(((item.get("user") or {}).get("generation") or {}).get("source") or "")
        for item in transcript or []
        if isinstance(item, dict)
    }
    return "model" if "model" in sources else ("glm" if "glm" in sources else ("rule_fallback" if "rule_fallback" in sources else "rule"))


def _generation_model(transcript: Any) -> str | None:
    for item in transcript or []:
        if not isinstance(item, dict):
            continue
        model = ((item.get("user") or {}).get("generation") or {}).get("model")
        if model:
            return str(model)
    return None


def _load_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default or {})
    return value if isinstance(value, dict) else dict(default or {})


def _tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-max(1, limit) :]
    except OSError:
        return []
    values: list[dict[str, Any]] = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            values.append(item)
    return values


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _empty_run() -> dict[str, Any]:
    return {
        "run_id": "",
        "status": "empty",
        "report": {},
        "report_path": "",
        "live_event_path": "",
        "transcript": [],
        "conversations": [],
        "turn_count": 0,
    }
