#!/usr/bin/env python3
"""运行独立的 500 人麻将馆仿真，保存可复核的工程及模型证据。"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mahjong_agent_runtime import JsonlTraceRecorder, SQLiteAgentStore  # noqa: E402
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient  # noqa: E402
from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock  # noqa: E402
from tests.simulation.sim_adapter import build_runtime  # noqa: E402
from tests.simulation.venue_actors import MockActor, ModelActor, SemanticJudge, generate_personas, write_personas  # noqa: E402
from tests.simulation.venue_channel import SandboxChannel  # noqa: E402
from tests.simulation.venue_cohort import VenueCohort  # noqa: E402
from tests.simulation.venue_dialogues import VenueDialogues  # noqa: E402
from tests.simulation.venue_fixture import VenueFixtureAgent  # noqa: E402
from tests.simulation.venue_models import (  # noqa: E402
    configure_venue_consent_prompt, opencode_auxiliary, opencode_config, venue_planner_config,
)
from tests.simulation.venue_scenario import VENUE, VenueScenario  # noqa: E402


def _client(prefix: str, *, fallbacks: dict[str, str] | None = None) -> OpenAICompatibleAgentClient:
    """为角色或判定器单独构造供应商客户端，主 Runtime 配置独立。"""
    fallback = fallbacks or {}
    api_key = os.environ.get(f"{prefix}_API_KEY") or fallback.get("api_key")
    model = os.environ.get(f"{prefix}_MODEL") or fallback.get("model")
    base_url = os.environ.get(f"{prefix}_BASE_URL") or fallback.get("base_url")
    if not api_key or not model or not base_url:
        raise ValueError(f"真实仿真需要 {prefix}_API_KEY、{prefix}_MODEL 和 {prefix}_BASE_URL")
    return OpenAICompatibleAgentClient(AgentLLMConfig(
        model=model, base_url=base_url, api_key=api_key, response_format="json_object",
        provider=os.environ.get(f"{prefix}_PROVIDER", "openai_compatible"),
        api_protocol=os.environ.get(f"{prefix}_API_PROTOCOL", "chat_completions"),
        max_tokens=1600, max_concurrency=1, retry_attempts=2,
    ))


def _real_roles(config: AgentLLMConfig | None = None) -> tuple[ModelActor, SemanticJudge]:
    """复用已有角色模型配置；判定器有显式、独立的配置命名空间。"""
    if config is not None:
        role = _client("MAHJONG_VENUE_ROLE") if os.environ.get("MAHJONG_VENUE_ROLE_API_KEY") else (
            opencode_auxiliary(config, max_tokens=1600, workers=config.max_concurrency)
        )
        judge = _client("MAHJONG_VENUE_JUDGE") if os.environ.get("MAHJONG_VENUE_JUDGE_API_KEY") else (
            opencode_auxiliary(config, max_tokens=2600, workers=1)
        )
        return ModelActor(role, repair_schema=True), SemanticJudge(judge)
    role_client = _client("MAHJONG_VENUE_ROLE", fallbacks={
        "api_key": os.environ.get("MAHJONG_SIM_GENERATOR_API_KEY", ""),
        "model": os.environ.get("MAHJONG_SIM_GENERATOR_MODEL", "glm-4.7-flash"),
        "base_url": os.environ.get("MAHJONG_SIM_GENERATOR_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
    })
    return ModelActor(role_client, repair_schema=True), SemanticJudge(_client("MAHJONG_VENUE_JUDGE"))


def _select_local_model(source: str, env_file: Path | None, *, workers: int) -> AgentLLMConfig | None:
    """显式模型来源优先，自动模式沿用进程或显式文件，然后查本机 OpenCode。"""
    inherited = bool(os.environ.get("MAHJONG_LLM_MODEL") and os.environ.get("MAHJONG_LLM_API_KEY"))
    if source == "project" or source == "auto" and (env_file is not None or inherited):
        return None
    codex_dir = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    try:
        return opencode_config(codex_dir, max_concurrency=workers)
    except FileNotFoundError:
        if source == "opencode-go":
            raise
        return None


def _model_execution(client, *, timeout_seconds: float | None) -> dict:
    """记录仿真实际选择的模型、协议与限制，保留跨端可复核的配置证据。"""
    config = getattr(client, "config", None)
    request = config.request_diagnostics() if isinstance(config, AgentLLMConfig) else {}
    return {
        "model": getattr(config, "model", "offline fixture"),
        "provider": getattr(config, "provider", "mock"),
        "protocol": getattr(config, "api_protocol", None),
        "max_output_tokens": getattr(config, "max_tokens", None),
        "max_concurrency": getattr(config, "max_concurrency", None),
        "retry_attempts": getattr(config, "retry_attempts", None),
        "attempt_timeout_seconds": getattr(config, "attempt_timeout_seconds", None),
        "reasoning_effort_on_wire": request.get("reasoning_effort"),
        "thinking_mode_on_wire": request.get("thinking_mode"),
        "timeout_seconds": timeout_seconds,
    }


def run_simulation(
    run_dir: Path, *, mode: str, seed: int, scenario_date: date, background_events: int,
    env_file: Path | None = None, model_source: str = "auto",
    scenario_name: str = "lifecycle", active_users: int = 12, workers: int = 3,
    dialogue_turns: int = 4, judge_users: int = 3, planner_low: bool = False, clarify_consent: bool = False,
) -> dict:
    """创建新沙盒目录，并将业务是否完成与模型是否实际参与分别报告。"""
    start_at = datetime.combine(scenario_date, time(19), tzinfo=DEFAULT_TZ)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "report.json"
    report = {"mode": mode, "scenario": scenario_name, "seed": seed,
              "scenario_date": scenario_date.isoformat(), "status": "failed"}
    try:
        if mode not in {"mock", "real"} or not 0 <= background_events <= 500:
            raise ValueError("mode must be mock or real and background_events must be in 0..500")
        if model_source not in {"auto", "project", "opencode-go"}:
            raise ValueError("模型来源必须为 auto、project 或 opencode-go")
        if mode != "real" and (planner_low or clarify_consent):
            raise ValueError("规划模型对照只适用于真实模型沙盒")
        if (scenario_name not in {"lifecycle", "dialogues", "formation", "no-match"}
                or not 1 <= active_users <= 500 or not 1 <= workers <= 32
                or not 2 <= dialogue_turns <= 8 or not 1 <= judge_users <= 20):
            raise ValueError("场景、活跃用户、并发、对话或语义抽样参数超出范围")
        main_override = None
        people = generate_personas(seed)
        write_personas(run_dir / "personas.json", people)
        (run_dir / "venue.json").write_text(json.dumps(VENUE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if mode == "real":
            if env_file:
                if not env_file.is_file():
                    raise ValueError("模型配置文件不存在")
                load_dotenv_defaults(env_file)
            open_code = _select_local_model(model_source, env_file, workers=min(workers, 4))
            load_dotenv_defaults(ROOT / ".env")
            actor, judge = _real_roles(open_code) if open_code is not None else _real_roles()
            if open_code is not None:
                main_override = OpenAICompatibleAgentClient(venue_planner_config(open_code, reasoning_low=planner_low))
            elif planner_low:
                raise ValueError("低推理对照需要 OpenCode Go 模型来源")
            fixture_agent = None
            report["model_source"] = "opencode-go" if open_code is not None else "project"
        else:
            actor, judge = MockActor(seed), None
            fixture_agent = VenueFixtureAgent(seed=seed)
        with using_clock(lambda: start_at - timedelta(minutes=70)):
            store = SQLiteAgentStore(run_dir / "agent.sqlite3")
            store.configure_rooms([room["id"] for room in VENUE["rooms"]])
            for block in VENUE["initial_blocks"]:
                store.reserve_room(
                    conversation_id="shop-inventory", game_id=None, room_id=block["room_id"], trace_id="shop-initial-state",
                    start_at=datetime.combine(scenario_date, time.fromisoformat(block["starts"]), tzinfo=DEFAULT_TZ),
                    end_at=datetime.combine(scenario_date, time.fromisoformat(block["ends"]), tzinfo=DEFAULT_TZ),
                )
            runtime, main_client = build_runtime(
                store, mode, seed=seed, offline_client=fixture_agent,
                model_client=main_override,
                trace_recorder=JsonlTraceRecorder(run_dir / "agent_trace.log"),
                local_coordination=True,
            )
            if clarify_consent:
                report["sandbox_consent_prompt_sha256"] = configure_venue_consent_prompt(runtime, run_dir)
            channel = SandboxChannel(runtime, room_id="venue-group",
                identities={person.sender_id: person.name for person in people}, archive_path=run_dir / "chat.sqlite3")
            initial_profiles = store.customers
            # 只比较实例应预填的公开字段，整份仿真画像不进入 Runtime 及其装配。
            initial_identity_only = len(initial_profiles) == len(people) and all(
                not any((item.preferred_games, item.preferred_stakes, item.preferred_time_tags, item.profile_facts,
                         item.notes, item.private_remark, item.gender, item.smoke_preference))
                for item in initial_profiles.values()
            )
        common = dict(people=people, actor=actor, judge=judge, channel=channel, store=store,
                      run_dir=run_dir, start_at=start_at, seed=seed, background_events=background_events,
                      fixture_agent=fixture_agent)
        if scenario_name in {"formation", "no-match"}:
            scenario = VenueCohort(**common, case=scenario_name, active_users=active_users, workers=workers,
                dialogue_turns=dialogue_turns, judge_users=judge_users)
        elif scenario_name == "dialogues":
            scenario = VenueDialogues(**common, active_users=active_users, workers=workers,
                dialogue_turns=dialogue_turns, judge_users=judge_users)
        else:
            scenario = VenueScenario(**common)
        report.update(scenario.run())
        report["checks"]["identity_only_at_start"] = initial_identity_only
        report["main_model"] = getattr(getattr(main_client, "config", None), "model", "offline fixture")
        report["main_provider"] = getattr(getattr(main_client, "config", None), "provider", "mock")
        report["model_execution"] = {
            "agent": _model_execution(main_client, timeout_seconds=runtime.llm_timeout_seconds),
            "role": _model_execution(getattr(actor, "client", None), timeout_seconds=getattr(actor, "timeout_seconds", None)),
            "judge": _model_execution(getattr(judge, "client", None), timeout_seconds=getattr(judge, "timeout_seconds", None)),
        }
        judgment = report.get("independent_judge") or {}
        role_ok = mode == "mock" or judgment.get("status") == "valid" and judgment.get("verdict") == "pass"
        checks_ok = bool(report["checks"]) and all(report["checks"].values()) and not report["failed_events"]
        if checks_ok and role_ok:
            report["status"] = "passed"
        elif judgment.get("verdict") != "fail" and report.get("acceptance", {}).get("status") != "failed" and (
            report.get("model_transport_errors")
            or report.get("acceptance", {}).get("status") == "inconclusive"
            or checks_ok and judgment.get("verdict") == "inconclusive"
        ):
            report["status"] = "inconclusive"
    except (OSError, TypeError, ValueError, RuntimeError, AttributeError, KeyError, StopIteration) as exc:
        report["error"] = str(exc)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    """离线默认运行；真实模型通过显式参数开启。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--scenario", choices=("lifecycle", "dialogues", "formation", "no-match"), default="lifecycle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date", type=date.fromisoformat, default=date(2026, 11, 25))
    parser.add_argument("--background-events", type=int, default=None)
    parser.add_argument("--active-users", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dialogue-turns", type=int, default=4)
    parser.add_argument("--judge-users", type=int, default=3)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model-source", choices=("auto", "project", "opencode-go"), default="auto")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data" / "venue_simulation" / "runs")
    parser.add_argument("--planner-low", action="store_true", help="仅在沙盒主模型开启显式低推理")
    parser.add_argument("--clarify-consent", action="store_true", help="加载独立副本报名授权对照提示")
    args = parser.parse_args()
    if args.scenario == "lifecycle":
        background_default = 3 if args.mode == "real" else 36
    elif args.scenario == "dialogues" and args.mode == "mock":
        background_default = 8
    else:
        background_default = 0
    limit = args.background_events if args.background_events is not None else background_default
    if not 0 <= limit <= 500:
        parser.error("--background-events must be in 0..500")
    default_users = {"formation": 1, "no-match": 2}.get(args.scenario, 4 if args.mode == "real" else 32)
    users = args.active_users if args.active_users is not None else default_users
    workers = args.workers if args.workers is not None else (3 if args.mode == "real" else 8)
    run_dir = args.runs_dir / f"{args.date.isoformat()}-{args.mode}-{uuid.uuid4().hex[:10]}"
    report = run_simulation(run_dir, mode=args.mode, seed=args.seed,
        scenario_date=args.date, background_events=limit, env_file=args.env_file, model_source=args.model_source,
        scenario_name=args.scenario, active_users=users, workers=workers,
        dialogue_turns=args.dialogue_turns, judge_users=args.judge_users,
        planner_low=args.planner_low, clarify_consent=args.clarify_consent)
    print(json.dumps({"status": report["status"], "mode": args.mode, "report": str(run_dir / "report.json"),
                      "failed_checks": [name for name, passed in report.get("checks", {}).items() if passed is False],
                      "not_run_checks": [name for name, passed in report.get("checks", {}).items() if passed is None],
                      "model_http_statuses": sorted({code for error in report.get("model_transport_errors", [])
                                                     for code in error["http_statuses"]}),
                      "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
