#!/usr/bin/env python3
"""运行独立的 500 人麻将馆仿真，保存可复核的工程及模型证据。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timedelta
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mahjong_agent_runtime import JsonlTraceRecorder, SQLiteAgentStore  # noqa: E402
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient  # noqa: E402
from mahjong_agent_runtime.models import CustomerProfile, DEFAULT_TZ, using_clock  # noqa: E402
from tests.simulation.sim_adapter import build_runtime  # noqa: E402
from tests.simulation.venue_actors import (  # noqa: E402
    MockActor, ModelActor, SemanticJudge, generate_personas, read_personas, write_personas,
)
from tests.simulation.venue_channel import SandboxChannel  # noqa: E402
from tests.simulation.venue_cohort import VenueCohort  # noqa: E402
from tests.simulation.venue_disturbance import (  # noqa: E402
    DISTURBANCE_CASES, DisturbanceFixtureActor, DisturbanceFixtureAgent, VenueDisturbances,
)
from tests.simulation.venue_dialogues import VenueDialogues  # noqa: E402
from tests.simulation.venue_fixture import VenueFixtureAgent  # noqa: E402
from tests.simulation.venue_friend_group_link import VenueFriendGroupLinkScenario  # noqa: E402
from tests.simulation.venue_friend_scenario import VenueFriendScenario  # noqa: E402
from tests.simulation.venue_history_scenario import HistoricalFixtureActor, VenueHistoricalScenario  # noqa: E402
from tests.simulation.venue_history_seed import (  # noqa: E402
    DEFAULT_HISTORY_CASE_ID, DEFAULT_HISTORY_DATASET, load_historical_group_seed,
)
from tests.simulation.venue_history_variants import HISTORY_GROUP_VARIANTS, load_group_history_variant  # noqa: E402
from tests.simulation.venue_models import (  # noqa: E402
    configure_venue_consent_prompt, deepseek_official_config, opencode_auxiliary, opencode_config, venue_planner_config,
)
from tests.simulation.venue_private_history_seed import PRIVATE_HISTORY_EPISODES, load_reviewed_private_episode  # noqa: E402
from tests.simulation.venue_role_journal import RoleSchedulerBusyError  # noqa: E402
from tests.simulation.venue_role_traits import blend_venue_role_traits, load_reviewed_venue_role_traits  # noqa: E402
from tests.simulation.venue_rolling_roles import RollingRoleTransportPaused, VenueRollingRoles  # noqa: E402
from tests.simulation.venue_scenario import VENUE, VenueScenario  # noqa: E402
from tests.simulation.venue_world import VenueWorld  # noqa: E402
from tests.simulation.venue_write_guard import VenueWriteGuard  # noqa: E402

SCENARIOS = ("lifecycle", "dialogues", "formation", "no-match", "mixed", "private-initiation", "friend-group-link",
             "disturbances", "historical-group", "historical-private", "rolling-roles")
MODEL_SOURCES = ("auto", "project", "opencode-go", "deepseek-official")
_IDENTITY_FIELDS = {"customer_id", "display_name", "public_name"}
_ROLLING_STATE = "rolling_start.json"


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


def _real_roles(config: AgentLLMConfig | None = None, *, force_opencode_role: bool = False) -> tuple[ModelActor, SemanticJudge]:
    """显式 OpenCode 来源同时约束顾客角色；独立裁判仍可单独配置。"""
    if config is not None:
        role = (opencode_auxiliary(config, max_tokens=1600, workers=config.max_concurrency)
                if force_opencode_role or not os.environ.get("MAHJONG_VENUE_ROLE_API_KEY")
                else _client("MAHJONG_VENUE_ROLE"))
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


def _initial_contact_evidence(store: SQLiteAgentStore, channel: SandboxChannel, nicknames: dict[str, str]) -> dict:
    """在角色行动之前核对好友、昵称、独立私聊和空白业务画像。"""
    profiles = store.customers
    channel_id = store.get_group_room_policy(channel.room_id).channel
    friends = channel.owner_friends
    routed = []
    for customer, nickname in friends.items():
        identity = store.get_channel_identity(channel_id, customer)
        if (identity is not None and identity.customer_id == customer and identity.public_name == nickname
                and identity.is_friend and identity.can_private_message
                and identity.private_conversation_id == channel.private_conversation_id(customer)):
            routed.append(identity.private_conversation_id)
    return {
        "group_members": len(nicknames),
        "owner_friends": len(friends),
        "friend_routing_matches_roster": friends == nicknames and len(routed) == len(nicknames),
        "unique_private_conversations": len(set(routed)),
        "private_conversations_with_history": sum(bool(store.recent_turns(thread, limit=1)) for thread in set(routed)),
        "initial_archive_messages": len(channel.events()),
        "nickname_only_profiles": sum(
            profile == CustomerProfile(customer_id=customer, display_name=nicknames[customer], public_name=nicknames[customer])
            for customer, profile in profiles.items() if customer in nicknames
        ),
        "stored_customers": len(profiles),
        "initial_customer_relationships": len(store.customer_relationships),
    }


def _profile_fields(profile: CustomerProfile) -> dict:
    """只比较画像字段，好友的身份和昵称变化不计作画像沉淀。"""
    return {key: value for key, value in profile.to_dict().items() if key not in _IDENTITY_FIELDS}


def _profile_snapshot(store: SQLiteAgentStore) -> dict[str, dict]:
    """从 SQLite 取出供本地差分的快照，客户键和字段值不写入评测汇总。"""
    return {key: _profile_fields(profile) for key, profile in store.customers.items()}


def _profile_evidence(store: SQLiteAgentStore, before: dict[str, dict], checks: dict) -> dict:
    """分别统计正式画像变动、当前任务记忆和待审长期候选，不暴露客户信息。"""
    default = _profile_fields(CustomerProfile(customer_id="", display_name=""))
    after = _profile_snapshot(store)
    changed = [{key for key, value in profile.items() if value != before.get(customer_id, default)[key]}
               for customer_id, profile in after.items()]
    changed_fields = Counter(key for fields in changed for key in fields)
    return {
        "source": "sqlite_store",
        "formal_profiles": {
            "customers_at_start": len(before),
            "customers_at_end": len(after),
            "customers_with_changed_fields": sum(bool(fields) for fields in changed),
            "customers_with_non_default_fields": sum(profile != default for profile in after.values()),
            "changed_fields": dict(sorted(changed_fields.items())),
        },
        "customer_relationship_records": len(store.customer_relationships),
        "current_task_memory_records_by_status": dict(sorted(Counter(
            item.status for item in store.task_memories.values()).items())),
        "long_term_candidate_records_by_status": dict(sorted(Counter(
            item.status for item in store.pending_memory_candidates.values()).items())),
        "visible_source_pending_candidate_check": checks.get("memory_from_visible_message"),
    }


def run_simulation(
    run_dir: Path, *, mode: str, seed: int, scenario_date: date, background_events: int,
    env_file: Path | None = None, model_source: str = "auto",
    scenario_name: str = "lifecycle", active_users: int = 12, workers: int = 3,
    dialogue_turns: int = 4, judge_users: int = 3, planner_low: bool = False, clarify_consent: bool = False,
    guard_writes: bool = False, disturbance_case: str | None = None, history_cutoff: str | None = None,
    history_dataset: Path | None = None, history_case: str | None = None, history_variant: str | None = None,
    private_cutoff_turn: int | None = None, private_history_episode: str | None = None, role_source: str = "generated",
    rolling_board: bool = False, rolling_history: bool = False, rolling_play: bool = False, show_verified_write_scope: bool = False,
    resume: bool = False,
) -> dict:
    """创建新沙盒目录，并将业务是否完成与模型是否实际参与分别报告。"""
    started = perf_counter()
    start_at = datetime.combine(scenario_date, time(19), tzinfo=DEFAULT_TZ)
    run_dir = run_dir.resolve()
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "report.json"
    report = {"mode": mode, "scenario": scenario_name, "seed": seed,
              "scenario_date": scenario_date.isoformat(), "status": "failed"}
    write_guard = None
    try:
        if mode not in {"mock", "real"} or not 0 <= background_events <= 500:
            raise ValueError("mode must be mock or real and background_events must be in 0..500")
        if model_source not in MODEL_SOURCES:
            raise ValueError(f"模型来源必须为 {', '.join(MODEL_SOURCES)}")
        if role_source not in {"generated", "history"}:
            raise ValueError("角色来源必须为 generated 或 history")
        if mode != "real" and (planner_low or clarify_consent or guard_writes):
            raise ValueError("规划模型对照只适用于真实模型沙盒")
        if show_verified_write_scope and not guard_writes:
            raise ValueError("规划器写入范围对照要求独立授权检查")
        if (scenario_name not in SCENARIOS
                or not 1 <= active_users <= 500 or not 1 <= workers <= 32
                or not 2 <= dialogue_turns <= 8 or not 1 <= judge_users <= 20):
            raise ValueError("场景、活跃用户、并发、对话或语义抽样参数超出范围")
        if disturbance_case is not None and scenario_name != "disturbances":
            raise ValueError("突发事件子案例只能用于 disturbances 场景")
        if any(value is not None for value in (history_cutoff, history_dataset, history_case)) and scenario_name != "historical-group":
            raise ValueError("真实群聊素材、用例和切点只能用于 historical-group 场景")
        if history_variant is not None and (scenario_name != "historical-group" or any(
                value is not None for value in (history_cutoff, history_dataset, history_case))):
            raise ValueError("历史变体根据自己的群聊素材和切点生成")
        if scenario_name == "historical-group" and (workers != 1 or background_events or active_users > 499):
            raise ValueError("历史群聊续演要求单线程、有至少一个未唤醒群友、且不安排额外背景事件")
        if (private_cutoff_turn is not None or private_history_episode is not None) and scenario_name != "historical-private":
            raise ValueError("真实私聊片段和切点只能用于 historical-private 场景")
        if scenario_name == "historical-private" and (workers != 1 or background_events or active_users != 1):
            raise ValueError("历史私聊续演只唤醒原对话中的一位沙盒顾客")
        if scenario_name == "rolling-roles" and background_events:
            raise ValueError("持续角色使用持久事件而非预设背景消息")
        if (rolling_board or rolling_history or rolling_play) and scenario_name != "rolling-roles":
            raise ValueError("滚动看板、历史开场和物理营业只能用于持续角色演练")
        if resume and scenario_name != "rolling-roles":
            raise ValueError("只能继续已有的持续角色运行")
        starting = json.loads(run_dir.joinpath(_ROLLING_STATE).read_text(encoding="utf-8")) if resume else None
        main_override = None
        role_seed_summary = None
        if resume:
            people = read_personas(run_dir / "personas.json")
            role_seed_summary = starting.get("role_seed_profiles")
            if not all(run_dir.joinpath(name).is_file() for name in ("agent.sqlite3", "chat.sqlite3")):
                raise ValueError("已有运行缺少业务库或聊天账本")
        else:
            people = generate_personas(seed)
            if scenario_name in {"disturbances", "rolling-roles"}:
                people = VenueWorld(seed, start_at).personas(people)
            if role_source == "history":
                traits = load_reviewed_venue_role_traits()
                blend = blend_venue_role_traits(people, seed=seed, traits=traits)
                people = blend.people
                role_seed_summary = {"source": "reviewed_chat_prefixes", "anonymous_speaker_samples": len(traits),
                    "source_cases": len({trait.case_id for trait in traits}), "derived_roles": len(blend.attributions),
                    "distinct_speech_profiles": len({person.speech_profile for person in people}),
                    "applied_style_fields": dict(Counter(field for row in blend.attributions for field in row.speech_dimensions_changed))}
                run_dir.joinpath("role_traits.json").write_text(json.dumps(
                    [entry.to_dict() for entry in blend.attributions], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            write_personas(run_dir / "personas.json", people)
            venue = {**VENUE, "opens": "09:00"} if scenario_name == "disturbances" else VENUE
            (run_dir / "venue.json").write_text(json.dumps(venue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if role_seed_summary:
            report["role_seed_profiles"] = role_seed_summary
        review_override = None
        if mode == "real":
            if env_file:
                if not env_file.is_file():
                    raise ValueError("模型配置文件不存在")
                load_dotenv_defaults(env_file)
            official = model_source == "deepseek-official"
            open_code = None if official else _select_local_model(model_source, env_file, workers=min(workers, 4))
            load_dotenv_defaults(ROOT / ".env")
            selected = deepseek_official_config(max_concurrency=min(workers, 4)) if official else open_code
            if official:
                # 显式官方对照的角色、店家、授权和两类评审都使用本次密钥，不能串用其他供应商配置。
                actor = ModelActor(opencode_auxiliary(selected, max_tokens=1600, workers=selected.max_concurrency), repair_schema=True)
                judge = SemanticJudge(opencode_auxiliary(selected, max_tokens=2600, workers=1))
                review_override = opencode_auxiliary(selected, max_tokens=16384, workers=selected.max_concurrency)
            else:
                actor, judge = (_real_roles(open_code, force_opencode_role=True) if model_source == "opencode-go"
                                else _real_roles(open_code) if open_code is not None else _real_roles())
            if selected is not None:
                main_override = OpenAICompatibleAgentClient(venue_planner_config(selected, reasoning_low=planner_low))
                if guard_writes:
                    write_guard = VenueWriteGuard(opencode_auxiliary(selected, max_tokens=700, workers=min(workers, 4)),
                        run_dir / "write_authorization.jsonl", repair_stale_citation=True, resume_audit=resume)
            elif planner_low:
                raise ValueError("低推理对照需要指定仿真模型来源")
            elif guard_writes:
                raise ValueError("独立写入检查需使用 OpenCode Go 或 DeepSeek 官方仿真配置")
            fixture_agent = None
            report["model_source"] = "deepseek-official" if official else "opencode-go" if open_code is not None else "project"
        else:
            if scenario_name == "disturbances":
                actor, judge, fixture_agent = DisturbanceFixtureActor(), None, DisturbanceFixtureAgent(seed=seed)
            else:
                actor, judge, fixture_agent = MockActor(seed), None, VenueFixtureAgent(seed=seed)
        with using_clock(lambda: start_at - timedelta(minutes=70)):
            store = SQLiteAgentStore(run_dir / "agent.sqlite3")
            if not resume:
                store.configure_rooms([room["id"] for room in VENUE["rooms"]])
                for block in VENUE["initial_blocks"]:
                    store.reserve_room(
                        conversation_id="shop-inventory", game_id=None, room_id=block["room_id"], trace_id="shop-initial-state",
                        start_at=datetime.combine(scenario_date, time.fromisoformat(block["starts"]), tzinfo=DEFAULT_TZ),
                        end_at=datetime.combine(scenario_date, time.fromisoformat(block["ends"]), tzinfo=DEFAULT_TZ),
                    )
            runtime, main_client = build_runtime(
                store, mode, seed=seed, offline_client=fixture_agent,
                model_client=main_override, reply_review_client=review_override,
                trace_recorder=JsonlTraceRecorder(run_dir / "agent_trace.log"),
                local_coordination=True,
                tool_gateway=write_guard.guarded_gateway(store) if write_guard else None,
            )
            if clarify_consent:
                report["sandbox_consent_prompt_sha256"] = configure_venue_consent_prompt(runtime, run_dir)
            nicknames = {person.sender_id: person.name for person in people}
            channel = SandboxChannel(runtime, room_id="venue-group",
                identities=nicknames, archive_path=run_dir / "chat.sqlite3",
                write_guard=write_guard, show_verified_write_scope=show_verified_write_scope)
            if show_verified_write_scope:
                report["show_verified_write_scope"] = True
            report["initial_contacts"] = contacts = (starting["contacts"] if starting else
                                                       _initial_contact_evidence(store, channel, nicknames))
            initial_profiles = starting["profiles"] if starting else _profile_snapshot(store)
        if scenario_name == "rolling-roles" and not resume:
            snapshot = {"schema_version": 1, "contacts": contacts, "profiles": initial_profiles,
                "role_seed_profiles": role_seed_summary, "options": {
                "mode": mode, "seed": seed, "scenario_date": scenario_date.isoformat(), "background_events": background_events,
                "env_file": str(env_file.resolve()) if env_file else None, "model_source": model_source,
                "active_users": active_users, "workers": workers, "dialogue_turns": dialogue_turns, "judge_users": judge_users,
                "planner_low": planner_low, "clarify_consent": clarify_consent, "guard_writes": guard_writes, "role_source": role_source,
                "rolling_board": rolling_board,
                "rolling_history": rolling_history,
                "rolling_play": rolling_play,
                "show_verified_write_scope": show_verified_write_scope,
            }}
            temporary = run_dir / f".{_ROLLING_STATE}.tmp"
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(run_dir / _ROLLING_STATE)
        common = dict(people=people, actor=actor, judge=judge, channel=channel, store=store,
                      run_dir=run_dir, start_at=start_at, seed=seed, background_events=background_events,
                      fixture_agent=fixture_agent)
        if scenario_name in {"formation", "no-match", "mixed"}:
            scenario = VenueCohort(**common, case=scenario_name, active_users=active_users, workers=workers,
                dialogue_turns=dialogue_turns, judge_users=judge_users)
        elif scenario_name == "private-initiation":
            scenario = VenueFriendScenario(**common, active_users=active_users, workers=workers,
                dialogue_turns=dialogue_turns, judge_users=judge_users)
        elif scenario_name == "friend-group-link":
            scenario = VenueFriendGroupLinkScenario(**common, active_users=active_users, workers=workers,
                dialogue_turns=dialogue_turns, judge_users=judge_users)
        elif scenario_name == "disturbances":
            scenario = VenueDisturbances(**common, selected_case=disturbance_case)
        elif scenario_name in {"historical-group", "historical-private"}:
            variant = load_group_history_variant(history_variant) if history_variant else None
            if variant:
                history = variant.seed
            elif scenario_name == "historical-group":
                history = load_historical_group_seed(history_dataset or DEFAULT_HISTORY_DATASET,
                    case_id=history_case or DEFAULT_HISTORY_CASE_ID, cutoff=history_cutoff or "m04")
            else:
                history = load_reviewed_private_episode(private_history_episode or "availability", cutoff_turn=private_cutoff_turn)
            scenario = VenueHistoricalScenario(history=history, people=people,
                actor=HistoricalFixtureActor(seed) if mode == "mock" else actor,
                judge=judge, channel=channel, run_dir=run_dir, start_at=start_at,
                active_users=active_users, dialogue_turns=dialogue_turns, judge_users=judge_users, variant=variant)
        elif scenario_name == "dialogues":
            scenario = VenueDialogues(**common, active_users=active_users, workers=workers,
                dialogue_turns=dialogue_turns, judge_users=judge_users)
        elif scenario_name == "rolling-roles":
            scenario = VenueRollingRoles(people=people, actor=actor, judge=judge, channel=channel,
                run_dir=run_dir, start_at=start_at, seed=seed, active_users=active_users,
                workers=workers, dialogue_turns=dialogue_turns, judge_users=judge_users, publish_boards=rolling_board,
                seed_public_history=rolling_history, physical_play=rolling_play)
        else:
            scenario = VenueScenario(**common)
        try:
            report.update(scenario.run())
        finally:
            report["profile_evidence"] = _profile_evidence(store, initial_profiles, report.get("checks", {}))
        report["checks"]["identity_only_at_start"] = (
            contacts["nickname_only_profiles"] == contacts["stored_customers"] == len(people)
            and contacts["initial_customer_relationships"] == 0
        )
        report["checks"]["all_customers_are_owner_friends"] = contacts["friend_routing_matches_roster"]
        report["checks"]["empty_separate_private_chats_at_start"] = (
            contacts["unique_private_conversations"] == len(people)
            and contacts["private_conversations_with_history"] == contacts["initial_archive_messages"] == 0
        )
        guard_ready = True
        if write_guard is not None:
            # 开放场景允许群聊或沉默。仅有实际到达的老板私聊时才需要前置私聊模型的审计结果。
            private_inbound = sum(
                row["kind"] == "private" and row["direction"] == "inbound" for row in channel.events()
            )
            guarded_case = scenario_name not in {"historical-group", "rolling-roles"} or private_inbound > 0
            report["write_authorization_coverage"] = "exercised" if guarded_case else "not_exercised"
            if guarded_case:
                guard_ready = write_guard.healthy(expected_inputs=private_inbound if resume else None)
                report["checks"]["independent_write_authorization_available"] = guard_ready
        report["main_model"] = getattr(getattr(main_client, "config", None), "model", "offline fixture")
        report["main_provider"] = getattr(getattr(main_client, "config", None), "provider", "mock")
        report["model_execution"] = {
            "agent": _model_execution(main_client, timeout_seconds=runtime.llm_timeout_seconds),
            "role": _model_execution(getattr(actor, "client", None), timeout_seconds=getattr(actor, "timeout_seconds", None)),
            "judge": _model_execution(getattr(judge, "client", None), timeout_seconds=getattr(judge, "timeout_seconds", None)),
        }
        if runtime.reply_self_review_client is not None:
            report["model_execution"]["reply_self_review"] = _model_execution(
                runtime.reply_self_review_client, timeout_seconds=runtime.reply_self_review_timeout_seconds)
        if write_guard is not None:
            report["model_execution"]["write_authorization"] = _model_execution(
                write_guard.client, timeout_seconds=write_guard.timeout_seconds)
        judgment = report.get("independent_judge") or {}
        role_ok = mode == "mock" or judgment.get("status") == "valid" and judgment.get("verdict") == "pass"
        checks_ok = bool(report["checks"]) and all(report["checks"].values()) and not report["failed_events"]
        if checks_ok and role_ok:
            report["status"] = "passed"
        elif judgment.get("verdict") != "fail" and report.get("acceptance", {}).get("status") != "failed" and (
            report.get("model_transport_errors")
            or report.get("acceptance", {}).get("status") == "inconclusive"
            or not guard_ready
            or checks_ok and judgment.get("verdict") == "inconclusive"
        ):
            report["status"] = "inconclusive"
    except RollingRoleTransportPaused as exc:
        report.update(status="paused_transport", provider_pause=exc.checkpoint)
    except RoleSchedulerBusyError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError, AttributeError, KeyError, StopIteration) as exc:
        report["error"] = str(exc)
    if write_guard is not None:
        report["write_authorization"] = write_guard.summary()
    report["elapsed_seconds"] = round(perf_counter() - started, 3)
    if report["status"] == "paused_transport":
        return report
    temporary_report = report_path.with_suffix(".tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    return report


def resume_rolling_simulation(run_dir: Path) -> dict:
    """继续已有营业演练，重用当初的人物、频道、时钟和业务库存。"""
    snapshot = json.loads(run_dir.joinpath(_ROLLING_STATE).read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or not isinstance(snapshot.get("options"), dict):
        raise ValueError("无效的持续角色运行起点")
    report_path = run_dir / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    options = snapshot["options"]
    options["scenario_date"] = date.fromisoformat(options["scenario_date"])
    options["env_file"] = Path(options["env_file"]) if options.get("env_file") else None
    return run_simulation(run_dir, scenario_name="rolling-roles", resume=True, **options)


def main() -> int:
    """离线默认运行；真实模型通过显式参数开启。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--scenario", choices=SCENARIOS, default="lifecycle")
    parser.add_argument("--disturbance-case", choices=DISTURBANCE_CASES, help="只重放一个营业突发案例")
    parser.add_argument("--history-cutoff", help="历史群聊最后一条可见消息，默认 m04")
    parser.add_argument("--history-dataset", type=Path, help="符合现有匿名群聊时间线合同的 JSONL")
    parser.add_argument("--history-case", help="历史群聊时间线在数据集中的案例编号")
    parser.add_argument("--history-variant", choices=HISTORY_GROUP_VARIANTS, help="固定真人切点的单条件对照")
    parser.add_argument("--private-cutoff-turn", type=int, help="历史私聊最后一条真人来往的轮次，默认使用所选片段的切点")
    parser.add_argument("--private-history-episode", choices=PRIVATE_HISTORY_EPISODES, help="选择已核实的真人私聊或独立留出片段")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date", type=date.fromisoformat, default=date(2026, 11, 25))
    parser.add_argument("--background-events", type=int, default=None)
    parser.add_argument("--active-users", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dialogue-turns", type=int, default=4)
    parser.add_argument("--judge-users", type=int, default=3)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--model-source", choices=MODEL_SOURCES, default="auto")
    parser.add_argument("--role-source", choices=("generated", "history"), default="generated", help="是否用审核后的真人前缀形成表达线索")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runtime_data" / "venue_simulation" / "runs")
    parser.add_argument("--resume-run", type=Path, help="继续已有持续角色运行，复用该目录中的起始配置")
    parser.add_argument("--planner-low", action="store_true", help="仅在沙盒主模型开启显式低推理")
    parser.add_argument("--clarify-consent", action="store_true", help="加载独立副本报名授权对照提示")
    parser.add_argument("--guard-writes", action="store_true", help="启用独立私聊写入授权检查")
    parser.add_argument("--rolling-board", action="store_true", help="在持续角色虚拟时钟执行正式群看板发布，并审计同局报名")
    parser.add_argument("--rolling-history", action="store_true", help="用已审核的真人群聊前缀唤醒持续角色")
    parser.add_argument("--rolling-play", action="store_true", help="用已确认四席和真实角色到离场决定推进四小时房态")
    parser.add_argument("--show-verified-write-scope", action="store_true", help="只向沙盒私聊规划器列出本轮已有独立授权的专用写工具")
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
    default_users = {"formation": 1, "no-match": 2, "mixed": 3, "private-initiation": 1, "friend-group-link": 1,
                     "historical-group": 3, "historical-private": 1,
                     "disturbances": 1 if args.disturbance_case else len(DISTURBANCE_CASES)}.get(
        args.scenario, 4 if args.mode == "real" else 32)
    users = args.active_users if args.active_users is not None else default_users
    workers = args.workers if args.workers is not None else (
        1 if args.scenario in {"disturbances", "historical-group", "historical-private"} else 3 if args.mode == "real" else 8
    )
    if args.resume_run:
        run_dir = args.resume_run.resolve()
        try:
            report = resume_rolling_simulation(run_dir)
        except (OSError, ValueError, TypeError, KeyError, RoleSchedulerBusyError) as exc:
            parser.error(str(exc))
    else:
        run_dir = args.runs_dir / f"{args.date.isoformat()}-{args.mode}-{uuid.uuid4().hex[:10]}"
        report = run_simulation(run_dir, mode=args.mode, seed=args.seed,
            scenario_date=args.date, background_events=limit, env_file=args.env_file, model_source=args.model_source,
            scenario_name=args.scenario, active_users=users, workers=workers,
            dialogue_turns=args.dialogue_turns, judge_users=args.judge_users,
            planner_low=args.planner_low, clarify_consent=args.clarify_consent, guard_writes=args.guard_writes,
            disturbance_case=args.disturbance_case, history_cutoff=args.history_cutoff,
            history_dataset=args.history_dataset, history_case=args.history_case, history_variant=args.history_variant,
            private_cutoff_turn=args.private_cutoff_turn, private_history_episode=args.private_history_episode,
            role_source=args.role_source, rolling_board=args.rolling_board, rolling_history=args.rolling_history,
            rolling_play=args.rolling_play,
            show_verified_write_scope=args.show_verified_write_scope)
    paused = report.get("provider_pause")
    rendered = run_dir / (RollingRoleTransportPaused.filename if paused else "report.json")
    print(json.dumps({"status": report["status"], "mode": report["mode"], "report": str(rendered),
                      "failed_checks": [name for name, passed in report.get("checks", {}).items() if passed is False],
                      "not_run_checks": [name for name, passed in report.get("checks", {}).items() if passed is None],
                      "model_http_statuses": paused["failed_http_statuses"] if paused else
                          sorted({code for error in report.get("model_transport_errors", []) for code in error["http_statuses"]}),
                      "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
