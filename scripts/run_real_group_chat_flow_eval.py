#!/usr/bin/env python3
"""将匿名化真实群消息回放到生产组件并验证状态与动作。"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_runtime import (  # noqa: E402
    AgentRuntime,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    OpenAICompatibleAgentClient,
    UserMessage,
)
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.llm import completion_diagnostics  # noqa: E402
from mahjong_agent_runtime.tracing import JsonlTraceRecorder  # noqa: E402
from mahjong_agent_runtime.group_chat import (  # noqa: E402
    BoardEngine,
    BoardItem,
    ClaimHandler,
    GroupMessage,
    GroupMessageHandler,
    GroupSessionClassifier,
    GroupSessionPipeline,
    L1RuleEngine,
    L2IntentRouter,
    MessageAccumulator,
    NotifyDispatcher,
    OwnerMessageParser,
    QuickFilter,
    ReplyConstraints,
    RoutingDecision,
    SessionCrystallizer,
    SessionMerger,
    SessionRouter,
)
from mahjong_agent_runtime.group_chat.owner_parser import _CHINESE_STAKE  # noqa: E402


TZ = ZoneInfo("Asia/Shanghai")
BASE_TIME = datetime(2026, 7, 22, 17, 0, tzinfo=TZ)
DEFAULT_DATASET = ROOT / "eval" / "golden" / "real_group_chat_20260722.jsonl"
DEFAULT_REPORT = ROOT / "runtime_data" / "real_group_chat_flow_eval_report.json"


def _configured_credentials(client: Any) -> tuple[str, ...]:
    """只读取实际配置的主/回退凭证作精确脱敏，不序列化客户端或请求头。"""
    pending, seen, credentials = [client], set(), set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        key = getattr(getattr(current, "config", None), "api_key", None)
        if isinstance(key, str) and key:
            credentials.add(key)
        pending.extend([getattr(current, "fallback_client", None), getattr(current, "delegate", None)])
    return tuple(sorted(credentials, key=len, reverse=True))


def _safe_evidence(value: Any, credentials: tuple[str, ...]) -> Any:
    """证据精确保留业务内容，仅替换已知凭证回显；原请求与运行时正文不作改写。"""
    if isinstance(value, str):
        for key in credentials:
            value = value.replace(key, "[redacted credential]")
        return value
    if isinstance(value, dict):
        return {_safe_evidence(key, credentials): _safe_evidence(item, credentials) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_evidence(item, credentials) for item in value]
    return value


def _write_evidence(path: Path, payload: Any, credentials: tuple[str, ...]) -> None:
    """写入本次独立目录中的安全快照；原模型输出和历史目录均不被修改。"""
    path.write_text(json.dumps(_safe_evidence(payload, credentials), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _EvidenceTraceRecorder(JsonlTraceRecorder):
    """复用生产追加式追踪，只在写盘前脱敏凭证，避免异常发生后才清理已泄露原迹。"""

    def __init__(self, path: Path, credentials: tuple[str, ...]) -> None:
        """不改变追踪格式或执行边界，也不另建运行时或复制一套追踪存储。"""
        super().__init__(path)
        self.credentials = credentials
        Path(self.path).touch(exist_ok=False)

    def record(self, trace_id: str, step: str, content: dict[str, Any], *, level: str = "INFO") -> None:
        """追加当前事件的脱敏副本，不影响主循环读取的原工具结果。"""
        super().record(_safe_evidence(trace_id, self.credentials), step, _safe_evidence(content, self.credentials), level=level)

    def get_trace(self, trace_id: str):
        """按同一脱敏身份回读已有事件，异常时也不依赖RuntimeResult已经返回。"""
        return super().get_trace(_safe_evidence(trace_id, self.credentials))


def _partial_state(store: Any, room_id: str | None) -> dict[str, Any]:
    """只保存领域计数和本房间公开看板，不把其他私聊、画像或参与者原文写入失败快照。"""
    board = store.get_group_board_state(room_id) if room_id else None
    return {"domain_counts": {"games": len(store.games), "invites": len(store.invite_drafts),
            "outbounds": len(store.outbound_message_drafts), "scheduled_tasks": len(store.scheduled_tasks)},
            "board_items": [project_board_item(item) for item in board.items] if board else []}


@dataclass(slots=True)
class CountingClient:
    """在调用前后追加实际请求/正文和安全诊断；异常不会抹去已经开始的模型请求。"""

    delegate: Any
    calls: list[dict[str, Any]] = field(default_factory=list)
    evidence_path: Path | None = None

    def _save_call(self, call: dict[str, Any]) -> None:
        """模型流水只追加状态快照，不保存config、请求头或提供方独立推理正文。"""
        if self.evidence_path is not None:
            with self.evidence_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(call, ensure_ascii=False) + "\n")

    def complete(self, messages, *, trace_id: str, timeout_seconds: float) -> str:
        """原参数原样委托，证据脱敏和JSON序列化不改变模型实际输入及返回值。"""
        started = time.perf_counter()
        credentials = _configured_credentials(self.delegate)
        call = _safe_evidence({"call_index": len(self.calls) + 1, "trace_id": trace_id, "status": "started",
                              "messages": deepcopy(messages)}, credentials)
        self.calls.append(call)
        self._save_call(call)
        try:
            content = self.delegate.complete(
                messages,
                trace_id=trace_id,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            call.update(_safe_evidence({"status": "error", "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": f"{type(exc).__name__}: {exc}", "completion_diagnostics": completion_diagnostics(exc)}, credentials))
            self._save_call(call)
            raise
        call.update(_safe_evidence({"status": "completed", "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "raw_response": str(content) if isinstance(content, str) else content,
            "completion_diagnostics": completion_diagnostics(content), "error": None}, credentials))
        self._save_call(call)
        return content


@dataclass(slots=True)
class _MainAgentCapture:
    """记录正式群入口传给同一主运行时的信封及结果，不执行额外规划或改写。"""

    runtime: AgentRuntime
    message: UserMessage | None = None
    result: Any = None
    trace_id: str | None = None

    def handle_user_message(self, message: UserMessage, *, trace_id: str):
        """原样委托真实主循环，让报告可审计生产入口实际采用的身份和输入。"""
        self.message = message
        self.trace_id = trace_id
        self.result = self.runtime.handle_user_message(message, trace_id=trace_id)
        return self.result


@dataclass(slots=True)
class _LocalGroupMessenger:
    """在本地记录正式群分支的输出，禁止评测触达外部通道或切到私聊。"""

    messages: list[dict[str, Any]] = field(default_factory=list)

    def send_group_message(self, room_id: str, text: str, *, metadata=None) -> str:
        """只记录本次房间的模拟交付证据，不发送网络请求。"""
        self.messages.append({"room_id": room_id, "text": text, "metadata": dict(metadata or {})})
        return f"eval-local-group-{len(self.messages)}"

    def send_private_message(self, external_user_id: str, text: str, *, metadata=None) -> str:
        """公开查询回放不得悄悄切私聊，选错分支时直接使评测失败。"""
        raise AssertionError("group query evaluation cannot send a private message")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取jsonl，不产生业务副作用。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_number(value: str | None) -> str | None:
    """比较已声明金额字段的等价表示，不从区间、规则码或自由文本猜出底注。"""
    raw_text = str(value or "").strip()
    text = raw_text.removesuffix("块").removesuffix("元")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        # 复用既有老板看板金额协议，但必须完整匹配；search 会误截区间或其他原话。
        matched = _CHINESE_STAKE.fullmatch(raw_text)
        if matched is None:
            return text
        chinese_number = OwnerMessageParser._chinese_number(matched.group(1))
        if chinese_number is None:
            return text
        number = float(chinese_number)
    return str(int(number)) if number.is_integer() else str(number)


def normalize_semantic_value(value: Any, *, path: str) -> Any:
    """按字段比较语义值；金额列表仅规范化元素，不改变原列表或偏好顺序。"""

    field_name = path.rsplit(".", 1)[-1]
    list_field, bracket, index_suffix = field_name.partition("[")
    # 路径由 subset_errors 递归产生；只接受金额多选字段的完整非负整数下标。
    accepted_stake_item = (
        list_field == "accepted_stakes"
        and bracket == "["
        and index_suffix.endswith("]")
        and index_suffix[:-1].isdigit()
    )
    if field_name in {"stake", "stakes"} or accepted_stake_item:
        return normalize_number(value)
    if field_name == "smoking":
        normalized = str(value or "").strip().lower()
        return {
            "no": "无烟",
            "no_smoking": "无烟",
            "non_smoking": "无烟",
            "无烟": "无烟",
            "yes": "有烟",
            "smoking": "有烟",
            "有烟": "有烟",
            "any": "不限",
            "不限": "不限",
            "都可": "不限",
        }.get(normalized, value)
    return value


def project_board_item(item: BoardItem) -> dict[str, Any]:
    return {
        "display_no": item.display_no,
        "game_type": item.game_type or None,
        "ruleset": item.ruleset,
        "participant_code": item.participant_code,
        "current_players": item.current_players,
        "missing_players": item.missing_players,
        "start_time": None if item.time == "人齐开" else item.time,
        "time_mode": "asap_when_full" if item.time == "人齐开" else "scheduled",
        "end_time": item.end_time,
        "duration_hours": item.duration_hours,
        "stake": normalize_number(item.stakes),
        "rule_code": item.rule_code,
        "smoking": item.smoking,
        "special_rules": [item.special_rules] if item.special_rules else [],
        "temporary_constraints": list(item.temporary_constraints),
        "status": item.status,
    }


def subset_errors(expected: Any, actual: Any, *, path: str = "actual") -> list[str]:
    """将期望值作为部分契约进行比较，允许实际数据包含额外字段。"""

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, got {type(actual).__name__}"]
        errors: list[str] = []
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
                continue
            errors.extend(subset_errors(value, actual[key], path=f"{path}.{key}"))
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected list, got {type(actual).__name__}"]
        if len(actual) < len(expected):
            return [f"{path}: expected at least {len(expected)} items, got {len(actual)}"]
        errors: list[str] = []
        for index, value in enumerate(expected):
            errors.extend(subset_errors(value, actual[index], path=f"{path}[{index}]"))
        return errors
    normalized_expected = normalize_semantic_value(expected, path=path)
    normalized_actual = normalize_semantic_value(actual, path=path)
    return (
        []
        if normalized_expected == normalized_actual
        else [f"{path}: expected {expected!r}, got {actual!r}"]
    )


class RealGroupChatFlowEvaluator:
    """驱动每个Gold用例依次经过解析器、会话分类器和可选的主Agent。"""

    def __init__(self, *, llm_client: Any | None, quiet_seconds: float = 10.0, evidence_dir: Path | None = None) -> None:
        """可选证据目录由CLI每次唯一分配；未提供时保留测试所需的内联追踪。"""
        self.llm_client = llm_client
        self.quiet_seconds = quiet_seconds
        self.evidence_dir = evidence_dir

    def evaluate(
        self,
        records: list[dict[str, Any]],
        *,
        dataset_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """逐例隔离异常并立即归档，某例失败不能丢掉前例结果或阻止后续独立用例。"""
        started = time.perf_counter()
        reports = []
        for index, record in enumerate(records, start=1):
            case_dir = None
            try:
                if self.evidence_dir is not None:
                    case_dir = self.evidence_dir / f"case_{index:04d}"
                    case_dir.mkdir(exist_ok=False)
                item = self.evaluate_case(record, evidence_dir=case_dir)
            except Exception as exc:
                # 包括用例包装或留证失败也必须失败，不能用其余断言成功掩盖异常。
                item = {"case_id": record.get("id"), "case_type": record.get("case_type"), "status": "failed",
                        "errors": [_safe_evidence(f"{type(exc).__name__}: {exc}", _configured_credentials(self.llm_client))]}
            reports.append(item)
            if self.evidence_dir is not None:
                _write_evidence(self.evidence_dir / "progress.json", {"status": "running", "cases": reports}, _configured_credentials(self.llm_client))
        executed = [item for item in reports if item["status"] != "skipped"]
        passed = [item for item in executed if item["status"] == "passed"]
        return {
            "dataset": str(dataset_path or DEFAULT_DATASET),
            "generated_at": datetime.now(TZ).isoformat(),
            "mode": "live" if self.llm_client is not None else "deterministic",
            "quiet_seconds": self.quiet_seconds,
            "evidence_dir": str(self.evidence_dir) if self.evidence_dir is not None else None,
            "summary": {
                "total": len(reports),
                "executed": len(executed),
                "passed": len(passed),
                "failed": len(executed) - len(passed),
                "skipped": len(reports) - len(executed),
                "pass_rate": round(len(passed) / len(executed), 4) if executed else 0.0,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            },
            "cases": reports,
        }

    def evaluate_case(self, case: dict[str, Any], *, evidence_dir: Path | None = None) -> dict[str, Any]:
        """先归档原用例，再捕获本例全部异常与已有状态；不执行任何异常后的补偿业务动作。"""
        store = InMemoryAgentStore()
        credentials = _configured_credentials(self.llm_client)
        counting = CountingClient(self.llm_client, evidence_path=evidence_dir / "model_calls.jsonl" if evidence_dir else None) if self.llm_client is not None else None
        try:
            if evidence_dir is not None:
                _write_evidence(evidence_dir / "source_case.json", case, credentials)
            result = self._evaluate_case(case, store=store, counting=counting, evidence_dir=evidence_dir)
        except Exception as exc:
            result = {"case_id": case.get("id"), "case_type": case.get("case_type"), "status": "failed",
                      "expected": case.get("expected"), "errors": [f"{type(exc).__name__}: {exc}"],
                      "actual": {"partial_state": _partial_state(store, (case.get("source") or {}).get("room_alias"))},
                      "model_calls": list(counting.calls) if counting is not None else []}
        result = _safe_evidence(result, credentials)
        if evidence_dir is not None:
            result["evidence_dir"] = str(evidence_dir)
            _write_evidence(evidence_dir / "result.json", result, credentials)
        return result

    def _evaluate_case(self, case: dict[str, Any], *, store: Any, counting: Any, evidence_dir: Path | None) -> dict[str, Any]:
        """沿用既有生产回放及断言，只把外部留证与异常隔离放在外层。"""
        started = time.perf_counter()
        expected = dict(case["expected"])
        semantic_required = expected.get("model_call_required") is not False and case["case_type"] in {
            "member_query",
            "fragmented_input",
        }
        if semantic_required and self.llm_client is None:
            return {
                "case_id": case["id"],
                "case_type": case["case_type"],
                "status": "skipped",
                "reason": "requires live model",
                "elapsed_ms": 0.0,
            }

        pipeline = self._pipeline(store, counting, case)
        ids: dict[str, str] = {}
        actions: list[str] = []
        outcomes = []
        before_snapshots: list[list[dict[str, Any]]] = []

        self._seed_precondition(case, pipeline, ids)
        self._seed_quoted_source(case, pipeline, ids)

        for index, raw in enumerate(case["messages"]):
            sent_at = BASE_TIME + timedelta(seconds=int(raw["offset_seconds"]))
            outcomes.extend(pipeline.flush_due(at=sent_at))
            message = self._message(case, raw, index=index, sent_at=sent_at, ids=ids)
            before = store.get_group_board_state(message.room_id)
            if before is not None:
                before_snapshots.append([project_board_item(item) for item in before.items])
            result = pipeline.accept(message, trace_id=f"trace_real_group_{case['id']}_{index}")
            actions.append(result.action)
            alias = raw.get("message_alias")
            if alias:
                ids[str(alias)] = message.message_id

        final_at = BASE_TIME + timedelta(
            seconds=max(int(item["offset_seconds"]) for item in case["messages"]) + self.quiet_seconds + 0.1
        )
        outcomes.extend(pipeline.flush_due(at=final_at))
        board = store.get_group_board_state(case["source"]["room_alias"])
        actual = {
            "pipeline_actions": actions,
            "board_items": [project_board_item(item) for item in board.items] if board else [],
            "session_count": len(pipeline.session_router.list_sessions(case["source"]["room_alias"])),
            "session_outcomes": [
                {
                    "action": item.action,
                    "session_id": item.session_id,
                    "classification": item.classification.to_dict() if item.classification else None,
                    "detail": item.detail,
                }
                for item in outcomes
            ],
            "model_call_count": len(counting.calls) if counting is not None else 0,
            "main_agent_invocation_count": 0,
            "main_agent": None,
        }

        if semantic_required and outcomes and counting is not None:
            actual["main_agent"] = self._run_main_agent(
                case=case,
                store=store,
                pipeline=pipeline,
                outcome=outcomes[-1],
                llm=counting,
                evidence_dir=evidence_dir,
            )
            actual["main_agent_invocation_count"] = 1
            actual["model_call_count"] = len(counting.calls)

        errors = self._check_case(
            case,
            expected=expected,
            actual=actual,
            before_snapshots=before_snapshots,
            pipeline=pipeline,
        )
        if actual.get("main_agent", {}) and actual["main_agent"].get("error"):
            errors.append("main Agent raised an exception: " + actual["main_agent"]["error"])
        elif actual.get("main_agent", {}) and actual["main_agent"].get("status") not in {"completed", "waiting_user"}:
            # Runtime可能已安全吞下提供方异常并返回技术终态；这仍不是该评测用例通过。
            errors.append("main Agent did not complete a valid evaluation turn: " + str(actual["main_agent"].get("status")))
        return {
            "case_id": case["id"],
            "case_type": case["case_type"],
            "status": "passed" if not errors else "failed",
            "expected": expected,
            "actual": actual,
            "errors": errors,
            "model_calls": list(counting.calls) if counting is not None else [],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _pipeline(self, store, llm, case) -> GroupSessionPipeline:
        owners = {
            str(item["actor_alias"])
            for item in case["messages"]
            if item["role"] == "operator"
        }
        classifier = GroupSessionClassifier(llm, timeout_seconds=30) if llm is not None else _NoModelClassifier()
        router = SessionRouter(clock=lambda: BASE_TIME)
        return GroupSessionPipeline(
            store=store,
            owner_parser=OwnerMessageParser(owner_external_ids=owners | {"seed_owner"}),
            quick_filter=QuickFilter(),
            accumulator=MessageAccumulator(
                quiet_seconds=self.quiet_seconds,
                continuation_seconds=120,
            ),
            session_router=router,
            session_merger=SessionMerger(router),
            crystallizer=SessionCrystallizer(),
            classifier=classifier,
        )

    @staticmethod
    def _seed_precondition(case, pipeline, ids: dict[str, str]) -> None:
        items = (case.get("precondition") or {}).get("board_items") or []
        if not items:
            return
        text = "\n".join(str(item["text"]) for item in items)
        message = GroupMessage(
            room_id=case["source"]["room_alias"],
            conversation_id=f"wechaty:room:{case['source']['room_alias']}",
            sender_external_id="seed_owner",
            sender_name="seed_owner",
            text=text,
            message_id=f"seed:{case['id']}",
            sent_at=BASE_TIME - timedelta(seconds=1),
            metadata={"is_owner": True},
        )
        pipeline.accept(message, trace_id=f"trace_seed_{case['id']}")
        ids["precondition"] = message.message_id

    @staticmethod
    def _seed_quoted_source(case, pipeline, ids: dict[str, str]) -> None:
        quoted = next((item for item in case["messages"] if item.get("quoted_text")), None)
        if quoted is None:
            return
        quoted_id = f"quoted:{case['id']}"
        message = GroupMessage(
            room_id=case["source"]["room_alias"],
            conversation_id=f"wechaty:room:{case['source']['room_alias']}",
            sender_external_id="seed_owner",
            sender_name="seed_owner",
            text=str(quoted["quoted_text"]),
            message_id=quoted_id,
            sent_at=BASE_TIME - timedelta(seconds=1),
            metadata={"is_owner": True},
        )
        pipeline.accept(message, trace_id=f"trace_quote_seed_{case['id']}")
        ids["quoted_source"] = quoted_id

    @staticmethod
    def _message(case, raw, *, index: int, sent_at: datetime, ids: dict[str, str]) -> GroupMessage:
        message_id = f"real:{case['id']}:{index}"
        metadata: dict[str, Any] = {"content_type": raw.get("content_type", "text")}
        if raw["role"] == "operator":
            metadata["is_owner"] = True
        if raw.get("quoted_text"):
            metadata["quoted_text"] = raw["quoted_text"]
        target_alias = raw.get("target_message_alias")
        if target_alias:
            metadata["target_message_id"] = ids.get(str(target_alias), str(target_alias))
        quoted_message_id = ids.get("quoted_source") if raw.get("quoted_text") else None
        return GroupMessage(
            room_id=case["source"]["room_alias"],
            conversation_id=f"wechaty:room:{case['source']['room_alias']}",
            sender_external_id=str(raw.get("actor_alias") or raw["role"]),
            sender_name=str(raw.get("actor_alias") or raw["role"]),
            text=str(raw["text"]),
            message_id=message_id,
            sent_at=sent_at,
            quoted_message_id=quoted_message_id,
            metadata=metadata,
        )

    @staticmethod
    def _run_main_agent(*, case, store, pipeline, outcome, llm, evidence_dir: Path | None = None) -> dict[str, Any]:
        """把单客户聚合原文交给正式群答分支，保存完整原始主循环证据。

        候选 Session 的分类仍单独验收，不把其自述、回复或整群历史当作主循环事实。
        这里只复用已经选中的生产 agent_loop 分支，不再跑一遍词法路由；主循环
        可见的公开看板仍由原 ContextBuilder 从受控 Game 投影构建。
        """
        credentials = _configured_credentials(llm)
        trace = (_EvidenceTraceRecorder(evidence_dir / "main_trace.jsonl", credentials)
                 if evidence_dir is not None else InMemoryTraceRecorder())
        messenger = _LocalGroupMessenger()
        capture, result, error = None, None, None
        try:
            session = next(item for item in pipeline.session_router.list_sessions(case["source"]["room_alias"])
                           if item.id == outcome.session_id)
            source_message = session.messages[-1]
            runtime = AgentRuntime(
                llm_client=llm, store=store, trace_recorder=trace,
                # 主循环沿用应用配置；候选 Session 分类的独立入口预算保持原值。
                llm_timeout_seconds=float(os.getenv("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", "45")),
                customer_visible_text_generation_enabled=False, reply_self_review_enabled=False,
            )
            capture = _MainAgentCapture(runtime)
            board = BoardEngine(store=store, messenger=messenger)
            notifier = NotifyDispatcher(store=store, messenger=messenger, runtime=capture)
            handler = GroupMessageHandler(
                rule_engine=L1RuleEngine(bot_id="eval-local-bot"), intent_router=L2IntentRouter(store, llm=llm),
                board_engine=board, claim_handler=ClaimHandler(board_engine=board, store=store, notify_dispatcher=notifier),
                notify_dispatcher=notifier, messenger=messenger, runtime=capture,
            )
            handler._handle_agent_loop(source_message,
                RoutingDecision(action="agent_loop", channel="group", reply_constraints=ReplyConstraints(max_length=20)),
                trace_id=f"trace_main_{case['id']}")
            result = capture.result
            if result is None or capture.message is None:
                raise RuntimeError("production group entry did not invoke the main Agent")
        except Exception as exc:
            # 只停止本次执行并读取已有证据；不补跑工具、不发兜底消息，也不把异常变成成功。
            error = _safe_evidence(f"{type(exc).__name__}: {exc}", credentials)
        if result is None and capture is not None:
            # 群入口可能在Runtime已返回后失败；保留确实捕获到的结果，不能伪装成未执行。
            result = capture.result
        trace_id = getattr(result, "trace_id", None) or (capture.trace_id if capture else None) or f"trace_main_{case['id']}"
        events = trace.get_trace(trace_id)
        tool_results = ([item.to_dict() for item in result.tool_results] if result is not None else
                        [item.content for item in events if item.step == "tool_result"])
        report = {
            "trace_id": trace_id,
            "input_message": capture.message.to_dict() if capture and capture.message else None,
            "status": "failed" if error else result.status,
            "captured_result_status": getattr(result, "status", None),
            "error": error,
            "final_reply": result.final_reply if result is not None else "",
            "tool_calls": [item.get("name") for item in tool_results if item.get("called")],
            "tool_results": tool_results,
            "trace_steps": [item.step for item in events],
            "trace_events": [item.to_dict() for item in events],
            "group_messages": messenger.messages,
            "partial_state": _partial_state(store, case["source"]["room_alias"]),
        }
        if evidence_dir is not None:
            report["raw_trace_path"] = str(evidence_dir / "main_trace.jsonl")
            _write_evidence(evidence_dir / "main_agent.json", report, credentials)
        return _safe_evidence(report, credentials)

    @staticmethod
    def _check_case(case, *, expected, actual, before_snapshots, pipeline) -> list[str]:
        errors: list[str] = []
        case_type = case["case_type"]
        items = actual["board_items"]
        if "items" in expected:
            if case_type == "owner_board_increment":
                by_number = {item["display_no"]: item for item in items}
                for expected_item in expected["items"]:
                    display_no = expected_item.get("display_no")
                    errors.extend(
                        subset_errors(
                            expected_item,
                            by_number.get(display_no),
                            path=f"board_items[display_no={display_no}]",
                        )
                    )
            else:
                errors.extend(subset_errors(expected["items"], items, path="board_items"))
        if case_type == "board_state_diff":
            if len(items) != 1 and expected.get("must_not_create_duplicate_game"):
                errors.append(f"board_items: expected one logical item, got {len(items)}")
            if before_snapshots:
                errors.extend(subset_errors(expected["before"], before_snapshots[-1][0], path="before"))
            if items:
                errors.extend(subset_errors(expected["after"], items[0], path="after"))
        if case_type == "quoted_state_update":
            if len(items) != 1:
                errors.append(f"quoted update expected one item, got {len(items)}")
            elif items[0]["status"] != "full" or items[0]["missing_players"] != 0:
                errors.append("quoted update did not mark the target game full")
            else:
                errors.extend(subset_errors(expected["target_requirement"], items[0], path="target"))
        if case_type == "quoted_requirement_update":
            if len(items) != 1:
                errors.append(f"quoted requirement update expected one logical item, got {len(items)}")
            elif expected.get("must_not_create_duplicate_game"):
                errors.extend(subset_errors(expected["after"], items[0], path="after"))
        if case_type == "quick_filter":
            if actual["model_call_count"] != expected["model_call_count"]:
                errors.append("noise reached the model")
            if actual["session_count"] != expected["session_count_delta"]:
                errors.append("noise created a session")
            if any(action != "filtered" for action in actual["pipeline_actions"]):
                errors.append(f"noise actions were {actual['pipeline_actions']!r}")
        if case_type == "message_revoke":
            sessions = pipeline.session_router.list_sessions(case["source"]["room_alias"])
            leaked = any("人" in message.text for session in sessions for message in session.messages)
            if leaked:
                errors.append("revoked text remains in session state")
        if case_type in {"member_query", "fragmented_input"}:
            classifications = [
                item["classification"] for item in actual["session_outcomes"] if item["classification"] is not None
            ]
            if not classifications:
                errors.append("semantic message produced no session classification")
                return errors
            final = classifications[-1]
            if final["intent"] != expected["intent"]:
                errors.append(f"intent: expected {expected['intent']!r}, got {final['intent']!r}")
            features = final["extracted_features"]
            requirement = expected.get("search_requirement") or {}
            classifier_requirement = {
                ("stakes" if key == "stake" else key): value
                for key, value in requirement.items()
            }
            errors.extend(
                subset_errors(
                    classifier_requirement,
                    features,
                    path="classification.extracted_features",
                )
            )
            for key in ("current_players", "missing_players", "urgency"):
                if key in expected and features.get(key) != expected[key]:
                    errors.append(f"{key}: expected {expected[key]!r}, got {features.get(key)!r}")
            for key in expected.get("must_not_invent_fields") or []:
                feature_key = "stakes" if key == "stake" else key
                if features.get(feature_key) not in (None, "", []):
                    errors.append(f"model invented {key}={features.get(feature_key)!r}")
            if expected.get("must_not_invent_smoking_or_time"):
                for key in ("smoking", "time"):
                    if features.get(key) not in (None, "", []):
                        errors.append(f"model invented {key}={features.get(key)!r}")
            for key in expected.get("missing_fields") or []:
                feature_key = "stakes" if key == "stake" else key
                if features.get(feature_key) not in (None, "", []):
                    errors.append(f"expected {key} to remain missing, got {features.get(feature_key)!r}")
            if expected.get("same_session") and actual["session_count"] != 1:
                errors.append(f"expected one session, got {actual['session_count']}")
            if expected.get("process_after_aggregation") and len(classifications) != 1:
                errors.append(f"expected one post-aggregation classification, got {len(classifications)}")
            if "main_agent_invocation_count" in expected and (
                actual["main_agent_invocation_count"] != expected["main_agent_invocation_count"]
            ):
                errors.append(
                    "main Agent invocation count: "
                    f"expected {expected['main_agent_invocation_count']}, "
                    f"got {actual['main_agent_invocation_count']}"
                )
            if expected.get("preferred_channel_action") and (
                final["channel_action"] != expected["preferred_channel_action"]
            ):
                errors.append(
                    f"channel_action: expected {expected['preferred_channel_action']!r}, "
                    f"got {final['channel_action']!r}"
                )
            main = actual.get("main_agent") or {}
            tool_calls = main.get("tool_calls") or []
            for name in expected.get("required_tool_calls") or []:
                if name not in tool_calls:
                    errors.append(f"main Agent did not call required tool {name}")
            for name in expected.get("forbidden_tool_calls") or []:
                if name in tool_calls:
                    errors.append(f"main Agent called forbidden tool {name}")
        return errors


class _NoModelClassifier:
    """在确定性回放中禁止意外调用远程模型的测试替身。"""

    def classify(self, **_kwargs):
        raise AssertionError("deterministic case unexpectedly reached the model")


def parse_args() -> argparse.Namespace:
    """解析args，并把非结构化输入转换为稳定合同。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--live", action="store_true", help="run semantic cases through the configured model and main Agent")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when any executed case fails")
    parser.add_argument("--quiet-seconds", type=float, default=10.0)
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".local/share/mahjong-ops-agent/.env")
    return parser.parse_args()


def main() -> int:
    """每次分配唯一证据目录，逐例归档后保留既有report_path汇总接口。"""
    args = parse_args()
    report_path = args.report_path.expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_parent = report_path.parent / "group_eval_evidence"
    evidence_parent.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now(TZ).strftime("%Y%m%dT%H%M%S%f_"), dir=evidence_parent))
    _write_evidence(run_dir / "run.json", {"status": "started", "dataset": str(args.dataset), "live": args.live}, ())
    llm = None
    if args.live:
        load_dotenv_defaults(args.env_file.expanduser())
        llm = OpenAICompatibleAgentClient.from_env()
        if llm is None:
            raise RuntimeError("live mode requires MAHJONG_LLM_MODEL and provider credentials")
    evaluator = RealGroupChatFlowEvaluator(llm_client=llm, quiet_seconds=args.quiet_seconds, evidence_dir=run_dir)
    report = evaluator.evaluate(read_jsonl(args.dataset), dataset_path=args.dataset)
    report["report_path"] = str(report_path)
    _write_evidence(run_dir / "summary.json", report, _configured_credentials(llm))
    _write_evidence(report_path, report, _configured_credentials(llm))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.strict and report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
