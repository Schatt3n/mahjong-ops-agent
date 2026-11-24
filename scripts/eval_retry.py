"""评测沙盒的用例补试账本，保留每次尝试和同轮续跑点。"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import monotonic, sleep
from typing import Any

from mahjong_agent_runtime.llm import (
    ExternalCallBudgetExhausted, completion_diagnostics, eval_transport_failure, external_call_deadline,
)

BLOCKED_EXIT_CODE = 75
SCOPE_ENV = 'MAHJONG_EVAL_RETRY_SCOPE'
WINDOW_ENV = 'MAHJONG_EVAL_RETRY_WINDOW'
MAX_RETRIES = 3
RETRY_SECONDS = 360.0
DEFERRED_AUXILIARY_JUDGE = {'stage': 'judge', 'category': 'evaluation_retry_budget', 'retryable': True,
                            'reason': 'shared_retry_window_exhausted_before_judge'}
_budget: ContextVar[CaseBudget | None] = ContextVar('eval_retry_budget', default=None)
_defer_retries: ContextVar[bool] = ContextVar('eval_defer_retries', default=False)


def write_json(path: Path, data: Any) -> None:
    """同目录原子替换，保证崩溃后读到完整的旧账本或新账本。"""
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def is_inconclusive(failure: Any) -> bool:
    """只接受评测器已从原始诊断证明的暂态故障。"""
    return (isinstance(failure, dict) and failure.get('retryable') is True
            and failure.get('stage') in {'agent', 'judge'}
            and isinstance(failure.get('category'), str) and isinstance(failure.get('reason'), str))


class CaseBudget:
    """同一验证单元的 Agent 和判定器共享次数、追加时限和追加记录。"""

    def __init__(self, folder: Path, window: int):
        self.folder = folder
        self.window = window
        self.deadline: float | None = None
        self.retries = 0
        self.last_failure: dict | None = None

    def remaining_seconds(self) -> float:
        """首次运行使用现有预算，补试开始后使用单调时钟的共同截止时间。"""
        return math.inf if self.deadline is None else max(0.0, self.deadline - monotonic())

    def limit_timeout(self, seconds: float) -> float:
        """保留调用自身预算，同时保证每次请求都不跨过共同截止时间。"""
        return min(seconds, self.remaining_seconds())

    def begin_retry(self, failure: dict, *, stage: str) -> bool:
        """先落盘扣减再调用，退避也占用补试预算。"""
        failure = {**failure, 'stage': stage}
        if _defer_retries.get() or not is_inconclusive(failure):
            return False
        self.last_failure = failure
        if self.retries >= MAX_RETRIES or self.remaining_seconds() <= 0:
            return False
        if self.deadline is None:
            self.deadline = monotonic() + RETRY_SECONDS
            external_call_deadline.set(self.deadline)
        self.retries += 1
        write_json(self.folder / f'window_{self.window:03d}_retry_{self.retries:02d}.json', failure)
        sleep(min(2 ** (self.retries - 1), self.remaining_seconds()))
        return self.remaining_seconds() > 0


def current_budget() -> CaseBudget | None:
    """判定器和调用限制器只在门禁沙盒用例内消费补试额度。"""
    return _budget.get()


def _auxiliary_messages_digest(messages: list[dict[str, str]]) -> str:
    """绑定消息顺序和已序列化的每条原请求，不因报告重排 JSON 键改变证据。"""
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _defer_auxiliary_judgment(assessment: dict, messages: list[dict[str, str]], budget: CaseBudget) -> None:
    """关联本用例此前落盘的非递归重试原因，不为尚未发出的判定伪造 HTTP 证据。"""
    for path in sorted(budget.folder.glob('window_*_retry_*.json'), reverse=True):
        trigger = json.loads(path.read_text(encoding='utf-8'))
        if is_inconclusive(trigger) and trigger != DEFERRED_AUXILIARY_JUDGE:
            assessment['frozen_judge_messages'] = messages
            assessment['deferred_judge'] = {'retry_trigger': trigger, 'trigger_record': path.name,
                                            'messages_sha256': _auxiliary_messages_digest(messages)}
            assessment['eval_inconclusive'] = dict(DEFERRED_AUXILIARY_JUDGE)
            return


def resumable_auxiliary_failure(frozen: dict, expected: dict, messages: list[dict[str, str]], *,
                                parse, safe_error) -> dict:
    """核对待补判的精确请求、初判合同状态和结构化传输证据，拒绝重打已有有效结论。"""
    budget = current_budget()
    if budget is None or any(frozen.get(key) != expected[key]
            for key in ('input', 'input_sha256', 'rubric', 'trace_id', 'judge_config', 'requested_timeout_seconds')):
        raise ValueError('auxiliary judge frozen candidate or assessment input differs')
    attempts = frozen.get('attempts')
    if (not isinstance(attempts, list) or len(attempts) > 1 or frozen.get('contract_repair_count') != len(attempts)
            or frozen.get('frozen_judge_messages') != messages):
        raise ValueError('auxiliary judge frozen request or contract stage differs')
    if attempts:
        first = attempts[0]
        if first.get('trace_id') != expected['trace_id'] or first.get('raw_response') != frozen.get('raw_response'):
            raise ValueError('auxiliary judge invalid-response evidence differs')
        try:
            parse(first['raw_response'])
        except (ValueError, TypeError) as exc:
            if first.get('error') != safe_error(exc):
                raise ValueError('auxiliary judge contract error differs') from exc
        else:
            raise ValueError('an effective auxiliary judgment cannot be retried')
    elif frozen.get('raw_response') is not None:
        raise ValueError('auxiliary judge has an unverified response')
    marker = frozen.get('eval_inconclusive')
    if marker == DEFERRED_AUXILIARY_JUDGE:
        deferred = frozen.get('deferred_judge') or {}
        trigger = deferred.get('retry_trigger')
        records = {path.name: path for path in budget.folder.glob('window_*_retry_*.json')}
        source = records.get(deferred.get('trigger_record'))
        if (source is None or not is_inconclusive(trigger) or trigger == DEFERRED_AUXILIARY_JUDGE
                or deferred.get('messages_sha256') != _auxiliary_messages_digest(messages)
                or json.loads(source.read_text(encoding='utf-8')) != trigger):
            raise ValueError('deferred auxiliary judgment lacks its earlier retry evidence')
        return marker
    failures = frozen.get('transport_failures') or []
    last = failures[-1] if failures else {}
    failure = eval_transport_failure(last.get('completion_diagnostics'))
    expected_trace = expected['trace_id'] + (':contract_repair_1' if attempts else '')
    if (not failure or failure.get('retryable') is not True or last.get('failure') != failure
            or last.get('trace_id') != expected_trace or last.get('messages_sha256') != _auxiliary_messages_digest(messages)
            or marker != {**failure, 'stage': 'judge'}):
        raise ValueError('auxiliary judge interruption lacks matching HTTP evidence')
    return failure


def complete_frozen_auxiliary_judgment(judge, messages: list[dict[str, str]], *, trace_id: str,
                                       timeout_seconds: float, assessment: dict, safe_error,
                                       retry_from: dict | None = None) -> str:
    """只在评测沙盒以同一条冻结输入补试传输；实际响应继续由原分类及合同验证器处理。"""
    budget = current_budget()
    if budget is None:
        return judge.complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
    frozen = deepcopy(messages)
    failure = retry_from
    if failure is not None and not budget.begin_retry(failure, stage='judge'):
        assessment['eval_inconclusive'] = {**failure, 'stage': 'judge'}
        raise RuntimeError('auxiliary judge retry budget exhausted')
    assessment.pop('eval_inconclusive', None)
    while True:
        effective_timeout = budget.limit_timeout(timeout_seconds)
        if effective_timeout <= 0:
            if failure:
                assessment['eval_inconclusive'] = {**failure, 'stage': 'judge'}
            else:
                _defer_auxiliary_judgment(assessment, frozen, budget)
            raise RuntimeError('auxiliary judge retry time budget exhausted')
        try:
            return judge.complete(deepcopy(frozen), trace_id=trace_id, timeout_seconds=effective_timeout)
        except Exception as exc:
            if isinstance(exc, ExternalCallBudgetExhausted):
                if failure:
                    # 补试在发出 HTTP 前到期，继续引用本判定已有的原始证据。
                    assessment['eval_inconclusive'] = {**failure, 'stage': 'judge'}
                else:
                    _defer_auxiliary_judgment(assessment, frozen, budget)
                raise
            failure = eval_transport_failure(exc)
            if not failure or failure.get('retryable') is not True:
                raise
            assessment['frozen_judge_messages'] = frozen
            assessment.setdefault('transport_failures', []).append({
                'trace_id': trace_id, 'completion_diagnostics': completion_diagnostics(exc),
                'error': safe_error(exc), 'failure': failure, 'messages_sha256': _auxiliary_messages_digest(frozen),
            })
            if not budget.begin_retry(failure, stage='judge'):
                assessment['eval_inconclusive'] = {**failure, 'stage': 'judge'}
                raise


@contextmanager
def defer_case_retries():
    """并发工作线程记录超时与共享时限，补试只由协调线程串行扣减。"""
    token = _defer_retries.set(True)
    try:
        yield
    finally:
        _defer_retries.reset(token)


def scope_directory() -> Path | None:
    """独立执行回放时保留原有命令行为，统一门禁提供独立账本路径。"""
    directory = os.environ.get(SCOPE_ENV)
    return Path(directory) if directory else None


def _save_final(folder: Path, result: dict, attempt_dir: Path) -> dict:
    """只保存有明确结论的当前用例；失败也冻结，避免续跑挑最好结果。"""
    final = {**result, 'attempt_dir': str(attempt_dir)}
    write_json(folder / 'final.json', final)
    return final


def checkpointed_case(key: str, run, *, case_dir: Path, resume_judge=None) -> dict:
    """在本轮验证内复用已定论用例，按失败阶段补试；每次执行写到新目录。"""
    scope = scope_directory()
    if scope is None:
        return run(case_dir)
    window = int(os.environ[WINDOW_ENV])
    folder = scope / hashlib.sha256(key.encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'final.json').exists():
        return json.loads((folder / 'final.json').read_text(encoding='utf-8'))
    blocked_path = folder / 'blocked.json'
    prior = json.loads(blocked_path.read_text(encoding='utf-8')) if blocked_path.exists() else None
    if prior and (prior['window'] >= window or (prior['result']['eval_inconclusive']['stage'] == 'judge' and not resume_judge)):
        write_json(scope / 'blocked.json', {**prior['result']['eval_inconclusive'], 'key': key, 'window': window})
        return prior['result']
    budget = CaseBudget(folder, window)
    token = _budget.set(budget)
    deadline_token = external_call_deadline.set(None)
    latest = prior['result'] if prior else None
    try:
        while True:
            failure = latest['eval_inconclusive'] if latest else None
            judge_resume = failure and failure['stage'] == 'judge'
            # 首次尝试无额外扣减。判定器补判入口自己扣减共享预算。
            if failure and not judge_resume and not budget.begin_retry(failure, stage='agent'):
                break
            if judge_resume and (not resume_judge or budget.retries >= MAX_RETRIES or budget.remaining_seconds() <= 0):
                break
            attempt_dir = Path(tempfile.mkdtemp(prefix=f'window_{window:03d}_attempt_', dir=folder)) / 'case'
            result = resume_judge(latest, attempt_dir) if judge_resume else run(attempt_dir)
            result = {**result, 'attempt_dir': str(attempt_dir)}
            attempt_dir.mkdir(parents=True, exist_ok=True)
            write_json(attempt_dir / 'eval_result.json', result)
            if result.get('status') == 'blocked' and not is_inconclusive(result.get('eval_inconclusive')):
                raise ValueError('blocked case lacks verified transport evidence')
            if result.get('status') != 'blocked':
                return _save_final(folder, result, attempt_dir)
            latest = result
            # 共享判定器在调用过程中已经按照本窗口耗尽条件处理；下个窗口才能补判。
            if result['eval_inconclusive']['stage'] == 'judge':
                break
        write_json(blocked_path, {'window': window, 'result': latest})
        write_json(scope / 'blocked.json', {**latest['eval_inconclusive'], 'key': key, 'window': window})
        return latest
    finally:
        external_call_deadline.reset(deadline_token)
        _budget.reset(token)


def agent_failure_diagnosis(events) -> dict | None:
    """根据终止证据区分本地执行、工具边界与外部请求失败。"""
    entries = [event if isinstance(event, dict) else event.to_dict() for event in events]
    open_tools = set()
    for event in entries:
        content = event['content']
        boundary = (content.get('tool_name'), content.get('step_index'), content.get('idempotency_key'))
        if event['step'] == 'tool_gateway_received':
            open_tools.add(boundary)
        elif event['step'] == 'tool_gateway_completed':
            open_tools.discard(boundary)
    if open_tools:
        return {'stage': 'agent', 'category': 'local_tool', 'retryable': False, 'reason': 'tool_execution_unverified'}
    if any(event['step'] == 'budget_checked' and event['content'].get('allowed') is False for event in entries):
        return {'stage': 'agent', 'category': 'local_budget', 'retryable': False, 'reason': 'turn_budget_rejected'}
    llm_events = [event for event in entries if event['step'] in {'llm_error', 'llm_request_started', 'llm_response'}]
    if not llm_events or llm_events[-1]['step'] != 'llm_error':
        return None
    content = llm_events[-1]['content']
    budget = current_budget()
    if content.get('error_type') == 'ExternalCallBudgetExhausted' and budget:
        return budget.last_failure
    failure = eval_transport_failure(content.get('completion_diagnostics'))
    return {**failure, 'stage': 'agent'} if failure else None


def latest_agent_transport_failure(events, *, read_only_tools=frozenset()) -> dict | None:
    """仅在无未核验业务副作用、无有效失败时允许重新初始化该沙盒用例。"""
    entries = [event if isinstance(event, dict) else event.to_dict() for event in events]
    if any(event['step'] in {'action_contract_error',
                            'customer_visible_generation_started', 'customer_visible_review_started',
                            'external_message_sent'} for event in entries):
        # 通过边界证据的独立多轮回放可以只传入出错轮，前序已验证轨迹仍原样保留。
        return None
    if not _verified_read_only_execution(entries, read_only_tools):
        return None
    failure = agent_failure_diagnosis(entries)
    return failure if is_inconclusive(failure) else None


def _verified_read_only_execution(entries, read_only_tools) -> bool:
    """实际网关轨迹需与只读声明和返回结果完全对应，副作用结果不明时拒绝补试。"""
    received, completed, results = Counter(), Counter(), Counter()
    started_batches, completed_batches = Counter(), Counter()
    for event in entries:
        step, content = event['step'], event['content']
        if step in {'tool_gateway_received', 'tool_gateway_completed', 'tool_result'}:
            name = content.get('tool_name') if step != 'tool_result' else content.get('name')
            key = content.get('idempotency_key')
            if name not in read_only_tools or not isinstance(key, str) or not key:
                return False
            if step == 'tool_gateway_received':
                received[name, key] += 1
            else:
                if content.get('called') is not True or content.get('allowed') is not True or content.get('error'):
                    return False
                (completed if step == 'tool_gateway_completed' else results)[name, key] += 1
        elif step in {'tool_batch_started', 'tool_batch_completed'}:
            names = content.get('tool_names', [])
            calls = tuple(content.get('call_ids', []))
            if not calls or len(calls) != len(names) or any(name not in read_only_tools for name in names):
                return False
            if step == 'tool_batch_completed' and content.get('succeeded') != dict.fromkeys(calls, True):
                return False
            counts = started_batches if step == 'tool_batch_started' else completed_batches
            counts[(content.get('step_index'), content.get('wave_index'), calls)] += 1
    return received == completed == results and started_batches == completed_batches
