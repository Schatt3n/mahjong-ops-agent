"""提供隔离的手动聊天室，复用正式 Runtime 和模拟证据格式，不连接微信。"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from simulation_observability import DEFAULT_RUNTIME_DIR, DEFAULT_ENV_FILE, ROOT, _atomic_write_json, _read_env_metadata, _safe_run_id

SIMULATION_PATH = ROOT / 'tests' / 'simulation'
if str(SIMULATION_PATH) not in sys.path:
    sys.path.insert(0, str(SIMULATION_PATH))
from behavior_policy import SimulationAction
from sim_adapter import RequestOutcome, build_runtime, simulation_execution_observation
from sim_factory import build_population
from sim_orchestrator import outcome_to_transcript_entry
from mahjong_agent_runtime import CustomerProfile, SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.follow_up import handle_follow_up_task
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler

_MANUAL_LOCK = threading.Lock()
MANUAL_CUSTOMER_ID = 'manual_customer'


def handle_manual_simulation(payload: dict, *, runtime_dir: Path | None = None, mode: str | None = None) -> dict:
    """执行白名单手动操作；写入仅限本服务创建的测试库，原运行不允许改写。"""
    action = str(payload.get('action') or '')
    if action not in {'create', 'send', 'recall', 'advance'}:
        raise ValueError('unsupported manual simulation action')
    request_id = str(payload.get('request_id') or '')
    if not request_id or len(request_id)>100:
        raise ValueError('缺少有效请求编号')
    base = (runtime_dir or DEFAULT_RUNTIME_DIR).expanduser().resolve()
    with _MANUAL_LOCK:
        if action == 'create':
            return _create_session(base, payload, request_id, mode)
        run_id = _safe_run_id(str(payload.get('run_id') or ''))
        path = base / 'runs' / run_id / 'report.json'
        if not run_id or not path.is_file():
            raise ValueError('找不到手动测试记录')
        with (path.parent / 'manual.lock').open('a+') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            report = json.loads(path.read_text())
            if not report.get('manual_session'):
                raise ValueError('只能操作手动测试会话，历史自动测试记录只读')
            previous = report.get('manual_requests', {}).get(request_id)
            if previous is not None:
                return {'ok': True, 'run_id': run_id, 'sequence': previous, 'deduplicated': True}
            store = SQLiteAgentStore(path.parent / 'test_sim.db')
            try:
                runtime, _ = build_runtime(store, report['llm_mode'])
                return _operate(runtime, path, report, payload, request_id)
            finally:
                # 每次请求用持久化数据库恢复上下文，避免进程缓存成为第二事实源或泄漏连接。
                store._connection.close()



def _create_session(base: Path, payload: dict, request_id: str, mode: str | None) -> dict:
    """创建固定种子沙盒和无默认玩法的手动客户；创建重试复用同一运行。"""
    digest = hashlib.sha256(request_id.encode()).hexdigest()[:16]
    runs = base / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    existing = list(runs.glob(f'*_manual_{digest}'))
    if existing:
        return {'ok': True, 'run_id': existing[0].name, 'manual_session': True}
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_manual_' + digest
    folder = runs / run_id
    folder.mkdir()
    store, _ = build_population(folder / 'test_sim.db', seed=42)
    try:
        name = str(payload.get('display_name') or '手动测试客户').strip()[:30] or '手动测试客户'
        store.upsert_customer(CustomerProfile(customer_id=MANUAL_CUSTOMER_ID, display_name=name, gender='未知',
                                             preferred_games=[], preferred_stakes=[], smoke_preference=None))
        selected_mode = mode or _read_env_metadata(DEFAULT_ENV_FILE).get('MAHJONG_PERIODIC_SIM_LLM_MODE', 'real')
        runtime, client = build_runtime(store, selected_mode)
        report = {'run_id': run_id, 'manual_session': True, 'manual_customer_name': name,
                  'status': 'interactive', 'quality_status': 'unknown', 'llm_mode': selected_mode,
                  'main_agent_model': getattr(getattr(client, 'config', None), 'model', None),
                  'main_agent_runtime_config': {'max_tokens_per_call': runtime.token_budget.max_tokens_per_call},
                  'message_generation_mode': 'manual', 'started_at': now().isoformat(),
                  'configuration': {'seed': 42, 'channel': 'private', 'external_channel': False},
                  'total_messages': 0, 'transcript': [], 'manual_requests': {}}
        _atomic_write_json(folder / 'report.json', report)
        return {'ok': True, 'run_id': run_id, 'manual_session': True}
    finally:
        store._connection.close()


def _operate(runtime, path: Path, report: dict, payload: dict, request_id: str) -> dict:
    """串行提交消息或真实撤回事件，追加证据后原子更新审阅投影。"""
    run_id = report['run_id']
    conversation_id = f'manual:{run_id}'
    command = payload['action']
    sequence = len(report['transcript'])+1
    text = str(payload.get('text') or '').strip()
    if command == 'send' and (not text or len(text)>4000):
        raise ValueError('消息须为1至4000个字符')
    target = None
    if command == 'recall':
        target = next((turn for turn in report['transcript'] if turn['sequence']==payload.get('sequence') and turn['event_type']=='text'
                       and turn['user']['generation']['source']=='manual'), None)
        if target is None:
            raise ValueError('只能撤回自己发送的消息')
        if target.get('recalled'):
            return {'ok': True, 'run_id': run_id, 'already_recalled': True}
        text = '撤回了一条消息'

    def record(result, *, content: str, event_type: str, source: str, latency_ms: float):
        """复用自动模拟的审计格式，明确手动输入和系统复查的来源。"""
        action = SimulationAction(due_simulated_seconds=0, sequence=len(report['transcript'])+1, channel='private',
                                  conversation_id=conversation_id, sender_id=MANUAL_CUSTOMER_ID,
                                  sender_name=report['manual_customer_name'], text=content, thread_id=conversation_id,
                                  event_type=event_type, generation_source=source)
        response = result.to_dict()
        response['simulation_observation'] = simulation_execution_observation(runtime, result.trace_id)
        response['operational_state'] = {'follow_ups': [task.to_dict() for task in runtime.store.scheduled_tasks.values()
                                                       if task.conversation_id==conversation_id and task.subject_id==MANUAL_CUSTOMER_ID]}
        entry = outcome_to_transcript_entry(RequestOutcome(action=action, sent=True, status_code=200, response=response, latency_ms=latency_ms))
        if event_type=='recall' and target is not None:
            entry['recalled_message_id']=target['message_id']
        report['transcript'].append(entry)
        with (path.parent/'live_events.jsonl').open('a') as output:
            output.write(json.dumps(entry,ensure_ascii=False)+'\n')

    if command == 'advance':
        pending = [task for task in runtime.store.scheduled_tasks.values() if task.task_type=='conversation_follow_up'
                   and task.conversation_id==conversation_id and task.subject_id==MANUAL_CUSTOMER_ID and task.status.value=='pending']
        if not pending:
            return {'ok': True, 'run_id': run_id, 'message': '当前没有待复查任务'}
        due = min(task.due_at for task in pending)
        def trigger(task, trace_id):
            """只触发本沙盒的复查，不改变宿主时钟或伪造其他客户活动。"""
            started = time.monotonic()
            result = handle_follow_up_task(runtime, task, trace_id, at=due)
            if result is not None:
                record(result,content='（测试触发：到达复查时间）',event_type='follow_up_timeout',source='scheduler',latency_ms=(time.monotonic()-started)*1000)
        scheduler = ScheduledAgentTaskScheduler(runtime.store, trigger, runtime.trace_recorder)
        scheduler.run_task_once(min(pending,key=lambda task:task.due_at).task_id,at=due)
        report['manual_virtual_time'] = due.isoformat()
    else:
        action = SimulationAction(due_simulated_seconds=0, sequence=sequence, channel='private', conversation_id=conversation_id,
                                  sender_id=MANUAL_CUSTOMER_ID, sender_name=report['manual_customer_name'], text=text)
        metadata = {'channel':'console', 'simulation':True}
        if command=='recall':
            metadata.update(event_type='message_recalled',target_message_id=target['message_id'])
        stamp = max(now(), datetime.fromisoformat(report.get('manual_virtual_time') or now().isoformat()))
        started = time.monotonic()
        result = runtime.handle_user_message(UserMessage(conversation_id=conversation_id,sender_id=MANUAL_CUSTOMER_ID,
                                                         sender_name=report['manual_customer_name'], text='' if command=='recall' else text,
                                                         message_id=action.message_id,sent_at=stamp,metadata=metadata))
        record(result,content=text,event_type='recall' if command=='recall' else 'text',source='manual',latency_ms=(time.monotonic()-started)*1000)
        if target is not None:
            target['recalled'] = True
    report['manual_requests'][request_id] = report['transcript'][-1]['sequence']
    report['total_messages'] = len(report['transcript'])
    failures = [turn for turn in report['transcript'] if turn['agent']['status'] in {'needs_help','needs_human'} or turn['agent']['objective_status']=='needs_tool']
    report.update(status='interactive',quality_status='degraded' if failures else 'passed',agent_internal_failure_count=len(failures),
                  updated_at=now().isoformat(),pending_follow_up_count=sum(task.task_type=='conversation_follow_up' and task.status.value=='pending' for task in runtime.store.scheduled_tasks.values()))
    _atomic_write_json(path, report)
    return {'ok': True, 'run_id':run_id, 'sequence':report['transcript'][-1]['sequence']}
