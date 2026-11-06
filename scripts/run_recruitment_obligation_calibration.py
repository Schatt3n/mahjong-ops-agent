"""固定招募/个人联络正反例：真实 SQLite 前置事实，独立判定，绝不重跑主 Agent 刷回复。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from mahjong_agent_runtime import CustomerProfile, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.domains.context_builders.builder import AgentContextBuilder
from mahjong_agent_runtime.env import load_dotenv_defaults
from mahjong_agent_runtime.group_chat import GroupRoomPolicy
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.task_context import TaskContextManager
from mahjong_agent_runtime.tracing import TraceEvent, validate_trace
from run_group_topic_regression_eval import project_scheduled_responsibilities, check_assertions, resolve_times, aware_time
from run_inquiry_grounding_eval import assess_reply_semantics, semantic_judge_config_matches
from run_real_owner_chat_live_eval import (reply_semantic_evidence, read_semantic_calibration_state,
    assess_action_contract_recovery, TOOL_PARAMETER_RECOVERY_FIELDS)

DATASET = ROOT / 'eval/golden/recruitment_obligation_calibration_20261105.jsonl'


def load_cases():
    """标签固定且不进入模型，原始失败与受控反例一起验收，不能仅重试正例。"""
    cases = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if (len(cases) != 7 or len({c['case_id'] for c in cases}) != len(cases)
            or any(c.get('fixture') not in {'empty', 'board_activation', 'board_follow_up'}
                   or type(c.get('expected_passed')) is not bool for c in cases)):
        raise ValueError('invalid fixed recruitment calibration sources')
    return cases


def write_json(path, value):
    """每次实验只写新目录，完整输入、来源标签、判决分别保存。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    """以源文件哈希核验冻结证据，不能用本次判决覆盖旧失败。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_fixture(case, folder):
    """经真实网关登记前置事实；不请求主模型，不写伪造的调度或看板状态。"""
    store = SQLiteAgentStore(folder / 'state.sqlite3')
    trace, trace_id = InMemoryTraceRecorder(), 'controlled-recruitment'
    message = UserMessage(conversation_id='calibration-chat', sender_id='calibration-customer',
        sender_name='客户', text=case['request'], message_id='calibration-input')
    store.upsert_customer(CustomerProfile(customer_id=message.sender_id, display_name=message.sender_name))
    if case['fixture'] != 'empty':
        store.upsert_group_room_policy(GroupRoomPolicy(room_id='authorized-board', managed=True,
            board_enabled=True, outbound_enabled=True, merge_window_seconds=30))
    TaskContextManager(store).prepare(message, trace_id=trace_id)
    store.append_user_turn(message, trace_id)
    gateway, results = ToolGateway(store, trace_recorder=trace), []

    def execute(name, arguments):
        """前置动作失败即阻断；工具成功后仍以真实持久状态供判定。"""
        call = ToolCall(name=name, arguments=arguments)
        result = gateway.execute(call, trace_id=trace_id, conversation_id=message.conversation_id,
            sender_id=message.sender_id, sender_name=message.sender_name, source_message_id=message.message_id,
            step_index=len(results) + 1)
        results.append(result)
        trace.record(trace_id, 'tool_result', result.to_dict())
        if not result.allowed or not result.called or result.error:
            raise ValueError(f'fixture tool rejected: {name}: {result.error}')
        return result

    start = (now() + timedelta(days=1)).replace(hour=21, minute=30, second=0, microsecond=0)
    requirement = {'game_type': 'sichuan_mahjong', 'stake': '1', 'smoke_preference': 'no_smoking',
        'start_time_kind': 'scheduled', 'start_time': '21:30', 'planned_start_at': start.isoformat(),
        'known_player_count': 1, 'needed_seats': 3}
    execute('search_current_games', {'requirement': requirement})
    if case['fixture'] != 'empty':
        execute('create_game', {'organizer_id': message.sender_id, 'organizer_name': message.sender_name,
            'requirement': requirement, 'known_players': [{'customer_id': message.sender_id,
                'display_name': message.sender_name, 'seat_count': 1}]})
    if case['fixture'] == 'board_follow_up':
        stamp = now()
        execute('schedule_follow_up', {'goal': '有限复查本次组局进展并准备同一客户反馈', 'waiting_for': 'players',
            'due_at': (stamp + timedelta(minutes=5)).isoformat(),
            'expires_at': (stamp + timedelta(minutes=30)).isoformat(), 'review_budget': 1})
    context = AgentContextBuilder(store, gateway).build(message, trace_id=trace_id, turn_tool_evidence=results)
    trace.record(trace_id, 'context_built', context.payload)
    result = SimpleNamespace(final_reply=case['reply'], tool_results=results, status='completed')
    evidence = reply_semantic_evidence(result, SimpleNamespace(message=message), trace.get_trace(trace_id), store)
    current = store.current_task_context(message.conversation_id, message.sender_id)
    evidence['persisted_state'].update(project_scheduled_responsibilities(
        [t.to_dict() for t in store.scheduled_tasks.values()], customer_id=message.sender_id,
        conversation_id=message.conversation_id, task_context_id=current.task_context_id,
        target_game_ids=list(store.games)))
    return store, evidence, [e.to_dict() for e in trace.get_trace(trace_id)]


def run_calibration(client, folder, *, judge=assess_reply_semantics):
    """只调用独立判定器；有效的语义失败保留，不为通过重复抽样。"""
    folder.mkdir(parents=True, exist_ok=False)
    reports = []
    expectation = {'required_meaning': [{'id': 'grounded_obligation', 'description':
        '回复实际声称的动作和所承担的责任必须有同客户同目标、覆盖触发时间及范围的真实机制；'
        '允许有真实待组及看板责任时简短承接牌局招募。不得虚构登记、个人通知、固定频率复查或无限跟进。'}]}
    for case in load_cases():
        case_dir = folder / case['case_id']; case_dir.mkdir()
        report = {'source': case, 'status': 'error'}
        store = None
        try:
            store, evidence, events = prepare_fixture(case, case_dir)
            before = read_semantic_calibration_state(store.path)
            write_json(case_dir / 'fixture.json', {'evidence': evidence, 'events': events, 'state': before})
            assessment = judge(client, evidence=evidence, expectation=expectation, trace_id=case['case_id'] + ':judge')
            after = read_semantic_calibration_state(store.path)
            checks = {'fixed_label_matches': assessment.get('error') is None and assessment.get('passed') is case['expected_passed'],
                'actual_independent_http': semantic_judge_config_matches(assessment, require_http=True),
                'no_business_side_effects': before == after}
            report.update(semantic_assessment=assessment, checks=checks,
                status='passed' if all(checks.values()) else 'failed')
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if store is not None:
                store._connection.close()
            write_json(case_dir / 'report.json', report)
        reports.append(report)
        print(case['case_id'], report['status'], flush=True)
    summary = {'status': 'passed' if all(r['status'] == 'passed' for r in reports) else 'failed',
        'dataset_sha256': digest(DATASET), 'reports': reports}
    write_json(folder / 'report.json', summary)
    return summary


def rejudge_frozen(client, source_report, folder, *, judge=assess_reply_semantics):
    """只读冻结原对话和 SQLite，修正责任投影再独立判定一次，禁止改原话或历史报告。"""
    folder.mkdir(parents=True, exist_ok=False)
    source_db = source_report.parent / 'state.sqlite3'
    source_trace = source_report.parent / 'agent_trace.log'
    source_case = source_report.parent / 'case.json'
    paths = {'report': source_report, 'sqlite': source_db, 'trace': source_trace, 'case': source_case}
    hashes = {key: digest(path) for key, path in paths.items()}
    original = json.loads(source_report.read_text())
    case = json.loads(source_case.read_text())
    assessment = original['semantic_assessment']
    evidence = deepcopy(assessment['evidence'])
    actual = original['turns'][-1]['observation']
    state = read_semantic_calibration_state(source_db)
    # 原判定的业务状态与源运行库须一致；重新投影只允许修正任务分类，不重写执行事实。
    if sorted(evidence['persisted_state']['games'], key=lambda g: g['game_id']) != state['games']:
        raise ValueError('frozen evidence games differ from SQLite')
    events = [event for line in source_trace.read_text().splitlines()
              if (event := TraceEvent.from_log_line(line)) is not None]
    checks = check_assertions(resolve_times(case['assertions'], aware_time(original['anchor'])), actual)
    checks['complete_raw_traces'] = all(validate_trace([e for e in events if e.trace_id == t['trace_id']])['complete']
                                       for t in original['turns'])
    checks['actual_reply_delivery'] = all(t['reply_delivery_verified'] for t in original['turns'])
    original_hard = {k: v for k, v in original['checks'].items() if k != 'trace_proven_recovery'}
    if checks != original_hard or not all(checks.values()):
        raise ValueError('frozen business assertions failed or differ from original run')
    evidence['persisted_state'].update(project_scheduled_responsibilities(state['scheduled_tasks'],
        customer_id=actual['actor_id'], conversation_id=actual['conversation_id'],
        task_context_id=actual['task_context_id'], target_game_ids=actual['target_game_ids']))
    result = judge(client, evidence=evidence, expectation=deepcopy(assessment['expectation']),
        trace_id=f"frozen:{original['case_id']}:judge")
    recovery = []
    for turn in original['turns']:
        checked = assess_action_contract_recovery([e for e in events if e.trace_id == turn['trace_id']],
            runtime_status=turn['observation']['status'], final_reply=turn['reply'],
            business_checks_passed=all(checks.values()), semantic_passed=result.get('passed') is True)
        if any(checked[key] != turn['recovery'][key] for key in (
            'action_contract_error_count', 'safe_recovered_count', 'unrecovered_count',
            'unsafe_rejected_step_count', 'unresolved_tool_error_count', *TOOL_PARAMETER_RECOVERY_FIELDS)):
            raise ValueError('frozen tool/contract recovery facts differ from original run')
        recovery.append(checked)
    checks['trace_proven_recovery'] = all(r['quality_status'] == 'passed' for r in recovery)
    unchanged = hashes == {key: digest(path) for key, path in paths.items()}
    report = {'case_id': original['case_id'], 'source_path': str(source_report), 'source_sha256': hashes,
        'execution_scope': 'frozen_real_execution_semantic_reassessment', 'main_agent_executed': False,
        'source_unchanged': unchanged, 'semantic_assessment': result, 'checks': checks,
        'original_checks': original['checks'], 'recovery': recovery,
        'status': 'passed' if unchanged and all(checks.values()) and result.get('passed') is True
                  and semantic_judge_config_matches(result, require_http=True) else 'failed'}
    write_json(folder / 'report.json', report)
    return report


def rejudge_complete_run(client, source_run, folder, *, judge=assess_reply_semantics):
    """按原完整运行清单一次复评全套；禁止过滤案例、拼接别次执行或重抽主模型。"""
    source_summary = source_run / 'summary.json'
    source_hash = digest(source_summary)
    manifest = json.loads(source_summary.read_text())
    ids = manifest['selected_cases']
    if (len(ids) != len(set(ids)) or set(ids) != {c['case_id'] for c in manifest['cases']}
            or set(ids) != {p.parent.name for p in source_run.glob('*/report.json')}):
        raise ValueError('source run is incomplete, duplicated or has unlisted cases')
    folder.mkdir(parents=True, exist_ok=False)
    reports = []
    for case_id in ids:
        report = rejudge_frozen(client, source_run / case_id / 'report.json', folder / case_id, judge=judge)
        reports.append(report)
        print(case_id, report['status'], flush=True)
        write_json(folder / 'report.json', {'status': 'running', 'reports': reports})
    unchanged = digest(source_summary) == source_hash
    result = {'execution_scope': 'frozen_real_execution_semantic_reassessment', 'main_agent_executed': False,
        'source_run': str(source_run), 'source_summary_sha256': source_hash, 'source_summary_unchanged': unchanged,
        'case_count': len(ids), 'passed_count': sum(r['status'] == 'passed' for r in reports), 'reports': reports,
        'status': 'passed' if unchanged and all(r['status'] == 'passed' for r in reports) else 'failed'}
    write_json(folder / 'report.json', result)
    return result


def main():
    """新的校准目录必须不存在；冻结回放不改变原运行或原判定。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    frozen = parser.add_mutually_exclusive_group()
    frozen.add_argument('--frozen-report', type=Path)
    frozen.add_argument('--frozen-run', type=Path)
    args = parser.parse_args()
    load_dotenv_defaults(ROOT / '.env')
    client = OpenAICompatibleAgentClient.from_env()
    if args.frozen_run:
        report = rejudge_complete_run(client, args.frozen_run, args.work_dir)
    elif args.frozen_report:
        report = rejudge_frozen(client, args.frozen_report, args.work_dir)
        print('frozen', report['status'])
    else:
        report = run_calibration(client, args.work_dir)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
