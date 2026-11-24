"""真实群语义入口的独立回放；原始追踪、SQLite终态和独立语义判定分别保存。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests/simulation'))

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.group_chat import ChannelIdentity, GroupRoomPolicy
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.task_context import TaskContextManager
from mahjong_agent_runtime.tracing import TraceEvent, validate_trace
from eval_retry import BLOCKED_EXIT_CODE, checkpointed_case, latest_agent_transport_failure, scope_directory, write_json
from manual_scenario import validate_scenario
from manual_group_simulation import process_manual_group
from sim_adapter import build_runtime
from run_inquiry_grounding_eval import assess_reply_semantics, resume_reply_semantics, _semantic_criteria_ids
from run_real_owner_chat_live_eval import (
    reply_semantic_evidence, assess_action_contract_recovery, eval_values_equal, read_semantic_calibration_state,
)

DEFAULT_DATASET = ROOT / 'eval/golden/group_delegation_acceptance_20261105.jsonl'
QUERY_POLICY_PATH = ROOT / 'eval/handoffs/20261109_group_query_positive_control_policy.json'
QUERY_POLICY_SHA256 = '536b8d07ade5f969e64d640d4de9d1f2ef7bccade039d2dd63fcb568599c02ed'
DELIVERY_POLICY_PATH = ROOT / 'eval/handoffs/20261123_group_topic_silent_delivery.json'
DELIVERY_POLICY_SHA256 = '4ec9a5bbb227affb9e676c2dc1350fc3c20d89de883499e74ffc1ca5aa6ef8eb'
ASSERTION_KEYS = {'target_message_index', 'expected_intent', 'expected_intent_any', 'expected_runtime_channel',
    'required_tools', 'forbidden_tools', 'require_persisted_recruitment', 'require_board_responsibility',
    'no_business_writes', 'requester_not_joined', 'requester_joined_seeded', 'expected_status', 'expected_conditions',
    'forbidden_invite_actor_ids', 'require_same_goal_continuation', 'require_participation_deadline',
    'expected_requester_game_count', 'require_same_game', 'deadline_clock'}


def successful(tool: dict) -> bool:
    """被拒或报错的工具提议不能充当已完成业务动作。"""
    return tool.get('called') is True and tool.get('allowed') is True and not tool.get('error')


def aware_time(value):
    """只接受带时区 ISO 证据，并按绝对时刻比较；不截掉日期或从自然语言猜时刻。"""
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
        return parsed if parsed is not None and parsed.tzinfo is not None else None
    except ValueError:
        return None


def project_scheduled_responsibilities(tasks: list[dict], *, customer_id: str, conversation_id: str,
                                       task_context_id: str | None, target_game_ids: list[str]) -> dict:
    """按真实任务类型及本人目标投影责任，不能把任意同客户任务叫作参与截止。

    已取消/已执行等终态仍保留供判定器核对，不能静默改成有效承诺。个人复查已有
    reply_semantic_evidence.follow_up_state 的专用投影，此处只补牌局激活和参与截止。
    """
    result = {'participation_deadlines': [], 'recruitment_activations': []}
    if not task_context_id:
        return result
    target_ids = set(target_game_ids)
    for task in tasks:
        payload = task.get('payload') or {}
        game_id = payload.get('game_id')
        if (task.get('subject_id') != customer_id or task.get('conversation_id') != conversation_id
                or payload.get('task_context_id') != task_context_id or game_id not in target_ids):
            continue
        if (task.get('task_type') == 'activate_game_recruitment' and task.get('aggregate_type') == 'game'
                and task.get('aggregate_id') == game_id):
            bucket = 'recruitment_activations'
            fields = ('event_type', 'game_id', 'task_context_id', 'planned_start_at', 'recruitment_opens_at')
        elif (task.get('task_type') == 'participation_deadline' and task.get('aggregate_type') == 'game_participant'
                and payload.get('customer_id') == customer_id and task.get('aggregate_id') == f'{game_id}:{customer_id}'):
            bucket = 'participation_deadlines'
            fields = ('game_id', 'customer_id', 'task_context_id', 'source_message_id', 'deadline_at',
                      'condition', 'effect', 'execution_outcome')
        else:
            continue
        result[bucket].append({**{key: deepcopy(task[key]) for key in (
            'task_id', 'task_type', 'aggregate_type', 'aggregate_id', 'conversation_id', 'subject_id',
            'status', 'due_at', 'lease_until', 'completed_at') if key in task},
            'payload': {key: deepcopy(payload[key]) for key in fields if key in payload}})
    return result


def matching_deadline_tasks(actual: dict) -> list[dict]:
    """核对当前本人、目标、牌局、原始消息和成功工具返回，任意 pending 不能充数。"""
    actor, conversation = actual.get('actor_id'), actual.get('conversation_id')
    context, source = actual.get('task_context_id'), actual.get('message_id')
    targets = set(actual.get('target_game_ids', [])) & {g['game_id'] for g in actual['own_games']}
    expected = aware_time(actual.get('expected_deadline_at'))
    if not all((actor, conversation, context, source, targets, expected)):
        return []
    evidence = [tool.get('result', {}).get('participation_deadline', {}) for tool in actual['tools']
                if successful(tool) and tool.get('name') == 'set_participation_deadline']
    valid = []
    for task in actual['deadline_tasks']:
        payload = task.get('payload') or {}
        if (task.get('task_type') != 'participation_deadline' or task.get('status') not in {'pending', 'processing'}
                or task.get('subject_id') != actor or task.get('conversation_id') != conversation
                or payload.get('customer_id') != actor or payload.get('task_context_id') != context
                or payload.get('source_message_id') != source or payload.get('game_id') not in targets
                or payload.get('condition') != 'game_not_ready' or payload.get('effect') != 'withdraw_self'
                or payload.get('execution_outcome') or aware_time(payload.get('deadline_at')) != expected
                or aware_time(task.get('due_at')) != expected):
            continue
        if task.get('status') == 'processing':
            lease = aware_time(task.get('lease_until'))
            observed = aware_time(actual.get('observed_at'))
            if lease is None or observed is None or lease <= observed:
                continue
        # 工具成功本身不足以证明最终仍有效；任务ID及关键责任字段必须与权威终态一致。
        if any(t.get('task_id') == task.get('task_id') and t.get('subject_id') == actor
               and t.get('conversation_id') == conversation and t.get('payload') == payload
               and aware_time(t.get('due_at')) == expected for t in evidence):
            valid.append(task)
    return valid


def check_assertions(rules: dict, actual: dict) -> dict[str, bool]:
    """逐项核对执行事实；未知规则直接拒绝，不能因拼错字段漏掉硬断言。"""
    unknown = set(rules) - ASSERTION_KEYS
    if unknown:
        raise ValueError(f'unknown assertion: {sorted(unknown)}')
    tools = [t for t in actual['tools'] if successful(t)]
    names = {t['name'] for t in tools}
    games = actual['own_games']
    checks = {'runtime_completed_without_error': actual['status'] in {'completed', 'waiting_user'}}
    if 'expected_intent_any' in rules:
        checks['expected_intent'] = actual['intent'] in rules['expected_intent_any']
    elif 'expected_intent' in rules:
        checks['expected_intent'] = actual['intent'] == rules['expected_intent']
    for rule, field in [('expected_runtime_channel', 'runtime_channel'), ('expected_status', 'status')]:
        if rule in rules:
            checks[rule] = actual[field] == rules[rule]
    if 'required_tools' in rules:
        checks['required_tools'] = set(rules['required_tools']) <= names
    if 'forbidden_tools' in rules:
        checks['forbidden_tools'] = not names.intersection(rules['forbidden_tools'])
    if rules.get('require_persisted_recruitment'):
        checks['require_persisted_recruitment'] = any(g.get('status') in {'forming', 'inviting', 'ready'} for g in games)
    if rules.get('require_board_responsibility'):
        game_ids = {g['game_id'] for g in games}
        checks['require_board_responsibility'] = any(a.get('game_id') in game_ids and
            a.get('linked_room_count', 0) > 0 and a.get('outbound_authorized_room_count', 0) > 0 and
            (a.get('publication_sent') is True or a.get('pending_task_count', 0) > 0)
            for a in actual['board_arrangements'])
    if rules.get('no_business_writes'):
        definitions = ToolGateway(InMemoryAgentStore()).tools
        allowed = {name for name, definition in definitions.items() if definition.execution_mode in {'read_only', 'audit_write'}}
        allowed |= {'record_user_memory', 'update_context_checkpoint', 'customer_visible_content_review', 'customer_visible_text_generation'}
        checks['no_business_writes'] = names <= allowed and actual['new_game_count'] == 0
    if rules.get('requester_not_joined'):
        checks['requester_not_joined'] = not games and not actual['requester_joined_seeded']
    if rules.get('requester_joined_seeded'):
        checks['requester_joined_seeded'] = bool(games) and actual['requester_joined_seeded'] is True
    if 'expected_requester_game_count' in rules:
        checks['expected_requester_game_count'] = len(games) == rules['expected_requester_game_count']
    if 'expected_conditions' in rules:
        expected = rules['expected_conditions']
        checks['expected_conditions'] = bool(games) and all(all(eval_values_equal(
            (sum(p.get('seat_count', 0) for p in g.get('participants', [])
                 if p.get('customer_id') == actual.get('actor_id') and p.get('status') in {'joined', 'confirmed'}) if k == 'party_size' else
             g.get('planned_start_at') if k == 'planned_start_at' else g.get('requirement', {}).get(k)), v, field_path=k)
            for k, v in expected.items()) for g in games)
    if 'forbidden_invite_actor_ids' in rules:
        checks['forbidden_invite_actor_ids'] = not set(rules['forbidden_invite_actor_ids']).intersection(actual.get('invited_actor_ids', []))
    if rules.get('require_same_goal_continuation'):
        checks['require_same_goal_continuation'] = bool(actual.get('goal_relation') == 'continue'
            and actual.get('classified_task_context_id') in actual.get('previous_task_context_ids', [])
            and actual.get('task_context_id') and actual.get('source_bound_to_task') is True
            and set(actual.get('previous_source_message_ids', [])) & set(actual.get('inherited_source_message_ids', [])))
    if rules.get('require_same_game'):
        checks['require_same_game'] = bool(actual.get('previous_own_game_ids')) and {g['game_id'] for g in games} == set(actual['previous_own_game_ids'])
    if rules.get('require_participation_deadline'):
        tasks = matching_deadline_tasks(actual)
        checks['require_participation_deadline'] = bool(tasks)
        if 'deadline_clock' in rules:
            expected = aware_time(actual.get('expected_deadline_at'))
            checks['deadline_clock'] = bool(tasks) and expected.strftime('%H:%M') == rules['deadline_clock']
    return checks


def apply_query_positive_control_policy(path: Path, raw: bytes, cases: list[dict]) -> list[dict]:
    """仅给指定原资产追加明确向店内询局的后继输入，原断言和历史证据不变。"""
    policy_raw = QUERY_POLICY_PATH.read_bytes()
    if hashlib.sha256(policy_raw).hexdigest() != QUERY_POLICY_SHA256:
        raise ValueError('group query policy changed without review')
    policy = json.loads(policy_raw)
    source = next((item for item in policy['datasets']
                   if (ROOT / item['path']).resolve() == path.resolve()), None)
    if source is None:
        return cases
    if hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('group query policy original dataset changed')
    current = deepcopy(cases)
    for revision in source['revisions']:
        matches = [case for case in current if case['case_id'] == revision['case_id']]
        if len(matches) != 1:
            raise ValueError('group query policy requires one original case')
        case = matches[0]
        canonical = json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(canonical).hexdigest() != revision['original_case_sha256']:
            raise ValueError('group query policy original case changed')
        message = case['scenario']['messages'][revision['message_index']]
        if message['text'] != revision['previous_text']:
            raise ValueError('group query policy previous input differs')
        message['text'] = revision['current_text']
        # 后继输入和原话明确分开追溯；元数据不进入给模型的 scenario。
        case['input_revision'] = {'revision_id': policy['revision_id'], 'policy_sha256': QUERY_POLICY_SHA256,
            'original_dataset': source['path'], 'original_dataset_sha256': source['sha256'], **deepcopy(revision)}
    return current


def apply_silent_delivery_policy(path: Path, raw: bytes, cases: list[dict]) -> None:
    """给固定源资产补充外发验收，保留原始案例和输入修订。"""
    policy_raw = DELIVERY_POLICY_PATH.read_bytes()
    if hashlib.sha256(policy_raw).hexdigest() != DELIVERY_POLICY_SHA256:
        raise ValueError('group delivery policy changed without review')
    policy = json.loads(policy_raw)
    dataset = next((item for item in policy['datasets'] if (ROOT / item['path']).resolve() == path.resolve()), None)
    if dataset is None:
        return
    if hashlib.sha256(raw).hexdigest() != dataset['sha256']:
        raise ValueError('group delivery policy original dataset changed')
    by_id = {case['case_id']: case for case in cases}
    for case_id, indices in dataset['silent_private_turns'].items():
        case = by_id.get(case_id)
        if not case or not all(type(index) is int and 0 <= index < len(case['scenario']['messages']) for index in indices):
            raise ValueError('group delivery policy references missing source turns')
        case['silent_private_turns'] = indices


def verify_intentional_silence(turn: dict, events: list) -> bool:
    """同时核对本轮显式静默终态和空收件箱，区分主动静默、未生成和被拦截。"""
    if turn['reply'] != '' or turn['outputs'] != [] or turn['observation']['status'] != 'completed':
        return False
    outputs = [event.content for event in events if event.step == 'final_output' and event.trace_id == turn['trace_id']]
    if len(outputs) != 1:
        return False
    final = outputs[0]
    reason = final.get('silence_reason')
    return final.get('response_mode') == 'silent' and (
        final.get('reply') == '' and final.get('objective_status') == 'completed'
        and isinstance(reason, str) and bool(reason.strip())
    )


def verify_reply_delivery(turn: dict, events: list, *, expect_silent_private: bool = False) -> bool:
    """核对已送达回复的收件人；指定静默回合须有来源绑定的路由及零外发证据。"""
    outputs = turn.get('outputs')
    if not isinstance(outputs, list):
        return False
    actual, source = turn['observation'], turn['group_input']
    private = actual['runtime_channel'] == 'private'
    if expect_silent_private:
        receipts = [event.content for event in events if event.step == 'group_eval_private_delivery']
        return (private and actual['intent'] in {'delegate', 'create'} and actual['goal_relation'] == 'new'
                and not outputs and len(receipts) == 1
                and receipts[0] == {'action': 'private_switch', 'source_message_id': source['message_id'],
                    'customer_id': actual['actor_id'], 'private_conversation_id': actual['conversation_id'],
                    'private_reply_suppressed': bool(turn['reply']), 'private_delivery_id': '', 'output_count': 0})
    if not turn['reply']:
        return verify_intentional_silence(turn, events)
    recipient = actual['actor_id'] if private else source['room_id']
    return any(output.get('channel') == actual['runtime_channel']
        and output.get('text') == turn['reply'] and output.get('recipient_id') == recipient
        and (output.get('metadata') or {}).get('source_message_id') == source['message_id']
        for output in outputs)


def load_cases(path: Path) -> list[dict]:
    """在任何模型调用前验证全部案例、人工标准及场景输入。"""
    raw = path.read_bytes()
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    cases = apply_query_positive_control_policy(path, raw, cases)
    apply_silent_delivery_policy(path, raw, cases)
    ids = set()
    for case in cases:
        if case['case_id'] in ids or not case['case_id'].replace('_', '').isalnum():
            raise ValueError('duplicate or invalid case_id')
        ids.add(case['case_id'])
        validate_scenario(case['scenario'])
        _semantic_criteria_ids(case['expectation'])
        if set(case['assertions']) - ASSERTION_KEYS:
            raise ValueError('unknown assertion in dataset')
        if case['assertions']['target_message_index'] != len(case['scenario']['messages']) - 1:
            raise ValueError('target must be the last message; subsequent inputs cannot repair a failed target')
    return cases


def resolve_times(value, anchor):
    """只展开评测资产的显式时间占位符，不解析或改写客户自然语言。"""
    if isinstance(value, dict):
        return {k: resolve_times(v, anchor) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_times(v, anchor) for v in value]
    if isinstance(value, str) and value.startswith('$tomorrow_'):
        hour, minute = map(int, value.removeprefix('$tomorrow_').split(':'))
        return (anchor + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat()
    return value


def seed_games(store, profiles: dict, setup: dict, *, run_id: str) -> list[dict]:
    """通过真实工具建前置局，并核对实际席位和时间；前置错误须在请求模型前失败。"""
    gateway, fixtures = ToolGateway(store), []
    for index, seed in enumerate(setup.get('games', [])):
        host = profiles[seed['organizer_actor_id']]
        conversation = f'eval:{run_id}:{host.customer_id}'
        message = UserMessage(conversation_id=conversation, sender_id=host.customer_id, sender_name=host.display_name,
            text='评测前置：按提供条件登记既有局', message_id=f'seed:{index}')
        TaskContextManager(store).prepare(message, trace_id=f'seed:{index}')
        store.append_user_turn(message, f'seed:{index}')
        players = [{'customer_id': profiles[p['actor_id']].customer_id,
                    'display_name': profiles[p['actor_id']].display_name, 'seat_count': p['seat_count']}
                   for p in seed['known_players']]
        created = gateway.execute(ToolCall(name='create_game', arguments={'organizer_id': host.customer_id,
            'organizer_name': host.display_name, 'requirement': seed['requirement'], 'known_players': players}),
            trace_id=f'seed:{index}', conversation_id=conversation, sender_id=host.customer_id,
            sender_name=host.display_name, source_message_id=message.message_id, step_index=1)
        if not successful(created.to_dict()):
            raise RuntimeError(f'fixture rejected: {created.error}')
        game = store.require_game(created.result['game']['game_id']).to_dict()
        expected_seats = {p['customer_id']: p['seat_count'] for p in players}
        actual_seats = {p['customer_id']: p['seat_count'] for p in game['participants']
                        if p['status'] in {'joined', 'confirmed'}}
        if len(expected_seats) != len(players) or actual_seats != expected_seats:
            raise ValueError('fixture participant identities or seat counts differ from setup')
        if any(not eval_values_equal(game['requirement'].get(k), v, field_path=k)
               for k, v in seed['requirement'].items() if k not in {'requesting_party', 'planned_start_at'}):
            raise ValueError('fixture requirement differs from setup')
        if ('planned_start_at' in seed['requirement'] and
                aware_time(game.get('planned_start_at')) != aware_time(seed['requirement']['planned_start_at'])):
            raise ValueError('fixture planned time differs from setup')
        declared_party = seed['requirement'].get('requesting_party', {}).get('seat_count')
        if declared_party is not None and expected_seats.get(host.customer_id) != declared_party:
            raise ValueError('fixture requester party disagrees with registered seats')
        fixtures.append({'setup': deepcopy(seed), 'tool_result': created.to_dict(), 'persisted_game': game})
    return fixtures


def _turn_recovery(report: dict, traces: dict, *, business_passed: bool, semantic_passed: bool) -> list[dict]:
    """复用动作审计器，判定缺失只能用于结构预核，不能计为最终安全恢复。"""
    return [assess_action_contract_recovery(traces[turn['trace_id']], runtime_status=turn['observation']['status'],
        final_reply=turn['reply'], business_checks_passed=business_passed, semantic_passed=semantic_passed)
        for turn in report['turns']]


def _business_checks(case: dict, report: dict, traces: dict) -> dict:
    """每次补判重新核对固化的业务断言、原始追踪和沙盒交付。"""
    checks = check_assertions(resolve_times(case['assertions'], aware_time(report['anchor'])),
                              report['turns'][-1]['observation'])
    checks['complete_raw_traces'] = all(validate_trace(traces[turn['trace_id']])['complete'] for turn in report['turns'])
    silent_turns = set(case.get('silent_private_turns', []))
    checks['actual_reply_delivery'] = all(
        turn['reply_delivery_verified']
        and verify_reply_delivery(turn, traces[turn['trace_id']], expect_silent_private=index in silent_turns)
        for index, turn in enumerate(report['turns'])
    )
    return checks


def _eligible_agent_timeout(case: dict, report: dict, traces: dict) -> dict | None:
    """仅重跑有传输证据且尚未出现已知失败的隔离场景。"""
    last = report['turns'][-1]
    # 本轮可能仅有沙盒失败提示，或没有输出；前置轮仍要求有效交付证据。
    failed_turn_delivery_known = last['reply_delivery_verified'] or (not last['reply'] and last['outputs'] == [])
    if (not last['trace_validation']['complete'] or not failed_turn_delivery_known
            or not all(turn['trace_validation']['complete'] and turn['reply_delivery_verified']
                       for turn in report['turns'][:-1])):
        return None
    if any(r['quality_status'] != 'passed' for r in _turn_recovery(
            {**report, 'turns': report['turns'][:-1]}, traces, business_passed=True, semantic_passed=True)):
        return None
    checks = check_assertions(case['assertions'], last['observation'])
    # 终态相关的正向期望还无法判断；已观察到的禁止动作、路由或交付错误不能被补跑掩盖。
    definitive = {'expected_intent', 'expected_runtime_channel', 'forbidden_tools', 'no_business_writes',
                  'requester_not_joined', 'forbidden_invite_actor_ids'}
    if any(not value for key, value in checks.items() if key in definitive):
        return None
    return latest_agent_transport_failure(traces[last['trace_id']])


def _finish_judgment(report: dict, checks: dict, traces: dict, semantic: dict | None) -> None:
    """写入本次有效判决或受阻结论，明确区分结构预核和实际通过。"""
    report.pop('eval_inconclusive', None)
    recovery = _turn_recovery(report, traces, business_passed=all(checks.values()),
                              semantic_passed=bool(semantic and semantic.get('passed') is True))
    for turn, audit in zip(report['turns'], recovery):
        turn['recovery'] = audit
    checks = {**checks, 'trace_proven_recovery': all(r['quality_status'] == 'passed' for r in recovery)}
    report.update(checks=checks, semantic_assessment=semantic,
                  status='passed' if all(checks.values()) and semantic and semantic.get('passed') is True else 'failed')
    pending = (semantic or {}).get('eval_inconclusive')
    if pending and pending.get('stage') == 'judge' and pending.get('retryable') is True:
        # 只有通过结构预核才会调用判定器，失败判决时不会出现此标记。
        report.update(status='blocked', eval_inconclusive=pending)


def _source_digests(folder: Path) -> dict:
    """封存当前输入、追踪和关闭后的 SQLite，补判时逐一核对内容。"""
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ('case.json', 'agent_trace.log', 'state.sqlite3')}


def _verify_frozen_judge_inputs(original: dict, traces: dict, db_path: Path) -> None:
    """独立读回冻结状态和原始输入输出，确认判定内容仍对应同一轮实际执行。"""
    turn = original['turns'][-1]
    evidence = original['semantic_assessment']['evidence']
    events = traces[turn['trace_id']]
    inputs = [event.content['message'] for event in events if event.step == 'user_input']
    finals = [event.content for event in events if event.step == 'final_output']
    actual = turn['observation']
    if (inputs != [evidence['request']] or len(finals) != 1 or finals[0].get('reply') != turn['reply']
            or finals[0].get('objective_status') != actual['status'] or evidence['reply'] != turn['reply']
            or evidence['runtime_status'] != actual['status']):
        raise ValueError('frozen group topic original input or reply differs from judge evidence')
    stored = read_semantic_calibration_state(db_path)
    projected = evidence['persisted_state']
    responsibilities = project_scheduled_responsibilities(stored['scheduled_tasks'],
        customer_id=actual['actor_id'], conversation_id=actual['conversation_id'],
        task_context_id=actual['task_context_id'], target_game_ids=actual['target_game_ids'])
    if (sorted(projected['games'], key=lambda game: game['game_id']) != stored['games']
            or any(sorted(projected[key], key=lambda task: task['task_id']) != tasks
                   for key, tasks in responsibilities.items())):
        raise ValueError('frozen group topic judge evidence differs from SQLite')


def resume_case_judgment(previous: dict, folder: Path) -> dict:
    """在同一次门禁内只补判已经核对的冻结输入，保留旧判决和原数据库。"""
    source = Path(previous['attempt_dir'])
    original = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    if original != {key: value for key, value in previous.items() if key != 'attempt_dir'}:
        raise ValueError('frozen group topic report differs from the case checkpoint')
    if original.get('eval_inconclusive', {}).get('stage') != 'judge' or original.get('status') != 'blocked':
        raise ValueError('frozen case has no incomplete judge verdict')
    if original.get('frozen_source_digests') != _source_digests(source):
        raise ValueError('frozen group topic evidence changed')
    case = json.loads((source / 'case.json').read_text(encoding='utf-8'))
    if original['case_id'] != case['case_id'] or original['semantic_assessment']['expectation'] != case['expectation']:
        raise ValueError('frozen group topic case identity or expectation changed')
    events = [event for line in (source / 'agent_trace.log').read_text(encoding='utf-8').splitlines()
              if (event := TraceEvent.from_log_line(line)) is not None]
    traces = {turn['trace_id']: [event for event in events if event.trace_id == turn['trace_id']]
              for turn in original['turns']}
    checks = _business_checks(case, original, traces)
    if (not all(checks.values()) or checks != {k: v for k, v in original['checks'].items() if k != 'trace_proven_recovery'}
            or any(r['quality_status'] != 'passed' for r in _turn_recovery(
                original, traces, business_passed=True, semantic_passed=True))):
        raise ValueError('frozen group topic business assertions failed or differ from original')
    # 复制已关闭且经摘要校验的证据，旧报告与所有物理文件不因本次补判改变。
    folder.mkdir(parents=True, exist_ok=False)
    for name in original['frozen_source_digests']:
        shutil.copy2(source / name, folder / name)
    _verify_frozen_judge_inputs(original, traces, folder / 'state.sqlite3')
    semantic = resume_reply_semantics(OpenAICompatibleAgentClient.from_env(), original['semantic_assessment'])
    result = deepcopy(original)
    _finish_judgment(result, checks, traces, semantic)
    write_json(folder / 'report.json', result)
    return result


def run_case(case: dict, folder: Path) -> dict:
    """复用正式群入口和模拟发送器，模型不能收到人工断言或期望答案。"""
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'case.json', case)
    store = SQLiteAgentStore(folder / 'state.sqlite3')
    report = {'case_id': case['case_id'], 'status': 'running', 'turns': []}
    traces = {}
    try:
        anchor = now()
        parsed = validate_scenario(case['scenario'])
        profiles = {a['id']: a['profile'] for a in parsed['actors']}
        run_id = case['case_id']
        sandbox = {'run_id': run_id, 'manual_actors': [p.to_dict() for p in profiles.values()]}
        store.configure_rooms(['eval_room_1', 'eval_room_2', 'eval_room_3'])
        store.upsert_group_room_policy(GroupRoomPolicy(room_id=f'manual-room:{run_id}', channel='console',
            managed=True, board_enabled=True, outbound_enabled=True))
        for actor in parsed['actors']:
            p = actor['profile']
            store.upsert_customer(p)
            store.upsert_channel_identity(ChannelIdentity(channel='console', external_user_id=p.customer_id,
                customer_id=p.customer_id, public_name=p.display_name, private_conversation_id=f'eval:{run_id}:{p.customer_id}',
                is_friend=actor['is_friend'], can_private_message=actor['is_friend']))
        report['anchor'] = anchor.isoformat()
        report['fixtures'] = seed_games(store, profiles, resolve_times(case.get('setup', {}), anchor), run_id=run_id)
        seeded_ids = set(store.games)
        runtime, client = build_runtime(store, 'real')
        report['model_config'] = {key: getattr(client.config, key) for key in
            ('model', 'provider', 'thinking_mode', 'reasoning_effort', 'max_tokens', 'retry_attempts', 'attempt_timeout_seconds')}
        silent_turns = set(case.get('silent_private_turns', []))
        for index, item in enumerate(parsed['messages']):
            actor = profiles[item['actor_id']]
            before_game_ids = set(store.games)
            prior_own_ids = [g.game_id for g in store.active_games() if any(p.customer_id == actor.customer_id
                and p.status in {'joined', 'confirmed'} for p in g.participants)]
            previous_turns = [t['observation'] for t in report['turns'] if t['input']['actor_id'] == item['actor_id']]
            trace_id = f'{run_id}:{index}'
            if item['channel'] != 'group':
                raise ValueError('this regression runner requires genuine group input')
            result, processing, source = process_manual_group(runtime, sandbox, actor.to_dict(), item['text'],
                message_id=f'{run_id}:{index}', stamp=now(), trace_id=trace_id)
            if index in silent_turns:
                # 使用正式入口执行结果和沙盒收件箱计数生成可复核证据，不采信模型的外发自述。
                routing = processing['routing']
                details = routing.get('detail') or {}
                runtime.trace_recorder.record(result.trace_id, 'group_eval_private_delivery', {
                    'action': routing['action'], 'source_message_id': source.message_id, 'customer_id': actor.customer_id,
                    'private_conversation_id': details.get('private_conversation_id'),
                    'private_reply_suppressed': details.get('private_reply_suppressed'),
                    'private_delivery_id': details.get('private_delivery_id'), 'output_count': len(processing['outputs']),
                })
            events = runtime.trace_recorder.get_trace(result.trace_id)
            traces[result.trace_id] = events
            route = processing.get('topic_analysis', {}).get('receipt', {})
            classification = route.get('message', {}).get('classification', {})
            own_games = [g.to_dict() for g in store.active_games() if any(p.customer_id == actor.customer_id and
                p.status in {'joined', 'confirmed'} for p in g.participants)]
            context = store.current_task_context(result.conversation_id, actor.customer_id)
            topic_id = (route.get('topic') or {}).get('topic_id')
            bindings = store.list_group_topic_customer_bindings(topic_id, actor.customer_id) if topic_id else []
            source_bindings = [b for b in bindings if b.get('source_message_id') == source.message_id
                and b.get('conversation_id') == result.conversation_id and context
                and b.get('task_context_id') == context.task_context_id]
            initial_context = next((e.content for e in events if e.step == 'context_built'), {})
            tools = [t.to_dict() for t in result.tool_results]
            actual = {'intent': classification.get('intent'), 'goal_relation': classification.get('goal_relation'),
                'actor_id': actor.customer_id, 'conversation_id': result.conversation_id, 'message_id': source.message_id,
                'observed_at': now().isoformat(), 'task_context_id': context.task_context_id if context else None,
                'classified_task_context_id': classification.get('task_context_id'), 'status': result.status,
                'source_bound_to_task': bool(source_bindings),
                'target_game_ids': sorted({g for b in source_bindings for g in b.get('game_ids', [])}),
                'previous_task_context_ids': [t['task_context_id'] for t in previous_turns if t.get('task_context_id')],
                'previous_source_message_ids': [t['message_id'] for t in previous_turns],
                'inherited_source_message_ids': [r['source_message_id'] for r in
                    (initial_context.get('group_topic_context') or {}).get('own_condition_sources', [])],
                'runtime_channel': 'private' if result.conversation_id == f'eval:{run_id}:{actor.customer_id}' else 'group',
                'tools': tools, 'own_games': own_games, 'new_game_count': len(set(store.games) - before_game_ids),
                'previous_own_game_ids': prior_own_ids,
                'board_arrangements': [{'game_id': g['game_id'], **store.game_board_arrangement(g['game_id'])} for g in own_games],
                'deadline_tasks': [t.to_dict() for t in store.scheduled_tasks.values()
                    if t.subject_id == actor.customer_id and t.task_type == 'participation_deadline'],
                'requester_joined_seeded': any(g['game_id'] in seeded_ids for g in own_games),
                'invited_actor_ids': [d.customer_id for d in store.invite_drafts.values()]}
            if case['assertions'].get('require_participation_deadline'):
                # 本组资产的 clock 简写明确指运行锚点次日；完整时刻是评测真值，不送给模型。
                actual['expected_deadline_at'] = resolve_times('$tomorrow_' + case['assertions']['deadline_clock'], anchor)
            turn = {'input': item, 'trace_id': result.trace_id, 'reply': result.final_reply, 'observation': actual,
                    'trace_validation': validate_trace(events), 'outputs': processing['outputs'],
                    'routing_receipt': route, 'group_input': source.to_dict()}
            turn['reply_delivery_verified'] = verify_reply_delivery(turn, events, expect_silent_private=index in silent_turns)
            report['turns'].append(turn)
            write_json(folder / 'report.json', report)
            if result.status not in {'completed', 'waiting_user'}:
                failure = _eligible_agent_timeout(case, report, traces)
                if failure:
                    report.update(status='blocked', eval_inconclusive=failure)
                    return report
                raise RuntimeError(f'agent stopped: {result.status}')
        checks = _business_checks(case, report, traces)
        if not all(checks.values()) or any(r['quality_status'] != 'passed' for r in _turn_recovery(
                report, traces, business_passed=True, semantic_passed=True)):
            _finish_judgment(report, checks, traces, None)
            return report
        original = next(e.content['message'] for e in events if e.step == 'user_input')
        # 复用既有最小语义证据投影；不把模型的计划、自评和分类理由交给判定器。
        own_message = SimpleNamespace(sender_id=actor.customer_id, conversation_id=result.conversation_id,
                                      to_dict=lambda: original)
        evidence = reply_semantic_evidence(result, SimpleNamespace(message=own_message), events, store)
        if verify_intentional_silence(report['turns'][-1], events):
            evidence['execution_lifecycle']['reply_delivery'] = {'mode': 'intentional_silence', 'delivered_count': 0}
        evidence['persisted_state'].update(project_scheduled_responsibilities(
            [t.to_dict() for t in store.scheduled_tasks.values()], customer_id=actor.customer_id,
            conversation_id=result.conversation_id, task_context_id=actual['task_context_id'],
            target_game_ids=actual['target_game_ids']))
        semantic = assess_reply_semantics(client, evidence=evidence, expectation=case['expectation'], trace_id=f'{trace_id}:judge')
        _finish_judgment(report, checks, traces, semantic)
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        store._connection.close()
        if report.get('eval_inconclusive', {}).get('stage') == 'judge':
            report['frozen_source_digests'] = _source_digests(folder)
        write_json(folder / 'report.json', report)
    return report


def _scope_run_folder(parent: Path) -> Path:
    """统一门禁续跑沿用本范围唯一清单；独立运行始终分配新目录。"""
    if scope_directory() is not None:
        existing = list(parent.glob('run_*'))
        if existing:
            if len(existing) != 1 or not existing[0].is_dir() or existing[0].is_symlink():
                raise ValueError('group topic gate scope has no unique execution directory')
            return existing[0]
    return Path(tempfile.mkdtemp(prefix='run_', dir=parent))


def _publish_case_artifacts(source: Path, canonical: Path) -> None:
    """以独立目录发布同一范围的成功证据，保留全部原始尝试。"""
    if source == canonical:
        return
    if canonical.exists():
        if (_source_digests(source) != _source_digests(canonical)
                or (source / 'report.json').read_bytes() != (canonical / 'report.json').read_bytes()):
            raise ValueError('published group topic evidence differs from the completed attempt')
        return
    with tempfile.TemporaryDirectory(prefix='.publish_', dir=canonical.parent) as staging:
        candidate = Path(staging) / 'case'
        shutil.copytree(source, candidate)
        candidate.replace(canonical)


def main() -> int:
    """门禁模式按最小独立用例补试；已知失败阻断，证据无结论可续跑。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--collect-all', action='store_true', help='continue running independent cases after a failure')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if args.case:
        if set(args.case) - {c['case_id'] for c in cases}:
            raise ValueError('unknown case filter')
        cases = [c for c in cases if c['case_id'] in args.case]
    if args.validate_only:
        print(json.dumps({'validated': len(cases)}))
        return 0
    parent = args.work_dir or ROOT / 'runtime_data/group_topic_regressions'
    parent.mkdir(parents=True, exist_ok=True)
    folder = _scope_run_folder(parent)
    summary = {'status': 'running', 'dataset': str(args.dataset), 'selected_cases': [c['case_id'] for c in cases], 'cases': []}
    prior_summary = folder / 'summary.json'
    if prior_summary.exists():
        prior = json.loads(prior_summary.read_text(encoding='utf-8'))
        if any(summary[key] != prior.get(key) for key in ('dataset', 'selected_cases')):
            raise ValueError('group topic gate scope input changed between attempts')
    print(json.dumps({'run_dir': str(folder)}), flush=True)
    for case in cases:
        case_dir = folder / case['case_id']
        result = checkpointed_case(case['case_id'], lambda attempt: run_case(case, attempt), case_dir=case_dir,
                                   resume_judge=resume_case_judgment)
        if result['status'] == 'passed':
            _publish_case_artifacts(Path(result.get('attempt_dir', case_dir)), case_dir)
        summary['cases'].append({'case_id': case['case_id'], 'status': result['status'], 'checks': result.get('checks'),
                                 'error': result.get('error')})
        if result.get('eval_inconclusive'):
            summary['cases'][-1].update(eval_inconclusive=result['eval_inconclusive'])
        write_json(folder / 'summary.json', summary)
        print(json.dumps(summary['cases'][-1], ensure_ascii=False), flush=True)
        if result['status'] != 'passed' and not args.collect_all:
            break
    if scope_directory() is not None and summary['cases'] and summary['cases'][-1].get('eval_inconclusive'):
        summary['status'] = 'blocked'
    else:
        summary['status'] = 'passed' if len(summary['cases']) == len(cases) and all(c['status'] == 'passed' for c in summary['cases']) else 'failed'
    write_json(folder / 'summary.json', summary)
    return BLOCKED_EXIT_CODE if summary['status'] == 'blocked' else int(summary['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
