"""真实主 Agent 的通用话题多轮回放：逐轮保存事实、原始模型来源及独立语义。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import uuid
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from mahjong_agent_runtime import CustomerProfile, InMemoryTraceRecorder, OpenAICompatibleAgentClient, QuotedMessageRef, SQLiteAgentStore, ToolCall, UserMessage
from mahjong_agent_runtime.env import load_dotenv_defaults
from mahjong_agent_runtime.agent_state import restore_user_message
from mahjong_agent_runtime.llm import validate_model_response_source
from mahjong_agent_runtime.customer_visible_contract import customer_visible_text_contract_violations
from mahjong_agent_runtime.stores.sqlite.serialization import tool_result_from_payload
from mahjong_agent_runtime.tracing import TraceEvent
import run_real_owner_chat_live_eval as owner
from run_inquiry_grounding_eval import assess_reply_semantics

DATASET = ROOT / 'eval/golden/conversation_topic_continuity_20261105.jsonl'
QUERY_MEMORY_POLICY = ROOT / 'eval/handoffs/20261105_query_memory_scope_policy.json'
QUERY_MEMORY_POLICY_SHA256 = '91dd0f74e646cac76a49106813ecb253e8da9df7fe3bf6eaeab71d115baff32f'


def load_cases() -> list[dict]:
    """验收输入只来自独立资产，不能把当前输出补入期望。"""
    cases = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if len(cases) != 9 or len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('topic continuity requires nine unique source conversations')
    for case in cases:
        if not 1 <= len(case['turns']) <= 4 or any(not turn.get('text') or not turn.get('meaning') for turn in case['turns']):
            raise ValueError('topic continuity lacks complete multi-turn source expectations')
    raw_policy = QUERY_MEMORY_POLICY.read_bytes()
    if hashlib.sha256(raw_policy).hexdigest() != QUERY_MEMORY_POLICY_SHA256:
        raise ValueError('query memory scope revision changed without review')
    policy = json.loads(raw_policy)
    targets = [case for case in cases if case['case_id'] == policy['source_case_id']]
    from eval_product_contracts import canonical_digest
    if (hashlib.sha256(DATASET.read_bytes()).hexdigest() != policy['source_asset_sha256']
            or len(targets) != 1 or canonical_digest(targets[0]) != policy['source_case_sha256']
            or targets[0]['turns'][policy['turn_index']] != policy['previous_turn']):
        raise ValueError('query memory scope original asset or source case differs')
    targets[0]['turns'][policy['turn_index']] = deepcopy(policy['current_turn'])
    targets[0]['policy_revision_ids'] = [policy['revision_id']]
    return cases


def expectation(turn: dict) -> dict:
    """当前话题与业务事实分别评分，普通术语不能替代隐私或执行真实性判断。"""
    return {'required_meaning': [{'id': 'current_topic_and_goal', 'description': turn['meaning']}],
        'forbidden_meaning': owner.common_forbidden_semantic_meanings()}


def state_snapshot(store, conversation: str) -> dict:
    """业务权威快照不包含日志/checkpoint，正常审计写入不冒充业务副作用。"""
    task = store.current_task_context(conversation, 'topic_customer')
    return {'games': [game.to_dict() for game in store.games.values()],
        'invites': [draft.to_dict() for draft in store.invite_drafts.values()],
        'outbound': [draft.to_dict() for draft in store.outbound_message_drafts.values()],
        'memories': store.task_memory_context(conversation, 'topic_customer'),
        'all_task_memories': [item.to_dict() for item in store.task_memories.values()],
        'pending_memory_candidates': [item.to_dict() for item in store.pending_memory_candidates.values()],
        'waiting': [item.to_dict() for item in store.waiting_demands.values()],
        'scheduled': [item.to_dict() for item in store.scheduled_tasks.values()],
        'reservations': [item.to_dict() for item in store.room_reservations.values()],
        'task_context_id': task.task_context_id if task else None}


def grounded_current_task_memory(before, after, tools, *, request: dict | None, trace_id: str | None) -> bool:
    """询局可追加本人本目标约束，旧记忆不得改写，当前原话和成功工具回执缺一不可。

    此处核对身份、来源和写入闭环；记录的事实含义由独立语义规则与实际查询约束共同
    验收，不用自然语言关键词猜意图。完整记忆表避免把越界目标写入藏在过滤视图之外。
    """
    if not request or not trace_id or not after.get('task_context_id'):
        return False
    if before['pending_memory_candidates'] != after['pending_memory_candidates']:
        return False
    previous = {item['memory_id']: item for item in before['all_task_memories']}
    current = {item['memory_id']: item for item in after['all_task_memories']}
    if (len(previous) != len(before['all_task_memories']) or len(current) != len(after['all_task_memories'])
            or any(current.get(key) != value for key, value in previous.items())):
        return False
    writes = [tool for tool in tools if tool.get('name') == 'record_user_memory'
        and tool.get('called') is True and tool.get('allowed') is True and not tool.get('error')]
    if any(tool.get('result', {}).get('pending_long_term_memories') for tool in writes):
        return False
    returned = [memory for tool in writes for memory in tool.get('result', {}).get('task_memories', [])]
    added = [value for key, value in current.items() if key not in previous]
    if any(memory not in returned for memory in added) or any(memory not in added for memory in returned):
        return False
    for memory in added:
        evidence = memory.get('evidence')
        if (memory.get('customer_id') != request.get('sender_id')
                or memory.get('conversation_id') != request.get('conversation_id')
                or memory.get('metadata', {}).get('task_context_id') != after['task_context_id']
                or memory.get('target_customer_id') is not None
                or memory.get('scope') not in {'current_task', 'session', 'today'}
                or memory.get('source_trace_id') != trace_id or memory.get('status') != 'active'
                or not isinstance(evidence, str) or not evidence.strip() or evidence not in request.get('text', '')
                or not any(transition.get('entity_type') == 'task_memory' and transition.get('entity_id') == memory['memory_id']
                    and transition.get('from_status') is None and transition.get('to_status') == 'active'
                    and transition.get('trace_id') == trace_id for tool in writes for transition in tool.get('state_transitions', []))):
            return False
    return True


def checks_for(turn, before, after, tools, *, original_game_id=None, original_task_id=None, request=None, trace_id=None) -> dict:
    """只用结构化执行与权威状态判定，禁止用话题关键词猜用户意图。"""
    games = [game for game in after['games'] if game['status'] in {'forming', 'inviting', 'ready'}]
    names = {tool['name'] for tool in tools if tool.get('called') and tool.get('allowed') and not tool.get('error')}
    checks = {'expected_game_count': len(games) == turn['game_count'],
        'required_tools_executed': set(turn.get('required_tools', [])) <= names,
        'same_business_goal': original_task_id is None or after['task_context_id'] == original_task_id
            or turn.get('cancel_original') and after['task_context_id'] is None,
        'same_game': original_game_id is None or len(after['games']) == 1 and after['games'][0]['game_id'] == original_game_id}
    if turn.get('cancel_original'):
        checks['original_game_cancelled'] = len(after['games']) == 1 and after['games'][0]['status'] == 'cancelled'
    if turn.get('no_business_change'):
        checks['no_unrequested_business_change'] = all(before.get(key) == after.get(key) for key in (
            'games', 'invites', 'outbound', 'memories', 'all_task_memories', 'pending_memory_candidates', 'waiting', 'scheduled', 'reservations'))
    if turn.get('no_operational_change'):
        checks['no_unrequested_operational_change'] = all(before[key] == after[key] for key in ('games', 'invites', 'outbound', 'waiting', 'scheduled', 'reservations'))
        checks['query_memory_changes_are_grounded'] = turn.get('allow_grounded_current_task_memory') is True and grounded_current_task_memory(
            before, after, tools, request=request, trace_id=trace_id)
    if 'max_duration_hours' in turn:
        checks['duration_constraint_persisted'] = any(item.get('field') == 'max_duration_hours'
            and item.get('value') == turn['max_duration_hours'] for item in after['memories']) or any(
                game.get('requirement', {}).get('max_duration_hours') == turn['max_duration_hours'] for game in games)
    if 'required_duration_filter' in turn:
        queries = [tool['result'] for tool in tools if tool['name'] == 'search_current_games' and tool.get('called') and not tool.get('error')]
        checks['nonempty_duration_query'] = any(query.get('requirement', {}).get('max_duration_hours') == turn['required_duration_filter']
            and len(query.get('matches', [])) == 1
            and query['matches'][0]['game']['requirement'].get('duration_hours') == 4
            and len(query.get('alternatives', [])) == 1
            and query['alternatives'][0].get('match_kind') == 'duration_confirmation_required' for query in queries)
    return checks


def run_case(case: dict, client, args, folder: Path, *, semantic_assessor=assess_reply_semantics) -> dict:
    """所有客户轮次走同一真实主流程，外部人数变化只经真实网关，禁止预写模型回复。"""
    folder.mkdir(parents=True, exist_ok=False)
    store = SQLiteAgentStore(folder / 'state.sqlite3')
    trace = InMemoryTraceRecorder()
    store.configure_rooms(['topic_room'])
    store.upsert_customer(CustomerProfile(customer_id='topic_customer', display_name='客户',
        preferred_games=['hangzhou_mahjong'], preferred_stakes=case.get('preferred_stakes', ['1']), smoke_preference='no_smoke'))
    owner.configure_authorized_eval_board(store)
    runtime = owner.build_runtime(client, store, trace, args)
    for index, duration in enumerate(case.get('inventory_duration_hours', [])):
        organizer = f'inventory_owner_{index}'
        store.upsert_customer(CustomerProfile(customer_id=organizer, display_name='发起人'))
        requirement = {'game_type': 'hangzhou_mahjong', 'stake': '1', 'smoke_preference': 'no_smoke', 'start_time_kind': 'asap_when_full'}
        if duration is not None: requirement['duration_hours'] = duration
        result = runtime.tool_gateway.execute(ToolCall(name='create_game', arguments={'organizer_id': organizer,
            'organizer_name': '发起人', 'requirement': requirement,
            'known_players': [{'customer_id': organizer, 'display_name': '发起人', 'seat_count': 3}]}),
            trace_id=f'inventory:{index}', conversation_id=f'inventory:{index}', sender_id=organizer,
            sender_name='发起人', source_message_id=f'inventory:{index}', step_index=1)
        if not result.called or result.error: raise ValueError('duration inventory fixture failed')
    run_id = uuid.uuid4().hex
    conversation = 'conversation:' + run_id
    report = {'case_id': case['case_id'], 'source': case, 'main_agent_executed': True,
        'status': 'failed', 'run_id': run_id, 'turns': [], 'db_path': str(store.path), 'external_events': [],
        'main_model_config': {key: getattr(client.config, key) for key in ('provider', 'model', 'max_tokens', 'thinking_mode', 'reasoning_effort', 'response_format')},
        'runtime_timeout_seconds': args.timeout_seconds}
    original_game_id = original_task_id = None
    try:
        for index, turn in enumerate(case['turns']):
            if turn.get('event_before'):
                if turn['event_before'] != 'join_one_guest' or len(store.active_games()) != 1:
                    raise ValueError('unsupported or unavailable real domain event')
                game = store.active_games()[0]
                store.upsert_customer(CustomerProfile(customer_id='topic_guest', display_name='另一客户'))
                event_trace = f'{run_id}:domain:{index}'
                guest_message = UserMessage(conversation_id='guest_conversation', sender_id='topic_guest', sender_name='另一客户',
                    text='帮我查杭麻1块无烟的现成局，有合适的我参加。', message_id=f'guest:{index}')
                runtime.task_context_manager.prepare(guest_message, trace_id=event_trace)
                store.append_user_turn(guest_message, event_trace)
                query = runtime.tool_gateway.execute(ToolCall(name='search_current_games', arguments={'requirement': {
                    'game_type': 'hangzhou_mahjong', 'stake': '1', 'smoke_preference': 'no_smoke'}}),
                    trace_id=event_trace, conversation_id='guest_conversation', sender_id='topic_guest', sender_name='另一客户',
                    source_message_id=guest_message.message_id, step_index=1)
                trace.record(event_trace, 'tool_result', query.to_dict())
                if not query.called or query.error: raise ValueError('guest query access proof failed')
                # 参与权限读取本人当前任务的持久工具轮，保留真实查询回执而非手工写授权。
                store.append_tool_turn(guest_message.conversation_id, json.dumps([query.to_dict()], ensure_ascii=False), event_trace)
                call = ToolCall(name='join_game', arguments={'game_id': game.game_id, 'customer_id': 'topic_guest', 'display_name': '另一客户'})
                outcome = runtime.tool_gateway.execute(call, trace_id=event_trace, conversation_id='guest_conversation',
                    sender_id='topic_guest', sender_name='另一客户', source_message_id=f'guest:{index}', step_index=2)
                trace.record(event_trace, 'tool_result', outcome.to_dict())
                report['external_events'].append({'before_turn': index, 'query': query.to_dict(), 'result': outcome.to_dict(),
                    'events': [event.to_dict() for event in trace.get_trace(event_trace)]})
                if not outcome.called or not outcome.allowed or outcome.error:
                    raise ValueError('declared domain event did not execute')
            before = state_snapshot(store, conversation)
            turn_dir = folder / f'turn_{index:02d}'; turn_dir.mkdir()
            before_db_path = turn_dir / 'before.sqlite3'
            with sqlite3.connect(before_db_path) as target:
                store._connection.backup(target)
            message = UserMessage(conversation_id=conversation, sender_id='topic_customer', sender_name='客户',
                text=turn['text'], message_id=f'{run_id}:{index}',
                quoted_message=QuotedMessageRef(**turn['quote']) if turn.get('quote') else None)
            result = runtime.handle_user_message(message, trace_id=f'{run_id}:turn:{index}')
            events = trace.get_trace(result.trace_id)
            scene = owner.LiveEvalScenario(scenario_id=case['case_id'], name=case['case_id'], message=message, setup=lambda store: None)
            evidence = owner.reply_semantic_evidence(result, scene, events, store)
            semantic = semantic_assessor(client, evidence=evidence, expectation=expectation(turn), trace_id=result.trace_id + ':semantic')
            after = state_snapshot(store, conversation)
            tools = [tool.to_dict() for tool in result.tool_results]
            checks = checks_for(turn, before, after, tools, original_game_id=original_game_id, original_task_id=original_task_id,
                request=message.to_dict(), trace_id=result.trace_id)
            checks['runtime_finished'] = result.status in {'completed', 'waiting_user'}
            checks['customer_visible_privacy'] = not customer_visible_text_contract_violations(result.final_reply)
            recovery = owner.assess_action_contract_recovery(events, runtime_status=result.status, final_reply=result.final_reply,
                business_checks_passed=all(checks.values()), semantic_passed=semantic.get('passed') is True)
            raw_path = turn_dir / 'trace.json'
            raw_path.write_text(json.dumps({'events': [event.to_dict() for event in events]}, ensure_ascii=False, indent=2)+'\n')
            db_path = turn_dir / 'state.sqlite3'
            with sqlite3.connect(db_path) as target:
                store._connection.backup(target)
            entry = {'index': index, 'request': message.to_dict(), 'trace_id': result.trace_id, 'reply': result.final_reply,
                'status': result.status, 'before': before, 'after': after, 'tools': tools, 'checks': checks,
                'semantic_assessment': semantic, 'recovery_metrics': recovery, 'raw_trace_path': str(raw_path),
                'db_path': str(db_path), 'before_db_path': str(before_db_path)}
            report['turns'].append(entry)
            if original_task_id is None: original_task_id = after['task_context_id']
            if original_game_id is None and after['games']: original_game_id = after['games'][0]['game_id']
        report['status'] = 'passed' if all(all(turn['checks'].values()) and turn['semantic_assessment'].get('passed') is True
            and turn['recovery_metrics']['quality_status'] == 'passed' for turn in report['turns']) else 'failed'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'.replace(client.config.api_key, '[redacted]')
    finally:
        store._connection.close()
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def validate_scope(payload: dict, report_path: Path) -> None:
    """门禁重算每轮硬断言并绑定原始HTTP、用户输入、实际回复和独立判决，拒绝冻结冒充执行。"""
    from run_evals import _validate_reported_semantic_assessment

    cases = load_cases()
    root = report_path.resolve().parent
    call_ids = set()
    if (payload.get('status') != 'passed' or payload.get('execution_scope') != 'live_multi_turn_topic_continuity'
            or payload.get('query_memory_policy_sha256') != QUERY_MEMORY_POLICY_SHA256
            or payload.get('dataset_sha256') != hashlib.sha256(DATASET.read_bytes()).hexdigest()
            or [item.get('case_id') for item in payload.get('reports', [])] != [case['case_id'] for case in cases]):
        raise ValueError('topic scope missing current complete live source coverage')
    for case, report in zip(cases, payload['reports']):
        if report.get('source') != case or report.get('main_agent_executed') is not True or len(report.get('turns', [])) != len(case['turns']):
            raise ValueError('topic report source or actual main execution differs')
        game_id = task_id = None
        declared_events = [index for index, turn in enumerate(case['turns']) if turn.get('event_before')]
        if [event.get('before_turn') for event in report.get('external_events', [])] != declared_events:
            raise ValueError('topic external event coverage differs from source')
        for index, (source, turn) in enumerate(zip(case['turns'], report['turns'])):
            paths = [Path(turn[name]).resolve() for name in ('raw_trace_path', 'db_path', 'before_db_path')]
            if any(not path.is_relative_to(root) or not path.is_file() for path in paths):
                raise ValueError('topic raw evidence does not belong to this run')
            events = [TraceEvent(trace_id=raw['trace_id'], step=raw['step'], content=raw['content'], level=raw['level'], occurred_at=raw['time']) for raw in json.loads(paths[0].read_text())['events']]
            if any(event.trace_id != turn['trace_id'] for event in events): raise ValueError('topic mixed trace identity')
            inputs = [event.content['message'] for event in events if event.step == 'user_input']
            outputs = [event.content for event in events if event.step == 'final_output']
            if (inputs != [turn['request']] or turn['request']['text'] != source['text']
                    or turn['request']['message_id'] != f'{report["run_id"]}:{index}'
                    or turn['request']['conversation_id'] != 'conversation:' + report['run_id']
                    or turn['request']['sender_id'] != 'topic_customer'
                    or turn['request'].get('quoted_message') != (QuotedMessageRef(**source['quote']).to_dict() if source.get('quote') else None)
                    or len(outputs) != 1 or outputs[0].get('reply') != turn['reply'] or outputs[0].get('objective_status') != turn['status']):
                raise ValueError('topic actual input or delivered reply changed')
            responses = [event for event in events if event.step == 'llm_response']
            for response in responses:
                config = report['main_model_config']; diagnostics = response.content.get('completion_diagnostics', {})
                call_id = response.content.get('response_source', {}).get('call_id')
                if (diagnostics.get('route') != 'primary' or diagnostics.get('provider') != config['provider']
                        or diagnostics.get('requested_model') != config['model'] or call_id in call_ids
                        or diagnostics.get('request') != {key: config[key] for key in ('max_tokens', 'thinking_mode', 'reasoning_effort', 'response_format')}):
                    raise ValueError('topic actual primary model configuration or unique call differs')
                call_ids.add(call_id)
                prompts = [event for event in events[:events.index(response)] if event.step in {'llm_prompt', 'llm_prompt_after_context_compaction'}
                    and event.content.get('step_index') == response.content.get('step_index')]
                if not prompts: raise ValueError('topic actual model request missing')
                messages = prompts[-1].content['messages']
                request_payload = json.loads(messages[-1]['content'])
                serialized = json.dumps(messages, ensure_ascii=False)
                if (request_payload.get('current_message') != turn['request'] or case['case_id'] in serialized
                        or any(item['meaning'] in serialized for item in case['turns'])):
                    raise ValueError('topic actual model request mismatched or leaked evaluation expectation')
            if not responses or any(not validate_model_response_source(event.content.get('response_source'),
                    raw_response=event.content.get('content'), trace_id=event.trace_id, step_index=event.content.get('step_index'),
                    diagnostics=event.content.get('completion_diagnostics', {}), require_http=True) for event in responses):
                raise ValueError('topic actual main HTTP evidence absent')
            tools = [event.content for event in events if event.step == 'tool_result']
            if tools != turn['tools']: raise ValueError('topic tool evidence changed')
            snapshot_store = SQLiteAgentStore(paths[1])
            try: after = state_snapshot(snapshot_store, turn['request']['conversation_id'])
            finally: snapshot_store._connection.close()
            if after != turn['after']: raise ValueError('topic persisted state differs')
            before_store = SQLiteAgentStore(paths[2])
            try: before = state_snapshot(before_store, turn['request']['conversation_id'])
            finally: before_store._connection.close()
            if before != turn['before']: raise ValueError('topic pre-execution state differs')
            if index and not source.get('event_before') and before != report['turns'][index-1]['after']:
                raise ValueError('topic uninterrupted persistent history differs')
            if source.get('event_before'):
                proof = next(event for event in report['external_events'] if event['before_turn'] == index)
                raw_events = proof.get('events', [])
                calls = [event['content'] for event in raw_events if event['step'] == 'tool_gateway_received']
                outcomes = [event['content'] for event in raw_events if event['step'] == 'tool_gateway_completed']
                results = [event['content'] for event in raw_events if event['step'] == 'tool_result']
                prior = report['turns'][index-1]['after']
                if (source['event_before'] != 'join_one_guest' or len(calls) != 2 or len(outcomes) != 2
                        or results != [proof['query'], proof['result']] or any(event['trace_id'] != f'{report["run_id"]}:domain:{index}' for event in raw_events)
                        or [call['call']['name'] for call in calls] != ['search_current_games', 'join_game']
                        or not proof['query'].get('called') or proof['query'].get('error')
                        or calls[1]['call']['arguments'] != {'game_id': game_id, 'customer_id': 'topic_guest', 'display_name': '另一客户'}
                        or not proof['result'].get('called') or not proof['result'].get('allowed') or proof['result'].get('error')
                        or any(outcome.get('outcome') != 'executed' for outcome in outcomes) or outcomes[1].get('idempotency_key') != proof['result'].get('idempotency_key')
                        or len(before['games']) != 1 or before['games'][0]['game_id'] != game_id
                        or proof['result'].get('result', {}).get('game') != before['games'][0]
                        or any(before[key] != prior[key] for key in ('invites', 'outbound', 'memories', 'task_context_id'))):
                    raise ValueError('topic external change lacks same-game successful gateway and persistent proof')
                joined = [participant for participant in before['games'][0]['participants'] if participant['customer_id'] == 'topic_guest']
                previous = [participant for participant in prior['games'][0]['participants'] if participant['customer_id'] == 'topic_guest']
                if previous or len(joined) != 1 or joined[0]['seat_count'] != 1:
                    raise ValueError('topic declared guest event did not add exactly its authorized participant')
                original, changed = prior['games'][0], before['games'][0]
                derived_seats = {'known_player_count', 'needed_seats', 'seat_claims', 'seat_format'}
                if ([participant for participant in changed['participants'] if participant['customer_id'] != 'topic_guest'] != original['participants']
                        or any(original.get(key) != changed.get(key) for key in ('game_id', 'conversation_id', 'organizer_id'))
                        or {key: value for key, value in original['requirement'].items() if key not in derived_seats}
                            != {key: value for key, value in changed['requirement'].items() if key not in derived_seats}
                        or changed['seat_summary']['claimed_seats'] != original['seat_summary']['claimed_seats'] + 1
                        or changed['seat_summary']['remaining_seats'] != original['seat_summary']['remaining_seats'] - 1):
                    raise ValueError('topic guest join changed original participants or agreed conditions')
            checks = checks_for(source, turn['before'], after, tools, original_game_id=game_id, original_task_id=task_id,
                request=turn['request'], trace_id=turn['trace_id'])
            checks['runtime_finished'] = turn['status'] in {'completed', 'waiting_user'}
            checks['customer_visible_privacy'] = not customer_visible_text_contract_violations(turn['reply'])
            if checks != turn['checks'] or not all(checks.values()): raise ValueError('topic hard assertions fail')
            semantic = turn['semantic_assessment']
            _validate_reported_semantic_assessment(semantic, expectation(source), scope='topic continuity')
            if semantic['evidence']['request'] != inputs[0] or semantic['evidence']['reply'] != outputs[0]['reply']:
                raise ValueError('topic semantic assessment judged another input or reply')
            evidence_store = SQLiteAgentStore(paths[1])
            try:
                scene = owner.LiveEvalScenario(scenario_id=case['case_id'], name=case['case_id'],
                    message=restore_user_message(inputs[0]), setup=lambda store: None)
                actual_evidence = owner.reply_semantic_evidence(SimpleNamespace(final_reply=turn['reply'], status=turn['status'],
                    tool_results=[tool_result_from_payload(tool) for tool in tools]), scene, events, evidence_store)
            finally: evidence_store._connection.close()
            # observed_at是原判定时刻，重算只比较同一持久责任，不重新解释时间有效性。
            actual_evidence['persisted_state']['follow_up_state']['observed_at'] = semantic['evidence']['persisted_state']['follow_up_state']['observed_at']
            if actual_evidence != semantic['evidence']:
                raise ValueError('topic semantic evidence differs from actual tools, context or persistent facts')
            recovery = owner.assess_action_contract_recovery(events, runtime_status=turn['status'], final_reply=turn['reply'],
                business_checks_passed=True, semantic_passed=True)
            if recovery != turn['recovery_metrics'] or recovery['quality_status'] != 'passed': raise ValueError('topic recovery proof fails')
            if task_id is None: task_id = after['task_context_id']
            if game_id is None and after['games']: game_id = after['games'][0]['game_id']


def main() -> int:
    """默认完整执行本资产，validate-only仅核资产，不调用任何模型。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    cases = load_cases()
    if args.validate_only: print(json.dumps({'case_count': len(cases), 'turn_count': sum(len(case['turns']) for case in cases)})); return 0
    load_dotenv_defaults(ROOT / ".env")
    client = OpenAICompatibleAgentClient.from_env()
    if client is None: raise ValueError('main model configuration is required')
    args.work_dir.mkdir(parents=True, exist_ok=False)
    runtime_args = SimpleNamespace(max_steps=10, max_tokens_per_call=64000, max_calls_per_turn=12,
        skip_text_generation=True, skip_review=False, timeout_seconds=90)
    reports = [run_case(case, client, runtime_args, args.work_dir / case['case_id']) for case in cases]
    payload = {'status': 'passed' if all(item['status'] == 'passed' for item in reports) else 'failed',
        'execution_scope': 'live_multi_turn_topic_continuity', 'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        'query_memory_policy_sha256': QUERY_MEMORY_POLICY_SHA256,
        'case_count': len(cases), 'reports': reports}
    report_path = args.work_dir / 'report.json'; report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    if payload['status'] == 'passed': validate_scope(payload, report_path)
    return 0 if payload['status'] == 'passed' else 1


if __name__ == '__main__': raise SystemExit(main())
