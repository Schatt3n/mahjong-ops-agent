"""多轮回放运行器的真实 HTTP 协议、SQLite、来源绑定及反例；不计模型语义成绩。"""
from copy import deepcopy
from email.message import Message
import json
from pathlib import Path
from shutil import copytree
import sys
from types import SimpleNamespace

import pytest
from mahjong_agent_runtime import AgentLLMConfig, OpenAICompatibleAgentClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_conversation_topic_eval as runner
import run_inquiry_grounding_eval as inquiry


class Response:
    """本地协议替身不接公网，业务工具与持久化仍完整执行。"""
    status = 200
    def __init__(self, raw): self.raw = raw; self.headers = Message()
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def read(self): return json.dumps({'model': 'unit-model', 'choices': [{'finish_reason': 'stop', 'message': {'content': self.raw}}]}).encode()


def action(reply='', tool=None, arguments=None):
    return {'goal': '完成本次客户目标', 'objective_status': 'needs_tool' if tool else 'completed',
        'reasoning_summary': '受控测试选择', 'objective_state': {},
        'tool_calls': [{'name': tool, 'arguments': arguments, 'reason': '真实执行'}] if tool else [],
        'reply_to_user': reply, 'needs_human': False,
        'stop_reason': {'can_stop': not tool, 'why': '当前步骤完成', 'pending_work': ['等待工具'] if tool else [], 'depends_on_tool_results': bool(tool)}}


def local_client(case, *, remember_query=False):
    """按显式脚本回合选择工具，仅验证执行链路，不把测试桩当语义理解器。"""
    counts = {}
    def transport(request, **kwargs):
        body = json.loads(request.data); payload = json.loads(body['messages'][-1]['content'])
        if 'review_items' in payload:
            raw = {'approved': True, 'needs_human': False, 'reasoning_summary': '本地审查桩', 'violations': [],
                'item_reviews': [{'item_id': item['item_id'], 'approved': True, 'suggested_safe_text': item['text'],
                    'reasoning_summary': '本地桩', 'violations': []} for item in payload['review_items']]}
        else:
            message = payload['current_message']; message_id = message['message_id']; counts[message_id] = counts.get(message_id, 0) + 1
            step = counts[message_id]
            source = case['turns'][int(message_id.rsplit(':', 1)[1])]
            req = {'game_type': 'hangzhou_mahjong', 'stake': '1', 'smoke_preference': 'no_smoke', 'start_time_kind': 'asap_when_full'}
            if source['kind'] == 'delegation':
                if step == 1: raw = action(tool='search_current_games', arguments={'requirement': req})
                elif step == 2: raw = action(tool='create_game', arguments={'organizer_id': 'topic_customer', 'organizer_name': '客户', 'requirement': req, 'known_players': []})
                elif step == 3:
                    created = next(item['result']['game']['game_id'] for item in payload['turn_tool_evidence'] if item['name'] == 'create_game')
                    raw = action(tool='search_customers', arguments={'game_id': created, 'requirement': req, 'exclude_customer_ids': ['topic_customer']})
                else: raw = action('好，给你记上了。')
            elif source['kind'] == 'pending':
                raw = action('打几块的？'); raw['objective_status'] = 'waiting_user'
            elif source['kind'] == 'mixed':
                raw = action(tool='record_user_memory', arguments={'task_memories': [{'customer_id': 'topic_customer',
                    'memory_type': 'constraint', 'field': 'max_duration_hours', 'value': 4,
                    'scope': 'current_task', 'evidence': source['text'], 'confidence': 1.0}]}) if step == 1 else action('行，最多四小时。')
            elif source['kind'] == 'cancel':
                raw = action(tool='update_game_status', arguments={'game_id': payload['active_games'][0]['game_id'],
                    'status': 'cancelled', 'reason': source['text']}) if step == 1 else action('好，取消了。')
            elif source['kind'] == 'duration_query':
                if remember_query and step == 1:
                    raw = action(tool='record_user_memory', arguments={'task_memories': [{'customer_id': 'topic_customer',
                        'memory_type': 'constraint', 'field': 'max_duration_hours', 'value': 4,
                        'scope': 'current_task', 'evidence': source['text'], 'confidence': 1.0}]})
                else:
                    raw = action(tool='search_current_games', arguments={'requirement': {**req, 'max_duration_hours': 4,
                        'requesting_party': {'seat_count': 1}}}) if step == (2 if remember_query else 1) else action('有一个约好打四小时的，还有一个时长没定。')
            elif source['kind'] == 'casual': raw = action('嗯，可以接着聊这个。')
            else: raw = action('现在还差三个人。')
        return Response(json.dumps(raw, ensure_ascii=False))
    return OpenAICompatibleAgentClient(config=AgentLLMConfig(api_key='offline-only', model='unit-model', base_url='https://example.invalid',
        provider='deepseek', thinking_mode='enabled', reasoning_effort='high', max_tokens=16384, retry_attempts=1), urlopen=transport)


@pytest.fixture(scope='module', params=[*((index, False) for index in range(9)), (8, True)])
def actual_case(tmp_path_factory, request):
    """每个场景生成一份主内核和独立判定的离线证据，供同参数的用例复用。"""
    def semantic_transport(request, **kwargs):
        payload = json.loads(json.loads(request.data)['messages'][-1]['content'])
        return Response(json.dumps({'reason': '离线接线判决', 'criteria': [
            {'id': item['id'], 'reason': '离线合同', 'verdict': 'clear'} for item in payload['expectation']['checks']]}))
    case = runner.load_cases()[request.param[0]]; client = local_client(case, remember_query=request.param[1])
    args = SimpleNamespace(max_steps=8, max_tokens_per_call=64000, max_calls_per_turn=10, skip_text_generation=True,
        skip_review=False, timeout_seconds=5, review_client=client)
    root = tmp_path_factory.mktemp('topic_baseline')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inquiry, 'OpenAICompatibleAgentClient',
            lambda *, config: OpenAICompatibleAgentClient(config=config, urlopen=semantic_transport))
        report = runner.run_case(case, client, args, root / 'case')
    assert report['status'] == 'passed', report
    payload = {'status': 'passed', 'execution_scope': 'live_multi_turn_topic_continuity',
        'query_memory_policy_sha256': runner.QUERY_MEMORY_POLICY_SHA256,
        'dataset_sha256': runner.hashlib.sha256(runner.DATASET.read_bytes()).hexdigest(), 'reports': [report]}
    return case, payload, root


@pytest.fixture
def isolated_actual_case(actual_case, tmp_path, monkeypatch):
    """每项测试用独立证据副本；注入故障前完整校验副本，避免路径错误让反例误通过。"""
    case, baseline, root = actual_case
    payload = deepcopy(baseline)
    copytree(root / 'case', tmp_path / 'case')
    report, = payload['reports']
    report['db_path'] = str(tmp_path / Path(report['db_path']).relative_to(root))
    for turn in report['turns']:
        for field in ('raw_trace_path', 'db_path', 'before_db_path'):
            turn[field] = str(tmp_path / Path(turn[field]).relative_to(root))
    (tmp_path / 'case' / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    monkeypatch.setattr(runner, 'load_cases', lambda: [case])
    path = tmp_path / 'report.json'
    runner.validate_scope(payload, path)
    return payload, path


def test_source_covers_every_general_topic_and_real_duration_contrast():
    cases = runner.load_cases()
    assert len(cases) == 9 and sum(len(case['turns']) for case in cases) == 29
    assert cases[-1]['inventory_duration_hours'] == [4, 5, None]
    assert all(case['source']['not_original_human_transcript'] for case in cases)


def test_actual_multi_turn_execution_preserves_game_and_task(isolated_actual_case):
    payload, _ = isolated_actual_case
    turns = payload['reports'][0]['turns']
    assert len({turn['after']['task_context_id'] for turn in turns}) == 1
    assert len({turn['after']['games'][0]['game_id'] for turn in turns if turn['after']['games']}) == 1
    assert all(turn['semantic_assessment']['passed'] and all(turn['checks'].values()) for turn in turns)


@pytest.mark.parametrize('fault', ['source', 'reply', 'request', 'state', 'semantic_reply', 'http', 'hard_check', 'frozen',
                                 'main_request', 'main_expectation_leak', 'before_state', 'semantic_state'])
def test_current_scope_rejects_changed_or_misbound_evidence(isolated_actual_case, fault):
    value, path = isolated_actual_case; report = value['reports'][0]; turn = report['turns'][min(1, len(report['turns'])-1)]
    if fault == 'source': report['source']['turns'][0]['text'] = 'other'
    elif fault == 'reply': turn['reply'] = 'changed'
    elif fault == 'request': turn['request']['text'] = 'other'
    elif fault == 'state': turn['after']['games'] = [{'corrupt': True}]
    elif fault == 'before_state': turn['before']['scheduled'] = [{'corrupt': True}]
    elif fault == 'semantic_state': turn['semantic_assessment']['evidence']['persisted_state']['games'] = [{'corrupt': True}]
    elif fault == 'semantic_reply': turn['semantic_assessment']['evidence']['reply'] = 'other'
    elif fault == 'hard_check': turn['checks']['no_unrequested_business_change'] = False
    elif fault == 'frozen': report['main_agent_executed'] = False
    else:
        raw_path = Path(turn['raw_trace_path']); raw = json.loads(raw_path.read_text())
        if fault == 'http': next(event for event in raw['events'] if event['step'] == 'llm_response')['content'].pop('response_source')
        else:
            for event in raw['events']:
                if event['step'] in {'llm_prompt', 'llm_prompt_after_context_compaction'}:
                    messages = event['content']['messages']
                    if fault == 'main_expectation_leak': messages[0]['content'] += report['source']['turns'][0]['meaning']
                    else:
                        context = json.loads(messages[-1]['content']); context['current_message']['text'] = 'other'
                        messages[-1]['content'] = json.dumps(context)
        raw_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError): runner.validate_scope(value, path)


@pytest.mark.parametrize('field', ['waiting', 'scheduled', 'reservations'])
def test_casual_hard_contract_keeps_all_pending_business_responsibilities(field):
    """取消/新增跟进或候补也属于业务副作用，不能只观察牌局人数。"""
    before = {key: [] for key in ('games', 'invites', 'outbound', 'memories', 'waiting', 'scheduled', 'reservations')}
    before['task_context_id'] = 'same'; after = deepcopy(before); after[field].append({'unexpected': True})
    checks = runner.checks_for({'game_count': 0, 'no_business_change': True}, before, after, [])
    assert checks['no_unrequested_business_change'] is False
