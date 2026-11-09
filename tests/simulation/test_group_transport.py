"""自动模拟必须走真实群入口，并以实际通道收据验证公开和私密消息隔离。"""
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import sys
import time

import pytest

from mahjong_agent_runtime.models import now

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behavior_policy import BehaviorPolicy, SimulationAction
from sim_adapter import SimulationAdapter, RequestOutcome, build_runtime, running_http_backend
from sim_factory import GROUP_ID, build_population
from sim_orchestrator import SimulationOrchestrator, outcome_to_transcript_entry, _intentional_silence


def action(user, text, sequence=1):
    """合成一个指定角色的群输入，绝不混入其他人的私聊上下文。"""
    return SimulationAction(due_simulated_seconds=0, sequence=sequence, channel='group',
        conversation_id=f'sim:group:{GROUP_ID}', sender_id=user.customer_id,
        sender_name=user.display_name, text=text, thread_id='request-thread')


def seed_private_option(store, host, *, occupied=3):
    """用其他客户的真实待组局测试发送条件，原请求者仍未入局。"""
    game, _ = store.create_game(conversation_id=f'sim:private:{host.customer_id}',
        organizer_id=host.customer_id, organizer_name=host.display_name,
        requirement={'game_type': 'national_standard_mahjong', 'stake': '1', 'smoke_preference': 'no_smoke',
                     'start_time_kind': 'asap_when_full', 'requesting_party': {'seat_count': occupied}},
        known_players=[{'customer_id': host.customer_id, 'display_name': host.display_name,
                        'status': 'confirmed', 'seat_count': occupied}], trace_id='seed-private-option')
    assert game.status.value == 'forming'
    return game


def successful_search(response):
    """读取本轮真实成功工具结果，区分找到选项和空局池。"""
    search, = [item for item in response['tool_results'] if item['name'] == 'search_current_games']
    assert search['called'] and search['allowed'] and not search['error']
    return search['result']


def test_friend_switch_private_is_not_broadcast_and_next_turn_stays_private(tmp_path):
    store, users = build_population(tmp_path/'test_sim.db')
    user, host = users[80:82]
    game = seed_private_option(store, host)
    runtime, client = build_runtime(store, 'mock')
    client.semantic_intent = 'delegate'
    request = action(user, '帮我约个无烟局')
    with running_http_backend(runtime) as url:
        adapter = SimulationAdapter(base_url=url, users=users)
        result = adapter.send(request, deadline=time.monotonic()+10)
        assert result.status_code == 200, result.response
        search = successful_search(result.response)
        assert [row['game']['game_id'] for row in search['matches']] == [game.game_id]
        group = result.response['group_processing']
        assert group['entrypoint'] == 'GroupMessageHandler'
        assert group['routing']['action'] == 'private_switch'
        private, = group['outputs']
        assert private['channel'] == 'private' and private['recipient_id'] == user.customer_id
        assert private['conversation_id'] == f'sim:private:{user.customer_id}'
        assert private['metadata']['source_message_id'] == request.message_id
        assert result.inbox_deliveries == 1
        received, = adapter.inbox_for(user.customer_id)
        assert received.channel == 'private' and received.text == private['text']
        assert all(not adapter.inbox_for(other.customer_id) for other in users if other != user)
        assert store.require_game(game.game_id).to_dict() == game.to_dict()
        transcript = outcome_to_transcript_entry(result)
        assert transcript['channel'] == 'group'
        assert transcript['group_processing']['outputs'] == [private]
        orchestrator = SimulationOrchestrator(users=users, behavior_policy=BehaviorPolicy(users), adapter=adapter,
            max_messages=3, max_duration_seconds=1, report_path=tmp_path/'unused.json')
        orchestrator._advance_dialog_after_outcome(result, allow_schedule=True)
        state = orchestrator.active_sessions[user.customer_id]
        assert state.last_channel == state.business_channel == 'private'
        assert state.last_conversation_id == private['conversation_id']
        queued = list(orchestrator._schedule.queue)
        assert queued and all(item.channel == 'private' and item.conversation_id == private['conversation_id'] for item in queued)
        again = adapter.send(request, deadline=time.monotonic()+10)
        assert again.response['delivery_deduplicated'] is True
        assert again.inbox_deliveries == 0 and client.call_count == 3
        assert client.semantic_call_count == 1
        assert adapter.inbox_for(user.customer_id) == [received]


def test_friend_switch_with_empty_pool_keeps_generated_reply_internal(tmp_path):
    """新委托找不到具体可选局时，仍处理本人上下文，但没有群回执或私聊送达。"""
    store, users = build_population(tmp_path/'test_sim.db')
    runtime, client = build_runtime(store, 'mock')
    client.semantic_intent = 'delegate'
    user = users[80]
    with running_http_backend(runtime) as url:
        adapter = SimulationAdapter(base_url=url, users=users)
        result = adapter.send(action(user, '帮我约个无烟局'), deadline=time.monotonic()+10)
        assert result.status_code == 200, result.response
        search = successful_search(result.response)
        assert search['matches'] == search['alternatives'] == []
        group = result.response['group_processing']
        assert group['routing']['action'] == 'private_switch'
        assert group['routing']['detail']['private_reply_suppressed'] is True
        internal, = group['runtime_results']
        assert internal['conversation_id'] == f'sim:private:{user.customer_id}' and internal['final_reply']
        assert group['outputs'] == group['blocked_outputs'] == []
        assert result.response['final_reply'] == '' and 'reply_delivery' not in result.response
        assert result.inbox_deliveries == 0 and all(not adapter.inbox_for(member.customer_id) for member in users)


@pytest.mark.parametrize(('occupied', 'requesting_seats', 'allow_public'),
                         [(2, 1, False), (3, 1, True), (2, 2, True), (1, 3, True)])
def test_nonfriend_public_option_fallback_requires_a_table_filling_match(
        tmp_path, monkeypatch, occupied, requesting_seats, allow_public):
    """无私聊权限时，精确选项足以补齐才允许公开征询，且座位仍由本人确认。"""
    store, users = build_population(tmp_path/'test_sim.db')
    user, host = users[80:82]
    identity = store.get_channel_identity('console', user.customer_id)
    identity.is_friend = identity.can_private_message = False
    store.upsert_channel_identity(identity)
    game = seed_private_option(store, host, occupied=occupied)
    runtime, client = build_runtime(store, 'mock')
    client.semantic_intent = 'delegate'
    original_tool_action = client._tool_action

    def query_with_party():
        result = original_tool_action()
        result['tool_calls'][0]['arguments']['requirement']['requesting_party'] = {'seat_count': requesting_seats}
        return result

    monkeypatch.setattr(type(client), '_tool_action', staticmethod(query_with_party))
    with running_http_backend(runtime) as url:
        adapter = SimulationAdapter(base_url=url, users=users)
        result = adapter.send(action(user, f'@老板 帮我们{requesting_seats}个人找个无烟的'), deadline=time.monotonic()+10)
        assert result.status_code == 200, result.response
        option, = successful_search(result.response)['matches']
        assert option['join_projection']['would_fill_game'] is allow_public
        processing = result.response['group_processing']
        assert processing['routing']['action'] == 'agent_loop'
        assert [item['channel'] for item in processing['outputs']] == (['group'] if allow_public else [])
        assert store.require_game(game.game_id).remaining_seats() == 4 - occupied


def test_nonfriend_uses_group_and_classified_chitchat_skips_main_model(tmp_path):
    store, users = build_population(tmp_path/'test_sim.db')
    user = users[80]
    identity = store.get_channel_identity('console', user.customer_id)
    identity.is_friend = identity.can_private_message = False
    store.upsert_channel_identity(identity)
    runtime, client = build_runtime(store, 'mock')
    with running_http_backend(runtime) as url:
        adapter = SimulationAdapter(base_url=url, users=users)
        response = adapter.send(action(user, '帮我约个无烟局'), deadline=time.monotonic()+10)
        assert response.status_code == 200, response.response
        processing = response.response['group_processing']
        assert processing['routing']['action'] == 'agent_loop'
        assert [item['channel'] for item in processing['outputs']] == ['group']
        assert response.inbox_deliveries == 100
        called = client.call_count
        client.semantic_intent = 'chitchat'
        ignored = adapter.send(action(user, '今天外面也太热了', 2), deadline=time.monotonic()+10)
        assert ignored.response['group_processing']['routing']['action'] == 'ignore'
        assert ignored.inbox_deliveries == 0 and client.call_count == called + 1
        assert client.semantic_call_count == 2
        assert _intentional_silence(ignored.response)
        assert ignored.response['group_processing']['external_wechat_sent'] is False


def test_legacy_group_label_alone_does_not_authorize_broadcast(tmp_path):
    store, users = build_population(tmp_path/'test_sim.db')
    adapter = SimulationAdapter(base_url='http://127.0.0.1:1', users=users)
    outcome = RequestOutcome(action=action(users[80], '还有吗'), sent=True,
        response={'final_reply':'私聊敏感正文', 'status':'completed'})
    assert adapter._deliver_reply(outcome) == 0
    assert all(not adapter.inbox_for(user.customer_id) for user in users)
    assert not _intentional_silence({'status':'completed','final_reply':'', 'group_processing':{
        'entrypoint':'GroupMessageHandler','routing':{'action':'agent_loop'},'outputs':[]}})


def test_group_message_identity_conflict_is_rejected_without_second_delivery(tmp_path):
    store, users = build_population(tmp_path/'test_sim.db')
    runtime, client = build_runtime(store, 'mock')
    with running_http_backend(runtime) as url:
        adapter = SimulationAdapter(base_url=url, users=users)
        original = action(users[80], '现在有人吗')
        first = adapter.send(original, deadline=time.monotonic()+10)
        assert first.status_code == 200 and first.inbox_deliveries == 100
        second = adapter.send(replace(original, text='帮我约个局'), deadline=time.monotonic()+10)
        assert second.status_code == 500 and 'conflict' in second.response['detail']
        assert client.call_count == 3
        assert client.semantic_call_count == 1


def test_superseded_private_option_preserves_fence_evidence_without_delivery(tmp_path, monkeypatch):
    """有真实匹配仍必须通过交付栅栏，失效回复只保留内部审计。"""
    store, users = build_population(tmp_path/'test_sim.db')
    user, host = users[80:82]
    game = seed_private_option(store, host)
    runtime, client = build_runtime(store, 'mock')
    client.semantic_intent = 'delegate'
    original = type(store).claim_assistant_reply_delivery
    monkeypatch.setattr(type(store), 'claim_assistant_reply_delivery', lambda self, *args, **kwargs:
                        False if self is store else original(self, *args, **kwargs))
    with running_http_backend(runtime) as url:
        adapter = SimulationAdapter(base_url=url, users=users)
        result = adapter.send(action(user, '帮我约个无烟局'), deadline=time.monotonic()+10)
        assert result.status_code == 200, result.response
        assert [row['game']['game_id'] for row in successful_search(result.response)['matches']] == [game.game_id]
        group = result.response['group_processing']
        assert group['routing']['action'] == 'private_switch' and group['outputs'] == []
        blocked, = group['blocked_outputs']
        assert blocked['reason'] == 'reply_no_longer_deliverable'
        assert blocked['conversation_id'] == f'sim:private:{user.customer_id}'
        assert blocked['trace_id'] == result.response['trace_id'] and blocked['text']
        assert result.inbox_deliveries == 0 and all(not adapter.inbox_for(member.customer_id) for member in users)
        assert result.response['status'] == 'superseded'
        assert result.response['final_reply'] == '' and 'reply_delivery' not in result.response


def test_empty_main_reply_without_silence_evidence_is_not_accepted(tmp_path, monkeypatch):
    """正式入口的空结果仍须有静默合同，不能一律当成群路由忽略。"""
    from mahjong_agent_runtime.models import AgentRuntimeResult
    store, users = build_population(tmp_path/'test_sim.db')
    runtime, _ = build_runtime(store, 'mock')
    monkeypatch.setattr(type(runtime), 'handle_user_message', lambda self, message, **kwargs:
        AgentRuntimeResult(trace_id=kwargs['trace_id'], conversation_id=message.conversation_id,
                           final_reply='', status='completed', actions=[]))
    with running_http_backend(runtime) as url:
        adapter = SimulationAdapter(base_url=url, users=users)
        result = adapter.send(action(users[80], '现在有人吗'), deadline=time.monotonic()+10)
        assert result.status_code == 200
        assert result.response['group_processing']['routing']['action'] == 'agent_loop'
        assert result.response['group_processing']['outputs'] == []
        assert not _intentional_silence(result.response)


def test_late_group_reply_is_fenced_after_same_customers_new_message(tmp_path, monkeypatch):
    """模型完成后尚未外发时到达的新消息使旧回复失效，同群新回复仍正常投递。"""
    import threading
    from group_transport import CapturedRuntime
    store, users = build_population(tmp_path/'test_sim.db')
    runtime, _ = build_runtime(store, 'mock')
    reached, release = threading.Event(), threading.Event()
    original = CapturedRuntime._claim_result
    def delayed(self, result):
        if result.trace_id == 'trace_sim_000001':
            reached.set()
            assert release.wait(10)
        return original(self, result)
    monkeypatch.setattr(CapturedRuntime, '_claim_result', delayed)
    results = {}
    with running_http_backend(runtime) as url:
        adapter = SimulationAdapter(base_url=url, users=users)
        first = threading.Thread(target=lambda: results.update(first=adapter.send(
            action(users[80], '现在有人吗', 1), deadline=time.monotonic()+15)))
        first.start()
        try:
            assert reached.wait(5)
            results['second'] = adapter.send(action(users[80], '今晚有局吗', 2), deadline=time.monotonic()+10)
            assert results['second'].status_code == 200
            assert results['second'].response['group_processing']['outputs']
        finally:
            release.set()
            first.join(10)
            assert not first.is_alive()
        assert results['first'].response['status'] == 'superseded'
        assert results['first'].response['group_processing']['outputs'] == []
        assert results['first'].inbox_deliveries == 0
        assert len(results['first'].response['group_processing']['blocked_outputs']) == 1


def _published_board_claim_fixture(tmp_path, *, start_at):
    """显式日期避免固定钟点随测试运行时刻失效，其他群认领前置保持真实。"""
    from group_transport import SimulationMessenger
    from mahjong_agent_runtime.group_chat import BoardEngine, GroupMessage
    store, users = build_population(tmp_path/'test_sim.db')
    poster, claimant = users[80:82]
    for user in (poster, claimant):
        identity = store.get_channel_identity('console', user.customer_id)
        identity.is_friend = identity.can_private_message = False
        store.upsert_channel_identity(identity)
    messenger = SimulationMessenger(store, GROUP_ID)
    board = BoardEngine(store=store, messenger=messenger)
    game = board.import_game_from_post(GroupMessage(room_id=GROUP_ID, conversation_id=f'sim:group:{GROUP_ID}',
        sender_external_id=poster.customer_id, sender_name=poster.display_name, channel='console',
        text=f'{start_at:%Y-%m-%d %H:%M} 0.5 无烟 371', message_id='original-post'), trace_id='post')
    snapshot = board.publish(GROUP_ID, trace_id='publish')
    assert snapshot is not None
    runtime, client = build_runtime(store, 'mock')
    return store, claimant, game, snapshot, runtime, client


def _send_quoted_board_claim(runtime, claimant, snapshot):
    """引用不可变看板，经本地HTTP和正式群入口处理，不直接调用占位接口。"""
    import json
    import urllib.request
    with running_http_backend(runtime) as url:
        payload = action(claimant, '1来').to_wechat_payload()
        payload['metadata']['raw_wechat_payload']['quoted_message_id'] = snapshot.external_message_id
        request = urllib.request.Request(url+'/api/message', data=json.dumps(payload).encode(),
                                         headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)


def test_board_quote_reaches_production_claim_and_nonfriend_only_gets_public_notice(tmp_path):
    """有效的真实看板引用可占位，非好友只收到公开通知；跨午夜也不落到过去。"""
    anchor = now()
    store, claimant, game, snapshot, runtime, client = _published_board_claim_fixture(
        tmp_path, start_at=anchor + timedelta(hours=1))
    assert game.planned_start_at > anchor and game.expires_at > anchor
    data = _send_quoted_board_claim(runtime, claimant, snapshot)
    assert data['group_processing']['routing']['action'] == 'claim'
    assert data['group_processing']['routing']['detail']['status'] == 'claimed'
    assert data['group_processing']['routing']['game_id'] == game.game_id
    assert all(item['channel'] == 'group' for item in data['group_processing']['outputs'])
    assert any('占上了' in item['text'] for item in data['group_processing']['outputs'])
    assert client.call_count == 0
    current = store.require_game(game.game_id)
    assert any(member.customer_id == claimant.customer_id and member.status in {'joined','confirmed'}
               for member in current.participants)


def test_expired_board_quote_is_rejected_without_claim_or_success_notice(tmp_path):
    """旧快照能解析不等于仍可参加；过期拒绝不得留下占位或成功外发。"""
    anchor = now()
    store, claimant, game, snapshot, runtime, client = _published_board_claim_fixture(
        tmp_path, start_at=anchor - timedelta(hours=5))
    assert game.expires_at < anchor
    before = store.require_game(game.game_id).to_dict()
    data = _send_quoted_board_claim(runtime, claimant, snapshot)
    routing = data['group_processing']['routing']
    assert routing['action'] == 'claim' and routing['game_id'] == game.game_id
    assert routing['detail']['status'] == 'rejected'
    assert routing['detail']['reason'] == 'claim_conflict'
    assert store.require_game(game.game_id).to_dict() == before
    claim_action = action(claimant, '1来').to_wechat_payload()
    assert store.get_game_claim_by_source(claim_action['conversation_id'], claim_action['message_id']) is None
    assert data['group_processing']['outputs'] == [] and client.call_count == 0
