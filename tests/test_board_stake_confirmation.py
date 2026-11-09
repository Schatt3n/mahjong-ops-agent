"""查到真实自招备选后只向桌主核实档位，验证双方会话、权限及入局边界。"""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, ToolResult, UserMessage
from mahjong_agent_runtime.group_chat import ChannelIdentity, GroupRoomPolicy
from mahjong_agent_runtime.group_chat.stake_confirmation import (
    handle_board_candidate_update,
    handle_board_stake_confirmation,
    handle_board_stake_update,
    pending_board_candidate_decisions,
)
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler, ScheduledTaskDeferred
from mahjong_agent_runtime.tracing import InMemoryTraceRecorder


@pytest.fixture(params=['memory', 'sqlite'])
def store(request, tmp_path):
    store = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path / 'stake.db')
    store.upsert_group_room_policy(GroupRoomPolicy(room_id='room', channel='console'))
    for actor in ('organizer', 'requester'):
        store.upsert_channel_identity(ChannelIdentity(channel='console', external_user_id=actor, customer_id=actor,
            public_name=actor, private_conversation_id=f'private:{actor}', can_private_message=True))
    yield store
    if request.param == 'sqlite':
        store._connection.close()


def execute(store, name, arguments, *, actor='requester', trace='inquiry'):
    return ToolGateway(store).execute(ToolCall(name, arguments, '核对原桌事实'), trace_id=trace,
        conversation_id=f'private:{actor}', sender_id=actor, sender_name=actor, step_index=1,
        source_message_id=trace)


def seed_inquiry(store):
    game, _ = store.create_game(conversation_id='group:room:customer:organizer', organizer_id='organizer',
        organizer_name='客户A', requirement={'game_type': 'hangzhou_mahjong', 'start_time': '19:00',
        'recruitment_mode': 'board_only', 'party_size': 3, 'preferred_gender': '女'}, known_players=[], trace_id='register')
    store.append_user_turn(UserMessage(conversation_id='private:requester', sender_id='requester', sender_name='客户B',
                                      text='晚上有人打牌吗', message_id='inquiry'), 'inquiry')
    lookup = execute(store, 'search_current_games', {'requirement': {'stake_range': {'min': 0.5, 'max': 2}}})
    assert lookup.result['alternatives'][0]['game']['game_id'] == game.game_id
    store.append_tool_turn('private:requester', json.dumps([lookup.to_dict()]), 'inquiry')
    return game


def test_stake_request_safely_retries_after_search_in_supplemented_turn(store):
    game = seed_inquiry(store)
    trace = 'supplemented-inquiry'
    store.append_user_turn(UserMessage('private:requester', 'requester', '客户B', '我一个人', message_id=trace), trace)
    args = {'game_id': game.game_id}
    before = set(store.scheduled_tasks)
    refused = execute(store, 'request_board_stake_confirmation', args, trace=trace)
    assert not refused.called and not refused.allowed and refused.result['retryable_no_effect']
    assert refused.result['next_step_policy']['required_tool'] == 'search_current_games'
    assert set(store.scheduled_tasks) == before
    lookup = execute(store, 'search_current_games', {'requirement': {'stake_range': {'min': 0.5, 'max': 2}}}, trace=trace)
    store.append_tool_turn('private:requester', json.dumps([lookup.to_dict()]), trace)
    retried = execute(store, 'request_board_stake_confirmation', args, trace=trace)
    assert retried.allowed and not retried.deduplicated
    assert retried.result['stake_update']['status'] == 'registered'


@pytest.mark.parametrize('gender', ['男', '女', '未知'])
def test_private_inquiry_can_ask_owner_then_only_owner_can_update_stake(store, gender):
    store.upsert_customer(CustomerProfile('requester', '客户B', gender=gender))
    game = seed_inquiry(store)
    assert not execute(store, 'request_board_stake_confirmation', {'game_id': game.game_id}, trace='other').allowed
    queued = execute(store, 'request_board_stake_confirmation', {'game_id': game.game_id})
    task_info = queued.result['stake_confirmation']
    update_id = queued.result['stake_update']['task_id']
    assert queued.allowed and task_info['status'] == 'pending' and 'organizer' not in str(queued.result)
    assert store.claim_scheduled_task(update_id, at=now()) is None
    task = store.claim_scheduled_task(task_info['task_id'], at=now())
    assert task and task.subject_id == 'organizer' and task.conversation_id == 'private:organizer'

    def generate(messages, *, trace_id, timeout_seconds):
        facts = json.loads(messages[1]['content'])
        expected = {'time': '19:00', 'game': 'hangzhou_mahjong'}
        if trace_id == 'requester-update':
            expected['stake_options'] = [{'min': '1', 'max': '1', 'cap': None}]
            if gender == '男':
                expected['host_acceptance_pending'] = True
            assert facts == expected
            return json.dumps({'text': requester_update}, ensure_ascii=False)
        if gender == '男':
            expected['candidate_gender'] = '男'
        assert facts == expected
        return json.dumps({'text': host_question}, ensure_ascii=False)

    def review(**kwargs):
        expected = 'requester' if kwargs['trace_id'] == 'requester-update' else 'organizer'
        assert kwargs['message'].sender_id == expected
        assert all(actor not in str(kwargs['context_payload']) for actor in ('requester', 'organizer'))
        return ToolResult('customer_visible_content_review', True, True, result={'approved': True})

    outputs = []
    host_question = ('有位男生在问七点那桌，男生也可以吗？你们这桌打多大的？'
                     if gender == '男' else '七点那桌你们打多大的？')
    requester_update = ('七点那桌打1块，能不能补位还得确认。'
                        if gender == '男' else '七点那桌档位问到了，打1块的。')

    def send(recipient, text, *, metadata):
        outputs.append((recipient, text, metadata['draft_id']))
        return 'real-sandbox-receipt'

    runtime = SimpleNamespace(store=store, input_receipt_lock=lambda _: nullcontext(), llm_timeout_seconds=45,
        llm_client=SimpleNamespace(complete=generate), trace_recorder=InMemoryTraceRecorder(),
        action_processor=SimpleNamespace(review_customer_visible_content=review))
    messenger = SimpleNamespace(send_private_message=send)
    assert handle_board_stake_confirmation(runtime, messenger, task, 'host-question') is not None
    assert outputs == [('organizer', host_question, task.idempotency_key)]
    assert handle_board_stake_confirmation(runtime, messenger, task, 'host-question-retry') is None
    assert len(outputs) == 1
    assert not execute(store, 'update_game_requirement', {'game_id': game.game_id,
        'requirement_patch': {'stake': '1'}, 'reason': '询局人不是桌主'}, trace='other').allowed
    assert not execute(store, 'update_game_requirement', {'game_id': game.game_id,
        'requirement_patch': {'start_time': '20:00'}, 'reason': '超出档位确认范围'}, actor='organizer', trace='time').allowed
    updated = execute(store, 'update_game_requirement', {'game_id': game.game_id,
        'requirement_patch': {'stake': '1'}, 'reason': '桌主确认'}, actor='organizer', trace='price')
    assert updated.allowed and store.require_game(game.game_id).requirement['stake'] == '1'
    follow_up = store.claim_scheduled_task(update_id, at=now())
    assert follow_up and follow_up.subject_id == 'requester'
    # 即使 A 先回答，B 的首条已知情况也必须先交付。
    with pytest.raises(ScheduledTaskDeferred, match='initial_reply_not_delivered'):
        handle_board_stake_update(runtime, messenger, follow_up, 'requester-update')
    assert len(outputs) == 1
    store.append_assistant_turn('private:requester', '七点有一桌，档位我问一下。', 'inquiry',
                               metadata={'delivery_status': 'delivered'})
    assert handle_board_stake_update(runtime, messenger, follow_up, 'requester-update') is not None
    assert handle_board_stake_update(runtime, messenger, follow_up, 'requester-update-retry') is None
    assert outputs[-1] == ('requester', requester_update, follow_up.idempotency_key)
    assert len(outputs) == 2
    assert store.require_game(game.game_id).remaining_seats() == 1 and not store.invite_drafts


@pytest.mark.parametrize(('draft', 'error'), [
    ('七点那桌打多大？你直接说就行，我不猜档位。', 'content review did not approve delivery'),
    ('七点那桌打多大？还有哪些要核实？', 'includes extra questions'),
    ('想跟你确认一下七点这桌的具体情况，我这边需要知道你们对档位怎么安排，麻烦你有空的时候回一下。打多大？',
     'failed the text contract'),
])
def test_stake_question_with_internal_copy_or_extra_questions_is_not_sent(store, draft, error):
    game = seed_inquiry(store)
    queued = execute(store, 'request_board_stake_confirmation', {'game_id': game.game_id})
    task = store.claim_scheduled_task(queued.result['stake_confirmation']['task_id'], at=now())

    def generate(messages, **_kwargs):
        assert '直接问这桌打多大' in messages[0]['content']
        assert '正文不复述写作要求' in messages[0]['content']
        return json.dumps({'text': draft}, ensure_ascii=False)

    def review(**kwargs):
        assert kwargs['review_items'][0]['text'] == draft
        assert '正文复述生成要求' in kwargs['context_payload']['customer_visibility_contract']['message_scope']
        return ToolResult('customer_visible_content_review', True, True,
                          result={'approved': False, 'violations': ['leaks_internal_workflow']})

    runtime = SimpleNamespace(store=store, llm_timeout_seconds=45, llm_client=SimpleNamespace(complete=generate),
        trace_recorder=InMemoryTraceRecorder(), action_processor=SimpleNamespace(review_customer_visible_content=review))

    def send(*_args, **_kwargs):
        pytest.fail('审查拒绝的文本不能发给桌主')

    with pytest.raises(RuntimeError, match=error):
        handle_board_stake_confirmation(runtime, SimpleNamespace(send_private_message=send), task, 'host-question')
    assert store.resolve_message_reference(conversation_id=task.conversation_id, message_id=task.idempotency_key) is None


@pytest.mark.parametrize('rewrite_approved', [True, False])
def test_stake_notice_regenerates_after_semantic_rejection_and_requires_fresh_approval(store, rewrite_approved):
    store.upsert_customer(CustomerProfile('requester', '客户B', gender='男'))
    game = seed_inquiry(store)
    queued = execute(store, 'request_board_stake_confirmation', {'game_id': game.game_id})
    host_task = store.claim_scheduled_task(queued.result['stake_confirmation']['task_id'], at=now())
    update_id = queued.result['stake_update']['task_id']
    host_question = '你们打多大啊，有个男生想打，行吗？'
    # 不带问号仍然是在索要回复，必须由独立审查识别。
    first_draft = '七点那桌一块的，你来不'
    suggested = '今晚七点那桌打1块，能不能补位还要确认。'
    second_draft = ('七点那桌打一块，补位还得确认。' if rewrite_approved
                    else '七点那桌打一块的，这档合适不')
    generation_prompts, reviewed, outputs = [], [], []

    def generate(messages, *, trace_id, **_kwargs):
        if trace_id == 'host-question':
            return json.dumps({'text': host_question}, ensure_ascii=False)
        assert json.loads(messages[1]['content'])['stake_options'] == [{'min': '1', 'max': '1', 'cap': None}]
        generation_prompts.append(messages)
        assert len(generation_prompts) <= 2
        draft = first_draft if len(generation_prompts) == 1 else second_draft
        return json.dumps({'text': draft}, ensure_ascii=False)

    def review(**kwargs):
        if kwargs['trace_id'] == 'host-question':
            return ToolResult('customer_visible_content_review', True, True, result={'approved': True})
        item, = kwargs['review_items']
        assert item['recipient_id'] == 'requester' and item['item_id'] == 'board_stake_update'
        assert kwargs['action'].objective_status == 'completed'
        reviewed.append(item['text'])
        approved = rewrite_approved and len(reviewed) == 2
        violations = [] if approved else ['semantic_contradiction']
        return ToolResult('customer_visible_content_review', True, True, result={
            'approved': approved, 'needs_human': False, 'violations': violations,
            'item_reviews': [{'item_id': item['item_id'], 'approved': approved, 'violations': violations,
                              'suggested_safe_text': item['text'] if approved else suggested}],
        })

    def send(recipient, text, *, metadata):
        outputs.append((recipient, text, metadata['draft_id']))
        return f'sandbox-receipt-{len(outputs)}'

    trace = InMemoryTraceRecorder()
    runtime = SimpleNamespace(store=store, input_receipt_lock=lambda _: nullcontext(), llm_timeout_seconds=45,
        llm_client=SimpleNamespace(complete=generate), trace_recorder=trace,
        action_processor=SimpleNamespace(review_customer_visible_content=review))
    messenger = SimpleNamespace(send_private_message=send)
    assert handle_board_stake_confirmation(runtime, messenger, host_task, 'host-question')
    store.append_assistant_turn('private:requester', '七点有一桌，档位我问一下。', 'inquiry',
                               metadata={'delivery_status': 'delivered'})
    assert execute(store, 'update_game_requirement', {'game_id': game.game_id,
        'requirement_patch': {'stake': '1'}, 'reason': '桌主确认'}, actor='organizer', trace='price').allowed

    def dispatch(task, trace_id):
        handle_board_stake_update(runtime, messenger, task, trace_id)

    scheduler = ScheduledAgentTaskScheduler(store, dispatch, trace, max_attempts=1)
    assert scheduler.run_task_once(update_id) == 1
    assert reviewed == [first_draft, second_draft]
    assert 'semantic_contradiction' in json.dumps(generation_prompts[1], ensure_ascii=False)
    assert scheduler.run_task_once(update_id) == 0
    receipt = store.resolve_message_reference(conversation_id='private:requester', message_id=update_id)
    if rewrite_approved:
        assert outputs[1:] == [('requester', second_draft, update_id)]
        assert receipt is not None and receipt.text == second_draft
        assert store.scheduled_tasks[update_id].status.value == 'completed'
    else:
        assert outputs[1:] == [] and receipt is None
        assert store.scheduled_tasks[update_id].status.value == 'failed'
    assert store.require_game(game.game_id).remaining_seats() == 1 and not store.invite_drafts


def test_unreachable_owner_does_not_create_task_or_use_public_fallback(store):
    game = seed_inquiry(store)
    identity = store.get_channel_identity('console', 'organizer')
    identity.can_private_message = False
    store.upsert_channel_identity(identity)
    result = execute(store, 'request_board_stake_confirmation', {'game_id': game.game_id})
    assert result.result['stake_confirmation']['status'] == 'unavailable'
    assert all(task.task_type == 'publish_group_board' for task in store.scheduled_tasks.values())


@pytest.mark.parametrize('stake_known', [False, True])
def test_host_decision_from_prior_message_notifies_original_requester_without_their_reply(store, stake_known):
    store.upsert_customer(CustomerProfile('requester', '客户B', gender='男'))
    game = seed_inquiry(store)
    queued = execute(store, 'request_board_stake_confirmation', {'game_id': game.game_id})
    question = store.claim_scheduled_task(queued.result['stake_confirmation']['task_id'], at=now())
    outputs = []

    def generate(messages, *, trace_id, **_kwargs):
        facts = json.loads(messages[1]['content'])
        if trace_id == 'host-question':
            assert facts['candidate_gender'] == '男'
            return '{"text":"你们打多大啊，有个男生想打，行吗？"}'
        assert facts['host_accepts_candidate']
        assert facts['stake_confirmed'] == stake_known
        assert 'stake_options' not in facts
        assert 'candidate_gender' not in facts and 'organizer' not in str(facts)
        return ('{"text":"七点那桌可以补，你还想打吗？"}' if stake_known
                else '{"text":"七点那桌可以补，档位还在问。"}')

    def send(recipient, text, *, metadata):
        outputs.append((recipient, text, metadata['conversation_id']))
        return f'sandbox-receipt-{len(outputs)}'

    runtime = SimpleNamespace(store=store, input_receipt_lock=lambda _: nullcontext(), llm_timeout_seconds=45,
        llm_client=SimpleNamespace(complete=generate), trace_recorder=InMemoryTraceRecorder(),
        action_processor=SimpleNamespace(review_customer_visible_content=lambda **_: ToolResult(
            'customer_visible_content_review', True, True, result={'approved': True})))
    messenger = SimpleNamespace(send_private_message=send)
    assert handle_board_stake_confirmation(runtime, messenger, question, 'host-question')
    store.append_assistant_turn('private:requester', '七点有一桌，档位我问一下。', 'inquiry',
                               metadata={'delivery_status': 'delivered'})
    if stake_known:
        assert execute(store, 'update_game_requirement', {'game_id': game.game_id,
            'requirement_patch': {'stake': '1'}, 'reason': '桌主确认'}, actor='organizer', trace='price').allowed
    for message_id, text in [('accept', '男生的话，也行吧'), ('hygiene', '他这人干净吗？')]:
        store.append_user_turn(UserMessage(conversation_id='private:organizer', sender_id='organizer',
            sender_name='客户A', text=text, message_id=message_id), message_id)
    pending = pending_board_candidate_decisions(store, conversation_id='private:organizer', sender_id='organizer', now=now())
    assert len(pending) == 1 and 'requester' not in str(pending)
    args = {'decision_task_id': pending[0]['decision_task_id'], 'decision': 'accepted', 'evidence_message_id': 'accept'}
    assert not execute(store, 'record_board_candidate_decision', args, trace='wrong-session').allowed
    result = execute(store, 'record_board_candidate_decision', args, actor='organizer', trace='hygiene')
    assert result.allowed
    decision = store.scheduled_tasks[args['decision_task_id']]
    assert decision.payload['evidence_message_id'] == 'accept'
    update_id = result.result['candidate_update']['task_id']
    update = store.claim_scheduled_task(update_id, at=now())
    assert update and handle_board_candidate_update(runtime, messenger, update, 'candidate-update')
    assert handle_board_candidate_update(runtime, messenger, update, 'candidate-update-retry') is None
    assert [row[0] for row in outputs] == ['organizer', 'requester']
    assert outputs[-1][2] == 'private:requester'
    assert store.require_game(game.game_id).remaining_seats() == 1 and not store.invite_drafts
