"""验证已知档位的补位协商只使用本次授权，双方私聊各自得到实际交付。"""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import CustomerProfile, ToolResult, UserMessage
from mahjong_agent_runtime.group_chat.stake_confirmation import (
    BOARD_CANDIDATE_DECISION_TASK_TYPE,
    BOARD_CANDIDATE_UPDATE_TASK_TYPE,
    BOARD_STAKE_CONFIRMATION_TASK_TYPE,
    BOARD_STAKE_UPDATE_TASK_TYPE,
    handle_board_candidate_update,
    handle_board_stake_confirmation,
    pending_board_candidate_decisions,
)
from mahjong_agent_runtime.models import ScheduledTaskStatus, now
from mahjong_agent_runtime.scheduled_tasks import ScheduledTaskDeferred
from mahjong_agent_runtime.tracing import InMemoryTraceRecorder
from tests.test_board_stake_confirmation import execute, store


def known_board(store, *, hard_gender=False):
    store.upsert_customer(CustomerProfile('requester', '客户B', gender='未知'))
    requirements = {'game_type': 'hangzhou_mahjong', 'start_time': '19:00', 'stake': '1',
                    'recruitment_mode': 'board_only', 'party_size': 3, 'preferred_gender': '女'}
    if hard_gender:
        requirements['required_gender'] = '女'
    game, _ = store.create_game(conversation_id='group:room:customer:organizer', organizer_id='organizer',
        organizer_name='客户A', requirement=requirements, known_players=[], trace_id='board')
    return game


def candidate_inquiry(store, game, *, trace='fit-inquiry', search=True, metadata=None):
    store.append_user_turn(UserMessage(conversation_id='private:requester', sender_id='requester', sender_name='客户B',
        text='我男生，帮我问问七点那桌男生能不能补位？', message_id=trace, metadata=metadata or {}), trace)
    if search:
        lookup = execute(store, 'search_current_games', {'requirement': {'stake_range': {'min': 0.5, 'max': 2}}}, trace=trace)
        option, = lookup.result['alternatives']
        assert option['game']['game_id'] == game.game_id
        assert option['match_kind'] == 'host_confirmation_required'
        assert option['decision_required_fields'] == ['host_confirmation']
        store.append_tool_turn('private:requester', json.dumps([lookup.to_dict()]), trace)


def request_fit(store, game_id, *, trace='fit-inquiry'):
    return execute(store, 'request_board_candidate_confirmation', {'game_id': game_id, 'candidate_gender': '男'}, trace=trace)


@pytest.mark.parametrize('bad_request', ['prior_turn', 'unknown_game', 'hard_gender', 'group_continuation'])
def test_fit_request_requires_own_current_option_and_soft_preference(store, bad_request):
    game = known_board(store, hard_gender=bad_request == 'hard_gender')
    candidate_inquiry(store, game, metadata={'source': 'group_private_continuation'}
                      if bad_request == 'group_continuation' else None)
    original_tasks = set(store.scheduled_tasks)
    if bad_request == 'prior_turn':
        candidate_inquiry(store, game, trace='new-turn-without-search', search=False)
        result = request_fit(store, game.game_id, trace='new-turn-without-search')
    elif bad_request == 'unknown_game':
        result = request_fit(store, 'invented-game-id')
    else:
        result = request_fit(store, game.game_id)

    if bad_request == 'hard_gender':
        assert result.result['candidate_confirmation'] == {'status': 'unavailable'}
    elif bad_request == 'group_continuation':
        assert not result.allowed and 'retryable_no_effect' not in result.result
    else:
        assert not result.allowed and result.result['next_step_policy']['required_tool'] == 'search_current_games'
    assert set(store.scheduled_tasks) == original_tasks
    assert not store.invite_drafts and not store.outbound_message_drafts


def test_authorized_self_report_asks_only_fit_and_notifies_requester_once(store):
    game = known_board(store)
    profile_before = store.customers['requester'].to_dict()
    game_before = store.require_game(game.game_id).to_dict()
    candidate_inquiry(store, game)
    queued = request_fit(store, game.game_id)
    assert queued.allowed and 'stake_update' not in queued.result
    info = queued.result['candidate_confirmation']
    assert info['status'] == 'pending'
    question = store.claim_scheduled_task(info['task_id'], at=now())
    assert question.task_type == BOARD_STAKE_CONFIRMATION_TASK_TYPE
    assert question.conversation_id == 'private:organizer'
    assert question.payload['candidate_gender'] == '男' and question.payload['inquiry_source_message_id'] == 'fit-inquiry'

    host_question = '七点那桌有个男生想打，行吗？'
    candidate_update = '七点那桌可以补，你还想打吗？'
    generation = {
        'host-question': ('organizer', host_question, {'time': '19:00', 'game': 'hangzhou_mahjong', 'candidate_gender': '男'}),
        'candidate-update': ('requester', candidate_update, {'time': '19:00', 'game': 'hangzhou_mahjong',
                            'host_accepts_candidate': True, 'stake_confirmed': True}),
    }
    outputs = []

    def generate(messages, *, trace_id, **_kwargs):
        _, text, facts = generation[trace_id]
        assert json.loads(messages[1]['content']) == facts
        assert '直接问这桌打多大' not in messages[0]['content']
        return json.dumps({'text': text}, ensure_ascii=False)

    def review(**kwargs):
        recipient, text, facts = generation[kwargs['trace_id']]
        item, = kwargs['review_items']
        assert (item['recipient_id'], item['text']) == (recipient, text)
        assert kwargs['context_payload']['active_game_visible_summaries'] == [facts]
        return ToolResult('customer_visible_content_review', True, True, result={'approved': True})

    def send(recipient, text, *, metadata):
        outputs.append((recipient, text, metadata['conversation_id']))
        return f'sandbox-fit-receipt-{len(outputs)}'

    runtime = SimpleNamespace(store=store, input_receipt_lock=lambda _: nullcontext(), llm_timeout_seconds=45,
        llm_client=SimpleNamespace(complete=generate), trace_recorder=InMemoryTraceRecorder(),
        action_processor=SimpleNamespace(review_customer_visible_content=review))
    messenger = SimpleNamespace(send_private_message=send)
    assert handle_board_stake_confirmation(runtime, messenger, question, 'host-question')
    assert handle_board_stake_confirmation(runtime, messenger, question, 'host-question-retry') is None
    assert outputs == [('organizer', host_question, 'private:organizer')]

    pending, = pending_board_candidate_decisions(store, conversation_id='private:organizer', sender_id='organizer', now=now())
    assert 'requester' not in json.dumps(pending) and pending['question_text'] == host_question
    store.append_user_turn(UserMessage(conversation_id='private:organizer', sender_id='organizer', sender_name='客户A',
        text='男生也行，没问题', message_id='host-accepts'), 'host-accepts')
    decision = {'decision_task_id': pending['decision_task_id'], 'decision': 'accepted', 'evidence_message_id': 'host-accepts'}
    accepted = execute(store, 'record_board_candidate_decision', decision, actor='organizer', trace='host-accepts')
    assert accepted.allowed and accepted.result['candidate_decision']['status'] == 'accepted'
    assert store.scheduled_tasks[pending['decision_task_id']].payload['evidence_message_id'] == 'host-accepts'
    update = store.claim_scheduled_task(accepted.result['candidate_update']['task_id'], at=now())
    assert update.conversation_id == 'private:requester'

    # 桌主可以先答，客户的初始回复确实送达后即可回告，无需客户再发一条消息。
    with pytest.raises(ScheduledTaskDeferred, match='initial_reply_not_delivered'):
        handle_board_candidate_update(runtime, messenger, update, 'candidate-update')
    store.append_assistant_turn('private:requester', '有这桌，我帮你问下。', 'fit-inquiry',
                                metadata={'delivery_status': 'delivered'})
    assert handle_board_candidate_update(runtime, messenger, update, 'candidate-update')
    assert handle_board_candidate_update(runtime, messenger, update, 'candidate-update-retry') is None
    assert outputs == [('organizer', host_question, 'private:organizer'), ('requester', candidate_update, 'private:requester')]
    repeated = execute(store, 'record_board_candidate_decision', decision, actor='organizer', trace='host-retry')
    assert repeated.allowed and not repeated.state_transitions and 'candidate_update' not in repeated.result

    tasks = store.scheduled_tasks.values()
    assert sum(task.task_type == BOARD_CANDIDATE_UPDATE_TASK_TYPE for task in tasks) == 1
    assert sum(task.task_type == BOARD_CANDIDATE_DECISION_TASK_TYPE and task.status == ScheduledTaskStatus.COMPLETED
               for task in tasks) == 1
    assert not any(task.task_type == BOARD_STAKE_UPDATE_TASK_TYPE for task in tasks)
    assert store.require_game(game.game_id).to_dict() == game_before
    assert store.customers['requester'].to_dict() == profile_before
    assert not store.invite_drafts and not store.outbound_message_drafts
