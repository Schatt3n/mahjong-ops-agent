"""个人等待截止的可信执行合同：复用持久调度，不能用记忆或招募触发替代到期责任。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
from threading import Barrier
import json
from pathlib import Path
import sys

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, StaticAgentClient, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.models import AgentAction, ToolResult, now
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler
from mahjong_agent_runtime.services.objective_continuation import blocking_continuation


@pytest.fixture(params=['memory', 'sqlite'])
def store(request, tmp_path):
    result = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path/'deadlines.sqlite3')
    yield result
    if request.param == 'sqlite':
        result._connection.close()


def seed(store, *, members='AB', party_size=1):
    stamp = now()
    planned = stamp+timedelta(hours=31)
    game, _ = store.create_game(conversation_id='private:A', organizer_id='A', organizer_name='甲',
        requirement={'game_type':'sichuan_mahjong', 'stake':'1', 'smoke_preference':'no_smoking',
                     'planned_start_at':planned.isoformat(), 'duration_hours':4, 'party_size':party_size},
        known_players=[{'customer_id':x, 'display_name':x} for x in members], trace_id='seed-game')
    return game, planned+timedelta(minutes=30)


def execute(store, name, args, *, actor='A', source='deadline-input', text=None):
    conversation = 'private:'+actor
    stamp = now()
    context = store.current_task_context(conversation, actor)
    if context is None:
        store.activate_task_context(conversation_id=conversation, customer_id=actor, trace_id=source,
            activity_at=stamp, started_at=stamp, reason='first_message', force_new=True, archive_previous=False)
    if store.find_conversation_turn(conversation, message_id=source) is None:
        store.append_user_turn(UserMessage(conversation_id=conversation, sender_id=actor, sender_name=actor,
            text=text or '明晚最多等到这个时间，还没成局我就不等了。', message_id=source, sent_at=stamp), source)
    return ToolGateway(store).execute(ToolCall(name, args), trace_id=source, step_index=1,
        conversation_id=conversation, sender_id=actor, sender_name=actor, source_message_id=source)


def schedule(store, game, deadline, **kwargs):
    result = execute(store, 'set_participation_deadline', {'game_id':game.game_id, 'deadline_at':deadline.isoformat()}, **kwargs)
    assert result.called and result.allowed and not result.error, result.to_dict()
    task = store.scheduled_tasks[result.result['participation_deadline']['task_id']]
    return result, task


def run_task(store, task_id, at):
    from mahjong_agent_runtime.participation_deadlines import handle_participation_deadline_task
    scheduler = ScheduledAgentTaskScheduler(store,
        lambda task, trace: handle_participation_deadline_task(store, task, trace, at=at), InMemoryTraceRecorder())
    return scheduler.run_task_once(task_id, at=at)


def active(game):
    return {p.customer_id for p in game.participants if p.status in {'joined','confirmed'}}


def test_future_deadline_over_24h_is_bound_to_real_game_and_durable_effect(store):
    game, deadline = seed(store)
    result, task = schedule(store, game, deadline)
    assert deadline-now() > timedelta(hours=24)
    assert task.task_type == 'participation_deadline' and task.due_at == deadline
    assert task.subject_id == 'A' and task.payload['game_id'] == game.game_id
    assert task.payload['deadline_at'] == deadline.isoformat()
    assert task.payload['condition'] == 'game_not_ready' and task.payload['effect'] == 'withdraw_self'
    assert task.payload['source_message_id'] == 'deadline-input'
    assert task.payload['task_context_id'] == store.current_task_context('private:A','A').task_context_id
    assert active(store.require_game(game.game_id)) == {'A','B'}
    assert run_task(store, task.task_id, deadline-timedelta(seconds=1)) == 0
    assert run_task(store, task.task_id, deadline) == 1
    state = store.require_game(game.game_id)
    assert active(state) == {'B'} and state.status.value in {'forming','inviting'}
    assert state.game_id == game.game_id and len(store.games) == 1
    assert result.result['participation_deadline']['status'] == 'pending'


def test_deadline_delivery_and_original_tool_replay_are_idempotent(store):
    game, deadline = seed(store)
    first, task = schedule(store, game, deadline)
    replay, repeated = schedule(store, game, deadline)
    assert replay.deduplicated and repeated.task_id == task.task_id
    assert run_task(store, task.task_id, deadline) == 1
    before = deepcopy(store.require_game(game.game_id).to_dict())
    assert run_task(store, task.task_id, deadline+timedelta(minutes=1)) == 0
    assert store.require_game(game.game_id).to_dict() == before
    assert len([t for t in store.scheduled_tasks.values() if t.task_type=='participation_deadline']) == 1


def test_deadline_survives_sqlite_restart(tmp_path):
    path = tmp_path/'restart.sqlite3'; first = SQLiteAgentStore(path)
    game, deadline = seed(first); _, task = schedule(first, game, deadline)
    first._connection.close()
    reopened = SQLiteAgentStore(path)
    try:
        assert reopened.scheduled_tasks[task.task_id].payload['source_message_id']=='deadline-input'
        run_task(reopened, task.task_id, deadline)
        assert active(reopened.require_game(game.game_id)) == {'B'}
    finally:
        reopened._connection.close()


def test_recovery_after_effect_before_scheduler_ack_never_repeats_withdrawal(store):
    game, deadline = seed(store); _, task = schedule(store, game, deadline)
    claim = store.claim_scheduled_task(task.task_id, at=deadline, lease_seconds=1)
    first, _ = store.execute_participation_deadline(claim, trace_id='effect-before-crash', at=deadline)
    assert first['outcome'] == 'withdraw_self'
    before = deepcopy(store.require_game(game.game_id).to_dict())
    assert run_task(store, task.task_id, deadline+timedelta(seconds=2)) == 1
    assert store.require_game(game.game_id).to_dict() == before
    assert store.scheduled_tasks[task.task_id].status.value == 'completed'


def test_withdrawal_and_board_refresh_rollback_together_on_failure(store,monkeypatch):
    game, deadline = seed(store); _, task = schedule(store, game, deadline)
    claim = store.claim_scheduled_task(task.task_id, at=deadline)
    method = '_ensure_game_board_arrangement_in_transaction' if isinstance(store,SQLiteAgentStore) else '_ensure_game_board_arrangement_locked'
    def fail(*args,**kwargs):raise RuntimeError('board transaction unavailable')
    monkeypatch.setattr(type(store),method,fail)
    with pytest.raises(RuntimeError,match='board transaction'):
        store.execute_participation_deadline(claim,trace_id='rollback',at=deadline)
    assert active(store.require_game(game.game_id)) == {'A','B'}
    assert 'execution_outcome' not in store.scheduled_tasks[task.task_id].payload


def test_expired_current_lease_is_retried_instead_of_completed_without_effect(store):
    from mahjong_agent_runtime.participation_deadlines import handle_participation_deadline_task
    game, deadline = seed(store); _, task = schedule(store,game,deadline)
    trace=InMemoryTraceRecorder()
    scheduler=ScheduledAgentTaskScheduler(store,
        lambda claim,trace_id:handle_participation_deadline_task(store,claim,trace_id,at=deadline+timedelta(seconds=2)),
        trace,lease_seconds=1)
    assert scheduler.run_task_once(task.task_id,at=deadline)==1
    retried=store.scheduled_tasks[task.task_id]
    assert retried.status.value=='pending' and 'execution_outcome' not in retried.payload
    assert active(store.require_game(game.game_id))=={'A','B'}
    assert not any(event.step=='scheduled_agent_task_completed' for event in trace.events)
    assert run_task(store,task.task_id,retried.due_at)==1
    assert active(store.require_game(game.game_id))=={'B'}


def test_rolled_back_deadline_failure_can_retry_same_input_once(store,monkeypatch):
    game,deadline=seed(store)
    method='_save_scheduled_agent_task' if isinstance(store,SQLiteAgentStore) else '_clear_participation_deadlines_locked'
    original=getattr(type(store),method)
    injected=[False]
    def fail_once(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        if not injected[0]:
            injected[0]=True
            raise RuntimeError('temporary persistence failure after write')
        return result
    monkeypatch.setattr(type(store),method,fail_once)
    args={'game_id':game.game_id,'deadline_at':deadline.isoformat()}
    failed=execute(store,'set_participation_deadline',args)
    assert not failed.allowed and failed.result['retryable_no_effect'] is True
    assert not [t for t in store.scheduled_tasks.values() if t.task_type=='participation_deadline']
    recovered=execute(store,'set_participation_deadline',args)
    assert recovered.allowed and not recovered.deduplicated
    replay=execute(store,'set_participation_deadline',args)
    assert replay.allowed and replay.deduplicated
    assert recovered.result['participation_deadline']['task_id']==replay.result['participation_deadline']['task_id']
    assert len([t for t in store.scheduled_tasks.values() if t.task_type=='participation_deadline'])==1


def test_unknown_post_commit_failure_is_not_marked_safe_to_retry(store,monkeypatch):
    game,deadline=seed(store);original=type(store).save_participation_deadline;calls=[]
    def committed_but_unknown(self,**kwargs):
        calls.append('attempt')
        original(self,**kwargs)
        raise RuntimeError('unknown outcome after commit')
    monkeypatch.setattr(type(store),'save_participation_deadline',committed_but_unknown)
    args={'game_id':game.game_id,'deadline_at':deadline.isoformat()}
    failed=execute(store,'set_participation_deadline',args)
    assert not failed.allowed and failed.result['retryable_no_effect'] is False
    repeated=execute(store,'set_participation_deadline',args)
    assert repeated.deduplicated and calls==['attempt']
    assert len([t for t in store.scheduled_tasks.values() if t.task_type=='participation_deadline'])==1


def test_missing_or_system_source_cannot_create_deadline(store):
    game, deadline = seed(store)
    result = ToolGateway(store).execute(ToolCall('set_participation_deadline',{
        'game_id':game.game_id,'deadline_at':deadline.isoformat()}), trace_id='forged',step_index=1,
        conversation_id='private:A',sender_id='A',sender_name='甲',source_message_id='invented')
    assert not result.allowed and result.result['validation_error']['code']=='deadline_source_required'
    assert not [t for t in store.scheduled_tasks.values() if t.task_type=='participation_deadline']


def test_context_deadline_projection_only_includes_same_customer_and_goal(store):
    from mahjong_agent_runtime import AgentContextBuilder
    game, deadline = seed(store)
    _, own = schedule(store,game,deadline)
    schedule(store,game,deadline+timedelta(minutes=15),actor='B',source='b-limit')
    message=UserMessage(conversation_id='private:A',sender_id='A',sender_name='甲',text='看一下',message_id='read')
    builder=AgentContextBuilder(store,ToolGateway(store))
    payload=builder.build(message,trace_id='context-deadline').payload
    assert [task['task_id'] for task in payload['active_participation_deadlines']]==[own.task_id]
    stamp=now()
    store.activate_task_context(conversation_id='private:A',customer_id='A',trace_id='new',activity_at=stamp,
        started_at=stamp,reason='new_goal',force_new=True,archive_previous=False)
    assert builder.build(message,trace_id='new-context').payload['active_participation_deadlines']==[]
    assert store.scheduled_tasks[own.task_id].status.value=='pending'


def test_full_or_ready_game_never_withdraws_customer_at_deadline(store):
    game, deadline = seed(store, members='ABC'); _, task = schedule(store, game, deadline)
    store.join_game(game_id=game.game_id, customer_id='D', display_name='丁', trace_id='last-person')
    assert store.require_game(game.game_id).status.value == 'ready'
    run_task(store, task.task_id, deadline)
    assert active(store.require_game(game.game_id)) == set('ABCD')
    assert store.require_game(game.game_id).status.value == 'ready'


def test_replaced_deadline_revokes_old_lease_and_only_new_time_applies(store):
    game, deadline = seed(store); _, first = schedule(store, game, deadline)
    old_claim = store.claim_scheduled_task(first.task_id, at=deadline)
    assert old_claim
    _, second = schedule(store, game, deadline+timedelta(minutes=30), source='changed-deadline', text='我可以多等半小时，按新的截止时间。')
    from mahjong_agent_runtime.participation_deadlines import handle_participation_deadline_task
    handle_participation_deadline_task(store, old_claim, 'stale-lease', at=deadline)
    assert active(store.require_game(game.game_id)) == {'A','B'}
    assert store.scheduled_tasks[first.task_id].status.value == 'cancelled'
    run_task(store, second.task_id, deadline+timedelta(minutes=30))
    assert active(store.require_game(game.game_id)) == {'B'}


def test_cleared_deadline_cannot_withdraw_later(store):
    game, deadline = seed(store); _, task = schedule(store, game, deadline)
    cleared = execute(store, 'clear_participation_deadline', {'game_id':game.game_id}, source='clear-deadline', text='不用按之前的时间退出了，我接着等。')
    assert cleared.allowed and cleared.called and not cleared.error
    assert store.scheduled_tasks[task.task_id].status.value == 'cancelled'
    assert run_task(store, task.task_id, deadline) == 0
    assert active(store.require_game(game.game_id)) == {'A','B'}


def test_invalid_replacement_preserves_previous_valid_responsibility(store):
    game, deadline = seed(store); _, task = schedule(store, game, deadline)
    refused = execute(store, 'set_participation_deadline', {
        'game_id':game.game_id, 'deadline_at':(game.expires_at+timedelta(days=30)).isoformat(),
    }, source='invalid-change')
    assert not refused.allowed
    assert store.scheduled_tasks[task.task_id].status.value == 'pending'
    run_task(store, task.task_id, deadline)
    assert active(store.require_game(game.game_id)) == {'B'}


def test_unrelated_new_goal_does_not_discard_existing_game_deadline(store):
    game, deadline = seed(store); _, task = schedule(store, game, deadline)
    stamp = now()
    replacement, _ = store.activate_task_context(conversation_id='private:A', customer_id='A',
        trace_id='unrelated-new-goal', activity_at=stamp, started_at=stamp,
        reason='new_goal', force_new=True, archive_previous=True)
    assert replacement.task_context_id != task.payload['task_context_id']
    run_task(store, task.task_id, deadline)
    assert active(store.require_game(game.game_id)) == {'B'}
    assert store.current_task_context('private:A','A').task_context_id == replacement.task_context_id


def test_two_customers_have_independent_deadlines_and_no_global_game_cancel(store):
    game, deadline = seed(store, members='ABC')
    _, first = schedule(store, game, deadline)
    _, second = schedule(store, game, deadline+timedelta(minutes=15), actor='B', source='b-deadline')
    run_task(store, first.task_id, deadline)
    assert active(store.require_game(game.game_id)) == {'B','C'}
    assert store.scheduled_tasks[second.task_id].status.value == 'pending'
    run_task(store, second.task_id, deadline+timedelta(minutes=15))
    state = store.require_game(game.game_id)
    assert active(state) == {'C'} and state.status.value != 'cancelled'


def test_other_known_registered_members_are_preserved_when_contact_exits(store):
    game, deadline = seed(store, members='ABC'); _, task = schedule(store, game, deadline)
    run_task(store, task.task_id, deadline)
    assert active(store.require_game(game.game_id)) == {'B','C'}
    assert all(p.status in {'joined','confirmed'} for p in store.require_game(game.game_id).participants if p.customer_id in {'B','C'})


def test_aggregated_party_without_independent_self_seat_requires_clarification(store):
    game, deadline = seed(store, members='A', party_size=3)
    assert game.participants[0].seat_count == 3
    result = execute(store, 'set_participation_deadline', {'game_id':game.game_id, 'deadline_at':deadline.isoformat()})
    assert not result.allowed and not result.called
    assert result.result.get('validation_error',{}).get('code') == 'participation_scope_ambiguous'
    assert not [t for t in store.scheduled_tasks.values() if t.task_type=='participation_deadline']
    assert store.require_game(game.game_id).seat_summary()['claimed_seats'] == 3


def test_deadline_cannot_target_outsider_or_outlive_known_game_window(store):
    game, deadline = seed(store)
    outsider = execute(store, 'set_participation_deadline', {'game_id':game.game_id, 'deadline_at':deadline.isoformat()}, actor='X')
    assert not outsider.allowed and 'unknown tool' not in (outsider.error or '')
    late = execute(store, 'set_participation_deadline', {'game_id':game.game_id, 'deadline_at':(game.expires_at+timedelta(days=30)).isoformat()}, source='arbitrary-time')
    assert not late.allowed and late.result.get('validation_error',{}).get('code') == 'deadline_outside_game_window'
    assert not [t for t in store.scheduled_tasks.values() if t.task_type=='participation_deadline']


def test_deadline_and_last_join_share_transaction_boundary(store):
    game, deadline = seed(store, members='ABC'); _, task = schedule(store, game, deadline)
    other = SQLiteAgentStore(store.path) if isinstance(store,SQLiteAgentStore) else store
    barrier = Barrier(2)
    def expire():
        barrier.wait(timeout=5)
        return run_task(store, task.task_id, deadline)
    def join():
        barrier.wait(timeout=5)
        return other.join_game(game_id=game.game_id, customer_id='D', display_name='丁', trace_id='concurrent-last-join')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            left,right=pool.submit(expire),pool.submit(join)
            left.result(timeout=10);right.result(timeout=10)
        state=store.require_game(game.game_id)
        assert active(state) in ({'A','B','C','D'}, {'B','C','D'})
        assert (state.status.value=='ready') == (active(state)=={'A','B','C','D'})
        assert state.status.value!='cancelled'
    finally:
        if other is not store:other._connection.close()


def final_action(*, status='completed', calls=None):
    tools=calls or []
    return json.dumps(AgentAction(goal='承接本人截止责任', objective_status='needs_tool' if tools else status,
        reasoning_summary='依据工具事实处理', reply_to_user='' if tools else ('暂时没处理好。' if status=='needs_human' else '行，到点还没齐就不排了。'),
        tool_calls=[ToolCall(**{'reason':'落实本人明确截止条件','call_id':f'call_{i}','depends_on':[],**x}) for i,x in enumerate(tools)], needs_human=status=='needs_human',
        stop_reason={'can_stop':not tools,'why':'先落实截止责任' if tools else '本轮结束',
                     'pending_work':['截止责任'] if tools else [],'depends_on_tool_results':True}).to_dict(),ensure_ascii=False)


@pytest.mark.parametrize('source_channel',['private','group'])
def test_rejected_24h_followup_cannot_complete_on_recruitment_task_and_memory_only(store,source_channel):
    game, deadline=seed(store)
    call={'name':'schedule_follow_up','arguments':{'goal':'到截止时仍未成局则退出本人','waiting_for':'players',
        'due_at':deadline.isoformat(),'expires_at':game.expires_at.isoformat(),'review_budget':1}}
    client=StaticAgentClient([final_action(calls=[call]),final_action(),final_action(status='needs_human')])
    trace=InMemoryTraceRecorder();runtime=AgentRuntime(llm_client=client,store=store,trace_recorder=trace,max_steps=3)
    message=UserMessage(conversation_id='private:A',sender_id='A',sender_name='甲',message_id='real-failed-plan',
        text='明晚九点半不变，最多等到十点，还没成我就不等了。',metadata={'source':source_channel})
    store.record_task_memory(conversation_id='private:A',customer_id='A',memory_type='time_constraint',
        field='latest_wait_at',value=deadline.isoformat(),metadata={'game_id':game.game_id},trace_id='prior-memory')
    result=runtime.handle_user_message(message,trace_id='failed-deadline')
    failed=[x for x in result.tool_results if x.name=='schedule_follow_up']
    assert failed and not failed[0].allowed
    assert store.scheduled_task_for_game(game.game_id).task_type=='activate_game_recruitment'
    assert result.status!='completed'
    assert any(event.step=='objective_continuation_rejected' for event in trace.events)
    assert active(store.require_game(game.game_id))=={'A','B'}


@pytest.mark.parametrize('source_channel',['private','group'])
def test_failed_followup_can_recover_through_real_matching_deadline(store,source_channel):
    game,deadline=seed(store)
    followup={'name':'schedule_follow_up','arguments':{'goal':'到截止仍未成局退出本人','waiting_for':'players',
        'due_at':deadline.isoformat(),'expires_at':game.expires_at.isoformat()}}
    deadline_call={'name':'set_participation_deadline','arguments':{'game_id':game.game_id,'deadline_at':deadline.isoformat()}}
    client=StaticAgentClient([final_action(calls=[followup]),final_action(calls=[deadline_call]),final_action()])
    runtime=AgentRuntime(llm_client=client,store=store,trace_recorder=InMemoryTraceRecorder(),max_steps=4)
    result=runtime.handle_user_message(UserMessage(conversation_id='private:A',sender_id='A',sender_name='甲',
        text='最多等到约定时间，没成局就退出我自己。',message_id='recover-with-deadline',metadata={'source':source_channel}),trace_id='recovery')
    assert result.status=='completed',result.to_dict()
    repaired=next(item for item in result.tool_results if item.name=='set_participation_deadline')
    assert repaired.result['continuation']['resolved_obligation_keys']==[
        'follow_up:'+store.current_task_context('private:A','A').task_context_id]
    assert len([task for task in store.scheduled_tasks.values() if task.task_type=='participation_deadline'])==1


@pytest.mark.parametrize('mismatch',['different_source','ambiguous_games','successful_notification'])
def test_deadline_cannot_discharge_unrelated_or_successful_followup(store,mismatch):
    from mahjong_agent_runtime.services.future_responsibility import resolve_replaced_follow_up
    game,deadline=seed(store)
    if mismatch=='successful_notification':
        game,_=store.update_game_requirement(game_id=game.game_id,requirement_patch={
            'planned_start_at':(now()+timedelta(hours=1)).isoformat()},trace_id='nearer-game',reason='测试近期通知')
        deadline=game.planned_start_at+timedelta(minutes=30)
    failed=execute(store,'schedule_follow_up',{'goal':'到时提醒本人确认','waiting_for':'players',
        'due_at':deadline.isoformat(),'expires_at':game.expires_at.isoformat()},source='old-attempt')
    assert failed.allowed is (mismatch=='successful_notification')
    if mismatch=='ambiguous_games':
        other,_=store.create_game(conversation_id='private:A',organizer_id='A',organizer_name='甲',
            requirement={**game.requirement,'planned_start_at':(game.planned_start_at+timedelta(minutes=30)).isoformat(),'duration_hours':3.5},
            known_players=[{'customer_id':'A','display_name':'甲'}],alternative_to_game_id=game.game_id,trace_id='alternative')
        assert other.expires_at==game.expires_at
    source='new-attempt' if mismatch=='different_source' else 'old-attempt'
    recovered,_=schedule(store,game,deadline,source=source)
    message=UserMessage(conversation_id='private:A',sender_id='A',sender_name='甲',text='最多等到这个时间',message_id=source)
    resolve_replaced_follow_up(store,recovered,[failed],message,source)
    assert not recovered.result['continuation'].get('resolved_obligation_keys')
    if mismatch=='successful_notification':
        assert store.scheduled_tasks[failed.result['follow_up']['task_id']].status.value=='pending'


def test_unrelated_success_cannot_discharge_specific_failed_obligation():
    failed=ToolResult(name='set_participation_deadline',called=False,allowed=False,error='database failure',result={
        'continuation':{'obligation_key':'deadline:g:A','can_stop':False,'allowed_terminal_statuses':['needs_human'],
                        'pending_capabilities':['persist_participation_deadline']}})
    irrelevant=ToolResult(name='search_customers',called=True,allowed=True,result={
        'continuation':{'can_stop':True,'pending_capabilities':[]}})
    action=AgentAction(goal='截止责任',objective_status='completed',reasoning_summary='已做其他查询',reply_to_user='好。')
    assert blocking_continuation(action,[failed,irrelevant]) is not None
    recovered=ToolResult(name='set_participation_deadline',called=True,allowed=True,result={
        'continuation':{'obligation_key':'deadline:g:A','can_stop':True,'pending_capabilities':[]}})
    assert blocking_continuation(action,[failed,irrelevant,recovered]) is None
    optional=ToolResult(name='search_customers',called=False,allowed=False,error='read unavailable')
    assert blocking_continuation(action,[optional]) is None


def test_ordinary_followup_keeps_24h_limit_not_unbounded_by_new_deadline_tool(store):
    stamp=now()
    result=execute(store,'schedule_follow_up',{'goal':'以后再看看','waiting_for':'players',
        'due_at':(stamp+timedelta(days=30)).isoformat(),'expires_at':(stamp+timedelta(days=31)).isoformat()})
    assert not result.allowed and '24' in result.error
    assert not store.scheduled_tasks


def test_production_application_dispatches_personal_deadline(store,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import agent_runtime_app as app
    import mahjong_agent_runtime.participation_deadlines as deadline_handler
    game, deadline = seed(store); _, task = schedule(store,game,deadline)
    runtime=AgentRuntime(llm_client=StaticAgentClient([]),store=store,trace_recorder=InMemoryTraceRecorder())
    monkeypatch.setattr(app,'get_runtime',lambda:runtime)
    monkeypatch.setattr(deadline_handler,'now',lambda:deadline)
    scheduler=ScheduledAgentTaskScheduler(store,app.handle_due_scheduled_agent_task,runtime.trace_recorder)
    assert scheduler.run_task_once(task.task_id,at=deadline)==1
    assert active(store.require_game(game.game_id))=={'B'}
    assert any(event.step=='participation_deadline_evaluated' for event in runtime.trace_recorder.events)


def test_manual_advance_uses_real_deadline_handler_and_retains_other_tasks(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from manual_simulation import handle_manual_simulation
    run_id=handle_manual_simulation({'action':'create','request_id':'new'},runtime_dir=tmp_path,mode='mock')['run_id']
    handle_manual_simulation({'run_id':run_id,'action':'send','request_id':'input','text':'还没齐我就不等了'},runtime_dir=tmp_path)
    path=tmp_path/'runs'/run_id/'test_sim.db'; source_store=SQLiteAgentStore(path)
    conversation=f'manual:{run_id}';customer='manual_customer';planned=now()+timedelta(hours=31)
    game,_=source_store.create_game(conversation_id=conversation,organizer_id=customer,organizer_name='甲',trace_id='seed',
        requirement={'game_type':'sichuan_mahjong','stake':'1','smoke_preference':'no_smoking','planned_start_at':planned.isoformat(),'duration_hours':4},
        known_players=[{'customer_id':customer,'display_name':'甲'},{'customer_id':'manual_customer_2','display_name':'乙'}])
    turn=next(t for t in reversed(source_store.recent_turns(conversation)) if t.role.value=='user')
    result=ToolGateway(source_store).execute(ToolCall('set_participation_deadline',{'game_id':game.game_id,'deadline_at':(planned+timedelta(minutes=30)).isoformat()}),
        trace_id=turn.trace_id,step_index=1,conversation_id=conversation,sender_id=customer,sender_name='甲',source_message_id=turn.metadata['source_message_id'])
    assert result.allowed,result.to_dict()
    task_id=result.result['participation_deadline']['task_id'];recruitment_id=source_store.scheduled_task_for_game(game.game_id).task_id
    source_store._connection.close()
    advanced=handle_manual_simulation({'run_id':run_id,'action':'advance','request_id':'deadline-advance'},runtime_dir=tmp_path)
    assert advanced['sequence']==2
    reopened=SQLiteAgentStore(path)
    try:
        assert active(reopened.require_game(game.game_id))=={'manual_customer_2'}
        assert reopened.scheduled_tasks[task_id].status.value=='completed'
        assert reopened.scheduled_tasks[recruitment_id].status.value=='pending'
    finally:reopened._connection.close()
    report=json.loads((path.parent/'report.json').read_text())
    assert report['transcript'][-1]['event_type']=='participation_deadline_due'
    assert report['transcript'][-1]['user']['generation']['source']=='scheduler'
    assert report['transcript'][-1]['agent']['reply']==''
