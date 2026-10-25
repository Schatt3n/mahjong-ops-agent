"""验证静默、撤回、条件人数和持久化复查组成可审计的运营闭环。"""

from datetime import timedelta
import json

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, SQLiteAgentStore, InMemoryTraceRecorder, UserMessage, ToolCall
from mahjong_agent_runtime.action_contract import parse_action
from mahjong_agent_runtime.models import now, ScheduledTaskStatus, WaitingDemand
from mahjong_agent_runtime.follow_up import handle_follow_up_task
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler
from mahjong_agent_runtime.domains.context_builders.task_facts import project_explicit_task_facts
from mahjong_agent_runtime.matching import waiting_demand_mismatch_reason


def action(*, reply='', silent=False, calls=None):
    tools = calls or []
    return json.dumps({
        'goal': '跟进当前组局需求', 'objective_status': 'needs_tool' if tools else 'completed',
        'reasoning_summary': '按实际责任处理当前消息', 'objective_state': {}, 'objective_plan': [],
        'plan_revision_reason': '当前事实已核对', 'reply_to_user': reply, 'tool_calls': tools,
        'response_mode': 'silent' if silent else 'reply', 'silence_reason': '无需新增回复' if silent else '',
        'needs_human': False, 'stop_reason': {'can_stop': not tools, 'why': '执行工具' if tools else '本轮已处理',
                                           'pending_work': ['工具执行'] if tools else [], 'depends_on_tool_results': False},
    }, ensure_ascii=False)


class QueueClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append(json.loads(messages[-1]['content']))
        return self.responses.pop(0)


@pytest.fixture(params=['memory', 'sqlite'])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path / 'operational.db')
    yield value
    if hasattr(value, 'close'):
        value.close()


def message(text, ident='m1'):
    return UserMessage(conversation_id='c1', sender_id='u1', sender_name='客户', text=text, message_id=ident)


def test_silent_is_explicit_and_never_accepts_accidentally_empty_failure():
    good, errors = parse_action(action(silent=True))
    assert not errors and good.response_mode == 'silent'
    _, errors = parse_action(action())
    assert 'completed requires non-empty reply_to_user' in errors
    raw = json.loads(action(silent=True))
    raw['silence_reason'] = ''
    assert parse_action(json.dumps(raw))[1]


def test_recall_is_silent_preserves_audit_and_invalidates_only_derived_context(store):
    client = QueueClient([action(reply='打什么档位？')])
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())
    runtime.handle_user_message(message('我们三个人，还差一个'))
    context = store.current_task_context('c1','u1')
    store.upsert_conversation_checkpoint(conversation_id='c1',summary='已有三人',facts={'known_player_count':3},open_questions=[],trace_id='summary',task_context_id=context.task_context_id)
    recall = message('', 'recall1')
    recall.metadata = {'event_type':'message_recalled','target_message_id':'m1'}
    result = runtime.handle_user_message(recall)
    assert result.final_reply == '' and result.actions[-1].response_mode == 'silent'
    assert len(client.calls) == 1
    assert store.find_conversation_turn('c1',message_id='m1').content == '我们三个人，还差一个'
    assert store.find_conversation_turn('c1',message_id='m1').metadata['context_invalidated']
    assert store.get_conversation_checkpoint('c1') is None
    built = runtime.context_builder.build(message('先等等','m2'),trace_id='check')
    assert not any('我们三个人' in t['content'] for t in built.payload['recent_conversation'])
    runtime.handle_user_message(recall)
    assert len(client.calls) == 1


@pytest.mark.parametrize('text',['371我可以来','371的时候叫我','等三缺一再叫我','不是371','朋友说371'])
def test_decoding_a_seat_code_does_not_bind_its_semantic_role(text):
    projection = project_explicit_task_facts([], {'text':text}, None)
    assert projection['candidates']['seat_format'] == '371'
    assert projection['facts'] == {} and projection['binding_fields'] == []


def schedule_call(due, expires):
    return {'name':'schedule_follow_up','arguments':{'goal':'确认今晚是否继续协调','waiting_for':'customer',
            'due_at':due.isoformat(),'expires_at':expires.isoformat(),'review_budget':2},'reason':'仍有已承诺的跟进责任'}


def test_timeout_reenters_model_with_fresh_context_and_does_not_repeat(store):
    stamp = now(); due=stamp+timedelta(minutes=3); expires=stamp+timedelta(hours=1)
    client=QueueClient([action(calls=[schedule_call(due,expires)]),action(silent=True), action(reply='今晚还打吗？')])
    runtime=AgentRuntime(llm_client=client,store=store,trace_recorder=InMemoryTraceRecorder())
    result=runtime.handle_user_message(message('我想一下，三分钟后你再问我'))
    assert result.final_reply == '' and len(store.scheduled_tasks)==1
    events=[]
    scheduler=ScheduledAgentTaskScheduler(store,lambda task,trace:events.append(handle_follow_up_task(runtime,task,trace,at=due)),runtime.trace_recorder)
    assert scheduler.run_due_once(at=stamp)==0
    assert scheduler.run_due_once(at=due)==1
    assert events[0].final_reply == '今晚还打吗？'
    assert client.calls[-1]['system_trigger']['trigger_type']=='conversation_follow_up_due'
    assert len(store.outbound_message_drafts)==1
    assert scheduler.run_due_once(at=due+timedelta(minutes=1))==0
    assert len(client.calls)==3


def test_timeout_can_deliberately_remain_silent_and_cancelled_plan_never_wakes(store):
    stamp=now();due=stamp+timedelta(minutes=3);expires=stamp+timedelta(hours=1)
    client=QueueClient([action(calls=[schedule_call(due,expires)]),action(silent=True),action(silent=True)])
    runtime=AgentRuntime(llm_client=client,store=store,trace_recorder=InMemoryTraceRecorder())
    runtime.handle_user_message(message('三分钟后再看要不要继续联系'))
    scheduler=ScheduledAgentTaskScheduler(store,lambda task,trace:handle_follow_up_task(runtime,task,trace,at=due),runtime.trace_recorder)
    scheduler.run_due_once(at=due)
    assert len(store.outbound_message_drafts)==0
    assert len(client.calls)==3
    new_due=due+timedelta(minutes=3)
    scheduled=runtime.tool_gateway.execute(ToolCall('schedule_follow_up',schedule_call(new_due,expires)['arguments'],'再次复查'),trace_id='replan',step_index=1,conversation_id='c1',sender_id='u1',sender_name='客户')
    assert scheduled.called and scheduled.allowed
    # 目标已经结束时，取消不能变成未来的一条自动催促。
    runtime.tool_gateway.execute(ToolCall('cancel_follow_up',{'reason':'客户已取消'},'取消'),trace_id='cancel',step_index=1,conversation_id='c1',sender_id='u1',sender_name='客户')
    assert not any(task.status==ScheduledTaskStatus.PENDING for task in store.scheduled_tasks.values())
    assert scheduler.run_due_once(at=due+timedelta(hours=1))==0


def test_matching_condition_does_not_fabricate_seats(store):
    game,_=store.create_game(conversation_id='other',organizer_id='other',organizer_name='发起人',
                             requirement={'game_type':'hangzhou_mahjong','stake':'1','smoke_preference':'any',
                                          'start_time_kind':'asap_when_full','known_player_count':1,'needed_seats':3},
                             known_players=[],trace_id='create')
    demand=WaitingDemand(demand_id='d1',conversation_id='c1',sender_id='u1',sender_name='客户',
                         demand={'stake':'1','smoke_preference':'any','time_preference':'不限','match_remaining_seats':1},
                         expires_at=now()+timedelta(hours=1))
    assert waiting_demand_mismatch_reason(demand,game)=='remaining_seats_mismatch'
    assert game.remaining_seats()==3


def test_recalling_only_the_timer_source_cancels_its_unexecuted_follow_up(store):
    stamp=now();due=stamp+timedelta(minutes=3);expires=stamp+timedelta(hours=1)
    client=QueueClient([action(calls=[schedule_call(due,expires)]),action(silent=True)])
    runtime=AgentRuntime(llm_client=client,store=store,trace_recorder=InMemoryTraceRecorder())
    runtime.handle_user_message(message('三分钟后再问我'))
    task=next(iter(store.scheduled_tasks.values()))
    assert task.payload['source_message_id']=='m1'
    event=message('', 'recall')
    event.metadata={'event_type':'message_recalled','target_message_id':'m1'}
    runtime.handle_user_message(event)
    assert store.scheduled_tasks[task.task_id].status==ScheduledTaskStatus.CANCELLED
    scheduler=ScheduledAgentTaskScheduler(store,lambda *args:pytest.fail('cancelled follow-up woke'),runtime.trace_recorder)
    assert scheduler.run_due_once(at=due)==0
    assert len(client.calls)==2


def test_follow_up_budget_cannot_be_reset_by_replanning_or_old_lease_completion(store):
    runtime=AgentRuntime(llm_client=QueueClient([]),store=store,trace_recorder=InMemoryTraceRecorder())
    stamp=now();expires=stamp+timedelta(hours=1)
    def schedule(index):
        due=stamp+timedelta(minutes=index*2)
        return due,runtime.tool_gateway.execute(ToolCall('schedule_follow_up',schedule_call(due,expires)['arguments'],'复查'),trace_id=f'p{index}',step_index=1,conversation_id='c1',sender_id='u1',sender_name='客户')
    due,first=schedule(1)
    assert first.allowed
    first_id=first.result['follow_up']['task_id']
    assert store.claim_scheduled_task(first_id,at=due)
    due,second=schedule(2)
    assert second.allowed
    second_id=second.result['follow_up']['task_id']
    store.complete_scheduled_task(first_id,trace_id='old')
    assert store.scheduled_tasks[first_id].status==ScheduledTaskStatus.CANCELLED
    assert store.scheduled_tasks[second_id].status==ScheduledTaskStatus.PENDING
    assert store.claim_scheduled_task(second_id,at=due)
    _,third=schedule(3)
    assert third.allowed is False
    assert len(store.scheduled_tasks)==2


def test_successful_auxiliary_projection_removes_repetition_and_preserves_raw_evidence():
    from copy import deepcopy
    from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_result_dict
    raw={'name':'customer_visible_content_review','called':True,'allowed':True,'error':'','result':{
        'approved':True,'review_scope':'invite_drafts',
        'items':[{'item_id':str(i),'text':'原始邀约说明'*100} for i in range(10)],
        'item_reviews':[{'item_id':str(i),'approved':True,'violations':[],'reasoning_summary':'逐项重复说明'*100} for i in range(10)]}}
    original=deepcopy(raw)
    compact=compact_tool_result_dict(raw)
    assert len(json.dumps(compact)) < len(json.dumps(raw))/5
    assert raw==original
    assert len(compact['result']['item_reviews'])==10
    assert compact['result']['approved'] is True


def test_rejected_review_projection_preserves_repair_details():
    from mahjong_agent_runtime.domains.context_builders.tool_results import compact_tool_result_dict
    raw={'name':'customer_visible_content_review','called':True,'allowed':True,'error':'','result':{
        'approved':False,'review_scope':'reply_to_user','violations':['unsupported_promise'],
        'item_reviews':[{'item_id':'reply_to_user','approved':False,'suggested_safe_text':'我再看看','violations':['unsupported_promise']}],
        'items':[{'item_id':'reply_to_user','text':'已经问过所有人'}]}}
    result=compact_tool_result_dict(raw)['result']
    assert result['approved'] is False
    assert result['violations']==['unsupported_promise']
    assert result['item_reviews'][0]['suggested_safe_text']=='我再看看'
    assert result['items'][0]['text']=='已经问过所有人'


def test_cancelling_absent_follow_up_is_idempotent_and_does_not_poison_next_plan(store):
    runtime=AgentRuntime(llm_client=QueueClient([]),store=store,trace_recorder=InMemoryTraceRecorder())
    result=runtime.tool_gateway.execute(ToolCall('cancel_follow_up',{'reason':'暂时不用跟进'},'取消'),trace_id='cancel',step_index=1,conversation_id='c1',sender_id='u1',sender_name='客户')
    assert result.allowed and result.result['already_closed']
    assert not store.scheduled_tasks
    stamp=now()
    result=runtime.tool_gateway.execute(ToolCall('schedule_follow_up',schedule_call(stamp+timedelta(minutes=3),stamp+timedelta(hours=1))['arguments'],'新需求'),trace_id='new',step_index=1,conversation_id='c1',sender_id='u1',sender_name='客户')
    assert result.allowed and len(store.scheduled_tasks)==1


def test_follow_up_identity_does_not_collide_on_colon_delimited_customer_ids(store):
    runtime=AgentRuntime(llm_client=QueueClient([]),store=store,trace_recorder=InMemoryTraceRecorder())
    stamp=now()
    for conversation,subject in [('a:b','c'),('a','b:c')]:
        result=runtime.tool_gateway.execute(ToolCall('schedule_follow_up',schedule_call(stamp+timedelta(minutes=3),stamp+timedelta(hours=1))['arguments'],'跟进'),trace_id=f'{conversation}-{subject}',step_index=1,conversation_id=conversation,sender_id=subject,sender_name='客户')
        assert result.allowed
    assert sum(task.status==ScheduledTaskStatus.PENDING for task in store.scheduled_tasks.values())==2
    runtime.tool_gateway.execute(ToolCall('cancel_follow_up',{'reason':'结束本人的目标'},'取消'),trace_id='cancel',step_index=1,conversation_id='a:b',sender_id='c',sender_name='客户')
    remaining=[task for task in store.scheduled_tasks.values() if task.status==ScheduledTaskStatus.PENDING]
    assert len(remaining)==1 and remaining[0].subject_id=='b:c'


@pytest.mark.parametrize('scope,recipient,allowed',[('reply_to_user','u1',True),('invite_drafts','another',False)])
def test_known_condition_omission_is_limited_to_the_current_customer_reply(scope,recipient,allowed):
    from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
    from mahjong_agent_runtime import AgentAction
    payload=build_reply_self_review_payload(message=message('烟都可以'),action=AgentAction(goal='收讫当前条件',objective_status='completed',reasoning_summary='只确认收到',reply_to_user='好'),
                                           review_items=[{'item_id':'reply','recipient_id':recipient,'source_text':'好，烟都可以','text':'好'}],
                                           context_payload={},review_scope=scope)
    assert payload['semantic_fidelity_contract']['allow_known_user_echo_omission'] is allowed
    assert payload['action_boundary']['goal']=='收讫当前条件'
