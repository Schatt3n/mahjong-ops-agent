#!/usr/bin/env python3
"""回放静默、条件候补、复查和跨会话参与，生成可在聊天室审阅的真实证据。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from datetime import timedelta
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests' / 'simulation'))
from sim_adapter import build_runtime, simulation_execution_observation
from sim_factory import build_population
from behavior_policy import MessageGenerationRequest
from message_generation import build_message_generator
from mahjong_agent_runtime import CustomerProfile, UserMessage
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.follow_up import handle_follow_up_task
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler


def run_participation_case(store, send, generate_follow_up, checks: dict, events: list) -> None:
    """先检查真实查询与参与状态，再推进有界客户续聊；前提失败不发送后续剧本。

    客户生成器只收到自己看见的对话和本人意愿。数据库事实仅用于测试前置条件与断言，
    不作为隐藏上下文塞给客户，更不能把一次候补登记算成已经加入牌局。
    """
    names = ['cross_conversation_game_was_queried', 'cross_conversation_offer_ready',
             'cross_conversation_join_persisted', 'cross_conversation_join_has_tool_evidence',
             'cross_conversation_game_is_full', 'cross_conversation_keeps_origin',
             'cross_conversation_withdrawal_persisted', 'cross_conversation_does_not_create_duplicate']
    checks.update({name: False for name in names})
    game, _ = store.create_game(conversation_id='operational:another_customer', organizer_id='fixture_organizer', organizer_name='回归发起人',
        requirement={'game_type':'national_standard_mahjong','stake':'1','smoke_preference':'any','start_time_kind':'asap_when_full',
                     'user_visible_summary':'国标1块，人齐开，三缺一'},
        known_players=[{'customer_id':'fixture_organizer','display_name':'回归发起人','seat_count':3}], trace_id='participation_fixture')
    history = []

    def exchange(text, generation=None):
        """只累计客户实际看到的原话与回复，不复制工具、状态或主模型计划。"""
        result = send('participation', text, generation=generation)
        history.extend([{'role':'customer','text':text}, {'role':'agent','text':result.final_reply}])
        return result

    def block(step, reason):
        """前置条件不足时明确记录阻塞，保留已发生的消息，不补写或强行续演。"""
        events.append({'case':'participation','step':step,'status':'blocked','after_customer_turn':len(history)//2,'reason':reason})

    query = exchange('有国标1块人齐开的局吗？我一个人，烟都行')
    reads = [t for t in query.tool_results if t.name=='search_current_games' and t.called and t.allowed and not t.error]
    checks['cross_conversation_game_was_queried'] = bool(reads)
    offered = any(isinstance(match,dict) and (match.get('game') or {}).get('game_id')==game.game_id
                  for tool in reads for field in ('matches','alternatives') for match in tool.result.get(field, []))
    checks['cross_conversation_offer_ready'] = bool(offered and query.final_reply and query.status not in {'needs_help','needs_human'})
    if not checks['cross_conversation_offer_ready']:
        block('confirm_join', '上一轮没有查到可参加的目标局或没有正常回复，停止入局回归，不发送预设确认。')
        return

    joined_results = []
    for _ in range(3):
        generated = generate_follow_up('join', list(history))
        if generated.get('error') or not generated.get('text'):
            block('confirm_join', '客户消息生成失败，未发送固定替代话术。')
            return
        joined_results.append(exchange(generated['text'], generation=generated))
        current = store.require_game(game.game_id)
        joined = any(p.customer_id=='operational_customer' and p.status=='confirmed' and p.seat_count==1 for p in current.participants)
        if joined:
            break
        if generated.get('wants_to_continue') is False:
            break
    checks['cross_conversation_join_persisted'] = joined
    checks['cross_conversation_join_has_tool_evidence'] = any(t.name in {'join_game','record_candidate_reply'} and t.called and t.allowed
                                                           for result in joined_results for t in result.tool_results)
    checks['cross_conversation_game_is_full'] = current.remaining_seats()==0
    checks['cross_conversation_keeps_origin'] = current.conversation_id=='operational:another_customer'
    checks['cross_conversation_does_not_create_duplicate'] = not any(g.conversation_id=='operational:participation' for g in store.games.values())
    if not joined:
        block('withdraw', '客户尚未真实加入目标局，停止退出步骤，不凭空假设已有位置。')
        return
    # 改主意是该回归明确设置的下一阶段意愿，表达仍依据客户刚看到的实际对话生成。
    generated = generate_follow_up('withdraw', list(history))
    if generated.get('error') or not generated.get('text'):
        block('withdraw', '客户取消消息生成失败，未发送固定替代话术。')
        return
    exchange(generated['text'], generation=generated)
    current = store.require_game(game.game_id)
    checks['cross_conversation_withdrawal_persisted'] = any(p.customer_id=='operational_customer' and p.status=='declined' for p in current.participants) and current.remaining_seats()==1


def main() -> int:
    """仅在显式选择真实或离线模式后运行；虚拟时间不改变宿主时钟。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['real', 'mock'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--case', choices=['all','conditional','smoke','followup','participation'], default='all')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').isalnum():
        raise ValueError('invalid run id')
    folder = ROOT / 'runtime_data' / 'periodic_chat_simulation' / 'runs' / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    global _REPORT_PATH
    _REPORT_PATH = folder / 'report.json'
    store, _ = build_population(folder / 'test_sim.db', seed=42)
    store.upsert_customer(CustomerProfile(customer_id='operational_customer', display_name='模拟客户', gender='男',
                                         preferred_games=['hangzhou_mahjong'], preferred_stakes=['1'], smoke_preference='any'))
    runtime, client = build_runtime(store, args.mode, seed=42)
    customer_generator = build_message_generator('model') if args.mode=='real' and args.case in {'all','participation'} else None
    turns = []
    checks = {}
    scenario_events = []
    started = now().isoformat()

    def save():
        """每轮落盘真实记录，不为结果补写对话或模型判断。"""
        report = {'run_id': args.run_id, 'status': 'running', 'quality_status': 'unknown', 'llm_mode': args.mode,
                  'main_agent_model': getattr(getattr(client, 'config', None), 'model', None),
                  'message_generation_mode': 'scripted_regression_with_model_followups' if customer_generator else 'scripted_regression',
                  'message_generation_model': customer_generator.config.model if customer_generator else None, 'started_at': started,
                  'main_agent_runtime_config': {'max_tokens_per_call': runtime.token_budget.max_tokens_per_call, 'max_steps': runtime.max_steps},
                  'configuration': {'seed': 42, 'virtual_time': True, 'case': args.case}, 'transcript': turns, 'checks': checks,
                  'scenario_events': scenario_events,
                  'total_messages': len(turns)}
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return report

    def record(case, text, result, latency, source='scripted_regression', transport_event=None, generation=None):
        """把普通输入和定时唤醒放进同一条可审阅时间线，并明确来源。"""
        entry = {'sequence': len(turns)+1, 'conversation_id': f'operational:{case}', 'thread_id': case,
                 'channel': 'private', 'event_type': 'follow_up_timeout' if source=='scheduler' else (transport_event or {}).get('event_type','text'),
                 'transport_event': transport_event or {},
                 'dialog_phase': 'business', 'observed_at': now().isoformat(),
                 'user': {'customer_id': 'operational_customer', 'display_name': '模拟客户', 'text': text,
                          'generation': generation or {'source': source, 'context': {'case': case}}},
                 'agent': {'reply': result.final_reply, 'trace_id': result.trace_id, 'status': result.status,
                           'objective_status': result.actions[-1].objective_status if result.actions else result.status,
                           'response_mode': result.actions[-1].response_mode if result.actions else 'silent',
                           'actions': [a.to_dict() for a in result.actions]},
                 'tool_results': [t.to_dict() for t in result.tool_results],
                 'state_transitions': [t.to_dict() for t in result.state_transitions],
                 'simulation_observation': simulation_execution_observation(runtime,result.trace_id),
                 'operational_state': {'follow_ups': [task.to_dict() for task in store.scheduled_tasks.values()
                                                     if task.conversation_id==f'operational:{case}']},
                 'latency_ms': round(latency*1000,2), 'status_code': 200, 'error': ''}
        turns.append(entry)
        with (folder/'live_events.jsonl').open('a') as out:
            out.write(json.dumps(entry,ensure_ascii=False)+'\n')
        save()
        print(json.dumps({'case':case,'reply':result.final_reply,'status':result.status,'tools':[t.name for t in result.tool_results]},ensure_ascii=False),flush=True)
        return result

    def send(case, text, metadata=None, generation=None):
        """经真实 Runtime 处理固定客户输入，保留完整工具和状态证据。"""
        started_turn=time.monotonic()
        result=runtime.handle_user_message(UserMessage(conversation_id=f'operational:{case}',sender_id='operational_customer',sender_name='模拟客户',
                                                       text=text,message_id=f'{case}_{len(turns)+1}',metadata=metadata or {}))
        return record(case,'撤回了一条消息' if (metadata or {}).get('event_type')=='message_recalled' else text,result,time.monotonic()-started_turn,transport_event=metadata,generation=generation)

    def generate_follow_up(intent, history):
        """复用独立客户模型；输入只含可见历史、本人条件与本阶段意愿，没有隐藏业务状态。"""
        if customer_generator is None:
            return {'text':'可以，帮我留一个位置' if intent=='join' else '我不来了，帮我取消', 'source':'scripted_regression',
                    'context':{'intent':intent,'visible_history':history},'wants_to_continue':True}
        facts = {'game_type':'national_standard_mahjong','stake':'1','party_size':1,'smoke_preference':'any','start_time_text':'人齐开',
                 'current_intent': '愿意参加合适的现成局；先回答老板的问题，只能根据已看到的回复判断有没有具体局。' if intent=='join' else '现在改变主意不参加了，向老板说明并取消自己刚确认的位置。'}
        request = MessageGenerationRequest(sender_id='operational_customer',sender_name='模拟客户',persona='active_gambler',preferred_game='national_standard_mahjong',
            channel='private',conversation_id='operational:participation',turn_count=len(history)//2,last_agent_reply=history[-1]['text'],fallback_text='',
            is_follow_up=True,dialog_phase='business',business_anchor=history[0]['text'],customer_facts=facts,visible_history=tuple(history))
        return {**asdict(customer_generator.generate(request)), 'context':asdict(request)}

    save()
    if args.case in {'all','conditional'}:
        first=send('conditional','371我可一来')
        send('conditional','我说的是有三缺一的时候叫我。我一个人，财敲1块，人齐开，烟都可以')
        checks['first_turn_does_not_assume_three_own_players']=not any(a.objective_state.get('known_facts',{}).get('known_player_count')==3 for a in first.actions)
        checks['first_turn_does_not_switch_to_new_group']=not any(w in first.final_reply for w in ['帮你组','组一个','另组'])
        checks['conditional_does_not_create_game']=not any(game.conversation_id=='operational:conditional' for game in store.games.values())
        demands=[d for d in store.waiting_demands.values() if d.conversation_id=='operational:conditional']
        checks['conditional_saved_as_match_requirement']=bool(demands and demands[-1].demand.get('match_remaining_seats')==1)
        checks['conditional_preserves_variant']=bool(demands and demands[-1].demand.get('game_variant')=='caiqiao')

    if args.case in {'all','smoke'}:
        # 这里是可见历史夹具，不伪装成刚由模型完成的业务动作。
        store.activate_task_context(conversation_id='operational:smoke',customer_id='operational_customer',trace_id='seed',reason='fixed_regression',activity_at=now(),started_at=now(),force_new=False,archive_previous=False)
        store.append_user_turn(UserMessage(conversation_id='operational:smoke',sender_id='operational_customer',sender_name='模拟客户',
                                           text='这次先记下我的烟况偏好，暂时不组局',message_id='smoke_seed'),'seed')
        store.append_assistant_turn('operational:smoke','这次有烟还是无烟？','seed')
        smoke=send('smoke','有烟无言都可以')
        checks['no_redundant_smoke_echo']=not any(fragment in smoke.final_reply for fragment in ['烟都行','烟都可以','有烟无烟都'])
        recall=send('smoke','',{'event_type':'message_recalled','target_message_id':f'smoke_{len(turns)}'})
        checks['recall_is_silent']=recall.final_reply=='' and recall.actions[-1].response_mode=='silent'
        checks['recall_does_not_call_model']=not simulation_execution_observation(runtime,recall.trace_id)['model_call_observed']
        unseeded=send('smoke_unseeded','先记一下，这次有烟无烟都可以，暂时不用组局')
        checks['record_only_reply_does_not_echo']=not any(fragment in unseeded.final_reply for fragment in ['烟都行','烟都可以','有烟无烟'])


    if args.case in {'all','followup'}:
        send('followup','我想一下，三分钟后再问我今晚还打不打，十分钟还没决定就算了')
        pending=[task for task in store.scheduled_tasks.values() if task.conversation_id=='operational:followup' and task.status.value=='pending']
        checks['followup_is_persistent']=bool(pending)
        if pending:
            due=pending[0].due_at
            def due_handler(task, trace):
                """推进虚拟时间触发真实主 Agent，展示到期后说或不说的结果。"""
                start=time.monotonic()
                result=handle_follow_up_task(runtime,task,trace,at=due)
                if result is not None:
                    record('followup','（复查到期，重新检查当前事实）',result,time.monotonic()-start,'scheduler')
            scheduler=ScheduledAgentTaskScheduler(store,due_handler,runtime.trace_recorder)
            checks['due_event_claimed_once']=scheduler.run_due_once(at=due)==1 and scheduler.run_due_once(at=due)==0
            send('followup','不打了，这事不用再跟进')
            checks['cancelled_goal_has_no_pending_timer']=not any(task.conversation_id=='operational:followup' and task.status.value=='pending' for task in store.scheduled_tasks.values())
        else:
            checks['due_event_claimed_once']=False
            checks['cancelled_goal_has_no_pending_timer']=False
    if args.case in {'all','participation'}:
        run_participation_case(store, send, generate_follow_up, checks, scenario_events)
    failures=[t for t in turns if t['agent']['status'] in {'needs_help','needs_human'} or t['agent']['objective_status']=='needs_tool']
    checks['no_internal_failure']=not failures
    report=save()
    report.update(status='completed',quality_status='passed' if all(checks.values()) else 'degraded',finished_at=now().isoformat(),
                  stop_reason='scenario_precondition_failed' if scenario_events else 'fixed_scenarios_completed',agent_internal_failure_count=len(failures))
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'run_id':args.run_id,'checks':checks},ensure_ascii=False),flush=True)
    return 0 if all(checks.values()) else 1


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        report_path = globals().get('_REPORT_PATH')
        if report_path is not None and report_path.exists():
            partial = json.loads(report_path.read_text())
            partial.update(status='failed',quality_status='failed',stop_reason='evaluation_error',finished_at=now().isoformat(),error_type=type(exc).__name__)
            report_path.write_text(json.dumps(partial,ensure_ascii=False,indent=2)+'\n')
        raise
