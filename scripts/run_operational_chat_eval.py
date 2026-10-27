#!/usr/bin/env python3
"""回放静默、条件候补、复查和跨会话参与，生成可在聊天室审阅的真实证据。"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests' / 'simulation'))
from sim_adapter import build_runtime, simulation_execution_observation
from sim_factory import build_population
from mahjong_agent_runtime import CustomerProfile, UserMessage
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.follow_up import handle_follow_up_task
from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler


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
    turns = []
    checks = {}
    started = now().isoformat()

    def save():
        """每轮落盘真实记录，不为结果补写对话或模型判断。"""
        report = {'run_id': args.run_id, 'status': 'running', 'quality_status': 'unknown', 'llm_mode': args.mode,
                  'main_agent_model': getattr(getattr(client, 'config', None), 'model', None),
                  'message_generation_mode': 'scripted_regression', 'started_at': started,
                  'main_agent_runtime_config': {'max_tokens_per_call': runtime.token_budget.max_tokens_per_call, 'max_steps': runtime.max_steps},
                  'configuration': {'seed': 42, 'virtual_time': True, 'case': args.case}, 'transcript': turns, 'checks': checks,
                  'total_messages': len(turns)}
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        return report

    def record(case, text, result, latency, source='scripted_regression', transport_event=None):
        """把普通输入和定时唤醒放进同一条可审阅时间线，并明确来源。"""
        entry = {'sequence': len(turns)+1, 'conversation_id': f'operational:{case}', 'thread_id': case,
                 'channel': 'private', 'event_type': 'follow_up_timeout' if source=='scheduler' else (transport_event or {}).get('event_type','text'),
                 'transport_event': transport_event or {},
                 'dialog_phase': 'business', 'observed_at': now().isoformat(),
                 'user': {'customer_id': 'operational_customer', 'display_name': '模拟客户', 'text': text,
                          'generation': {'source': source, 'context': {'case': case}}},
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

    def send(case, text, metadata=None):
        """经真实 Runtime 处理固定客户输入，保留完整工具和状态证据。"""
        started_turn=time.monotonic()
        result=runtime.handle_user_message(UserMessage(conversation_id=f'operational:{case}',sender_id='operational_customer',sender_name='模拟客户',
                                                       text=text,message_id=f'{case}_{len(turns)+1}',metadata=metadata or {}))
        return record(case,'撤回了一条消息' if (metadata or {}).get('event_type')=='message_recalled' else text,result,time.monotonic()-started_turn,transport_event=metadata)

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
        # 仅设置可见局池夹具；客户是否参加由真实模型依据对话自主决定并执行工具。
        game, _ = store.create_game(conversation_id='operational:another_customer', organizer_id='fixture_organizer', organizer_name='回归发起人',
            requirement={'game_type':'national_standard_mahjong','stake':'1','smoke_preference':'any','start_time_kind':'asap_when_full',
                         'user_visible_summary':'国标1块，人齐开，三缺一'},
            known_players=[{'customer_id':'fixture_organizer','display_name':'回归发起人','seat_count':3}], trace_id='participation_fixture')
        query=send('participation','有国标1块人齐开的局吗？我一个人，烟都行')
        checks['cross_conversation_game_was_queried']=any(t.name=='search_current_games' and t.allowed for t in query.tool_results)
        joined=send('participation','就刚才说的那个，我一个人，给我留一个位置，人齐叫我')
        current=store.require_game(game.game_id)
        checks['cross_conversation_join_persisted']=any(p.customer_id=='operational_customer' and p.status=='confirmed' and p.seat_count==1 for p in current.participants)
        checks['cross_conversation_join_has_tool_evidence']=any(t.name in {'join_game','record_candidate_reply'} and t.called and t.allowed for t in joined.tool_results)
        checks['cross_conversation_game_is_full']=current.remaining_seats()==0
        checks['cross_conversation_keeps_origin']=current.conversation_id=='operational:another_customer'
        cancelled=send('participation','刚才那个我不来了，把我的位置放掉吧')
        current=store.require_game(game.game_id)
        checks['cross_conversation_withdrawal_persisted']=any(p.customer_id=='operational_customer' and p.status=='declined' for p in current.participants) and current.remaining_seats()==1
        checks['cross_conversation_does_not_create_duplicate']=not any(g.conversation_id=='operational:participation' for g in store.games.values())
    failures=[t for t in turns if t['agent']['status'] in {'needs_help','needs_human'} or t['agent']['objective_status']=='needs_tool']
    checks['no_internal_failure']=not failures
    report=save()
    report.update(status='completed',quality_status='passed' if all(checks.values()) else 'degraded',finished_at=now().isoformat(),
                  stop_reason='fixed_scenarios_completed',agent_internal_failure_count=len(failures))
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
