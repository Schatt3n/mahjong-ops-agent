#!/usr/bin/env python3
"""回放未知需求、可靠画像与明确需求，保存主 Agent 回复及独立结论范围检查。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'/'simulation'))
from sim_adapter import build_runtime, simulation_execution_observation
from mahjong_agent_runtime import AgentAction, CustomerProfile, SQLiteAgentStore, ToolCall, UserMessage
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.tools.search_tools import search_current_games
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient
from mahjong_agent_runtime.models import now


CASES = (
    ('unknown_empty', '现在有人吗', (), None, True),
    ('unknown_pool', '现在有人吗', ('national_standard_mahjong','sichuan_mahjong'), None, True),
    ('partial_request', '0.5现在有人吗', (), None, True),
    ('explicit_no_profile', '现在国标1块人齐开的有局吗？我一个人，烟都行', ('national_standard_mahjong',), None, False),
    ('reliable_profile', '现在有人齐开的局吗', ('national_standard_mahjong',), 'national_standard_mahjong', False),
    ('explicit_no_match', '现在国标1块人齐开的有局吗？我一个人，烟都行', (), None, False),
)


def main() -> int:
    """显式运行真实模型，并将逐轮原始回复、工具证据与独立判定写入审阅目录。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum():
        raise ValueError('invalid run id')
    folder=ROOT/'runtime_data'/'periodic_chat_simulation'/'runs'/args.run_id
    folder.mkdir(parents=True,exist_ok=False)
    global _REPORT_PATH
    _REPORT_PATH=folder/'report.json'
    report={'run_id':args.run_id,'status':'running','quality_status':'unknown','llm_mode':'real',
            'message_generation_mode':'scripted_regression','started_at':now().isoformat(),
            'configuration':{'cases':[case[0] for case in CASES],'customer_messages':'fixed_regression'},
            'transcript':[],'checks':{},'total_messages':0}

    def save():
        """追加真实回合后保存投影，不更改历史人工标注或已有运行。"""
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

    save()
    for case_id,text,game_types,preferred_game,clarify in CASES:
        path=folder/case_id/'test_sim.db';path.parent.mkdir()
        store=SQLiteAgentStore(path)
        try:
            profile=CustomerProfile(customer_id='customer',display_name='需求测试客户',
                preferred_games=[preferred_game] if preferred_game else [],preferred_stakes=['1'] if preferred_game else [],
                smoke_preference='any' if preferred_game else None)
            store.upsert_customer(profile)
            for index,game_type in enumerate(game_types):
                store.create_game(conversation_id=f'fixture:{index}',organizer_id=f'owner:{index}',organizer_name='发起人',
                    requirement={'game_type':game_type,'stake':'1','smoke_preference':'any','start_time_kind':'asap_when_full',
                                 'user_visible_summary':('国标' if game_type=='national_standard_mahjong' else '川麻')+'1块，人齐开，三缺一'},
                    known_players=[{'customer_id':f'owner:{index}','display_name':'发起人','seat_count':3}],trace_id='fixture')
            runtime,client=build_runtime(store,'real')
            report['main_agent_model']=client.config.model
            report['main_agent_runtime_config']={'max_tokens_per_call':runtime.token_budget.max_tokens_per_call}
            message=UserMessage(conversation_id=f'inquiry:{case_id}',sender_id='customer',sender_name='需求测试客户',text=text,message_id=case_id)
            started=time.monotonic();result=runtime.handle_user_message(message)
            entry={'sequence':len(report['transcript'])+1,'conversation_id':message.conversation_id,'thread_id':case_id,'channel':'private',
                'event_type':'text','dialog_phase':'business','observed_at':now().isoformat(),
                'user':{'customer_id':'customer','display_name':'需求测试客户','text':text,'generation':{'source':'scripted_regression','context':{'case':case_id}}},
                'agent':{'reply':result.final_reply,'trace_id':result.trace_id,'status':result.status,
                         'objective_status':result.actions[-1].objective_status if result.actions else result.status,
                         'actions':[a.to_dict() for a in result.actions]},'tool_results':[t.to_dict() for t in result.tool_results],
                'state_transitions':[t.to_dict() for t in result.state_transitions],
                'simulation_observation':simulation_execution_observation(runtime,result.trace_id),
                'latency_ms':round((time.monotonic()-started)*1000,2),'status_code':200,'error':''}
            report['transcript'].append(entry);report['total_messages']=len(report['transcript'])
            with (folder/'live_events.jsonl').open('a') as stream:
                stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
            save()
            # 判定器只读实际客户回复与已知条件，不能把主模型的计划当作已完成证据。
            judge=OpenAICompatibleAgentClient(config=replace(client.config,temperature=0,max_tokens=700))
            evidence={'request':text,'reply':result.final_reply,'known_profile':{'preferred_games':profile.preferred_games,'preferred_stakes':profile.preferred_stakes},
                      'requires_clarification':clarify,'public_search_results':[t.result for t in result.tool_results if t.name=='search_current_games' and t.allowed]}
            raw=judge.complete([{'role':'system','content':
                '检查麻将馆回复的结论依据，输出JSON {"passed":boolean,"reason":"简短依据"}。需求未知时应自然澄清玩法或档位，'
                '不得先断言没有适合客户的局；requires_clarification=true时先问需求，不应附带有无牌局的先行判断（包括没有登记局等换词）。'
                '只问需求时不得声称已经持续找人或跟进，也不应重复问已明确的现在、档位或唯一可靠画像玩法。'
                '没有画像但原话条件完整时可以查局，不应机械要求画像。明确条件查询后没有对应局可以如实说没有；不要求固定措辞。'
                'requires_clarification=true时应有必要的澄清问题，其余场景不得无依据重问已知条件。'},
                {'role':'user','content':json.dumps(evidence,ensure_ascii=False)}],trace_id=f'judge:{case_id}',timeout_seconds=45)
            assessment=json.loads(raw);entry['inquiry_assessment']=assessment
            report['checks'][case_id]=assessment.get('passed') is True and result.status not in {'needs_help','needs_human'}
            if clarify:
                report['checks'][case_id+'_waits_for_customer']=result.status=='waiting_user'
            save()
            print(json.dumps({'case':case_id,'reply':result.final_reply,'status':result.status,'assessment':assessment},ensure_ascii=False),flush=True)
            if case_id=='unknown_empty':
                bad='现在还没有合适的局，想打什么玩法、几点，我帮你留意着。'
                read=search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':{}}),'probe',message.conversation_id,'customer','需求测试客户')
                action=AgentAction(goal='回答客户当前有没有合适局',objective_status='completed',reasoning_summary='复现人工标注原稿',reply_to_user=bad)
                payload=build_reply_self_review_payload(message=message,action=action,
                    review_items=[{'item_id':'reply_to_user','source':'reply_to_user','recipient_id':'customer','text':bad,'source_text':bad}],
                    context_payload={'sender_profile':{'preferred_games':[],'preferred_stakes':[]},'turn_tool_evidence':[read.to_dict()]},review_scope='reply_to_user')
                reviewed=json.loads(client.complete([{'role':'system','content':(ROOT/'src/mahjong_agent_runtime/prompts/agent_runtime_reply_self_review.md').read_text()},
                    {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],trace_id='probe:unsupported_fit',timeout_seconds=45))
                report['reviewer_probe']=reviewed
                report['checks']['annotated_original_reply_rejected']=reviewed.get('approved') is False and not reviewed.get('needs_human')
                save()
        finally:
            store._connection.close()
    report.update(status='completed',quality_status='passed' if all(report['checks'].values()) else 'degraded',finished_at=now().isoformat())
    save()
    return 0 if all(report['checks'].values()) else 1


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        path=globals().get('_REPORT_PATH')
        if path and path.exists():
            report=json.loads(path.read_text())
            report.update(status='failed',quality_status='failed',finished_at=now().isoformat(),error_type=type(exc).__name__)
            path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        raise
