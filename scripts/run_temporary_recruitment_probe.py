#!/usr/bin/env python3
"""仅观察临时补位现有能力：独立SQLite、固定业务时钟、真实主Agent，不修生产行为。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
from types import SimpleNamespace
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
for entry in (ROOT/'src',ROOT/'scripts'):
    if str(entry) not in sys.path:sys.path.insert(0,str(entry))
from mahjong_agent_runtime import CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, InMemoryTraceRecorder, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.env import load_dotenv_defaults
from mahjong_agent_runtime.customer_visible_contract import customer_visible_text_contract_violations
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient
import run_real_owner_chat_live_eval as owner
from run_inquiry_grounding_eval import assess_reply_semantics

DATASET=ROOT/'eval/golden/temporary_recruitment_parallel_v4.json'
STAGES={'original_ready','after_d_cancelled','parallel_forming','early_committed'}
ACTORS=tuple('ABCDEFG')


def dump(path: Path, value: Any) -> None:
    """保留原始JSON证据，调用方只使用本轮新建目录。"""
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def load_dataset(path: Path=DATASET) -> dict[str,Any]:
    """验证探针源与分支覆盖；答案仅供后端评分，禁止进入主模型消息。"""
    source=json.loads(path.read_text());cases=source.get('cases')
    if source.get('schema_version')!=4 or source.get('kind')!='temporary_recruitment_parallel_baseline' or not isinstance(cases,list) or len(cases)!=10:
        raise ValueError('temporary recruitment requires ten reviewed cases')
    ids=[c.get('case_id') for c in cases]
    if len(set(ids))!=len(ids) or any(not isinstance(i,str) or not i for i in ids):raise ValueError('invalid case identities')
    for case in cases:
        if case.get('fixture_stage') not in STAGES or not case.get('turns') or not case.get('required_meaning'):
            raise ValueError('missing fixture, turns or original expectations')
        for turn in case['turns']:
            if turn.get('actor') not in ACTORS or not isinstance(turn.get('text'),str) or not turn['text'].strip():
                raise ValueError('invalid actor or message')
            datetime.strptime(turn['at'],'%H:%M')
    return source


class BusinessClock:
    """仅冻结本进程领域时间；HTTP单调计时、连接读取和重试使用真实时间。"""
    def __init__(self,day: str):
        self.day=day;self.current=datetime.fromisoformat(day+'T18:40:00+08:00');self.patches=[]

    def set(self,clock: str) -> None:
        """每条消息显式推进业务时间，不因真实推理延迟偷偷改变开局约定。"""
        self.current=datetime.fromisoformat(self.day+'T'+clock+':00+08:00')

    def __enter__(self):
        parent=self
        class DateTimeMeta(type):
            """让既有真实时间对象仍通过领域类型校验，避免冻结时间改变参数语义。"""
            def __instancecheck__(cls,obj):
                return isinstance(obj,datetime)
        class FrozenDateTime(datetime,metaclass=DateTimeMeta):
            """仅提供固定业务当前时刻，其余时间解析和运算沿用标准实现。"""
            @classmethod
            def now(cls,tz=None):
                return parent.current.astimezone(tz) if tz else parent.current.replace(tzinfo=None)
        self.replacement=FrozenDateTime
        self.refresh()
        return self

    def refresh(self) -> None:
        """补齐按需导入的领域模块；提供方客户端和网络诊断不改写。"""
        for name,module in list(sys.modules.items()):
            if (module is not None and name.startswith('mahjong_agent_runtime')
                    and (name.endswith('.models') or '.stores.' in name or '.domains.' in name)
                    and getattr(module,'datetime',None) is datetime):
                self.patches.append(module);module.datetime=self.replacement

    def __exit__(self,*args):
        for module in reversed(self.patches):module.datetime=datetime


def cid(actor: str) -> str:
    """稳定沙盒客户ID，不能用游戏或会话ID冒充身份。"""
    return 'temporary_customer_'+actor.lower()


def conversation(actor: str) -> str:
    """每个客户保留独立私聊边界。"""
    return 'temporary_private_'+actor.lower()


def stamp(day: str,clock: str) -> str:
    """把数据集明确时刻转为带时区瞬时，不让模型猜日期。"""
    return day+'T'+clock+':00+08:00'


def setup_fixture(store: Any, source: dict[str,Any], stage: str, clock: BusinessClock) -> dict[str,str]:
    """真实Store事务建立预置状态；双方案用同会话真实备选接口预置，不伪装成Agent已实现双建。"""
    store.configure_rooms(['temporary_room_1','temporary_room_2'])
    owner.configure_authorized_eval_board(store)
    for actor in ACTORS:
        store.upsert_customer(CustomerProfile(customer_id=cid(actor),display_name=actor,public_name=actor,
            preferred_games=['hangzhou_mahjong'],preferred_stakes=['0.5'],smoke_preference='no_smoke'))
    requirement={**source['game'],'start_time_kind':'scheduled','planned_start_at':stamp(clock.day,'19:00'),'start_time':'19:00'}
    game,_=store.create_game(conversation_id=conversation('A'),organizer_id=cid('A'),organizer_name='A',
        requirement=requirement,known_players=[{'customer_id':cid(a),'display_name':a,'seat_count':1} for a in 'ABCD'],trace_id='fixture_original_ready')
    # 原约定由报名联系人A集中确认四人：不虚构D本人答复或独立接管座位。
    store.record_candidate_reply(game_id=game.game_id,customer_id=cid('A'),display_name='A',status='confirmed',trace_id='fixture_original_confirmed')
    ids={'original':game.game_id}
    store.append_user_turn(UserMessage(conversation_id=conversation('A'),sender_id=cid('A'),sender_name='A',
        text='今晚七点财敲0.5无烟，我们A、B、C、D四个人都确认参加。',message_id='fixture_original_agreement',sent_at=clock.current),'fixture_history')
    store.append_assistant_turn(conversation('A'),'七点财敲0.5无烟，四位的约定已记下。','fixture_history')
    if stage!='original_ready':
        clock.set('18:49')
        store.record_candidate_reply(game_id=game.game_id,customer_id=cid('D'),display_name='D',status='declined',trace_id='fixture_d_cancelled')
        store.append_user_turn(UserMessage(conversation_id=conversation('A'),sender_id=cid('A'),sender_name='A',
            text='D今天临时有事来不了，A、B、C继续等补位。',message_id='fixture_d_cancelled',sent_at=clock.current),'fixture_history')
    if stage in {'parallel_forming','early_committed'}:
        store.update_game_requirement(game_id=game.game_id,requirement_patch={'start_time':'19:30','planned_start_at':stamp(clock.day,'19:30')},reason='夹具：ABC已接受七点半备选',trace_id='fixture_early')
        late,_=store.create_game(conversation_id=conversation('A'),organizer_id=cid('A'),organizer_name='A',alternative_to_game_id=game.game_id,
            requirement={**source['game'],'start_time_kind':'scheduled','start_time':'20:00','planned_start_at':stamp(clock.day,'20:00')},
            known_players=[{'customer_id':cid(a),'display_name':a,'seat_count':1} for a in 'EAB'],trace_id='fixture_late')
        ids.update(early=game.game_id,late=late.game_id)
        store.append_user_turn(UserMessage(conversation_id=conversation('A'),sender_id=cid('A'),sender_name='A',
            text='七点半ABC，八点ABE，两边都是待成局三缺一。我和B两边都能等，哪边先落实再确认；C只等到七点半，E八点能到。',
            message_id='fixture_parallel_agreement',sent_at=clock.current),'fixture_history')
    if stage=='early_committed':
        store.join_game(game_id=game.game_id,customer_id=cid('F'),display_name='F',seat_count=1,trace_id='fixture_early_confirmed')
    clock.set('18:50');clock.refresh()
    return ids


def snapshot(store: Any) -> dict[str,Any]:
    """从真实Store读取局、安排与草稿，不用模型摘要当作执行结果。"""
    return {'games':[g.to_dict() for g in store.games.values()],
            'board_arrangements':{g.game_id:store.game_board_arrangement(g.game_id) for g in store.games.values()},
            'scheduled_tasks':[t.to_dict() for t in store.scheduled_tasks.values()],
            'invite_drafts':[d.to_dict() for d in store.invite_drafts.values()],
            'outbound_drafts':[d.to_dict() for d in store.outbound_message_drafts.values()]}


def active_games(state: dict[str,Any]) -> list[dict[str,Any]]:
    """保留本次仍开放或已成局的记录；不把取消历史混成人数。"""
    return [g for g in state['games'] if g['status'] in {'forming','inviting','ready','playing'}]


def members(game: dict[str,Any]) -> list[str]:
    """只统计领域已占座状态，候选和协商状态不占席。"""
    return sorted(p['customer_id'] for p in game['participants'] if p['status'] in {'joined','confirmed'})


def evaluate_state(state: dict[str,Any], expected: dict[str,Any], fixture_game_ids: dict[str,str] | None=None) -> list[dict[str,Any]]:
    """按先验声明比较权威人数/时刻/状态；不按回复关键词执行任何业务动作。"""
    games=active_games(state);checks=[]
    def check(name,passed,actual):checks.append({'name':name,'passed':bool(passed),'actual':actual})
    check('capacity_never_exceeds_four',all(sum(p.get('seat_count',1) for p in g['participants'] if p['status'] in {'joined','confirmed'})<=4 for g in games),
          {g['game_id']:members(g) for g in games})
    for actor in ('A','B'):
        ready=[g['game_id'] for g in games if g['status'] in {'ready','playing'} and cid(actor) in members(g)]
        check('no_overlapping_ready_commitment_'+actor,len(ready)<=1,ready)
    for requested in expected.get('plans',[]):
        found=[g for g in games if str(g.get('planned_start_at') or g['requirement'].get('planned_start_at') or '')[11:16]==requested['clock']
               and members(g)==sorted(cid(a) for a in requested['members']) and g['status'] in requested['allowed_statuses']
               and g['remaining_seats']==requested['remaining']]
        check('plan_'+requested['clock'],len(found)==1,[{'game_id':g['game_id'],'time':g.get('planned_start_at'),'status':g['status'],'members':members(g),'remaining':g['remaining_seats']} for g in games])
    for actor in expected.get('absent_from_all',[])+expected.get('not_counted',[]):
        check('not_counted_'+actor,all(cid(actor) not in members(g) for g in games),{g['game_id']:members(g) for g in games})
    if 'max_active_games' in expected:check('no_duplicate_unrelated_game',len(games)<=expected['max_active_games'],len(games))
    if expected.get('no_ready'):check('no_premature_ready',all(g['status'] not in {'ready','playing'} for g in games),[g['status'] for g in games])
    if expected.get('preserve_original_game'):
        original_id=(fixture_game_ids or {}).get('original')
        check('original_game_identity_preserved',bool(original_id) and any(g['game_id']==original_id for g in games),original_id)
        check('no_cancelled_replacement_game',all(g['status']!='cancelled' for g in state['games']),[g['status'] for g in state['games']])
    if expected.get('same_demand_options'):
        roots={g.get('option_root_game_id') or g['game_id'] for g in games}
        check('same_demand_option_relation',len(games)>=2 and len(roots)==1,sorted(roots))
    if expected.get('recruitment_progress'):
        arrangements=state['board_arrangements']
        # 每个仍需招募的方案都必须有自己的持久责任，不能拿另一桌或取消历史的任务冒充。
        ready=bool(games) and all(
            (a:=arrangements.get(g['game_id'],{})).get('status')=='scheduled'
            and a.get('pending_task_count',0)>0 and a.get('outbound_authorized_room_count',0)>0 for g in games)
        check('durable_board_recruitment_responsibility',ready,arrangements)
    return checks


def discover_for_actor(store: Any,actor: str,clock: BusinessClock,trace: InMemoryTraceRecorder) -> None:
    """陌生客户经真实只读Gateway获得可见选项，不拷贝其他人的原始私聊。"""
    if actor not in {'E','F','G'}:return
    conv=conversation(actor);identity=cid(actor);tid='fixture_discovery_'+actor
    store.activate_task_context(conversation_id=conv,customer_id=identity,trace_id=tid,activity_at=clock.current,started_at=clock.current,reason='夹具：咨询临时待组局',force_new=False,archive_previous=False)
    message=UserMessage(conversation_id=conv,sender_id=identity,sender_name=actor,text='今晚财敲0.5无烟有哪些待组的局？',message_id=tid,sent_at=clock.current)
    store.append_user_turn(message,tid)
    result=ToolGateway(store,trace_recorder=trace).execute(ToolCall('search_current_games',{'requirement':{'game_type':'hangzhou_mahjong','game_variant':'caiqiao','stake':'0.5','smoke_preference':'no_smoke'},'limit':10}),
        trace_id=tid,conversation_id=conv,sender_id=identity,sender_name=actor,source_message_id=message.message_id,step_index=1)
    if not result.called or not result.allowed or result.error:raise ValueError('fixture discovery failed: '+str(result.error))
    store.append_tool_turn(conv,json.dumps([result.to_dict()],ensure_ascii=False),tid)
    summaries=result.result.get('customer_reply_contract',{}).get('matched_result_summaries',[])
    if not summaries:
        summaries=[str(item['game']['requirement'].get('start_time') or item['game']['requirement'].get('planned_start_at'))
            +' 财敲0.5无烟，缺'+str(item['game'].get('seat_summary',{}).get('remaining_seats','待核对'))+'人'
            for item in result.result.get('matches',[])]
    store.append_assistant_turn(conv,'当前可选局：'+'；'.join(summaries)+'。你考虑哪一桌？',tid)


def semantic_expectation(source: dict[str,Any],case: dict[str,Any]) -> dict[str,Any]:
    """复用先验业务规则，不把case_id、答案标签或本轮模型自评交给判定器。"""
    return {'required_meaning':[{'id':'temporary_required_'+str(i),'description':s} for i,s in enumerate(case['required_meaning'])]+
            [{'id':'temporary_common_'+str(i),'description':s} for i,s in enumerate(source['common_required'])],
            'forbidden_meaning':[]}


def source_manifest() -> dict[str,str]:
    """绑定本轮正式源码和数据，不将本机路径、凭证或运行库放入版本控制。"""
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(set(names)) if n and (ROOT/n).is_file()}


def run_case(client: Any,args: Any,source: dict[str,Any],case: dict[str,Any],work: Path) -> dict[str,Any]:
    """每个分支独立落库，连续回合不人工修状态；失败也保留前后快照和完整trace。"""
    case_dir=work/case['case_id'];case_dir.mkdir();store=SQLiteAgentStore(case_dir/'state.sqlite3');trace=InMemoryTraceRecorder()
    report={'case_id':case['case_id'],'fixture_stage':case['fixture_stage'],'status':'failed','turns':[],'db_path':str(store.path)}
    try:
        with BusinessClock(source['logical_date']) as clock:
            ids=setup_fixture(store,source,case['fixture_stage'],clock);report['fixture_game_ids']=ids
            report['fixture_state']=snapshot(store);dump(case_dir/'fixture_state.json',report['fixture_state'])
            runtime=owner.build_runtime(client,store,trace,args);seen=set()
            for index,turn in enumerate(case['turns']):
                clock.set(turn['at']);clock.refresh();actor=turn['actor'];tid='temporary_probe_'+uuid.uuid4().hex
                if actor not in seen:discover_for_actor(store,actor,clock,trace);seen.add(actor)
                message=UserMessage(conversation_id=conversation(actor),sender_id=cid(actor),sender_name=actor,text=turn['text'],message_id=tid,sent_at=clock.current)
                row={'index':index+1,'input':message.to_dict(),'before':snapshot(store),'trace_id':tid}
                try:
                    result=runtime.handle_user_message(message,trace_id=tid);events=trace.get_trace(tid)
                    after=snapshot(store);expected=case['expected_final'] if index==len(case['turns'])-1 else {'not_counted':['E'],'no_ready':True}
                    checks=evaluate_state(after,expected,ids)
                    checks.append({'name':'runtime_completed_or_waiting','passed':result.status in {'completed','waiting_user'},'actual':result.status})
                    checks.append({'name':'deterministic_visible_contract','passed':not customer_visible_text_contract_violations(result.final_reply),'actual':customer_visible_text_contract_violations(result.final_reply)})
                    expectation=semantic_expectation(source,case)
                    # 每轮用本轮实际目标判定；后续尚未发生的确认不能成为前轮必须完成的要求。
                    if index<len(case['turns'])-1:
                        expectation['required_meaning']=[{'id':'temporary_current_turn','description':'理解当前到店时间或等待协商，询问或承接真正缺项；不能提前占座、擅自替他人同意或宣布四人已确定。'}]+[r for r in expectation['required_meaning'] if r['id'].startswith('temporary_common_')]
                    evidence=owner.reply_semantic_evidence(result,SimpleNamespace(message=message),events,store)
                    assessment=assess_reply_semantics(client,evidence=evidence,expectation=expectation,trace_id=tid+'_judge')
                    trace.record(tid,'temporary_probe_semantic_assessment',assessment)
                    hard=all(c['passed'] for c in checks)
                    recovery=owner.assess_action_contract_recovery(events,runtime_status=result.status,final_reply=result.final_reply,business_checks_passed=hard,semantic_passed=assessment['passed'])
                    row.update(reply=result.final_reply,runtime_status=result.status,tool_results=[t.to_dict() for t in result.tool_results],after=after,checks=checks,
                        semantic_assessment=assessment,recovery_metrics=recovery,status='passed' if hard and assessment['passed'] and recovery['quality_status']=='passed' else 'failed')
                except Exception as exc:
                    row.update(status='failed',error=type(exc).__name__+': '+str(exc),after=snapshot(store))
                finally:
                    path=case_dir/('turn_'+str(index+1)+'_trace.json');dump(path,{'trace_id':tid,'input':message.to_dict(),'events':[e.to_dict() for e in trace.get_trace(tid)]});row['raw_trace_path']=str(path)
                report['turns'].append(row);dump(case_dir/'report.json',report)
                print(json.dumps({'case_id':case['case_id'],'turn':index+1,'status':row['status'],'reply':row.get('reply'),'error':row.get('error')},ensure_ascii=False),flush=True)
            report['final_state']=snapshot(store)
            report['status']='passed' if all(r['status']=='passed' for r in report['turns']) else 'failed'
    except Exception as exc:report['error']=type(exc).__name__+': '+str(exc)
    finally:
        store._connection.close();dump(case_dir/'report.json',report)
    return report


def main() -> int:
    """只运行独立能力探针，不自动接入既有通过门禁，不改期望或重跑有效失败。"""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,default=DATASET);parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--case',action='append',default=[],help='仅补跑指定分支；报告明确不是全部集合。');parser.add_argument('--max-steps',type=int,default=8);parser.add_argument('--strict',action='store_true')
    args=parser.parse_args();load_dotenv_defaults(ROOT/'.env');source=load_dataset(args.dataset)
    selected=[c for c in source['cases'] if not args.case or c['case_id'] in args.case]
    if not selected or set(args.case)-{c['case_id'] for c in source['cases']}:raise ValueError('unknown selected case')
    client=OpenAICompatibleAgentClient.from_env()
    if client is None:raise SystemExit('missing model credentials; no live observation executed')
    parent=ROOT/'runtime_data/temporary_recruitment_trials';parent.mkdir(parents=True,exist_ok=True)
    work=args.work_dir if args.work_dir else Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%dT%H%M%S')+'_',dir=parent))
    work.mkdir(parents=True,exist_ok=True)
    if any(work.iterdir()):raise ValueError('probe requires a new empty directory')
    args.max_tokens_per_call=64000;args.max_calls_per_turn=8;args.timeout_seconds=90;args.skip_text_generation=False;args.skip_review=False
    before=source_manifest();dotenv=hashlib.sha256((ROOT/'.env').read_bytes()).hexdigest()
    config={k:getattr(client.config,k) for k in ('model','provider','thinking_mode','reasoning_effort','max_tokens','attempt_timeout_seconds')}
    report={'kind':'temporary_recruitment_parallel_probe','status':'running','not_full_gate':True,'main_agent_executed':True,'external_messages_sent':False,
        'execution_scope':'selected_cases' if args.case else 'all_cases','selected_case_ids':[c['case_id'] for c in selected],
        'dataset_path':str(args.dataset.relative_to(ROOT)),'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),'source_manifest':before,'dotenv_sha256':dotenv,'main_model_config':config,'reports':[]}
    dump(work/'report.json',report);print('WORK_DIR='+str(work),flush=True)
    for case in selected:
        report['reports'].append(run_case(client,args,source,case,work));dump(work/'report.json',report)
    after=source_manifest();report['changed_source_files']=[n for n in before.keys()|after.keys() if before.get(n)!=after.get(n)]
    report['dotenv_unchanged']=dotenv==hashlib.sha256((ROOT/'.env').read_bytes()).hexdigest()
    report['passed_count']=sum(r['status']=='passed' for r in report['reports']);report['failed_count']=len(report['reports'])-report['passed_count']
    report['status']='passed' if report['passed_count']==len(selected) and not report['changed_source_files'] and report['dotenv_unchanged'] else 'failed'
    dump(work/'report.json',report);print(json.dumps({k:report[k] for k in ('status','passed_count','failed_count','changed_source_files','dotenv_unchanged')},ensure_ascii=False),flush=True)
    return int(args.strict and report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
