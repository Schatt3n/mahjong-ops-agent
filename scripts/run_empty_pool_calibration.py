"""完整空池推论的正反校准：真网关查询与持久事实，独立判定不读取标签。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import sqlite3
import tempfile
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from mahjong_agent_runtime import CustomerProfile, InMemoryTraceRecorder, OpenAICompatibleAgentClient, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.env import load_dotenv_defaults
from run_inquiry_grounding_eval import assess_reply_semantics, project_inquiry_tool_results

DATASET = ROOT/'eval/golden/empty_pool_evidence_calibration_20261105.jsonl'
EXPECTATION = {'required_meaning': [{'id': 'query_scope_and_duration_grounding', 'description':
    '回复必须由同一次真实查询及其完整范围证据支持。成功完整可见可用池为空且supports_narrower_empty_claim=true可支持更窄四小时条件的无局结论，即使未显式过滤时长；非空、截断、失败或false/缺失标志不能使用此推论。实际介绍的局须有约定时长证据，不得虚构过滤、结果或执行。'}]}


def load_cases():
    """固定六条覆盖有效推论、有限范围、非空、截断与失败，不改历史原样本。"""
    cases=[json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if len(cases)!=6 or len({c['case_id'] for c in cases})!=6 or sum(c['expected_passed'] for c in cases)!=2:
        raise ValueError('empty pool calibration fixed source coverage differs')
    return cases


def state(store):
    """只读业务态，查询审计允许新增，业务写入和外发一律不允许。"""
    return {'games':[g.to_dict() for g in store.games.values()], 'invites':len(store.invite_drafts),
        'outbound':len(store.outbound_message_drafts), 'waiting':len(store.waiting_demands)}


def run_case(case, client, folder, *, assessor=assess_reply_semantics):
    """模型只接收真实查询及客户文本，固定标签仅在判定完成后比较。"""
    folder.mkdir(parents=True,exist_ok=False); store=SQLiteAgentStore(folder/'state.sqlite3'); trace=InMemoryTraceRecorder()
    gateway=ToolGateway(store,trace_recorder=trace)
    store.configure_rooms(['calibration_room'])
    store.upsert_customer(CustomerProfile(customer_id='scope_customer',display_name='客户'))
    try:
        durations=[] if case['fixture'] in {'empty','failed'} else [5,4] if case['fixture']=='limited' else [4]
        for i,duration in enumerate(durations):
            customer=f'seed_owner_{i}';store.upsert_customer(CustomerProfile(customer_id=customer,display_name='发起人'))
            created=gateway.execute(ToolCall(name='create_game',arguments={'organizer_id':customer,'organizer_name':'发起人',
                'requirement':{'game_type':'hangzhou_mahjong','stake':'1','smoke_preference':'no_smoke','start_time_kind':'asap_when_full','duration_hours':duration},
                'known_players':[{'customer_id':customer,'display_name':'发起人','seat_count':3}]}),
                trace_id=f'prepare:{i}',conversation_id=f'prepare:{i}',sender_id=customer,sender_name='发起人',step_index=1)
            if not created.called or created.error:raise ValueError('empty pool calibration actual fixture failed')
        before=state(store)
        query=gateway.execute(ToolCall(name='search_current_games',arguments={'requirement':case['requirement'],'limit':case['limit']}),
            trace_id='scope-query',conversation_id='scope-conversation',sender_id='scope_customer',sender_name='客户',step_index=1)
        trace.record('scope-query','tool_result',query.to_dict())
        evidence={'request':case['request'],'reply':case['reply'],'tool_results':project_inquiry_tool_results([query.to_dict()]),
            'persisted_state':{'before':before,'after':state(store)}}
        semantic=assessor(client,evidence=evidence,expectation=EXPECTATION,trace_id='scope-judge')
        passed=semantic.get('error') in (None,'') and semantic.get('passed') is case['expected_passed'] and before==state(store)
        report={'case_id':case['case_id'],'source':case,'status':'passed' if passed else 'failed',
            'main_agent_executed':False,'semantic_assessment':semantic,'query':query.to_dict(),'state':before,'db_path':str(store.path),
            'events':[event.to_dict() for event in trace.events]}
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        return report
    finally:store._connection.close()


def validate_scope(payload, report_path):
    """原始判决、SQLite和实际网关结果三方绑定，不用期望标签自证语义通过。"""
    from run_evals import _validate_reported_semantic_assessment
    cases=load_cases();root=report_path.resolve().parent
    if (payload.get('status')!='passed' or payload.get('execution_scope')!='controlled_empty_pool_semantic_calibration'
            or payload.get('dataset_sha256')!=hashlib.sha256(DATASET.read_bytes()).hexdigest()
            or [r.get('case_id') for r in payload.get('reports',[])]!=[c['case_id'] for c in cases]):
        raise ValueError('empty pool calibration source or coverage differs')
    for case,report in zip(cases,payload['reports']):
        if report.get('source')!=case or report.get('status')!='passed' or report.get('main_agent_executed') is not False:
            raise ValueError('empty pool calibration changed original source')
        db=Path(report['db_path']).resolve()
        if not db.is_relative_to(root) or not db.is_file():raise ValueError('empty pool calibration evidence outside run')
        store=SQLiteAgentStore(db)
        try:actual=state(store)
        finally:store._connection.close()
        if actual!=report['state']:raise ValueError('empty pool calibration persistent state differs')
        query=report['query'];events=report['events']
        # 在数据库副本重放同一只读查询，报告与trace同时被误接也不能自洽通过。
        with tempfile.TemporaryDirectory(prefix='empty_scope_verify_') as temporary:
            copied=Path(temporary)/'state.sqlite3'
            with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as original, sqlite3.connect(copied) as target:
                original.backup(target)
            verification=SQLiteAgentStore(copied)
            try:
                actual_query=ToolGateway(verification).execute(ToolCall(name='search_current_games',arguments={
                    'requirement':case['requirement'],'limit':case['limit']}),trace_id='scope-query',conversation_id='scope-conversation',
                    sender_id='scope_customer',sender_name='客户',step_index=1).to_dict()
            finally:verification._connection.close()
        if any(actual_query[key]!=query[key] for key in ('name','called','allowed','result','error','state_transitions')):
            raise ValueError('empty pool query result contradicts actual authoritative inventory')
        received=[e['content'] for e in events if e['step']=='tool_gateway_received' and e['trace_id']=='scope-query']
        results=[e['content'] for e in events if e['step']=='tool_result' and e['trace_id']=='scope-query']
        if (query['name']!='search_current_games' or results!=[query] or len(received)!=1
                or received[0]['call']['arguments']!={'requirement':case['requirement'],'limit':case['limit']}):
            raise ValueError('empty pool calibration queried changed source')
        completed=[e['content'] for e in events if e['step']=='tool_gateway_completed' and e['trace_id']=='scope-query']
        # schema失败在执行前拒绝；其余源必须有成功进入真实业务查询的网关完成事件。
        if case['fixture']=='failed':
            if (query.get('called') or not query.get('error') or len(completed)!=1
                    or completed[0].get('outcome')!='blocked'):raise ValueError('failed query was not rejected safely')
        elif not query.get('called') or query.get('error') or len(completed)!=1:
            raise ValueError('empty pool calibration actual query absent')
        semantic=report['semantic_assessment'];_validate_reported_semantic_assessment(semantic,EXPECTATION,scope='empty pool calibration',expected_passed=case['expected_passed'])
        expected={'request':case['request'],'reply':case['reply'],'tool_results':project_inquiry_tool_results([query]),'persisted_state':{'before':actual,'after':actual}}
        if semantic['evidence']!=expected:raise ValueError('empty pool calibration judged mismatched facts')


def main():
    """独立真实审查校准可单跑，validate-only仅检查资产。"""
    parser=argparse.ArgumentParser();parser.add_argument('--work-dir',type=Path,required=True);parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    cases=load_cases()
    if args.validate_only:print(json.dumps({'case_count':len(cases)}));return 0
    load_dotenv_defaults(ROOT / ".env");client=OpenAICompatibleAgentClient.from_env()
    if client is None:raise ValueError('model configuration required')
    args.work_dir.mkdir(parents=True,exist_ok=False)
    reports=[run_case(case,client,args.work_dir/case['case_id']) for case in cases]
    payload={'status':'passed' if all(r['status']=='passed' for r in reports) else 'failed','execution_scope':'controlled_empty_pool_semantic_calibration',
        'dataset_sha256':hashlib.sha256(DATASET.read_bytes()).hexdigest(),'reports':reports}
    path=args.work_dir/'report.json';path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    if payload['status']=='passed':validate_scope(payload,path)
    return 0 if payload['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
