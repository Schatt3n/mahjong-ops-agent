#!/usr/bin/env python3
"""以原场景的真实SQLite和Gateway证据校准回复范围判定，仅调用独立judge、不跑主Agent。"""
from __future__ import annotations

import argparse
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT/'src', ROOT/'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from mahjong_agent_runtime import AgentContextBuilder, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.env import load_dotenv_defaults
from mahjong_agent_runtime.llm import OpenAICompatibleAgentClient
import run_real_owner_chat_live_eval as owner
import run_inquiry_grounding_eval as inquiry
import reply_money_eval as money

DATASET_PATH = ROOT/'eval/golden/reply_scope_calibration.jsonl'
CALIBRATION_REVISION_PATH = DATASET_PATH.with_name('reply_scope_calibration_revisions.jsonl')
CALIBRATION_REVISION_SHA256 = '1ddefc08cee0c3ae689f9f98ab7fca6877ce9773f5391d028c911a6f1048e332'
SCENARIO_IDS = {'flexible_time_followup_matches_evening_game', 'single_profile_game_used_for_query'}
# 固定评测义务的证据路由，不改原期望，也不把硬比较伪装成general judge的识别结果。
MONEY_CRITERION_ROLES = {'flexible_time_followup_matches_evening_game.evening_preserve_one_yuan_and_profile_game':'stake'}


def digest(value: Any) -> str:
    """绑定精确JSON类型的证据，避免数字/布尔或字段变动被宽松相等吞掉。"""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _apply_allowed_violation_revision(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """仅追加已有换档规则的重叠违例许可；原样本、必需错档条件及历史分数不变。

    固定摘要绑定审阅后的单一声明，并复核原校准文件、完整原case和原业务规则。
    修订只用于后端比较允许集，不加入判定器输入，也不追改旧运行的判决或成绩。
    """
    raw = CALIBRATION_REVISION_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CALIBRATION_REVISION_SHA256:
        raise ValueError('reply scope calibration revision source changed without review')
    declarations = [json.loads(line, object_pairs_hook=inquiry._unique_json_object)
                    for line in raw.decode().splitlines() if line.strip()]
    fields = {'schema_version','kind','revision_id','source','source_dataset_path','source_dataset_sha256',
              'target_case_id','target_scenario_id','source_case_sha256','previous_allowed_violation_ids',
              'current_allowed_violation_ids','preserved_required_violation_ids','preserved_expected_passed',
              'rule_source','historical_scores_unchanged'}
    if len(declarations) != 1 or not isinstance(declarations[0], dict) or set(declarations[0]) != fields:
        raise ValueError('reply scope calibration requires one complete scoped revision')
    record = declarations[0]
    target_id, scenario_id = 'known_option_wrong_stake_20261103', 'flexible_time_followup_matches_evening_game'
    criterion = scenario_id + '.evening_no_unnecessary_time_or_stake_change'
    source = record['source']
    if (type(record['schema_version']) is not int or record['schema_version'] != 1
            or record['kind'] != 'reply_scope_calibration_allowed_violation_revision'
            or record['revision_id'] != 'known_option_wrong_stake_overlapping_violation_20261103'
            or not isinstance(source, dict) or source.get('kind') != 'original_rule_overlap_review'
            or source.get('not_original_human_transcript') is not True
            or not isinstance(source.get('basis'), str) or not source['basis'].strip()
            or record['source_dataset_path'] != 'eval/golden/reply_scope_calibration.jsonl'
            or record['source_dataset_sha256'] != hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
            or record['target_case_id'] != target_id or record['target_scenario_id'] != scenario_id
            or record['historical_scores_unchanged'] is not True):
        raise ValueError('reply scope calibration revision source or identity differs')
    targets = [case for case in cases if case['case_id'] == target_id and case['scenario_id'] == scenario_id]
    if len(targets) != 1:
        raise ValueError('reply scope calibration original target is missing or duplicated')
    target = targets[0]
    if (record['source_case_sha256'] != digest(target)
            or digest(record['previous_allowed_violation_ids']) != digest(target['allowed_violation_ids'])
            or digest(record['current_allowed_violation_ids']) != digest(target['allowed_violation_ids'] + [criterion])
            or digest(record['preserved_required_violation_ids']) != digest(target['required_violation_ids'])
            or record['preserved_expected_passed'] is not False or target['expected_passed'] is not False):
        raise ValueError('reply scope calibration revision changed original labels or more than the overlap')
    # 依据必须来自既有原规则，不能由本次模型理由或当前输出临时创造新允许项。
    originals = [case for item in owner.read_product_records()
                 if item.get('id') == 'owner_chat_flexible_time_and_help_offer_20261029_001'
                 for case in item.get('eval_cases', []) if case.get('id') == scenario_id]
    if len(originals) != 1:
        raise ValueError('reply scope calibration original business case is missing or duplicated')
    rules = [item for item in originals[0]['expected']['forbidden_meaning']
             if item['id'] == 'evening_no_unnecessary_time_or_stake_change']
    expected_rule = {'path':'eval/golden/real_owner_chat_golden.jsonl',
                     'record_id':'owner_chat_flexible_time_and_help_offer_20261029_001','case_id':scenario_id,
                     'case_sha256':digest(originals[0]),'criterion_id':criterion,
                     'description':rules[0]['description'] if len(rules) == 1 else None}
    if len(rules) != 1 or digest(record['rule_source']) != digest(expected_rule):
        raise ValueError('reply scope calibration revision business rule differs from its original source')
    result = deepcopy(cases)
    next(case for case in result if case['case_id'] == target_id)['allowed_violation_ids'].append(criterion)
    return result


def load_cases() -> list[dict[str, Any]]:
    """固定四个正反对照，标签仅由后端验收，不能以当前被测输出改写标准。"""
    cases = [json.loads(line) for line in DATASET_PATH.read_text().splitlines() if line.strip()]
    fields = {'schema_version','kind','case_id','scenario_id','reply','expected_passed','required_violation_ids','allowed_violation_ids','source'}
    if len(cases) != 4 or any(not isinstance(case, dict) or set(case) != fields for case in cases):
        raise ValueError('reply scope calibration requires its four complete source cases')
    ids = [case['case_id'] for case in cases]
    if len(set(ids)) != 4 or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError('reply scope calibration identities are invalid')
    for case in cases:
        if (type(case['schema_version']) is not int or case['schema_version'] != 1 or case['kind'] != 'reply_scope_calibration'
                or case['scenario_id'] not in SCENARIO_IDS or not isinstance(case['reply'], str) or not case['reply'].strip()
                or type(case['expected_passed']) is not bool or not isinstance(case['source'], dict)
                or case['source'].get('kind') != 'controlled_original_scenario_boundary'
                or case['source'].get('not_original_human_transcript') is not True):
            raise ValueError('reply scope calibration source or typed labels are invalid')
        for key in ('required_violation_ids','allowed_violation_ids'):
            values = case[key]
            if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
                raise ValueError('reply scope calibration violation identities are invalid')
        if (not set(case['required_violation_ids']) <= set(case['allowed_violation_ids'])
                or bool(case['required_violation_ids']) is case['expected_passed']
                or (case['expected_passed'] and case['allowed_violation_ids'])):
            raise ValueError('reply scope calibration must distinguish allowed positive and negative boundaries')
    if any(sum(case['scenario_id']==sid for case in cases)!=2 or sum(case['scenario_id']==sid and case['expected_passed'] for case in cases)!=1 for sid in SCENARIO_IDS):
        raise ValueError('reply scope calibration requires one positive/negative pair per original scenario')
    return _apply_allowed_violation_revision(cases)


def original_scenarios() -> dict[str, Any]:
    """从正式目录取原两场景；不复制或降低人工语义标准。"""
    result = {scenario.scenario_id: scenario for scenario in owner.live_eval_scenarios() if scenario.scenario_id in SCENARIO_IDS}
    if set(result) != SCENARIO_IDS:
        raise ValueError('original reply scope scenarios are missing')
    return result


def query_call(scenario_id: str) -> ToolCall:
    """仅按受控夹具声明准备原场景的真实只读工具，不实现自然语言路由。"""
    return ToolCall(name='search_current_games', arguments={'requirement': {'game_type':'hangzhou_mahjong','stake':'1'}, 'limit':10},
                    reason='受控原场景只读证据') if scenario_id=='flexible_time_followup_matches_evening_game' else ToolCall(
                    name='check_room_availability', arguments={}, reason='受控原场景只读证据')


def sqlite_snapshot(path: Path) -> dict[str, Any]:
    """只读事务快照包含所有持久表；隐藏写入或只改payload不能冒充只读校准。"""
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)) as connection:
        connection.execute('BEGIN')
        names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {name: sorted([list(row) for row in connection.execute('SELECT * FROM "'+name.replace('"','""')+'"')], key=lambda row:json.dumps(row,sort_keys=True)) for name in names}


def combined_assessment(assessment: dict[str, Any], money_check: dict[str, Any], expectation: dict[str, Any]) -> dict[str, Any]:
    """独立记录模型与数值硬证据；不篡改模型原始passed或criteria。"""
    general={item['id'] for item in assessment['criteria'] if not item['passed']}
    ids=set(inquiry._semantic_criteria_ids(expectation));hard=set()
    for criterion,role in MONEY_CRITERION_ROLES.items():
        if criterion in ids and money_check['status']=='conflict' and any(
                row['status']=='mismatch' and row['scope']=='existing_option_fact' and row['role']==role
                for row in money_check['comparisons']):
            hard.add(criterion)
    return {'general_semantic_passed':assessment['passed'],'general_violation_ids':sorted(general),
            'money_fact_violation_ids':sorted(hard),'effective_violation_ids':sorted(general|hard),
            'passed':not general and not money_check['blocks']}


def assessment_matches(case: dict[str, Any], assessment: Any, *, expectation: dict[str, Any], evidence: dict[str, Any], trace_id: str,
                       money_check: Any, tool_results: list[Any], money_trace_id: str) -> bool:
    """原始判定与金额独立证据共同验收原义务，不能借硬检查隐藏判定器误读。"""
    if not money.validate_money_fact_check(money_check,reply=case['reply'],tool_results=tool_results,trace_id=money_trace_id):
        return False
    if (not isinstance(assessment, dict) or assessment.get('error') is not None
            or digest(assessment.get('expectation')) != digest(expectation) or digest(assessment.get('evidence')) != digest(evidence)
            or assessment.get('rubric') != inquiry.REPLY_SEMANTIC_JUDGE_RUBRIC
            or digest(assessment.get('judge_expectation')) != digest(inquiry._project_judge_expectation(expectation))
            or assessment.get('trace_id') != trace_id):
        return False
    attempts = assessment.get('attempts')
    if (not inquiry.semantic_judge_config_matches(assessment,require_http=True)
            or not isinstance(attempts,list) or len(attempts) not in (1,2) or any(not isinstance(a,dict) for a in attempts)
            or type(assessment.get('contract_repair_count')) is not int or assessment['contract_repair_count']!=len(attempts)-1
            or attempts[0].get('trace_id')!=trace_id or attempts[-1].get('error') is not None
            or attempts[-1].get('raw_response')!=assessment.get('raw_response')):
        return False
    try:
        ids = inquiry._semantic_criteria_ids(expectation)
        if not set(case['allowed_violation_ids']) <= set(ids):
            return False
        if len(attempts)==2:
            if not isinstance(attempts[0].get('error'),str) or not attempts[0]['error'].strip() or attempts[1].get('trace_id')!=trace_id+':contract_repair_1':
                return False
            try:
                inquiry._parse_semantic_assessment(attempts[0].get('raw_response',''),ids)
            except ValueError:
                pass
            else:
                return False
        parsed = inquiry._parse_semantic_assessment(assessment.get('raw_response',''),ids)
    except (ValueError,TypeError):
        return False
    if (any(digest(assessment.get(key))!=digest(parsed[key]) for key in ('reason','criteria','passed'))
            or attempts[-1].get('passed') is not parsed['passed']):
        return False
    combined=combined_assessment(parsed,money_check,expectation)
    violated=set(combined['effective_violation_ids'])
    # 错金额负例必须真实冻结原字面、盲分类为现成局底注并与同一候选硬比冲突；
    # 判定器故障、作用域不明或未覆盖不能冒充发现了预期负例。
    required_money=set(case['required_violation_ids'])&set(MONEY_CRITERION_ROLES)
    return (required_money<=set(combined['money_fact_violation_ids']) and combined['passed'] is case['expected_passed']
            and set(case['required_violation_ids'])<=violated<=set(case['allowed_violation_ids']))


def run_calibration(client: Any, args: Any, work_dir: Path) -> dict[str, Any]:
    """新目录逐例执行真实只读网关，再独立判定固定文本；不调主Agent或外发。"""
    cases, scenarios = load_cases(), original_scenarios()
    work_dir.mkdir(parents=True,exist_ok=False)
    reports=[]
    for index,case in enumerate(cases,1):
        case_dir=work_dir/case['case_id'];case_dir.mkdir()
        store=SQLiteAgentStore(case_dir/'calibration.sqlite3');trace=InMemoryTraceRecorder();trace_id=f'reply_scope_{index:02d}'
        item={'case_id':case['case_id'],'trace_id':trace_id,'source':deepcopy(case),'status':'failed','db_path':str(store.path)}
        try:
            scenario=scenarios[case['scenario_id']];scenario.setup(store);message=deepcopy(scenario.message)
            store.append_user_turn(message,trace_id)
            gateway=ToolGateway(store=store,trace_recorder=trace)
            context=AgentContextBuilder(store,gateway).build(message,trace_id=trace_id).payload
            trace.record(trace_id,'context_built',deepcopy(context))
            before=sqlite_snapshot(store.path);call=query_call(case['scenario_id'])
            result=gateway.execute(call,trace_id=trace_id,conversation_id=message.conversation_id,sender_id=message.sender_id,
                sender_name=message.sender_name,step_index=1,source_message_id=message.message_id)
            trace.record(trace_id,'tool_result',result.to_dict())
            after=sqlite_snapshot(store.path)
            controlled=SimpleNamespace(final_reply=case['reply'],status='waiting_user',tool_results=[result])
            evidence=owner.reply_semantic_evidence(controlled,SimpleNamespace(message=message),trace.get_trace(trace_id),store)
            expectation=owner.reply_semantic_expectation(scenario)
            assessment=inquiry.assess_reply_semantics(client,evidence=evidence,expectation=expectation,
                trace_id=trace_id+':judge',judge_timeout_seconds=getattr(args,'judge_timeout_seconds',None))
            trace.record(trace_id,'reply_scope_assessment',deepcopy(assessment))
            money_check=money.check_reply_money_facts(client,reply=case['reply'],tool_results=[result],
                trace_id=trace_id+':money',timeout_seconds=args.timeout_seconds)
            trace.record(trace_id,'reply_money_fact_check',deepcopy(money_check))
            final=sqlite_snapshot(store.path)
            read_only=all(before[name]==after[name] for name in before if name!='runtime_idempotency_ledger') and after==final
            passed=result.called and result.allowed and not result.error and read_only and assessment_matches(
                case,assessment,expectation=expectation,evidence=evidence,trace_id=trace_id+':judge',
                money_check=money_check,tool_results=[result],money_trace_id=trace_id+':money')
            item.update(status='passed' if passed else 'failed',request=message.to_dict(),tool_call=call.to_dict(),
                tool_result=result.to_dict(),expectation=expectation,semantic_assessment=assessment,money_fact_check=money_check,
                combined_assessment=combined_assessment(assessment,money_check,expectation) if assessment.get('criteria') else None,read_only=read_only)
            raw={'case_id':case['case_id'],'trace_id':trace_id,'run_id':work_dir.name,'source':deepcopy(case),
                'request':message.to_dict(),'tool_call':call.to_dict(),'tool_result':result.to_dict(),
                'before_tools':before,'before_judge':after,'after_judge':final,
                'evidence':evidence,'expectation':expectation,'semantic_assessment':assessment,'money_fact_check':money_check,
                'combined_assessment':item['combined_assessment'],
                'events':[event.to_dict() for event in trace.get_trace(trace_id)]}
        except Exception as exc:
            error=inquiry._semantic_judge_error(exc,getattr(client,'config',None))
            item.update(error=error)
            raw={'case_id':case['case_id'],'trace_id':trace_id,'run_id':work_dir.name,'error':error,
                 'events':[event.to_dict() for event in trace.get_trace(trace_id)]}
        finally:
            store._connection.close()
        raw_path=case_dir/'raw_trace.json';raw_path.write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
        item.update(raw_trace_path=str(raw_path),trace_steps=[event['step'] for event in raw['events']],run_id=work_dir.name)
        reports.append(item)
    passed=sum(item['status']=='passed' for item in reports)
    return {'kind':'reply_scope_calibration_report','status':'passed' if passed==len(cases) else 'failed',
        'run_id':work_dir.name,'dataset_path':'eval/golden/reply_scope_calibration.jsonl',
        'dataset_sha256':hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),'case_count':len(cases),
        'passed_count':passed,'failed_count':len(cases)-passed,'main_agent_executed':False,'reports':reports}



def validate_report(payload: dict[str, Any], report_path: Path) -> dict[str, Any]:
    """门禁重读本次源、原始轨迹、独立判定和SQLite，不信子报告自报通过。"""
    from datetime import datetime
    from mahjong_agent_runtime import CustomerProfile
    from mahjong_agent_runtime.domains.context_builders.tool_results import turn_payload_for_context
    from mahjong_agent_runtime.stores.sqlite.serialization import customer_from_payload, turn_from_payload

    cases,scenarios=load_cases(),original_scenarios();root=report_path.resolve().parent
    if (not isinstance(payload,dict) or payload.get('kind')!='reply_scope_calibration_report' or payload.get('status')!='passed'
            or payload.get('main_agent_executed') is not False or payload.get('dataset_path')!='eval/golden/reply_scope_calibration.jsonl'
            or payload.get('dataset_sha256')!=hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
            or any(type(payload.get(key)) is not int or payload[key]!=value for key,value in (('case_count',4),('passed_count',4),('failed_count',0)))):
        raise ValueError('reply scope calibration source, coverage or scope is invalid')
    reports=payload.get('reports')
    if (not isinstance(reports,list) or len(reports)!=4 or any(not isinstance(item,dict) for item in reports)
            or {item.get('case_id') for item in reports}!={case['case_id'] for case in cases}):
        raise ValueError('reply scope calibration must execute every fixed case exactly once')
    by_id={item['case_id']:item for item in reports};seen_paths=set()
    for index,source in enumerate(cases,1):
        item=by_id[source['case_id']];trace_id=f'reply_scope_{index:02d}'
        if (item.get('status')!='passed' or item.get('source')!=source or item.get('trace_id')!=trace_id
                or item.get('run_id')!=payload.get('run_id') or item.get('read_only') is not True):
            raise ValueError('reply scope calibration source or case identity differs')
        paths=[]
        for key in ('raw_trace_path','db_path'):
            value=item.get(key)
            if not isinstance(value,str) or not value:
                raise ValueError('reply scope original evidence path missing')
            path=Path(value).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path in seen_paths:
                raise ValueError('reply scope evidence is missing, reused or outside this run')
            seen_paths.add(path);paths.append(path)
        raw_path,db_path=paths
        if raw_path.parent!=db_path.parent or raw_path.parent.name!=source['case_id'] or raw_path.parent.parent.name!=payload['run_id']:
            raise ValueError('reply scope evidence is not bound to its unique current case directory')
        raw=json.loads(raw_path.read_text());events=raw.get('events')
        if (raw.get('case_id')!=source['case_id'] or raw.get('source')!=source or raw.get('trace_id')!=trace_id
                or raw.get('run_id')!=payload['run_id'] or not isinstance(events,list) or not events
                or any(not isinstance(event,dict) or event.get('trace_id')!=trace_id or not isinstance(event.get('content'),dict) for event in events)
                or item.get('trace_steps')!=[event.get('step') for event in events]):
            raise ValueError('reply scope original trace source or complete events are invalid')
        grouped={step:[event['content'] for event in events if event.get('step')==step] for step in
                 ('context_built','tool_gateway_received','tool_gateway_completed','tool_result','reply_scope_assessment','reply_money_fact_check')}
        if (any(len(values)!=1 for values in grouped.values())
                or any(event.get('step') in {'llm_prompt','llm_response','state_transition','tool_exception',
                    'customer_visible_text_generation_response','customer_visible_content_review_response'} for event in events)):
            raise ValueError('reply scope requires exactly one real readonly Gateway call and one independent judgement, with no main Agent')
        scenario=scenarios[source['scenario_id']];expected=owner.reply_semantic_expectation(scenario)
        call=query_call(source['scenario_id']).to_dict();request=raw.get('request');result=raw.get('tool_result')
        if (not isinstance(request,dict) or any(request.get(key)!=scenario.message.to_dict()[key] for key in
                ('text','conversation_id','sender_id','sender_name','message_id')) or item.get('request')!=request
                or raw.get('tool_call')!=call or item.get('tool_call')!=call or grouped['tool_gateway_received'][0].get('call')!=call
                or not isinstance(result,dict) or item.get('tool_result')!=result or grouped['tool_result']!=[result]
                or result.get('name')!=call['name'] or result.get('called') is not True or result.get('allowed') is not True
                or result.get('error') is not None or result.get('deduplicated') is not False
                or digest(raw.get('expectation'))!=digest(expected) or digest(item.get('expectation'))!=digest(expected)):
            raise ValueError('reply scope input, original expectation or actual tool execution differs')
        completed=grouped['tool_gateway_completed'][0]
        if any(completed.get(key)!=value for key,value in (('tool_name',call['name']),('called',True),('allowed',True),
                ('outcome','executed'),('error',None),('idempotency_key',result.get('idempotency_key')))):
            raise ValueError('reply scope Gateway completion disagrees with actual result')
        state=sqlite_snapshot(db_path);before=raw.get('before_tools');after=raw.get('before_judge')
        if (not isinstance(before,dict) or set(before)!=set(state) or digest(after)!=digest(state)
                or digest(raw.get('after_judge'))!=digest(state)
                or any(before[name]!=state[name] for name in state if name!='runtime_idempotency_ledger')
                or before['runtime_idempotency_ledger'] or len(state['runtime_idempotency_ledger'])!=1):
            raise ValueError('reply scope read or judge changed persistent business state')
        known=source['scenario_id']=='flexible_time_followup_matches_evening_game'
        counts={'runtime_customers':1,'runtime_rooms':1,'runtime_idempotency_ledger':1,
                'runtime_conversation_turns':3 if known else 1}
        if known:
            counts.update(runtime_games=1,runtime_game_participants=1,runtime_scheduled_agent_tasks=1,runtime_state_transitions=2)
        if any(len(rows)!=counts.get(name,0) for name,rows in state.items()):
            raise ValueError('reply scope SQLite differs from the original minimal fixture or contains hidden work')
        with closing(sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True)) as connection:
            profile=customer_from_payload(json.loads(connection.execute('SELECT payload FROM runtime_customers WHERE customer_id=?',(request['sender_id'],)).fetchone()[0])).to_model_context()
            wanted=CustomerProfile(customer_id=request['sender_id'],display_name=request['sender_name'],
                preferred_games=['hangzhou_mahjong'] if known else ['national_standard_mahjong'],
                preferred_stakes=['1','2'] if known else [],smoke_preference='any' if known else None).to_model_context()
            history=[turn_payload_for_context(turn_from_payload(json.loads(row[0]))) for row in connection.execute(
                'SELECT payload FROM runtime_conversation_turns WHERE conversation_id=? ORDER BY id',(request['conversation_id'],))]
            ledger=connection.execute('SELECT payload FROM runtime_idempotency_ledger WHERE idempotency_key=?',(result.get('idempotency_key'),)).fetchone()
        if profile!=wanted or ledger is None or digest(json.loads(ledger[0]))!=digest(result):
            raise ValueError('reply scope profile or persisted Gateway result differs from original facts')
        if [turn['content'] for turn in history]!=(['下午有人打吗','打多大的，几点方便？',request['text']] if known else [request['text']]):
            raise ValueError('reply scope necessary original dialogue is missing or changed')
        context=grouped['context_built'][0]
        if context.get('sender_profile')!=profile or context.get('recent_conversation')!=history:
            raise ValueError('reply scope context contradicts persisted original history/profile')
        business=owner.read_semantic_calibration_state(db_path);games=business['games'];body=result.get('result',{})
        if known:
            game=games[0];requirement=game['requirement'];matches=body.get('matches',[])
            if (requirement.get('game_type')!='hangzhou_mahjong' or requirement.get('stake')!='1'
                    or requirement.get('user_visible_summary')!='晚上七点1块，三缺一' or game['seat_summary']['claimed_seats']!=3
                    or game['remaining_seats']!=1 or game['conversation_id']==request['conversation_id']
                    or datetime.fromisoformat(game['planned_start_at']).strftime('%H:%M')!='19:00'
                    or datetime.fromisoformat(game['planned_start_at']).date()!=datetime.fromisoformat(request['sent_at']).date()
                    or len(matches)!=1 or matches[0].get('game',{}).get('game_id')!=game['game_id']
                    or body.get('requirement',{}).get('game_type')!='hangzhou_mahjong' or body.get('requirement',{}).get('stake')!='1'):
                raise ValueError('reply scope actual known option/price/time is not the original scenario')
        elif (games or body.get('configured') is not True or body.get('room_count')!=1 or body.get('available_count')!=1
                or (datetime.fromisoformat(body['end_at'])-datetime.fromisoformat(body['start_at'])).total_seconds()!=4*3600):
            raise ValueError('reply scope unknown-time evidence does not establish just one available room')
        evidence=raw.get('evidence');assessment=item.get('semantic_assessment')
        observed=evidence.get('persisted_state',{}).get('follow_up_state',{}).get('observed_at') if isinstance(evidence,dict) else None
        if not isinstance(observed,str) or datetime.fromisoformat(observed).tzinfo is None:
            raise ValueError('reply scope follow-up observation identity missing')
        expected_evidence={'request':request,'reply':source['reply'],'recent_conversation':history,'known_profile':profile,
            'tool_results':owner.summarize_tool_results([SimpleNamespace(**result)]),
            'persisted_state':{'sender_profile':profile,'related_active_games':[],'games':games,'sender_task_memories':[],
                'invite_statuses':[],'follow_up_state':{'observed_at':observed,'conversation_id':request['conversation_id'],
                    'customer_id':request['sender_id'],'current_task_context_id':None,'tasks':[]}},
            'runtime_status':'waiting_user','execution_lifecycle':{'turn':{'status':'waiting_user','scope':'synchronous_turn_only'},'activities':[]}}
        money_check=item.get('money_fact_check')
        if (digest(evidence)!=digest(expected_evidence) or digest(raw.get('semantic_assessment'))!=digest(assessment)
                or grouped['reply_scope_assessment']!=[assessment]
                or digest(raw.get('money_fact_check'))!=digest(money_check) or grouped['reply_money_fact_check']!=[money_check]
                or not assessment_matches(source,assessment,expectation=expected,evidence=expected_evidence,trace_id=trace_id+':judge',
                    money_check=money_check,tool_results=[result],money_trace_id=trace_id+':money')
                or digest(item.get('combined_assessment'))!=digest(combined_assessment(assessment,money_check,expected))
                or digest(raw.get('combined_assessment'))!=digest(item.get('combined_assessment'))):
            raise ValueError('reply scope judge input, original attempts or fixed positive/negative discrimination failed')
    return {'total':4,'passed':4,'failed':0,'skipped':0,'executed_case_ids':[case['case_id'] for case in cases]}

def main() -> int:
    """仅显式运行本scope时创建唯一证据目录；无密钥或校准失败返回非零，不重复语义判定。"""
    # 与其它真实runner一致，先加载项目环境，再求默认超时；显式CLI参数仍优先。
    load_dotenv_defaults(ROOT / '.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-path',type=Path,required=True);parser.add_argument('--strict',action='store_true')
    parser.add_argument('--timeout-seconds',type=float,default=float(os.getenv('MAHJONG_AGENT_LLM_TIMEOUT_SECONDS','90')),
        help='仅用于独立金额作用域分类的调用超时；复杂语义判定使用独立judge预算。')
    parser.add_argument('--judge-timeout-seconds',type=float,default=None,
        help='独立复杂语义judge预算；优先于MAHJONG_SEMANTIC_JUDGE_TIMEOUT_SECONDS及180秒默认。')
    args=parser.parse_args()
    parent=args.report_path.parent;parent.mkdir(parents=True,exist_ok=True)
    name=Path(tempfile.mkdtemp(prefix='reply_scope_',dir=parent));name.rmdir()
    payload=run_calibration(OpenAICompatibleAgentClient.from_env(),args,name)
    args.report_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    return int(args.strict and payload['status']!='passed')


if __name__=='__main__':
    raise SystemExit(main())
