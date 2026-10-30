"""解析真人产品口径的追加修订，保留原案例、旧合同与生效版本，不调用模型。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

PRODUCT_REVISION_KIND = 'owner_chat_product_contract_revision'
DATASET_PATH = Path(__file__).resolve().parents[1]/'eval/golden/real_owner_chat_golden.jsonl'
LEGACY_ARCHIVE_PATH = DATASET_PATH.with_name('query_strategy_legacy_contracts_7b97042.json')
LEGACY_ARCHIVE_SHA256 = '24e0a86880a23449c03a18dbffeba355e72acd9b1fcb0c9f605db4b41cf4e100'
LEGACY_SOURCE_COMMIT = '7b97042e9889b6ee1b0eb58d74a8bc220ca4bfbe'
OWNER_SUPPLEMENT_PATH = DATASET_PATH.with_name('query_strategy_profile_legacy_contract_7b97042.json')
OWNER_SUPPLEMENT_SHA256 = 'ee36d29358c7ea860f10e0caf25689b3b7f9956b4df5eae26f9e69e1aadd34f4'


def canonical_digest(value: Any) -> str:
    """按精确 JSON 类型生成来源摘要，布尔与数字、数组顺序均不能混为一谈。"""
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_product_records(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """读取原资产与追加元数据，既有行只读，不重写历史期望。"""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_frozen_legacy_contracts(path: Path = LEGACY_ARCHIVE_PATH) -> dict[str, Any]:
    """只读固定摘要的 JSON 旧合同，不执行存档函数，也不依赖运行机器的 Git 状态。"""
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=LEGACY_ARCHIVE_SHA256:
        raise ValueError('frozen legacy contract archive hash mismatch')
    archive=json.loads(raw)
    if archive.get('source_commit')!=LEGACY_SOURCE_COMMIT:
        raise ValueError('frozen legacy source commit mismatch')
    return archive


def load_frozen_owner_supplement(path: Path = OWNER_SUPPLEMENT_PATH) -> dict[str, Any]:
    """增补另一原case的固定JSON基线，不改既有存档，不接受元数据自报旧标准。"""
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=OWNER_SUPPLEMENT_SHA256:
        raise ValueError('frozen owner supplement archive hash mismatch')
    archive=json.loads(raw)
    if archive.get('source_commit')!=LEGACY_SOURCE_COMMIT:
        raise ValueError('frozen owner supplement source commit mismatch')
    return archive


def _validate_description_only_revision(revision: dict[str, Any]) -> None:
    """声明局部消歧时，仅准指定既有条目的description变化，禁止夹带其他标准或catalog变动。"""
    if 'changed_criterion_ids' not in revision:
        return
    ids=revision['changed_criterion_ids']
    if (not isinstance(ids,list) or not ids or any(not isinstance(item,str) or not item.strip() for item in ids)
            or len(set(ids))!=len(ids)):
        raise ValueError('changed_criterion_ids must be a non-empty unique string list')
    previous=revision['previous_expectation'];current=revision['current_expectation'];restored=deepcopy(current)
    _expectation(previous,'description-only revision previous')
    old={item['id']:item for group in ('required_meaning','forbidden_meaning') for item in previous[group]}
    new={item['id']:item for group in ('required_meaning','forbidden_meaning') for item in current[group]}
    if any(key not in old or key not in new or old[key]['description']==new[key]['description'] for key in ids):
        raise ValueError('changed_criterion_ids must identify changed existing descriptions')
    for group in ('required_meaning','forbidden_meaning'):
        for item in restored[group]:
            if item['id'] in ids:
                item['description']=old[item['id']]['description']
    if canonical_digest(restored)!=canonical_digest(previous):
        raise ValueError('description-only revision must preserve all undeclared expectation fields')
    if canonical_digest(revision.get('previous_scenario_contract'))!=canonical_digest(revision.get('current_scenario_contract')):
        raise ValueError('description-only revision must preserve catalog contract')


def _expectation(value: Any, label: str) -> None:
    """修订合同必须提供非空正反语义，不能以空要求吞掉事实或安全验收。"""
    if not isinstance(value,dict):
        raise ValueError(f'{label}: expectation must be an object')
    seen=set()
    for field in ('required_meaning','forbidden_meaning'):
        entries=value.get(field)
        if not isinstance(entries,list) or not entries:
            raise ValueError(f'{label}: {field} must be non-empty')
        for entry in entries:
            if not isinstance(entry,dict) or any(not isinstance(entry.get(k),str) or not entry[k].strip() for k in ('id','description')):
                raise ValueError(f'{label}: malformed semantic criterion')
            if entry['id'] in seen:
                raise ValueError(f'{label}: duplicate semantic criterion id')
            seen.add(entry['id'])


def _nonempty(value: Any, label: str) -> None:
    """拒绝缺失来源及空版本号，元数据无法追溯时不能静默生效。"""
    if not isinstance(value,str) or not value.strip():
        raise ValueError(f'{label} must be a non-empty string')


def resolve_product_contract_revisions(
    records: list[dict[str, Any]], *, inquiry_inputs: dict[str, dict[str, Any]] | None = None,
    scenario_contracts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """顺序解析唯一原案例、旧快照和显式产品修订，禁止未来目标、重复版本与断链。

    inquiry 基线是原代码合同的存档，不冒充微信原始对话；owner 来源摘要覆盖
    未改的原 case。当前 catalog 与修订后的字段也必须一致，避免自动生成旧条目残留。
    """
    owners={}; originals={}; origins={}; live_ids={}; inquiry={}; revision_ids={}; seen_revisions=set(); seen_baselines=set(); seen_records=set(); catalog_chain={}
    frozen=load_frozen_legacy_contracts()
    for record in records:
        for case in record.get('eval_cases',[]) if isinstance(record.get('eval_cases'),list) else []:
            if not isinstance(case,dict) or not isinstance(case.get('id'),str):
                continue
            case_id=case['id']
            if case_id in owners:
                raise ValueError(f'duplicate product contract source case: {case_id}')
            owners[case_id]=deepcopy(case.get('expected')); originals[case_id]=case;origins[case_id]=record.get('id');live_ids[case_id]=case.get('live_scenario_id')
        if record.get('kind')!=PRODUCT_REVISION_KIND:
            continue
        record_id=record.get('id');_nonempty(record_id,'product revision record id')
        if record_id in seen_records:
            raise ValueError('duplicate product revision record id')
        seen_records.add(record_id)
        if type(record.get('schema_version')) is not int or record['schema_version']!=1:
            raise ValueError('product revision schema_version must be integer 1')
        if any(k in record for k in ('eval_cases','messages','input','expected')):
            raise ValueError('product revision must be metadata-only')
        source=record.get('source')
        if not isinstance(source,dict) or source.get('kind')!='current_thread_direct_user_instruction' or source.get('not_original_human_transcript') is not True:
            raise ValueError('product revision requires explicit current-thread user instruction provenance')
        for field in ('source_thread_id','client_date','scope'):
            _nonempty(source.get(field),'product source '+field)
        if not isinstance(source.get('user_quotes'),list) or not source['user_quotes'] or any(not isinstance(q,str) or not q.strip() for q in source['user_quotes']):
            raise ValueError('product source requires verbatim user_quotes')
        if source.get('message_ids_unavailable') is not True or source.get('message_id') is not None:
            raise ValueError('product source must disclose unavailable message id without inventing one')
        archive_ref=record.get('legacy_archive')
        if archive_ref!={'path':'eval/golden/query_strategy_legacy_contracts_7b97042.json',
                         'sha256':LEGACY_ARCHIVE_SHA256,'source_commit':LEGACY_SOURCE_COMMIT}:
            raise ValueError('product revision frozen legacy archive reference mismatch')
        owner_baselines=frozen['owner_cases']
        if 'owner_baseline_archive' in record:
            if record['owner_baseline_archive']!={'path':'eval/golden/query_strategy_profile_legacy_contract_7b97042.json',
                    'sha256':OWNER_SUPPLEMENT_SHA256,'source_commit':LEGACY_SOURCE_COMMIT}:
                raise ValueError('product revision frozen owner supplement reference mismatch')
            supplement=load_frozen_owner_supplement()
            if (supplement.get('base_legacy_archive')!=archive_ref
                    or any(frozen['source_sha256'].get(path)!=value for path,value in supplement['source_sha256'].items())
                    or set(owner_baselines)&set(supplement['owner_cases'])):
                raise ValueError('frozen owner supplement must extend the same trusted source without replacement')
            owner_baselines={**owner_baselines,**supplement['owner_cases']}
        baseline_source=record.get('inquiry_baseline_source')
        baselines=record.get('inquiry_expectation_baselines',[])
        if baselines:
            if not isinstance(baseline_source,dict) or not isinstance(baseline_source.get('function_snapshot'),str) or hashlib.sha256(baseline_source['function_snapshot'].encode()).hexdigest()!=baseline_source.get('function_sha256'):
                raise ValueError('inquiry legacy function snapshot hash mismatch')
            if (baseline_source['function_snapshot']!=frozen['inquiry_function_snapshot']
                    or baseline_source['function_sha256']!=frozen['inquiry_function_sha256']):
                raise ValueError('inquiry legacy function differs from trusted frozen source')
        if not isinstance(baselines,list):
            raise ValueError('inquiry baselines must be a list')
        if baselines and {item.get('case_id') for item in baselines if isinstance(item,dict)}!=set(frozen['inquiry_cases']):
            raise ValueError('inquiry baselines must preserve all six frozen cases')
        for baseline in baselines:
            if not isinstance(baseline,dict):
                raise ValueError('inquiry baseline must be an object')
            case_id=baseline.get('case_id');_nonempty(case_id,'inquiry baseline case id')
            if case_id in inquiry or case_id in seen_baselines:
                raise ValueError('duplicate inquiry baseline case')
            input_snapshot=baseline.get('input_snapshot')
            if not isinstance(input_snapshot,dict) or canonical_digest(input_snapshot)!=baseline.get('source_input_sha256'):
                raise ValueError('inquiry baseline input hash mismatch')
            if inquiry_inputs is not None and input_snapshot!=inquiry_inputs.get(case_id):
                raise ValueError('inquiry baseline does not match actual fixed input')
            _expectation(baseline.get('expectation'),'inquiry baseline')
            legacy=frozen['inquiry_cases'].get(case_id)
            if (legacy is None or canonical_digest(input_snapshot)!=canonical_digest(legacy['input_snapshot'])
                    or canonical_digest(baseline['expectation'])!=canonical_digest(legacy['expectation'])):
                raise ValueError('inquiry baseline differs from trusted frozen contract')
            inquiry[case_id]=deepcopy(baseline['expectation']);seen_baselines.add(case_id)
        collections=(('owner',record.get('case_expectation_revisions',[])),('inquiry',record.get('inquiry_expectation_revisions',[])))
        if not any(bool(items) for _,items in collections):
            raise ValueError('product revision requires non-empty expectation revisions')
        for domain,items in collections:
            if not isinstance(items,list):
                raise ValueError('product expectation revisions must be lists')
            for revision in items:
                if not isinstance(revision,dict):
                    raise ValueError('product expectation revision must be an object')
                if domain=='owner' and 'owner_baseline_archive' in record and 'changed_criterion_ids' not in revision:
                    raise ValueError('supplemented owner revision requires changed_criterion_ids')
                for field in ('revision_id','target_case_id','reason'):
                    _nonempty(revision.get(field),'product revision '+field)
                version=revision['revision_id'];target=revision['target_case_id'];key=domain+':'+target
                if version in seen_revisions:
                    raise ValueError('duplicate product revision_id')
                seen_revisions.add(version)
                target_map=owners if domain=='owner' else inquiry
                if target not in target_map:
                    raise ValueError('product revision target must be a unique prior case')
                if domain=='owner':
                    if origins[target]!=revision.get('source_record_id') or live_ids[target]!=revision.get('live_scenario_id'):
                        raise ValueError('product revision source record or live scenario mismatch')
                    if canonical_digest(originals[target])!=revision.get('source_case_sha256'):
                        raise ValueError('product revision original case hash mismatch')
                    previous=revision.get('previous_scenario_contract');current=revision.get('current_scenario_contract')
                    if not isinstance(previous,dict) or not previous or not isinstance(current,dict) or not current:
                        raise ValueError('product revision requires old and current catalog contracts')
                    if set(previous)!=set(current):
                        raise ValueError('product revision catalog snapshots must cover the same fields')
                    source_hash=revision.get('source_catalog_sha256_before_revision')
                    if not isinstance(source_hash,str) or len(source_hash)!=64 or any(c not in '0123456789abcdef' for c in source_hash):
                        raise ValueError('product revision requires original catalog source sha256')
                    if target in catalog_chain and canonical_digest(previous)!=canonical_digest(catalog_chain[target]):
                        raise ValueError('product catalog contract revision is discontinuous')
                    if target not in catalog_chain:
                        legacy=owner_baselines.get(target)
                        if (legacy is None or legacy['source_record_id']!=origins[target]
                                or legacy['live_scenario_id']!=live_ids[target]
                                or legacy['source_case_sha256']!=revision['source_case_sha256']
                                or canonical_digest(previous)!=canonical_digest(legacy['scenario_contract'])
                                or source_hash!=frozen['source_sha256']['scripts/run_real_owner_chat_live_eval.py']):
                            raise ValueError('first owner catalog baseline differs from trusted frozen source')
                    catalog_chain[target]=deepcopy(current)
                if canonical_digest(revision.get('previous_expectation'))!=canonical_digest(target_map[target]):
                    raise ValueError('product expectation previous snapshot mismatch')
                _expectation(revision.get('current_expectation'),'product revision current')
                _validate_description_only_revision(revision)
                if canonical_digest(revision['previous_expectation'])==canonical_digest(revision['current_expectation']):
                    raise ValueError('product revision must change expectation')
                target_map[target]=deepcopy(revision['current_expectation'])
                revision_ids.setdefault(key,[]).append(version)
    # 历史版本只核对前后连续性；只有每条链的最终版本才与当前可执行catalog比较。
    if scenario_contracts is not None:
        for target,current in catalog_chain.items():
            actual=scenario_contracts.get(live_ids[target])
            if actual is None or any(canonical_digest(actual.get(field))!=canonical_digest(value) for field,value in current.items()):
                raise ValueError('product revision current catalog contract mismatch')
    return {'owner':owners,'inquiry':inquiry,'revision_ids':revision_ids}
