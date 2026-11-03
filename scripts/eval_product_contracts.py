"""解析真人产品口径的追加修订，保留原案例、旧合同与生效版本，不调用模型。"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
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
REPLY_PRESENTATION_REVISION_KIND = 'owner_reply_presentation_policy_revision'
ACTIVE_REVISION_FIXTURE_PATH = DATASET_PATH.with_name('active_input_revision.jsonl')
ACTIVE_REVISION_POLICY_PATH = DATASET_PATH.with_name('active_input_revision_policy.jsonl')
ACTIVE_REVISION_POLICY_SHA256 = 'a96f01bf48cc94c976c95a0fa70016535408d14039d695fd10600d12ed121462'
QUERY_OPTION_POLICY_PATH = DATASET_PATH.with_name('query_option_reply_policy.jsonl')
QUERY_OPTION_POLICY_SHA256 = '4c3b0d86a66734a000e0e7a50bbab7d52852d77c1d90d2504a99b41a2485e3a3'


def canonical_digest(value: Any) -> str:
    """按精确 JSON 类型生成来源摘要，布尔与数字、数组顺序均不能混为一谈。"""
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_product_records(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """读取原资产与追加元数据，既有行只读，不重写历史期望。"""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def apply_active_revision_policy(fixture: dict[str, Any]) -> dict[str, Any]:
    """应用用户明示的查询沟通口径，原输入、旧标准和旧失败报告均保持只读。

    固定摘要绑定已审阅的产品修订，不能把当前被测输出反向生成标准。新版本仅
    允许告知/征询两条路径，并由独立语义判定核验回复与同步状态是否一致。
    """
    raw = ACTIVE_REVISION_POLICY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ACTIVE_REVISION_POLICY_SHA256:
        raise ValueError('active revision user policy source changed without review')
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(records) != 1:
        raise ValueError('active revision requires one explicit user policy revision')
    record = records[0]
    original = ACTIVE_REVISION_FIXTURE_PATH.read_bytes()
    if (record.get('source_fixture_path') != 'eval/golden/active_input_revision.jsonl'
            or record.get('source_fixture_sha256') != hashlib.sha256(original).hexdigest()
            or canonical_digest(json.loads(original)) != canonical_digest(fixture)
            or record.get('source_fixture_id') != fixture.get('id')
            or record.get('previous_expectation') != fixture.get('expected')):
        raise ValueError('active revision original fixture or expectation drifted')
    result = deepcopy(fixture)
    result['expected'] = deepcopy(record['current_expectation'])
    result['product_contract_revision_ids'] = [record['revision_id']]
    return result


def _query_option_revisions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只应用两条经用户确认的纯查询口径；旧来源及全部非询问义务保持精确不变。"""
    raw = QUERY_OPTION_POLICY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != QUERY_OPTION_POLICY_SHA256:
        raise ValueError('query option policy source changed without review')
    declarations = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(declarations) != 1:
        raise ValueError('query option policy requires one fixed metadata record')
    record = declarations[0]
    source, baseline = record.get('source', {}), record.get('source_catalog_baseline', {})
    prefix = record.get('original_dataset_prefix_bytes')
    dataset = DATASET_PATH.read_bytes()
    if (not isinstance(source, dict) or not isinstance(baseline, dict)
            or type(record.get('schema_version')) is not int or record['schema_version'] != 1
            or record.get('kind') != 'owner_query_option_reply_policy_revision'
            or source.get('kind') != 'current_thread_direct_user_instruction'
            or source.get('not_original_human_transcript') is not True
            or source.get('message_ids_unavailable') is not True or source.get('message_id') is not None
            or source.get('source_thread_id') != 'current_thread_id_unavailable'
            or source.get('user_quotes') != ['两种都允许，状态与实际回复一致']
            or baseline.get('kind') != 'reviewed_code_catalog' or baseline.get('not_original_human_transcript') is not True
            or baseline.get('script_path') != 'scripts/run_real_owner_chat_live_eval.py'
            or not isinstance(baseline.get('script_sha256_before_policy'), str) or len(baseline['script_sha256_before_policy']) != 64
            or type(prefix) is not int or not 0 < prefix <= len(dataset)
            or hashlib.sha256(dataset[:prefix]).hexdigest() != record.get('original_dataset_prefix_sha256')):
        raise ValueError('query option policy provenance or original dataset prefix is invalid')
    targets = {
        'flexible_time_followup_matches_evening_game': ('owner_chat_flexible_time_and_help_offer_20261029_001',
            'flexible_time_followup_matches_evening_game', 'evening_offer_verified_option'),
        'profile_range_matches_middle_stake': ('owner_chat_live_semantic_regression_20261029_001',
            'synthetic_live_profile_range_matches_middle_stake', 'offer_actual_middle_stake_game'),
    }
    revisions = record.get('revisions')
    if (not isinstance(revisions, list) or len(revisions) != len(targets)
            or any(not isinstance(item, dict) for item in revisions)
            or {item.get('live_scenario_id') for item in revisions} != set(targets)):
        raise ValueError('query option policy must cover exactly its two authorized query scenes')
    versions = set()
    for revision in revisions:
        scene = revision['live_scenario_id']
        record_id, case_id, criterion = targets[scene]
        originals = [case for item in records if item.get('id') == record_id
            for case in (item.get('eval_cases', []) if isinstance(item.get('eval_cases'), list) else [])
            if isinstance(case, dict) and case.get('id') == case_id]
        version = revision.get('revision_id')
        if (revision.get('source_record_id') != record_id or revision.get('target_case_id') != case_id
                or len(originals) != 1 or revision.get('source_case_sha256') != canonical_digest(originals[0])
                or not isinstance(version, str) or not version or version in versions
                or canonical_digest(revision.get('previous_expectation')) != canonical_digest(originals[0].get('expected'))):
            raise ValueError('query option policy original case or revision identity drifted')
        versions.add(version)
        before, after = revision['previous_expectation'], revision.get('current_expectation')
        _expectation(after, 'query option policy current')
        old_entries = [item for item in before['required_meaning'] if item['id'] == criterion]
        new_entries = [item for item in after['required_meaning'] if item['id'] == criterion]
        restored = deepcopy(after)
        if (revision.get('changed_criterion_ids') != [criterion] or len(old_entries) != 1 or len(new_entries) != 1
                or old_entries[0]['description'] == new_entries[0]['description']):
            raise ValueError('query option policy must change only its declared confirmation criterion')
        next(item for item in restored['required_meaning'] if item['id'] == criterion)['description'] = old_entries[0]['description']
        if scene == 'flexible_time_followup_matches_evening_game':
            if (before.get('expected_runtime_status') != 'waiting_user'
                    or 'expected_runtime_status' in after
                    or after.get('expected_runtime_status_any') != ['completed', 'waiting_user']):
                raise ValueError('query option policy status alternatives differ from authorized paths')
            restored.pop('expected_runtime_status_any')
            restored['expected_runtime_status'] = 'waiting_user'
        if canonical_digest(restored) != canonical_digest(before):
            raise ValueError('query option policy changed factual, permission or other semantic obligations')
        old_catalog, new_catalog = revision.get('previous_scenario_contract'), revision.get('current_scenario_contract')
        if not isinstance(old_catalog, dict) or not old_catalog or not isinstance(new_catalog, dict):
            raise ValueError('query option policy requires exact old/current reviewed catalog snapshots')
        expected = deepcopy(old_catalog)
        if scene == 'flexible_time_followup_matches_evening_game':
            if old_catalog.get('expected_status_any') != ['waiting_user']:
                raise ValueError('query option policy original waiting status drifted')
            expected['expected_status_any'] = ['completed', 'waiting_user']
        if canonical_digest(new_catalog) != canonical_digest(expected):
            raise ValueError('query option policy changed facts or a non-authorized catalog field')
        guard = revision.get('evaluation_guard')
        if not isinstance(guard, dict) or set(guard) != {'requires_query_reply_state_check'} or guard['requires_query_reply_state_check'] is not True:
            raise ValueError('query option policy requires the independent blind reply-state guard')
    return revisions


def apply_query_option_policy_to_scenarios(scenarios: list[Any], records: list[dict[str, Any]]) -> list[Any]:
    """应用前逐字段核对真实catalog，并显式开启只属于这两纯查询场景的盲状态验收。"""
    result = list(scenarios)
    for revision in _query_option_revisions(records):
        indexes = [index for index, scene in enumerate(result) if scene.scenario_id == revision['live_scenario_id']]
        if len(indexes) != 1:
            raise ValueError('query option policy live scenario is absent or duplicated')
        index = indexes[0]
        # 完整旧catalog来自代码审阅，不能以删掉某字段的快照绕过该字段的精确比对。
        contract_fields = {item.name for item in fields(result[index])} - {
            'scenario_id', 'name', 'message', 'setup', 'requires_query_reply_state_check'}
        if (set(revision['previous_scenario_contract']) != contract_fields
                or any(canonical_digest(getattr(result[index], field)) != canonical_digest(value)
                for field, value in revision['previous_scenario_contract'].items())
                or getattr(result[index], 'requires_query_reply_state_check') is not False):
            raise ValueError('query option policy original catalog or guard drifted')
        result[index] = replace(result[index], **deepcopy(revision['current_scenario_contract']), requires_query_reply_state_check=True)
    return result


def apply_query_option_policy_to_cases(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """语义采用同版最小修订，来源、原行和旧失败保持只读，不重复改已正确的其他场景。"""
    result = deepcopy(cases)
    for revision in _query_option_revisions(records):
        targets = [item for item in result if item['record_id'] == revision['source_record_id'] and item['case_id'] == revision['target_case_id']]
        if len(targets) != 1 or canonical_digest(targets[0]['expected']) != canonical_digest(revision['previous_expectation']):
            raise ValueError('query option policy golden link is absent, duplicated or changed')
        targets[0]['expected'] = deepcopy(revision['current_expectation'])
        targets[0]['product_contract_revision_ids'].append(revision['revision_id'])
    return result


def _reply_presentation_revisions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """校验用户明示的话术口径修订；旧资产只读，修订不能触及工具、人数或权限事实。

    这里限定为已知条件的复述和自然确认措辞，不能把产品口径修订当作任意放宽
    安全断言的通道。前缀摘要及原case摘要绑定当时依据，新增资产仍可继续追加。
    """
    # 非列表资产交给原验证器报告结构错误；索引不能先抛 TypeError，也不补造缺失来源。
    originals = {(record.get('id'), case.get('id')): case for record in records
                 for case in (record.get('eval_cases', []) if isinstance(record.get('eval_cases'), list) else [])
                 if isinstance(case, dict)}
    revisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_terms = {'0.5', '无烟', '打吗', '来吗', '来不'}
    echo_only_terms = {'accept_existing_offer_marks_game_ready': {'七点','无烟','0.5'},
                       'duration_limit_update_should_persist': {'5小时','五小时'},
                       'resume_status_after_casual_chat': {'0.5','无烟'}}
    raw = DATASET_PATH.read_bytes()
    for record in records:
        if record.get('kind') != REPLY_PRESENTATION_REVISION_KIND:
            continue
        source = record.get('source', {})
        prefix = record.get('original_dataset_prefix_bytes')
        if (not isinstance(source, dict)
                or type(record.get('schema_version')) is not int or record['schema_version'] != 1
                or source.get('kind') != 'current_thread_direct_user_instruction'
                or source.get('not_original_human_transcript') is not True
                or source.get('message_ids_unavailable') is not True
                or not isinstance(source.get('user_quotes'), list) or not source['user_quotes']
                or any(not isinstance(q, str) or not q.strip() for q in source['user_quotes'])
                or type(prefix) is not int or not 0 < prefix <= len(raw)
                or hashlib.sha256(raw[:prefix]).hexdigest() != record.get('original_dataset_prefix_sha256')):
            raise ValueError('reply presentation revision lacks unchanged source and direct user provenance')
        items = record.get('revisions')
        if not isinstance(items, list) or not items:
            raise ValueError('reply presentation revisions must be nonempty')
        for revision in items:
            version = revision.get('revision_id')
            original = originals.get((revision.get('source_record_id'), revision.get('target_case_id')))
            if (not isinstance(version, str) or not version or version in seen or original is None
                    or canonical_digest(original) != revision.get('source_case_sha256')
                    or revision.get('live_scenario_id') not in {'profile_default_matched_game','later_people_count_query',*echo_only_terms}):
                raise ValueError('reply presentation revision source or identity is invalid')
            before, after = revision.get('previous_expectation'), revision.get('current_expectation')
            if not isinstance(before, dict) or not isinstance(after, dict) or canonical_digest(before) != canonical_digest(original.get('expected')):
                raise ValueError('reply presentation previous expectation differs from original')
            expected = deepcopy(before)
            if revision['live_scenario_id']=='profile_default_matched_game':
                expected['forbidden_reply_contains'] = [term for term in before['forbidden_reply_contains']
                                                        if term not in {'0.5', '无烟'}]
            if revision['live_scenario_id'] in echo_only_terms and 'forbidden_reply_contains' in expected:
                expected['forbidden_reply_contains']=[term for term in expected['forbidden_reply_contains']
                                                    if term not in echo_only_terms[revision['live_scenario_id']]]
            if canonical_digest(after) != canonical_digest(expected):
                raise ValueError('reply presentation revision changed factual or safety expectations')
            old, new = revision.get('previous_scenario_contract'), revision.get('current_scenario_contract')
            if (not isinstance(old, dict) or not isinstance(new, dict)
                    or set(old) != {'required_reply_any', 'forbidden_reply_contains'} or set(new) != set(old)):
                raise ValueError('reply presentation catalog scope is invalid')
            expected_catalog = deepcopy(old)
            if revision['live_scenario_id']=='profile_default_matched_game':
                expected_catalog['forbidden_reply_contains'] = [term for term in old['forbidden_reply_contains']
                                                               if term not in allowed_terms]
                expected_catalog['required_reply_any'][2] += ['打吗', '来吗', '来不']
            elif revision['live_scenario_id'] in echo_only_terms:
                # 裸数值/烟况不是业务错误，拒绝换条件和错误承诺的语义及状态硬断言原样保留。
                expected_catalog['forbidden_reply_contains']=[term for term in old['forbidden_reply_contains']
                    if term not in echo_only_terms[revision['live_scenario_id']]]
            else:
                # 只问人数时保留准确人数要求；时间和参加征询不能成为强制追加目标。
                # 原golden语义、工具、人数后态和隐私禁项均不变，原失败不追改。
                if len(old['required_reply_any'])!=3 or old['required_reply_any'][0]!=['两个人','2个人','两个']:
                    raise ValueError('count-only presentation original anchors changed')
                expected_catalog['required_reply_any']=deepcopy(old['required_reply_any'][:1])
            if canonical_digest(new) != canonical_digest(expected_catalog):
                raise ValueError('reply presentation catalog changed non-presentation assertions')
            seen.add(version)
            revisions.append(revision)
    return revisions


def validate_reply_presentation_policy(records: list[dict[str, Any]]) -> None:
    """核验追加话术和纯查询口径的来源与最小变更范围，不执行模型或修改历史记录。"""
    _reply_presentation_revisions(records)
    _query_option_revisions(records)


def apply_reply_presentation_to_scenarios(scenarios: list[Any], records: list[dict[str, Any]]) -> list[Any]:
    """从保留的原catalog应用用户话术修订，版本失配就拒绝，不改领域硬断言。"""
    result = list(scenarios)
    for revision in _reply_presentation_revisions(records):
        indexes = [i for i, item in enumerate(result) if item.scenario_id == revision['live_scenario_id']]
        if len(indexes) != 1:
            raise ValueError('reply presentation live scenario is absent or duplicated')
        index = indexes[0]
        if any(getattr(result[index], field) != value for field, value in revision['previous_scenario_contract'].items()):
            raise ValueError('reply presentation original catalog drifted')
        result[index] = replace(result[index], **deepcopy(revision['current_scenario_contract']))
    return result


def apply_reply_presentation_to_cases(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """让独立语义与硬检查采用同版用户口径，原golden内容及原失败判分保持不动。"""
    result = deepcopy(cases)
    for revision in _reply_presentation_revisions(records):
        targets = [item for item in result if item['record_id'] == revision['source_record_id']
                   and item['case_id'] == revision['target_case_id']]
        if len(targets) != 1 or targets[0]['expected'] != revision['previous_expectation']:
            raise ValueError('reply presentation golden link is absent, duplicated or changed')
        targets[0]['expected'] = deepcopy(revision['current_expectation'])
        targets[0]['product_contract_revision_ids'].append(revision['revision_id'])
    return result


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
    """只准声明的语义描述及catalog字段变化，旧局部消歧默认仍不得更改catalog。"""
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
    old_catalog=revision.get('previous_scenario_contract');new_catalog=revision.get('current_scenario_contract')
    fields=revision.get('changed_catalog_fields',[])
    if (not isinstance(fields,list) or any(not isinstance(field,str) or not field for field in fields)
            or len(set(fields))!=len(fields)):
        raise ValueError('changed_catalog_fields must list unique catalog fields')
    restored_catalog=deepcopy(new_catalog)
    for field in fields:
        if (field not in old_catalog or field not in new_catalog
                or canonical_digest(old_catalog[field])==canonical_digest(new_catalog[field])):
            raise ValueError('changed_catalog_fields must identify changed existing catalog fields')
        restored_catalog[field]=deepcopy(old_catalog[field])
    if canonical_digest(old_catalog)!=canonical_digest(restored_catalog):
        raise ValueError('description-only revision must preserve undeclared catalog fields')


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
                    if target in catalog_chain:
                        prior=deepcopy(catalog_chain[target])
                        # 旧链未包含的字段只能从已绑定原case的硬快照精确补作基线，不能取当前输出。
                        snapshot=originals[target].get('existing_hard_assertions_snapshot',{})
                        for field in set(previous)-set(prior):
                            if field not in snapshot or canonical_digest(previous[field])!=canonical_digest(snapshot[field]):
                                raise ValueError('new catalog field differs from original hard assertion snapshot')
                            prior[field]=deepcopy(snapshot[field])
                        if canonical_digest(previous)!=canonical_digest(prior):
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
