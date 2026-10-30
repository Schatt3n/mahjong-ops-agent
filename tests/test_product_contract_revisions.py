"""产品口径修订必须可追溯地生效，不能覆盖原人工案例或悄悄留下旧自动判据。"""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from eval_product_contracts import canonical_digest, read_product_records, resolve_product_contract_revisions
from run_inquiry_grounding_eval import inquiry_input_catalog, inquiry_reply_expectation
from run_real_owner_chat_live_eval import golden_reply_cases, live_eval_scenarios, reply_semantic_expectation


def _record(records):
    """只选择本次真人产品口径，不把其他审查修订当作本次授权。"""
    return next(r for r in records if r.get('kind')=='owner_chat_product_contract_revision')


def _catalog():
    """从实际当前catalog读取会自动生成旧语义的两个字段，不执行夹具或模型。"""
    return {s.scenario_id:{'required_reply_any':s.required_reply_any,'forbidden_reply_contains':s.forbidden_reply_contains}
            for s in live_eval_scenarios()}


def _resolve(records):
    """校验实际固定输入、原案例和当前catalog三个来源，不能只让metadata自证。"""
    return resolve_product_contract_revisions(records,inquiry_inputs=inquiry_input_catalog(),scenario_contracts=_catalog())


def test_effective_product_revision_preserves_legacy_bytes_quotes_and_nonconflicting_contracts():
    import hashlib
    records=read_product_records();before=deepcopy(records);record=_record(records)
    raw=(ROOT/'eval/golden/real_owner_chat_golden.jsonl').read_bytes()
    assert hashlib.sha256(raw[:record['original_dataset_prefix_bytes']]).hexdigest()==record['original_dataset_prefix_sha256']
    source=record['source']
    assert source['source_thread_id']=='01a18cf2-3fff-7332-9b15-f60e3014d685'
    assert source['message_id'] is None and source['message_ids_unavailable'] is True
    assert source['not_original_human_transcript'] is True and len(source['user_quotes'])==3
    assert '如果没有冲突' in source['user_quotes'][2]
    effective=_resolve(records)
    assert records==before
    owner=record['case_expectation_revisions'][0]
    original=next(c for r in records for c in r.get('eval_cases',[]) if c['id']==owner['target_case_id'])
    assert original['expected']==owner['previous_expectation']
    assert original['expected']!=effective['owner'][original['id']]
    assert 'ask_open_game_question' in {item['id'] for item in original['expected']['required_meaning']}
    assert owner['previous_scenario_contract']['required_reply_any']
    assert not owner['current_scenario_contract']['required_reply_any']
    assert set(owner['previous_scenario_contract']['forbidden_reply_contains'])-set(owner['current_scenario_contract']['forbidden_reply_contains'])=={'杭麻还是川麻','川麻还是杭麻'}
    assert set(effective['revision_ids'])=={'owner:'+original['id'],'owner:synthetic_live_profile_range_empty_inquiry',
        'inquiry:unknown_empty','inquiry:unknown_pool','inquiry:partial_request'}
    assert len(record['inquiry_expectation_baselines'])==6
    for baseline in record['inquiry_expectation_baselines']:
        case_id=baseline['case_id'];old=baseline['expectation'];current=effective['inquiry'][case_id]
        if baseline['input_snapshot']['criteria_incomplete']:
            assert 'clarify_unknown_need' in {r['id'] for r in old['required_meaning']}
            assert 'reasonable_contextual_progress' in {r['id'] for r in current['required_meaning']}
        else:
            assert old==current
    untouched=next(s for s in live_eval_scenarios() if s.scenario_id=='single_profile_game_complete_query')
    assert untouched.require_help_offer is True and untouched.max_reply_chars==80


@pytest.mark.parametrize('field,value,error',[
    ('source_record_id','wrong','source record or live scenario mismatch'),
    ('target_case_id','unknown','unique prior case'),
    ('live_scenario_id','profile_range_empty_inquiry','source record or live scenario mismatch'),
    ('source_case_sha256','0'*64,'original case hash mismatch'),
    ('previous_expectation',{},'previous snapshot mismatch'),
    ('current_expectation',{},'required_meaning must be non-empty'),
    ('current_scenario_contract',{},'old and current catalog contracts'),
    ('source_catalog_sha256_before_revision',None,'original catalog source sha256'),
])
def test_product_revision_rejects_untraceable_or_discontinuous_owner_expectations(field,value,error):
    records=read_product_records();_record(records)['case_expectation_revisions'][0][field]=value
    with pytest.raises(ValueError,match=error):_resolve(records)


@pytest.mark.parametrize('mode', ['duplicate_version','future_source','original_tampered','catalog_leftover','inquiry_input_tampered','inquiry_previous_tampered','function_snapshot_tampered','missing_user_quotes','invented_message_id'])
def test_product_revision_rejects_source_or_execution_contract_drift(mode):
    records=read_product_records();record=_record(records)
    if mode=='duplicate_version':record['case_expectation_revisions'].append(deepcopy(record['case_expectation_revisions'][0]))
    elif mode=='future_source':records.remove(record);records.insert(0,record)
    elif mode=='original_tampered':
        target=record['case_expectation_revisions'][0]['target_case_id']
        next(c for r in records for c in r.get('eval_cases',[]) if c['id']==target)['expected']['reply_reference']='篡改原文'
    elif mode=='catalog_leftover':record['case_expectation_revisions'][0]['current_scenario_contract']['required_reply_any']=[['什么玩法']]
    elif mode=='inquiry_input_tampered':record['inquiry_expectation_baselines'][0]['input_snapshot']['text']='替换原输入'
    elif mode=='inquiry_previous_tampered':record['inquiry_expectation_revisions'][0]['previous_expectation']['required_meaning']=[]
    elif mode=='function_snapshot_tampered':record['inquiry_baseline_source']['function_snapshot']='伪造旧合同'
    elif mode=='missing_user_quotes':record['source']['user_quotes']=[]
    else:record['source']['message_id']='invented-message-id'
    with pytest.raises(ValueError):_resolve(records)


def test_owner_runner_and_inquiry_runner_use_effective_contract_without_old_generated_question():
    scene=next(s for s in live_eval_scenarios() if s.scenario_id=='unknown_game_requires_open_question')
    expectation=reply_semantic_expectation(scene)
    assert expectation['product_contract_revision_ids']==['user-flexible-owner-unknown-game-20261030']
    assert not any(item['id']=='scenario_reply_alternatives_0' or item['id'].endswith('.ask_open_game_question') for item in expectation['required_meaning'])
    effective=next(c for c in golden_reply_cases() if c['scenario_id']==scene.scenario_id)
    assert any(item['id']=='reasonable_unknown_game_progress' for item in effective['expected']['required_meaning'])
    for case_id in ('unknown_empty','unknown_pool','partial_request'):
        current=inquiry_reply_expectation(True,case_id=case_id)
        assert current['product_contract_revision_ids']==['user-flexible-inquiry-'+case_id+'-20261030']
        ids={item['id'] for field in ('required_meaning','forbidden_meaning') for item in current[field]}
        assert {'respect_known_conditions','reasonable_contextual_progress','unsupported_personal_fit_or_inventory','irrelevant_drift','unauthorized_business_side_effect'}<=ids
        assert not {'clarify_unknown_need','availability_before_need'}&ids


@pytest.mark.parametrize('version_count',[2,3])
def test_legal_continuous_owner_revisions_compare_only_final_catalog(version_count):
    """两版及三版连续链均合法，旧current不能被要求与最终可执行catalog相等。"""
    records=read_product_records();record=_record(records);first=record['case_expectation_revisions'][0]
    previous=first
    for index in range(2,version_count+1):
        revision=deepcopy(previous)
        revision['revision_id']='same-policy-version-'+str(index)
        revision['previous_expectation']=deepcopy(previous['current_expectation'])
        revision['current_expectation']=deepcopy(previous['current_expectation'])
        revision['current_expectation']['reply_reference']='想了解哪一种？' if index==2 else '想看看哪种玩法？'
        revision['previous_scenario_contract']=deepcopy(previous['current_scenario_contract'])
        revision['current_scenario_contract']=deepcopy(previous['current_scenario_contract'])
        revision['current_scenario_contract']['forbidden_reply_contains'].append('我是AI' if index==2 else '我是机器人')
        record['case_expectation_revisions'].append(revision);previous=revision
    catalog=_catalog();catalog[first['live_scenario_id']]=deepcopy(previous['current_scenario_contract'])
    result=resolve_product_contract_revisions(records,inquiry_inputs=inquiry_input_catalog(),scenario_contracts=catalog)
    assert result['owner'][first['target_case_id']]==previous['current_expectation']
    assert len(result['revision_ids']['owner:'+first['target_case_id']])==version_count
    with pytest.raises(ValueError,match='current catalog contract mismatch'):
        _resolve(records)


@pytest.mark.parametrize('mode',['known_description','unknown_description','first_catalog','first_source_hash','function_and_hash','frozen_reference'])
def test_first_product_baselines_are_bound_to_trusted_frozen_source(mode):
    import hashlib
    records=read_product_records();record=_record(records)
    if mode in {'known_description','unknown_description'}:
        case_id='explicit_no_profile' if mode=='known_description' else 'unknown_empty'
        baseline=next(row for row in record['inquiry_expectation_baselines'] if row['case_id']==case_id)
        baseline['expectation']['required_meaning'][0]['description']='任何回复都满足，无须保留已知条件'
    elif mode=='first_catalog':record['case_expectation_revisions'][0]['previous_scenario_contract']['required_reply_any']=[]
    elif mode=='first_source_hash':record['case_expectation_revisions'][0]['source_catalog_sha256_before_revision']='0'*64
    elif mode=='function_and_hash':
        record['inquiry_baseline_source']['function_snapshot']='raise AssertionError("must not execute archived source")\n'
        record['inquiry_baseline_source']['function_sha256']=hashlib.sha256(record['inquiry_baseline_source']['function_snapshot'].encode()).hexdigest()
    else:record['legacy_archive']['sha256']='0'*64
    with pytest.raises(ValueError,match='trusted frozen|frozen legacy'):_resolve(records)


def test_frozen_archive_rejects_modified_contract_even_with_valid_json(tmp_path):
    from eval_product_contracts import LEGACY_ARCHIVE_PATH,load_frozen_legacy_contracts
    archive=json.loads(LEGACY_ARCHIVE_PATH.read_text())
    archive['inquiry_cases']['explicit_no_profile']['expectation']['required_meaning'][0]['description']='任何回复都满足'
    modified=tmp_path/'modified_archive.json';modified.write_text(json.dumps(archive,ensure_ascii=False))
    with pytest.raises(ValueError,match='archive hash mismatch'):load_frozen_legacy_contracts(modified)


def _profile_record(records):
    """选择后追加的profile局部消歧，不改上一条产品修订的字节或范围。"""
    return next(record for record in records if record['id']=='owner_chat_profile_empty_response_revision_20261031_001')


def test_profile_revision_changes_only_conflicting_description_and_preserves_frozen_sources():
    """新增effective描述保留所有其他原文、catalog、已知条件及执行硬验收。"""
    import hashlib
    from eval_product_contracts import LEGACY_ARCHIVE_PATH, LEGACY_ARCHIVE_SHA256, load_frozen_owner_supplement
    records=read_product_records();record=_profile_record(records);revision=record['case_expectation_revisions'][0]
    raw=(ROOT/'eval/golden/real_owner_chat_golden.jsonl').read_bytes()
    assert len(records)==11 and record is records[-1]
    assert raw[:record['original_dataset_prefix_bytes']]==b''.join(raw.splitlines(keepends=True)[:10])
    assert hashlib.sha256(raw[:record['original_dataset_prefix_bytes']]).hexdigest()==record['original_dataset_prefix_sha256']=='693f8dcd0e64132ed9d1140fa287e3ee1ce47c4a4de6064382dd0a022096a6f4'
    assert hashlib.sha256(LEGACY_ARCHIVE_PATH.read_bytes()).hexdigest()==LEGACY_ARCHIVE_SHA256
    assert record['source']['client_date']=='2026-10-31'
    assert record['source']['message_id'] is None and record['source']['message_ids_unavailable'] is True
    assert record['source']['user_quotes'][-1]=='当然，这也是仅限于有冲突的。如果没有冲突，那可以按时不用修改'
    frozen=load_frozen_owner_supplement()['owner_cases'][revision['target_case_id']]
    original=next(case for row in records for case in row.get('eval_cases',[]) if case['id']==revision['target_case_id'])
    assert original==frozen['case_snapshot'] and original['expected']==revision['previous_expectation']==frozen['expectation']
    assert revision['source_case_sha256']==canonical_digest(original)
    assert revision['previous_scenario_contract']==revision['current_scenario_contract']==frozen['scenario_contract']
    assert revision['changed_criterion_ids']==['answer_current_inquiry']
    restored=deepcopy(revision['current_expectation'])
    changed=next(item for item in restored['required_meaning'] if item['id']=='answer_current_inquiry')
    assert '不强制显式复述空结果' in changed['description'] and '省略会误导' in changed['description']
    changed['description']=next(item['description'] for item in original['expected']['required_meaning'] if item['id']==changed['id'])
    assert restored==original['expected']
    effective=_resolve(records)
    assert effective['owner'][original['id']]==revision['current_expectation']
    assert effective['revision_ids']['owner:'+original['id']]==[revision['revision_id']]
    scene=next(scene for scene in live_eval_scenarios() if scene.scenario_id==revision['live_scenario_id'])
    assert scene.require_help_offer is True and scene.max_reply_chars==80
    assert scene.required_tool_names==original['existing_hard_assertions_snapshot']['required_tool_names']
    assert scene.forbidden_tool_names==original['existing_hard_assertions_snapshot']['forbidden_tool_names']
    assert scene.expected_tool_result_paths==original['existing_hard_assertions_snapshot']['expected_tool_result_paths']
    assert scene.expected_active_game_count==original['existing_hard_assertions_snapshot']['expected_active_game_count']==0
    projected=reply_semantic_expectation(scene)
    criterion=next(item for item in projected['required_meaning'] if item['id']==original['id']+'.answer_current_inquiry')
    assert criterion['description']==revision['current_expectation']['required_meaning'][0]['description']
    assert projected['product_contract_revision_ids']==[revision['revision_id']]
    assert 'offer_finding_players' in {item['id'] for item in projected['required_meaning']}


@pytest.mark.parametrize('mode', ['archive_missing','archive_hash','archive_path','archive_commit','source_hash','source_case',
    'old_catalog','current_catalog','previous_description','other_description','reference','drop_criterion','change_id',
    'scope_missing','scope_empty','scope_duplicate','scope_unknown','scope_unchanged','future_source'])
def test_profile_revision_rejects_untrusted_baseline_or_changes_outside_declared_description(mode):
    records=read_product_records();record=_profile_record(records);revision=record['case_expectation_revisions'][0]
    if mode=='archive_missing':record.pop('owner_baseline_archive')
    elif mode=='archive_hash':record['owner_baseline_archive']['sha256']='0'*64
    elif mode=='archive_path':record['owner_baseline_archive']['path']='other.json'
    elif mode=='archive_commit':record['owner_baseline_archive']['source_commit']='0'*40
    elif mode=='source_hash':revision['source_catalog_sha256_before_revision']='0'*64
    elif mode=='source_case':revision['source_case_sha256']='0'*64
    elif mode=='old_catalog':revision['previous_scenario_contract']['forbidden_reply_contains']=[]
    elif mode=='current_catalog':revision['current_scenario_contract']['forbidden_reply_contains']=[]
    elif mode=='previous_description':revision['previous_expectation']['required_meaning'][0]['description']='伪造旧描述'
    elif mode=='other_description':revision['current_expectation']['required_meaning'][1]['description']='不必复用可靠画像'
    elif mode=='reference':revision['current_expectation']['reply_reference']='替换原参考句'
    elif mode=='drop_criterion':revision['current_expectation']['forbidden_meaning'].pop()
    elif mode=='change_id':revision['current_expectation']['required_meaning'][0]['id']='changed-id'
    elif mode=='scope_missing':revision.pop('changed_criterion_ids')
    elif mode=='scope_empty':revision['changed_criterion_ids']=[]
    elif mode=='scope_duplicate':revision['changed_criterion_ids']*=2
    elif mode=='scope_unknown':revision['changed_criterion_ids']=['nonexistent-id']
    elif mode=='scope_unchanged':revision['changed_criterion_ids'].append('accept_range_as_known')
    else:records.remove(record);records.insert(0,record)
    with pytest.raises(ValueError):_resolve(records)


def test_profile_supplement_is_fixed_json_and_rejects_modified_valid_archive(tmp_path):
    from eval_product_contracts import OWNER_SUPPLEMENT_PATH, load_frozen_owner_supplement
    archive=json.loads(OWNER_SUPPLEMENT_PATH.read_text())
    archive['owner_cases']['synthetic_live_profile_range_empty_inquiry']['expectation']['required_meaning'][1]['description']='篡改已知条件'
    path=tmp_path/'modified-owner-supplement.json';path.write_text(json.dumps(archive,ensure_ascii=False))
    with pytest.raises(ValueError,match='supplement archive hash mismatch'):load_frozen_owner_supplement(path)
