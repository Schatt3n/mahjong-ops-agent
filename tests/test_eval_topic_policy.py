"""追加政策独立核验最新视图，旧话术/查询修订继续验证其历史阶段而不改原资产。"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import eval_topic_policy as policy
import run_real_owner_chat_live_eval as owner


def test_latest_policy_preserves_every_business_hard_assertion_and_original_asset():
    declaration=policy.topic_policy(); old=owner.live_eval_scenarios(apply_current_topic_policy=False)
    new=owner.live_eval_scenarios(); assert len(old)==len(new)
    for previous,current in zip(old,new):
        before,after=asdict(previous),asdict(current)
        before['message'].pop('sent_at');after['message'].pop('sent_at')
        assert previous.scenario_id==current.scenario_id
        assert not set(declaration['removed_generic_terms']) & set(current.forbidden_reply_contains)
        for key in ('forbidden_reply_contains','required_reply_any'):before.pop(key);after.pop(key)
        if previous.scenario_id in declaration['casual_scenario_ids']:
            assert before.pop('max_reply_chars') == 80
            assert after.pop('max_reply_chars') == 200
        assert before==after
        if previous.scenario_id not in declaration['casual_scenario_ids']:
            assert previous.required_reply_any==current.required_reply_any
        else: assert current.required_reply_any==[]
    assert policy.POLICY_SHA256==policy.hashlib.sha256(policy.POLICY_PATH.read_bytes()).hexdigest()


def test_topic_semantic_revision_keeps_business_state_privacy_and_original_source_read_only():
    originals=owner.golden_reply_cases(apply_current_topic_policy=False); saved=deepcopy(originals)
    revised=policy.apply_topic_policy_to_cases(originals); assert originals==saved
    for before,after in zip(originals,revised):
        assert before['record_id']==after['record_id'] and before['case_id']==after['case_id']
        for name in ('must_preserve_slots','should_not_mutate_active_game','expected_active_game_requirement'):
            assert before['expected'].get(name)==after['expected'].get(name)
    expected=owner.reply_semantic_expectation(next(s for s in owner.live_eval_scenarios() if s.scenario_id=='casual_chat_should_not_pollute_business_state'))
    assert any(item['id']=='privacy' for item in expected['forbidden_meaning'])
    assert any(item['id'].endswith('unsolicited_resume_or_evasion') for item in expected['forbidden_meaning'])


def test_changed_policy_hash_is_rejected(monkeypatch,tmp_path):
    changed=tmp_path/'policy.json';changed.write_bytes(policy.POLICY_PATH.read_bytes()+b' ')
    monkeypatch.setattr(policy,'POLICY_PATH',changed)
    with pytest.raises(ValueError,match='changed without review'):policy.topic_policy()
