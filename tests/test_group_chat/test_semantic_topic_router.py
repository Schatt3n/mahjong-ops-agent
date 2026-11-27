"""语义路由合同回归：测试边界与来源，不用关键词替身证明模型理解能力。"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent_runtime.group_chat.models import GroupMessage
from mahjong_agent_runtime.group_chat.semantic_topic_router import (
    SemanticTopicClassifier, SemanticTopicContractError, SemanticTopicInputError,
)


class RecordingLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({'messages': messages, **kwargs})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


def message(text, *, message_id='m1', quote=None):
    return GroupMessage(room_id='room', conversation_id='group-room', sender_external_id='external-a',
                        sender_name='阿甲', message_id=message_id, text=text, quoted_message_id=quote,
                        sent_at=datetime(2026, 11, 5, 11, 20, tzinfo=ZoneInfo('Asia/Shanghai')))


def decision(intent='query', **changes):
    return {'intent': intent, 'topic_action': 'new', 'topic_id': None, 'goal_relation': 'new',
            'task_context_id': None, 'public_summary': {'game_type': 'hangzhou_mahjong'},
            'constraints': {}, 'source_message_id': 'm1', 'confidence': 0.9,
            'reasoning': '按当前用户表达区分查询和授权。', **changes}


def card(topic_id='topic-hang', **changes):
    return {'topic_id': topic_id, 'room_id': 'room', 'version': 1,
            'public_summary': {'game_type': 'hangzhou_mahjong'}, **changes}


@pytest.mark.parametrize(('text', 'intent'), [
    ('明晚七点半杭麻0.5我这边两个人，先看看有没有能一起坐的。', 'query'),
    ('老板，帮我找明天下午两点国标1块无烟的，我一个人。', 'delegate'),
    ('老板，帮我找明晚九点半川麻1块无烟的，我一个人。', 'delegate'),
])
def test_language_inputs_are_sent_whole_to_semantic_model(text, intent):
    llm = RecordingLLM(decision(intent))
    result = SemanticTopicClassifier(llm).classify(message=message(text), candidates=[], trace_id='trace')
    assert result['intent'] == intent
    payload = json.loads(llm.calls[0]['messages'][1]['content'])
    assert payload['message']['text'] == text
    assert payload['message']['sent_at'] == '2026-11-05T11:20:00+08:00'
    assert 'channel_action' not in result and 'response' not in result
    assert result['audit']['model_calls'] == 1


def test_optional_null_date_times_are_omitted_and_keep_actual_constraints_and_raw_audit():
    """空时点不增加个人约束，保留有效绝对时间和原模型输出。"""
    planned = '2026-11-05T19:30:00+08:00'
    raw = decision(topic_action='none', goal_relation='none', public_summary={'time_window': {'start': '19:30', 'end': None}},
                   constraints={'planned_start_at': planned, 'latest_arrival_at': None, 'latest_wait_at': None, 'end_at': None})
    result = SemanticTopicClassifier(RecordingLLM(raw)).classify(
        message=message('七点半那桌还有位吗？我得晚点到。'), candidates=[], trace_id='optional-clock')
    assert result['constraints'] == {'planned_start_at': planned}
    assert result['public_summary'] == raw['public_summary']
    assert result['goal_relation'] == result['topic_action'] == 'none'
    assert result['audit']['model_calls'] == 1 and result['audit']['contract_recovery_count'] == 0
    assert result['audit']['attempts'][0]['output'] == json.dumps(raw, ensure_ascii=False)


@pytest.mark.parametrize("constraints,error", [
    ({'latest_arrival_at': '2026-11-05T20:00:00'}, 'constraint_datetime_requires_timezone'),
    ({'latest_arrival_at': False}, 'constraint_datetime_requires_timezone'),
    ({'party_size': None}, 'invalid_party_size'),
    ({'latest_arrival_at': None, 'extra_clock': None}, 'constraint_fields_mismatch'),
    ({'source_message_id': None}, 'constraint_source_message_mismatch'),
])
def test_other_invalid_constraints_are_not_masked_by_optional_null_date_time_handling(constraints, error):
    """非空时间仍须带时区，来源、人数与未知字段仍按原合同校验。"""
    raw = decision(topic_action='none', goal_relation='none', constraints=constraints)
    with pytest.raises(SemanticTopicContractError) as failed:
        SemanticTopicClassifier(RecordingLLM(raw), max_contract_retries=0).classify(
            message=message('七点半那桌还有位吗？我得晚点到。'), candidates=[], trace_id='invalid-clock')
    assert failed.value.audit['attempts'][0]['error'] == error


def test_bare_seat_code_prompt_distinguishes_table_size_from_stake():
    output = decision('self_recruitment', goal_relation='none', public_summary={},
                      constraints={'party_size': 1, 'start_when_full': True})
    llm = RecordingLLM(output)
    result = SemanticTopicClassifier(llm).classify(
        message=message('173，有烟，人齐开，四小时'), candidates=[], trace_id='bare-seat-code')
    prompt = llm.calls[0]['messages'][0]['content']
    assert all(rule in prompt for rule in ('173 = 1 缺 3', '272 = 2 缺 2', '371 = 3 缺 1'))
    assert '只有原文另有明确档位' in prompt
    assert result['constraints']['party_size'] == 1
    assert 'stakes' not in result['public_summary'] and 'stakes' not in result['constraints']


def test_same_user_update_uses_present_owned_task_not_other_customers():
    llm = RecordingLLM(decision('update', topic_action='existing', topic_id='topic-hang',
                                goal_relation='continue', task_context_id='task-a'))
    result = SemanticTopicClassifier(llm).classify(
        message=message('我委托的杭麻时间改成明晚八点吧，还是0.5无烟，我一个人。'),
        candidates=[card(own_task_context_id='task-a')],
        own_context={'customer_id':'customer-a','tasks':[{'customer_id':'customer-a',
            'task_context_id':'task-a','topic_id':'topic-hang','goal_summary':'明晚杭麻0.5无烟一人'}]}, trace_id='trace')
    assert result['task_context_id'] == 'task-a'
    assert result['goal_relation'] == 'continue'


@pytest.mark.parametrize('intent,text', [
    ('claim', '1来，19点杭麻1块底，给我留个'),
    ('query', '这张里一号还有位么？我不抽烟'),
])
def test_board_item_is_separate_from_topic_when_interpreting_natural_text(intent, text):
    reference = {'snapshot_id': 'snapshot-old', 'display_no': 1}
    bad = decision(intent, topic_action='existing', topic_id='snapshot-old')
    ready = decision(intent, board_reference=reference)
    llm = RecordingLLM(bad, ready)
    result = SemanticTopicClassifier(llm).classify(
        message=message(text, quote='board-old'), candidates=[],
        board=[{**reference, 'game_id': 'game-a', 'reference_status': 'quoted'}], trace_id='board-reference')
    assert result['intent'] == intent and result['topic_id'] is None
    assert result['board_reference'] == reference
    assert result['audit']['attempts'][0]['error'] == 'topic_reference_not_in_candidates'
    assert result['audit']['contract_recovery_count'] == 1
    correction = llm.calls[1]['messages'][-1]['content']
    assert 'topic_id只取candidates' in correction and '属于board_reference' in correction


@pytest.mark.parametrize(('topic_action', 'error', 'hint'), [
    ('existing', 'topic_reference_not_in_candidates', '群友通报须省略该字段'),
    ('none', 'board_reference_requires_customer_board_request', '保留member_discussion并省略board_reference'),
])
def test_already_joined_group_announcement_omits_board_reference_without_becoming_claim(topic_action, error, hint):
    """重放群友已报上的通报，纠正缺失候选与不适用的看板引用仍保留群友交流。"""
    reference = {'snapshot_id': 'snapshot-latest', 'display_no': 1}
    summary = {'game_type': 'hangzhou_mahjong', 'stakes': ['1'], 'smoking': 'no_smoking',
               'time_window': {'start': '19:00', 'end': None}}
    announcement = decision('member_discussion', topic_action='none', goal_relation='none', public_summary=summary)
    llm = RecordingLLM({**announcement, 'topic_action': topic_action, 'board_reference': reference}, announcement)

    result = SemanticTopicClassifier(llm).classify(
        message=message('晚上7点那桌杭麻，1块底无烟，我位子定好了，同场的到时见。'), candidates=[],
        board=[{**reference, **summary, 'game_id': 'game-a', 'reference_status': 'latest'}], trace_id='share-joined')

    assert result['intent'] == 'member_discussion' and result['topic_action'] == result['goal_relation'] == 'none'
    assert result['topic_id'] is None and result['task_context_id'] is None and result['constraints'] == {}
    assert 'board_reference' not in result
    assert result['audit']['attempts'][0]['error'] == error and result['audit']['contract_recovery_count'] == 1
    assert hint in llm.calls[1]['messages'][-1]['content']
    assert '引用看板本身不会使通报变成报名' in llm.calls[0]['messages'][0]['content']


def test_game_family_is_not_accepted_as_ruleset_in_board_constraint():
    """玩法大类冒用细规则时给出具体合同错误，修正回合仍需引用现有看板。"""
    reference = {'snapshot_id': 'snapshot-latest', 'display_no': 1}
    corrected = decision('claim', board_reference=reference, constraints={'game_type': 'hangzhou_mahjong'})
    first = {**corrected, 'constraints': {'ruleset': 'hangzhou_mahjong'}}
    llm = RecordingLLM(first, corrected)

    result = SemanticTopicClassifier(llm).classify(
        message=message('晚上7点的杭麻1号桌我参加，给我留一个。'), candidates=[],
        board=[{**reference, 'game_id': 'game-a', 'reference_status': 'latest'}], trace_id='claim-ruleset')

    error = 'constraints.ruleset_invalid:use_explicit_canonical_ruleset_or_omit'
    assert result['audit']['attempts'][0]['error'] == error and error in llm.calls[1]['messages'][-1]['content']
    assert result['intent'] == 'claim' and result['board_reference'] == reference
    assert result['constraints'] == corrected['constraints']


@pytest.mark.parametrize('reference', [
    {'snapshot_id': 'snapshot-new', 'display_no': 1},
    {'snapshot_id': 'snapshot-old', 'display_no': 2},
])
def test_board_reference_requires_same_projected_snapshot_and_item(reference):
    llm = RecordingLLM(decision('claim', board_reference=reference))
    with pytest.raises(SemanticTopicContractError) as error:
        SemanticTopicClassifier(llm, max_contract_retries=0).classify(
            message=message('一号我去', quote='board-old'), candidates=[],
            board=[{'snapshot_id': 'snapshot-old', 'display_no': 1}], trace_id='board-reference')
    assert error.value.audit['attempts'][0]['error'] == 'board_reference_not_in_public_board'


@pytest.mark.parametrize('changes', [
    {'goal_relation': 'new'},
    {'constraints': {'party_size': 1}},
])
def test_member_discussion_cannot_bind_personal_goal_or_conditions(changes):
    output = decision('member_discussion', goal_relation='none')
    output.update(changes)
    llm = RecordingLLM(output)
    with pytest.raises(SemanticTopicContractError) as error:
        SemanticTopicClassifier(llm, max_contract_retries=0).classify(
            message=message('我也想打，你们打什么？'), candidates=[], trace_id='trace')
    assert error.value.audit['attempts'][0]['error'] == 'member_discussion_cannot_set_personal_goal'


@pytest.mark.parametrize('topic_action', ['new', 'existing'])
@pytest.mark.parametrize('preferred_gender', ['男', '女'])
def test_self_recruitment_retains_private_constraints_without_operator_goal(topic_action, preferred_gender):
    """自招可登记或续接公开话题；本人匹配条件留在来源收据，不升级店方任务或公开性别。"""
    summary = {'game_type': 'hangzhou_mahjong', 'ruleset': '财敲', 'smoking': 'no_smoking'}
    constraints = {**summary, 'party_size': 3, 'preferred_gender': preferred_gender, 'source_message_id': 'm1'}
    output = decision('self_recruitment', topic_action=topic_action,
                      topic_id='topic-hang' if topic_action == 'existing' else None,
                      goal_relation='none', public_summary=summary, constraints=constraints)
    llm = RecordingLLM(output)
    result = SemanticTopicClassifier(llm, max_contract_retries=0).classify(
        message=message('六点371，财敲，无烟，再来一个女生' if preferred_gender == '女' else '六点371，财敲，无烟，再来一个男生'),
        candidates=[card()] if topic_action == 'existing' else [], trace_id='self-recruitment')
    assert result['intent'] == 'self_recruitment' and result['topic_action'] == topic_action
    assert result['goal_relation'] == 'none' and result['task_context_id'] is None
    assert result['constraints'] == constraints
    assert result['public_summary'] == summary and 'preferred_gender' not in result['public_summary']
    assert result['audit']['model_calls'] == 1 and result['audit']['contract_recovery_count'] == 0


@pytest.mark.parametrize('changes', [
    {'goal_relation': 'new'},
    {'goal_relation': 'continue', 'topic_action': 'existing', 'topic_id': 'topic-hang', 'task_context_id': 'task-a'},
    {'topic_action': 'none'},
    {'topic_action': 'uncertain'},
    {'public_summary': {'game_type': 'hangzhou_mahjong', 'preferred_gender': '女'}},
    {'constraints': {'party_size': 3, 'preferred_gender': '女生'}},
    {'constraints': {'party_size': 3, 'preferred_gender': '未知'}},
    {'constraints': {'party_size': 3, 'preferred_gender': ['女']}},
    {'public_summary': {'game_type': 'hangzhou_mahjong', 'required_gender': '女'}},
    {'constraints': {'party_size': 3, 'preferred_gender': '女', 'required_gender': '男'}},
])
def test_self_recruitment_rejects_operator_goal_and_invalid_or_public_gender(changes):
    """即使提供真实本人任务引用，自招也不能冒充接手授权；性别仅接受私有约束枚举。"""
    output = decision('self_recruitment', goal_relation='none',
                      constraints={'party_size': 3, 'preferred_gender': '女'})
    output.update(changes)
    llm = RecordingLLM(output)
    with pytest.raises(SemanticTopicContractError) as error:
        SemanticTopicClassifier(llm, max_contract_retries=0).classify(
            message=message('六点371，财敲，无烟，再来一个女生'),
            candidates=[card(own_task_context_id='task-a')],
            own_context={'customer_id': 'customer-a', 'tasks': [{
                'customer_id': 'customer-a', 'task_context_id': 'task-a', 'topic_id': 'topic-hang',
            }]}, trace_id='self-recruitment-invalid')
    assert len(llm.calls) == 1
    assert error.value.audit['attempts'][0]['output'] == json.dumps(output, ensure_ascii=False)


def test_self_recruitment_preserves_explicit_gender_restriction_in_private_constraints():
    output = decision('self_recruitment', goal_relation='none',
                      constraints={'game_type': 'hangzhou_mahjong', 'party_size': 3, 'required_gender': '女'})
    result = SemanticTopicClassifier(RecordingLLM(output)).classify(
        message=message('七点371，财敲，无烟，最后一位只接受女生'), candidates=[], trace_id='hard-preference')
    assert result['constraints']['required_gender'] == '女'
    assert 'required_gender' not in result['public_summary']


def test_self_recruitment_corrects_all_shared_field_disagreements_in_one_retry():
    summary = {'game_type': 'hangzhou_mahjong', 'ruleset': '财敲', 'stakes': ['0.5', '1'], 'smoking': 'no_smoking'}
    incomplete = decision('self_recruitment', goal_relation='none', public_summary=summary,
                          constraints={'party_size': 3, 'smoking': 'smoking'})
    corrected = {**incomplete, 'constraints': {**summary, 'stakes': ['1.0', '0.50'], 'party_size': 3}}
    llm = RecordingLLM(incomplete, corrected)

    result = SemanticTopicClassifier(llm).classify(
        message=message('财敲371，无烟，半块或一块都行'), candidates=[], trace_id='shared-fields')

    error = 'self_recruitment_constraint_mismatch:missing=game_type,ruleset,stakes;conflicting=smoking'
    assert result['audit']['attempts'][0]['error'] == error
    assert error in llm.calls[1]['messages'][-1]['content']
    assert result['audit']['contract_recovery_count'] == 1
    assert result['constraints'] == corrected['constraints']


def test_game_type_contract_identifies_noncanonical_label_in_self_recruitment():
    label = {'game_type': '杭麻', 'ruleset': '财敲'}
    canonical = {**label, 'game_type': 'hangzhou_mahjong'}
    first = decision('self_recruitment', goal_relation='none', public_summary=label,
                     constraints={**label, 'party_size': 3})
    second = {**first, 'public_summary': canonical, 'constraints': {**canonical, 'party_size': 3}}
    llm = RecordingLLM(first, second)

    result = SemanticTopicClassifier(llm).classify(
        message=message('财敲371'), candidates=[], trace_id='canonical-type')

    error = 'public_summary.game_type_invalid:use_canonical_requested_game_in_summary_and_constraints'
    assert result['audit']['attempts'][0]['error'] == error
    assert error in llm.calls[1]['messages'][-1]['content']
    terminology = json.loads(llm.calls[0]['messages'][1]['content'])['domain_terminology']
    term = next(row for row in terminology if row['term_id'] == 'game_variant_caiqiao')
    assert term['canonical']['requested_game'] == 'hangzhou_mahjong'
    assert 'game_type' not in term['canonical']


def test_distinct_game_types_are_preserved_as_separate_candidates():
    llm = RecordingLLM(decision('delegate', topic_action='existing', topic_id='sichuan',
                                public_summary={'game_type':'sichuan_mahjong'}))
    candidates = [card('guobiao', public_summary={'game_type':'national_standard_mahjong','stakes':['1']}),
                  card('sichuan', public_summary={'game_type':'sichuan_mahjong','stakes':['1']})]
    result = SemanticTopicClassifier(llm).classify(message=message('明晚川麻1块帮我找'), candidates=candidates, trace_id='t')
    assert result['topic_id'] == 'sichuan'
    payload = json.loads(llm.calls[0]['messages'][1]['content'])
    assert len(payload['candidates']) == 2
    assert payload['candidates'][0]['public_summary']['game_type'] != payload['candidates'][1]['public_summary']['game_type']


def test_unknown_topic_reference_is_rejected_and_corrected_without_silent_ignore():
    llm = RecordingLLM(decision(topic_action='existing', topic_id='invented'), decision())
    result = SemanticTopicClassifier(llm).classify(message=message('有局吗'), candidates=[card()], trace_id='t')
    assert result['topic_action'] == 'new'
    assert result['audit']['contract_recovery_count'] == 1
    assert len(llm.calls) == 2
    assert 'invented' not in json.dumps(llm.calls[1]['messages'],ensure_ascii=False)


def test_exhausted_contract_failure_has_evidence_and_is_not_chitchat():
    llm = RecordingLLM('{}', '{}')
    with pytest.raises(SemanticTopicContractError) as error:
        SemanticTopicClassifier(llm).classify(message=message('帮我找'), candidates=[], trace_id='t')
    assert error.value.attempts == 2
    assert error.value.audit['model_calls'] == 2
    assert len(error.value.audit['attempts']) == 2


def test_provider_failure_is_not_converted_to_ignore():
    llm = RecordingLLM(TimeoutError('provider timed out'))
    with pytest.raises(TimeoutError):
        SemanticTopicClassifier(llm).classify(message=message('帮我找'), candidates=[], trace_id='t')


def test_projection_drops_shared_history_notes_and_other_customer_data():
    llm = RecordingLLM(decision())
    candidates = [card(messages=[{'text':'OTHER_PRIVATE_BODY'}],participants=['SECRET_NAME'],
                       summary='FREE_TEXT_SECRET', private_remark='NOTE_SECRET',
                       public_summary={'game_type':'hangzhou_mahjong','participants':['ANOTHER_SECRET']})]
    own = {'customer_id':'customer-a','notes':'OWN_INTERNAL_NOTE',
           'tasks':[{'customer_id':'customer-other','task_context_id':'other-task','goal_summary':'OTHER_TASK_SECRET'},
                    {'customer_id':'customer-a','task_context_id':'task-a','topic_id':'topic-hang','goal_summary':'本人组局目标'}],
           'recent_messages':[{'customer_id':'customer-other','text':'OTHER_RAW'},
                              {'customer_id':'customer-a','message_id':'past-a','text':'本人已确认的条件'}]}
    SemanticTopicClassifier(llm).classify(message=message('有局吗'), candidates=candidates, own_context=own,
        board=[{'game_id':'game-a','game_type':'hangzhou_mahjong','participants':['BOARD_SECRET']}], trace_id='t')
    sent = json.dumps(llm.calls[0]['messages'],ensure_ascii=False)
    for secret in ['OTHER_PRIVATE_BODY','SECRET_NAME','FREE_TEXT_SECRET','NOTE_SECRET','ANOTHER_SECRET',
                   'OWN_INTERNAL_NOTE','OTHER_TASK_SECRET','OTHER_RAW','BOARD_SECRET']:
        assert secret not in sent
    assert '本人组局目标' in sent and '本人已确认的条件' in sent


def test_current_user_existing_target_and_quote_rank_before_recent_strangers():
    llm = RecordingLLM(decision())
    candidates = [card(f'other-{i}') for i in range(7)]
    candidates += [card('mine', own_task_context_id='mine-task'), card('quoted', quoted_reference=True)]
    SemanticTopicClassifier(llm, max_candidates=2).classify(message=message('改八点',quote='old-message'),
        candidates=candidates, own_context={'customer_id':'customer-a','tasks':[
            {'customer_id':'customer-a','task_context_id':'mine-task','topic_id':'mine'}]}, trace_id='t')
    payload = json.loads(llm.calls[0]['messages'][1]['content'])
    assert {x['topic_id'] for x in payload['candidates']} == {'mine','quoted'}
    assert payload['context_limits']['candidate_count_omitted'] == 7


def test_hard_budget_and_raw_message_size_do_not_silently_truncate():
    llm = RecordingLLM(decision())
    with pytest.raises(SemanticTopicInputError):
        SemanticTopicClassifier(llm, max_input_tokens=100).classify(message=message('有局吗'),candidates=[],trace_id='t')
    with pytest.raises(SemanticTopicInputError):
        SemanticTopicClassifier(llm).classify(message=message('字'*4001),candidates=[],trace_id='t')
    assert llm.calls == []


def test_oversized_optional_history_is_omitted_with_audit_under_budget():
    llm = RecordingLLM(decision())
    result = SemanticTopicClassifier(llm, max_input_tokens=2500).classify(message=message('有局吗'), candidates=[],
        own_context={'customer_id':'customer-a','recent_messages':[
            {'customer_id':'customer-a','message_id':str(i),'text':'旧'*1000} for i in range(30)]},trace_id='t')
    assert result['audit']['estimated_input_tokens'] <= 2500
    assert result['audit']['omitted_own_messages'] > 0


@pytest.mark.parametrize('change', [
    {'channel_action':'private_switch'}, {'response':'直接发给用户'},
    {'task_context_id':'someone-elses-task','goal_relation':'continue'},
    {'source_message_id':'another-message'},
    {'public_summary':{'game_type':'张三今天请病假'}},
    {'public_summary':{'participants':['张三']}},
    {'constraints':{'party_size':True}},
])
def test_model_cannot_expand_output_contract_or_reference_authority(change):
    llm = RecordingLLM(decision(**change))
    with pytest.raises(SemanticTopicContractError):
        SemanticTopicClassifier(llm,max_contract_retries=0).classify(message=message('看看'),candidates=[],trace_id='t')


def test_new_goal_does_not_inherit_previous_task_even_for_same_person():
    llm = RecordingLLM(decision('delegate', public_summary={'game_type':'national_standard_mahjong'}))
    result = SemanticTopicClassifier(llm).classify(message=message('另外帮我找明天下午的国标'),
        candidates=[card(own_task_context_id='old-hang-task')], trace_id='t')
    assert result['goal_relation'] == 'new' and result['task_context_id'] is None


def test_oversized_goal_is_omitted_whole_with_explicit_evidence():
    llm = RecordingLLM(decision())
    result = SemanticTopicClassifier(llm).classify(message=message('看看'),candidates=[],
        own_context={'customer_id':'a','tasks':[{'customer_id':'a','task_context_id':f't{i}',
            'topic_id':f'p{i}','goal_summary':'条件'*300+'不要参加'} for i in range(10)]},trace_id='t')
    assert result['audit']['omitted_own_summaries'] == 10
    assert result['audit']['omitted_own_tasks'] == 2
    payload = result['audit']['input']
    assert all('goal_summary' not in row and row['goal_summary_omitted'] for row in payload['own_context']['tasks'])


def test_public_board_values_cannot_smuggle_names_or_arbitrary_text():
    llm = RecordingLLM(decision())
    result = SemanticTopicClassifier(llm).classify(message=message('看看'),candidates=[],trace_id='t',
        board=[{'id':'game-1','game_type':'杭麻','stakes':'0.5','smoking':'无烟','time':'19:30',
                'participants':['NAME_SECRET']},
               {'id':'真实姓名','game_type':'NAME_SECRET','stakes':'INTERNAL_SECRET',
                'time':'PRIVATE_REASON','smoking':{'name':'SECRET'}}])
    rows = result['audit']['input']['public_board']
    assert rows[0] == {'id':'game-1','game_type':'hangzhou_mahjong','stakes':['0.5'],
                       'smoking':'no_smoking','time':'19:30'}
    assert rows[1] == {}


def test_cross_game_topic_merge_is_rejected_then_recovered_as_new_topic():
    output = decision('delegate',topic_action='existing',topic_id='guobiao',
                      public_summary={'game_type':'sichuan_mahjong'})
    llm = RecordingLLM(output,decision('delegate',public_summary={'game_type':'sichuan_mahjong'}))
    result = SemanticTopicClassifier(llm).classify(message=message('帮我找川麻'),
        candidates=[card('guobiao',public_summary={'game_type':'national_standard_mahjong'})],trace_id='t')
    assert result['topic_action'] == 'new'
    assert result['audit']['attempts'][0]['error'] == 'existing_topic_game_type_conflict'
