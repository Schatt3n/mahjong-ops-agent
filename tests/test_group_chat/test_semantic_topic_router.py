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
