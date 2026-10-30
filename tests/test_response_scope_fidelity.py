"""完整局况的保真义务先受真实应答范围约束，公开字段不自动成为本轮必答清单。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentContextBuilder, AgentRuntime, CustomerProfile,
    InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, TokenBudget, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.copywriting import build_customer_visible_text_generation_payload
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['memory', 'sqlite'])
def scoped_store(request, tmp_path):
    """真实双实现保存公开昵称与无关私有备注，用来检查投影和后端执行边界。"""
    store = InMemoryAgentStore() if request.param == 'memory' else SQLiteAgentStore(tmp_path/'scope.sqlite3')
    for customer_id, public_name in [('summer', '夏日'), ('smile', '笑脸'), ('kexing', '可星')]:
        store.upsert_customer(CustomerProfile(customer_id=customer_id, display_name='PRIVATE_'+customer_id,
            public_name=public_name, private_remark='PRIVATE_REMARK_'+customer_id))
    store.create_game(conversation_id='current', organizer_id='summer', organizer_name='PRIVATE_summer',
        requirement={'game_type':'hangzhou_mahjong','stake':'0.5','start_time_kind':'asap_when_full',
                     'user_visible_summary':'七点三缺一'},
        known_players=[{'customer_id':customer_id,'display_name':'PRIVATE_'+customer_id} for customer_id in ('summer','smile','kexing')],
        trace_id='fixture')
    yield store
    if request.param == 'sqlite':
        store._connection.close()


def _build(store, text):
    """所有提问都经同一构建器，不按自然语言删除背景或选择合同。"""
    message = UserMessage(conversation_id='current', sender_id='customer', sender_name='客户', text=text, message_id='scope-question')
    built = AgentContextBuilder(store, ToolGateway(store)).build(message, trace_id='scope')
    return message, built.payload


@pytest.mark.parametrize('text', ['哪些人啊', '现在几个人了', '抽不抽烟', '今天辛苦了'])
def test_authorized_projection_retains_summary_but_conditions_its_preservation(scoped_store, text):
    """保留相同的公开事实与完整摘要合同，适用性由模型按真实问题判断。"""
    _, payload = _build(scoped_store, text)
    summary = payload['active_game_visible_summaries'][0]
    contract = summary['status_query_reply_contract']
    assert summary['public_participant_names'] == ['夏日', '笑脸', '可星']
    assert contract['preferred_reply_text'] == '七点三缺一'
    assert contract['preservation_mode'] == 'all_decision_anchors'
    assert contract['required_semantic_source'] == 'preferred_reply_text'
    assert '本轮回答该局局况、人数、缺口或进展' in contract['applicability']
    assert '人数、缺口和进展不属于这种豁免' in contract['applicability']
    assert '可披露不等于必须披露' in contract['applicability']
    assert '仅在 applicability 成立' in contract['rule']
    assert 'PRIVATE_' not in json.dumps(summary, ensure_ascii=False)


@pytest.mark.parametrize('expanded_proposal', [False, True])
def test_review_payload_keeps_actual_request_authoritative_over_expanded_draft(scoped_store, expanded_proposal):
    """复现首次正确目标及被错误反馈污染后的目标，后者也不能产生用户授权。"""
    message, context = _build(scoped_store, '哪些人啊')
    names = '夏日、笑脸、可星'
    draft = '七点三缺一，现在是夏日、笑脸、可星。' if expanded_proposal else names
    action = AgentAction(goal='答名单并保留局况' if expanded_proposal else '答名单', objective_status='completed',
        reasoning_summary='', reply_to_user=names, objective_state={'reply_scope':{
            'requested_information':'当前有哪些人',
            'allowed_response_facts':['公开昵称', '七点三缺一'] if expanded_proposal else ['公开昵称'],
        }})
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context, review_scope='reply_to_user',
        review_items=[{'item_id':'reply_to_user','recipient_id':message.sender_id,'text':names,'source_text':draft}])
    fidelity = payload['semantic_fidelity_contract']
    assert payload['current_message']['text'] == '哪些人啊'
    assert fidelity['response_scope_before_anchor_preservation'] is True
    assert fidelity['visible_facts_are_not_mandatory_response_facts'] is True
    assert fidelity['progress_queries_require_all_decision_anchors'] is True
    assert fidelity['narrow_reply_must_not_add_unrequested_status_or_confirmation'] is True
    assert 'source_text 和审查建议本身不能扩大用户请求' in fidelity['response_scope_source']
    assert any('all_decision_anchors 先核对其 applicability' in rule for rule in fidelity['rules'])
    assert payload['review_items'][0]['source_text'] == draft
    assert payload['active_game_visible_summaries'][0]['status_query_reply_contract']['preferred_reply_text'] == '七点三缺一'
    assert payload['active_game_visible_summaries'][0]['public_participant_names'] == ['夏日','笑脸','可星']


@pytest.mark.parametrize('question', ['哪些人啊', '现在几个人了'])
def test_generation_scope_is_selected_from_request_without_erasing_decision_facts(question):
    """合同保持原文事实，同时声明只对真正局况/选项强制完整锚点；此处不模拟语义通过。"""
    original = '七点三缺一，夏日、笑脸、可星。'
    payload = build_customer_visible_text_generation_payload(
        message=UserMessage(conversation_id='current',sender_id='customer',sender_name='客户',text=question),
        action=AgentAction(goal='回复客户',objective_status='completed',reasoning_summary='',reply_to_user=original),
        items=[{'item_id':'reply_to_user','source':'reply_to_user','text':original}],
        context_payload={'PRIVATE_SHOULD_NOT_ENTER_GENERATOR':'secret'},generation_scope='reply_to_user')
    assert payload['current_request']['text'] == question
    assert payload['items'][0]['text'] == original
    rule = payload['reply_relevance_contract']['coherent_option_summary_rule']
    assert 'First check whether the actual current_request or authorized proposal requires' in rule
    assert 'Only in that scope' in rule and 'names-only or non-progress single-attribute question may omit' in rule
    assert 'Counts, shortages and progress are status queries, not single-attribute exemptions' in rule
    assert 'information actually necessary to identify the requested object' in rule
    assert 'invite/draft recipients still require all their decision facts' in payload['style_quality_contract']['must_preserve_scope']
    assert 'PRIVATE_SHOULD_NOT_ENTER_GENERATOR' not in json.dumps(payload)


@pytest.mark.parametrize('question,draft,final', [
    ('哪些人啊', '七点三缺一，现在是夏日、笑脸、可星。', '夏日、笑脸、可星'),
    ('现在几个人了', '七点三缺一', '七点三缺一'),
])
def test_full_response_pipeline_passes_same_scope_evidence_without_bypassing_review(scoped_store, question, draft, final):
    """模型替身只验证生成、审查与最终交付接线，真实语义另由真实模型探针验收。"""
    calls = []
    class Main:
        def complete(self, messages, **kwargs):
            calls.append('main')
            return json.dumps(AgentAction(goal='回答客户本次问题',objective_status='completed',reasoning_summary='',reply_to_user=draft,
                stop_reason={'can_stop':True,'why':'本次已答','pending_work':[],'depends_on_tool_results':False}).to_dict(),ensure_ascii=False)
    class Generator:
        def complete(self, messages, **kwargs):
            calls.append('generation')
            payload = json.loads(messages[-1]['content'])
            assert payload['current_request']['text'] == question
            assert payload['items'][0]['text'] == draft
            assert 'Only in that scope' in payload['reply_relevance_contract']['coherent_option_summary_rule']
            return json.dumps({'reasoning_summary':'受控接线测试','item_rewrites':[{'item_id':'reply_to_user',
                'final_text':final,'semantic_preserved':True,'used_facts':[final],
                'withheld_facts':[] if draft==final else ['未问局况'], 'style_checks':['事实保真'],'change_summary':'按受控结果改写'}]},ensure_ascii=False)
    class Reviewer:
        def complete(self, messages, **kwargs):
            calls.append('review')
            payload = json.loads(messages[-1]['content'])
            assert payload['current_message']['text'] == question
            assert payload['review_items'][0]['text'] == final
            assert payload['review_items'][0]['source_text'] == draft
            assert payload['semantic_fidelity_contract']['response_scope_before_anchor_preservation'] is True
            assert payload['active_game_visible_summaries'][0]['status_query_reply_contract']['preservation_mode']=='all_decision_anchors'
            return json.dumps({'approved':True,'needs_human':False,'reasoning_summary':'受控接线结果','violations':[],
                'item_reviews':[{'item_id':'reply_to_user','approved':True,'suggested_safe_text':final,
                    'reasoning_summary':'受控接线结果','violations':[]}]},ensure_ascii=False)
    before = [game.to_dict() for game in scoped_store.games.values()]
    runtime = AgentRuntime(store=scoped_store,llm_client=Main(),trace_recorder=InMemoryTraceRecorder(),
        customer_visible_text_generation_enabled=True,customer_visible_text_generation_client=Generator(),
        reply_self_review_enabled=True,reply_self_review_client=Reviewer())
    result = runtime.handle_user_message(UserMessage(conversation_id='current',sender_id='customer',sender_name='客户',text=question,message_id='scope-pipeline'))
    assert result.status=='completed' and result.final_reply==final
    assert calls==['main','generation','review']
    assert [game.to_dict() for game in scoped_store.games.values()]==before
    assert not [item for item in result.tool_results if item.name not in {'customer_visible_text_generation','customer_visible_content_review'}]


@pytest.mark.parametrize('question,source_text,text,label,safe', [
    ('哪些人啊', '七点三缺一，现在是夏日、笑脸、可星。', '七点三缺一，现在是夏日、笑脸、可星。',
     'unsolicited_goal_expansion', '夏日、笑脸、可星。'),
    ('现在几个人了', '七点三缺一', '三个人，还差一个', 'semantic_fact_loss', '七点三缺一'),
])
def test_negative_scope_verdict_is_not_overridden_by_true_background_facts(scoped_store, question, source_text, text, label, safe):
    """替身仅核验真实拒绝标签和修正反馈的传递，不能当作生产模型正反例通过。"""
    message, context = _build(scoped_store, question)
    observed = []
    class Reviewer:
        def complete(self, messages, **kwargs):
            payload=json.loads(messages[-1]['content']);observed.append(payload)
            assert payload['current_message']['text']==question
            assert payload['semantic_fidelity_contract']['progress_queries_require_all_decision_anchors'] is True
            assert payload['semantic_fidelity_contract']['narrow_reply_must_not_add_unrequested_status_or_confirmation'] is True
            return json.dumps({'approved':False,'needs_human':False,'reasoning_summary':'固定负例反馈接线',
                'violations':[label],'item_reviews':[{'item_id':'reply_to_user','approved':False,
                    'suggested_safe_text':safe,'reasoning_summary':'固定负例反馈接线','violations':[label]}]},ensure_ascii=False)
    before=[game.to_dict() for game in scoped_store.games.values()]
    runtime=AgentRuntime(store=scoped_store,llm_client=Reviewer(),trace_recorder=InMemoryTraceRecorder(),
                         reply_self_review_enabled=True,reply_self_review_client=Reviewer())
    action=AgentAction(goal='只回应本次问题',objective_status='completed',reasoning_summary='',reply_to_user=text)
    result=runtime.action_processor.review_customer_visible_content(message=message,trace_id='scope-negative',action=action,
        context_payload=context,review_scope='reply_to_user',turn_budget=TokenBudget(max_calls_per_turn=1),
        review_items=[{'item_id':'reply_to_user','recipient_id':message.sender_id,'text':text,'source_text':source_text}])
    assert len(observed)==1 and result is not None
    assert result.result['approved'] is result.result['raw_approved'] is False
    assert label in result.result['violations']
    assert result.result['item_reviews'][0]['suggested_safe_text']==safe
    assert [game.to_dict() for game in scoped_store.games.values()]==before


def test_prompt_contracts_exclude_progress_from_narrow_attribute_omission():
    """静态合同回归保护同一范围定义，真实语义还须用独立真实模型证据验收。"""
    prompt_dir=ROOT/'src/mahjong_agent_runtime/prompts'
    generation=(prompt_dir/'customer_visible_text_generation.md').read_text()
    review=(prompt_dir/'agent_runtime_reply_self_review.md').read_text()
    assert '人数、缺口和进展不属于此豁免' in generation
    assert '人数、缺口和进展属于局况查询，不是可省略时间等锚点的单项属性豁免' in review
    assert '待审文本擅自附带未请求的局况、进展、招揽或参加确认' in review
    assert '必须以 `unsolicited_goal_expansion` 拒绝' in review
    assert '不机械按字段删除' in generation and '不机械按字段删减' in review
