"""局池事实与个人适合性结论必须有不同的证据范围。"""

import json
from pathlib import Path

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentContextBuilder, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder,
    SQLiteAgentStore, ToolCall, ToolGateway, UserMessage,
)
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.copywriting import build_customer_visible_text_generation_payload
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract
from mahjong_agent_runtime.domains.tools.search_tools import search_current_games
from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract


@pytest.fixture(params=['memory','sqlite'])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param=='memory' else SQLiteAgentStore(tmp_path/'inquiry.sqlite3')
    yield value
    if request.param=='sqlite':
        value._connection.close()


@pytest.mark.parametrize('populated',[False,True])
def test_unfiltered_inventory_does_not_establish_personal_suitability(store, populated):
    if populated:
        store.create_game(conversation_id='other',organizer_id='owner',organizer_name='发起人',
            requirement={'game_type':'national_standard_mahjong','stake':'1','start_time_kind':'asap_when_full'},
            known_players=[{'customer_id':'owner','display_name':'发起人','seat_count':3}],trace_id='setup')
    result = search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':{}}),
                                  'query','customer-chat','customer','客户')
    contract = result.result['customer_reply_contract']
    assert contract['evidence_scope']['kind']=='unfiltered_inventory'
    assert contract['evidence_scope']['applied_filter_fields']==[]
    assert contract['evidence_scope']['criteria_completeness']=='not_assessed_by_retrieval'
    assert contract['search_result_semantics']['status']==('unfiltered_pool_listing' if populated else 'unfiltered_pool_empty')
    assert contract['search_result_semantics']['actionable_match_count']==0
    assert contract['evidence_scope']['unfiltered_available_pool_empty'] is (not populated)
    assert contract['evidence_scope']['supports_narrower_empty_claim'] is (not populated)
    assert len(result.result['matches'])==(1 if populated else 0)


@pytest.mark.parametrize('requirement',[{'stake':'1'},{'game_type':'national_standard_mahjong','stake':'1'}])
def test_filtered_search_reports_only_submitted_scope_not_criteria_completeness(store, requirement):
    result = search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':requirement}),
                                  'query','customer-chat','customer','客户')
    contract = result.result['customer_reply_contract']
    assert contract['evidence_scope']['kind']=='submitted_filters_only'
    assert set(requirement)<=set(contract['evidence_scope']['applied_filter_fields'])
    assert contract['evidence_scope']['criteria_completeness']=='not_assessed_by_retrieval'
    assert contract['search_result_semantics']['status']=='no_actionable_match'


def test_non_scoring_query_argument_cannot_hide_open_games_or_claim_filter_was_applied(store):
    store.create_game(conversation_id='other',organizer_id='owner',organizer_name='发起人',
        requirement={'game_type':'national_standard_mahjong','stake':'1','start_time_kind':'asap_when_full'},
        known_players=[{'customer_id':'owner','display_name':'发起人','seat_count':3}],trace_id='setup')
    result=search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':{'time_preference':'现在'}}),
                                'query','customer-chat','customer','客户')
    assert len(result.result['matches'])==1
    assert result.result['requirement']=={'time_preference':'现在'}
    assert result.result['customer_reply_contract']['evidence_scope']['kind']=='unfiltered_inventory'
    assert result.result['customer_reply_contract']['evidence_scope']['applied_filter_fields']==[]


def test_empty_superset_query_establishes_only_queried_registered_scope(store):
    """可报告本次没有返回，但非空局池下不能把评分结果当成任意子范围的缺失证明。"""
    store.create_game(conversation_id='other', organizer_id='owner', organizer_name='发起人',
        requirement={'game_type':'red_center_mahjong','stake':'5','start_time_kind':'asap_when_full'},
        known_players=[{'customer_id':'owner','display_name':'发起人','seat_count':3}],trace_id='setup')
    result=search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':{
        'game_type':'red_center_mahjong','stake_range':{'min':1,'max':3}}}),
        'query','customer-chat','customer','客户')
    contract=result.result['customer_reply_contract']
    assert result.result['matches']==[] and result.result['alternatives']==[]
    assert len(store.active_games())==1
    assert contract['evidence_scope']['unfiltered_available_pool_empty'] is False
    assert contract['evidence_scope']['supports_narrower_empty_claim'] is False
    assert contract['evidence_scope']['criteria_completeness']=='not_assessed_by_retrieval'
    assert contract['matched_query_requirement']['stake_range']=={'min':1.0,'max':3.0}
    assert 'start_time' not in contract['evidence_scope']['applied_filter_fields']
    assert 'adding filters may raise a candidate' in contract['evidence_scope']['claim_boundary']
    assert contract['game_query_strategy_contract']==game_query_strategy_contract()
    assert 'game_query_strategy_contract.query_result_reporting' in contract['evidence_scope']['claim_boundary']
    assert 'No registered actionable game does not mean no potential players' in contract['search_result_semantics']['instruction']
    assert 'without a preliminary availability judgment' not in contract['reply_shape']


def test_limited_nonempty_result_never_establishes_empty_subscope(store):
    """limit 后的非空结果不能证明未返回的时段没有局，避免把一条概况外推成完整空集。"""
    for index in range(2):
        store.create_game(conversation_id=f'other-{index}',organizer_id=f'owner-{index}',organizer_name='发起人',
            requirement={'game_type':'red_center_mahjong','stake':str(index+1),'start_time_kind':'asap_when_full'},
            known_players=[{'customer_id':f'owner-{index}','display_name':'发起人','seat_count':3}],trace_id='setup')
    result=search_current_games(store,ToolCall(name='search_current_games',arguments={
        'requirement':{'game_type':'red_center_mahjong'},'limit':1}),
        'query','customer-chat','customer','客户')
    contract=result.result['customer_reply_contract']
    assert len(result.result['matches'])==1 and len(store.active_games())==2
    assert contract['evidence_scope']['unfiltered_available_pool_empty'] is False
    assert contract['evidence_scope']['supports_narrower_empty_claim'] is False
    assert 'Non-empty or truncated results cannot establish absence' in contract['evidence_scope']['claim_boundary']


@pytest.mark.parametrize('requirement', [{}, {'cap_score':16}])
def test_contract_factory_without_backend_pool_evidence_never_infers_narrower_absence(requirement):
    """合同工厂没有数据库读取权，不能从调用方传入的空列表臆测未过滤局池为空。"""
    scope = current_game_search_reply_contract(requirement, [])['evidence_scope']
    assert scope['unfiltered_available_pool_empty'] is None
    assert scope['supports_narrower_empty_claim'] is False
    assert 'empty_result_is_complete_for_submitted_filters' not in scope


def test_cap_unknown_empty_query_can_revive_after_adding_game_filter(store):
    """真实双实现复现新增条件提高总分，保留原匹配资格但不再给出错误空子集证明。"""
    game, _ = store.create_game(conversation_id='other',organizer_id='owner',organizer_name='PRIVATE_OWNER',
        requirement={'game_type':'red_center_mahjong','stake':'1','start_time_kind':'asap_when_full',
                     'user_visible_summary':'PRIVATE_AUXILIARY_SUMMARY'},
        known_players=[{'customer_id':'owner','display_name':'PRIVATE_OWNER','seat_count':3}],trace_id='setup')
    before = [item.to_dict() for item in store.games.values()]
    broad = search_current_games(store,ToolCall(name='search_current_games',arguments={
        'requirement':{'cap_score':16}}),'query','customer-chat','customer','客户').result
    assert broad['matches']==[] and broad['alternatives']==[]
    assert broad['customer_reply_contract']['evidence_scope']['unfiltered_available_pool_empty'] is False
    assert broad['customer_reply_contract']['evidence_scope']['supports_narrower_empty_claim'] is False
    assert game.game_id not in json.dumps(broad)
    assert 'PRIVATE_' not in json.dumps(broad)
    narrow = search_current_games(store,ToolCall(name='search_current_games',arguments={
        'requirement':{'cap_score':16,'game_type':'red_center_mahjong'}}),
        'query','customer-chat','customer','客户').result
    assert [item['game']['game_id'] for item in narrow['matches']]==[game.game_id]
    assert narrow['matches'][0]['match_kind']=='exact'
    assert 'cap_score_unknown' in narrow['matches'][0]['reasons']
    assert [item.to_dict() for item in store.games.values()]==before
    assert not store.invite_drafts and not store.outbound_message_drafts and not store.scheduled_tasks


def test_original_afternoon_inquiry_retains_empty_pool_proof_without_more_slots(store):
    """原查询未带下午时段，全池确空仍提供下午事实证据，但不规定本轮固定复述。"""
    root=Path(__file__).resolve().parents[1]
    handoff=json.loads((root/'eval/handoffs/20261030_full18_failures.json').read_text())
    original=next(case for case in handoff['cases'] if case['scenario_id']=='profile_range_empty_inquiry')
    requirement=original['query_results'][0]['requirement']
    assert original['input']=='下午有人打吗' and 'start_time' not in requirement
    result=search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':requirement}),
        'query','range_empty','range_customer','客户A').result
    contract=result['customer_reply_contract']
    assert result['matches']==[] and result['alternatives']==[]
    assert contract['evidence_scope']['unfiltered_available_pool_empty'] is True
    assert contract['evidence_scope']['supports_narrower_empty_claim'] is True
    assert contract['evidence_scope']['criteria_completeness']=='not_assessed_by_retrieval'
    assert contract['game_query_strategy_contract']==game_query_strategy_contract()
    assert '不强制每轮先复述空结果' in contract['game_query_strategy_contract']['query_result_reporting']
    assert not store.games and not store.waiting_demands and not store.scheduled_tasks


@pytest.mark.parametrize('explicit_result', [False, True])
def test_empty_query_offer_review_receives_contextual_reporting_without_fixed_result_prefix(store, explicit_result):
    """实际空查询接入既有征询正例，检查合同接线而不以模型替身冒充语义通过。"""
    root=Path(__file__).resolve().parents[1]
    cases=[json.loads(line) for line in (root/'eval/golden/customer_visible_review_contract.jsonl').read_text().splitlines() if line.strip()]
    original=next(case for case in cases if case['case_id']=='review_ask_people_offer_question')
    assert original['expected_approved'] is True
    assert original['text']=='要我帮你问问有没有人吗？'
    message=UserMessage(conversation_id='offer-chat',sender_id='customer',sender_name='客户',text='下午有人打吗')
    gateway=ToolGateway(store)
    context=AgentContextBuilder(store,gateway).build(message,trace_id='offer-context').payload
    query=gateway.execute(ToolCall(name='search_current_games',arguments={'requirement':original['requirement']}),
        trace_id='offer-query',conversation_id=message.conversation_id,sender_id=message.sender_id,
        sender_name=message.sender_name,step_index=1)
    assert query.called and query.allowed and query.error is None
    assert query.result['matches']==[] and query.result['alternatives']==[]
    context['turn_tool_evidence']=[query.to_dict()]
    text=('当前没查到现成局，' if explicit_result else '')+original['text']
    action=AgentAction(goal='征询是否继续帮忙',objective_status='waiting_user',reasoning_summary='',reply_to_user=text)
    payload=build_reply_self_review_payload(message=message,action=action,context_payload=context,review_scope='reply_to_user',
        review_items=[{'item_id':'reply_to_user','recipient_id':message.sender_id,'text':text,'source_text':text,
                       'action_evidence':{'contact_started':False}}])
    generation=build_customer_visible_text_generation_payload(message=message,action=action,
        items=[{'item_id':'reply_to_user','source':'reply_to_user','text':text}],
        context_payload=context,generation_scope='reply_to_user')
    strategy=game_query_strategy_contract()
    assert payload['game_query_strategy_contract']==strategy
    assert generation['game_query_strategy_contract']['query_result_reporting']==strategy['query_result_reporting']
    assert query.result['customer_reply_contract']['game_query_strategy_contract']==strategy
    assert payload['review_items'][0]['text']==text
    assert payload['turn_tool_evidence']==[query.to_dict()]
    assert payload['turn_tool_evidence'][0]['result']['customer_reply_contract']['evidence_scope']['supports_narrower_empty_claim'] is True
    effective_contract=json.dumps({'query':query.result['customer_reply_contract'],
        'fidelity':payload['semantic_fidelity_contract'],
        'main_prompt':(root/'src/mahjong_agent_runtime/prompts/agent_runtime_system.md').read_text(),
        'review_prompt':(root/'src/mahjong_agent_runtime/prompts/agent_runtime_reply_self_review.md').read_text()},ensure_ascii=False)
    for forced_order in ('First answer the current inquiry','仍先回答实际检索事实，再问','先如实回答真实查询已证明的事实，再问',
                         'do not omit the known query result merely because seat count or precise time'):
        assert forced_order not in effective_contract
    assert not store.games and not store.waiting_demands and not store.invite_drafts
    assert not store.outbound_message_drafts and not store.scheduled_tasks


def test_auxiliary_pool_check_keeps_requester_and_conversation_relationship_scope(store):
    """空池证明保留当前任务关系约束，不跨会话套用，也不返回受约束牌局或私有理由。"""
    game, _=store.create_game(conversation_id='owner-chat',organizer_id='owner',organizer_name='PRIVATE_OWNER',
        requirement={'game_type':'red_center_mahjong','stake':'1','start_time_kind':'asap_when_full'},
        known_players=[{'customer_id':'owner','display_name':'PRIVATE_OWNER','seat_count':3}],trace_id='setup')
    store.record_task_memory(conversation_id='customer-chat',customer_id='customer',memory_type='relationship',
        field='avoid_playing',value=True,target_customer_id='owner',evidence='PRIVATE_RELATION_REASON',
        confidence=1,trace_id='relationship')
    before=[item.to_dict() for item in store.games.values()]
    for conversation_id,sender_id,expected_empty in [
        ('customer-chat','customer',True),('different-chat','customer',False),('customer-chat','other-customer',False),
    ]:
        result=search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':{'cap_score':16}}),
            'query',conversation_id,sender_id,'客户').result
        scope=result['customer_reply_contract']['evidence_scope']
        assert scope['available_pool_scope']=='same_requester_and_conversation'
        assert scope['unfiltered_available_pool_empty'] is expected_empty
        assert scope['supports_narrower_empty_claim'] is expected_empty
        assert game.game_id not in json.dumps(result) and 'PRIVATE_' not in json.dumps(result)
    assert len(store.active_games())==1
    assert [item.to_dict() for item in store.games.values()]==before


@pytest.mark.parametrize('populated',[False,True])
def test_unfiltered_query_reuses_its_own_pool_evidence_without_another_read(store, monkeypatch, populated):
    """未过滤读取本身足以判断可用局池有无，不重复查库或新增模型工具调用。"""
    if populated:
        store.create_game(conversation_id='other',organizer_id='owner',organizer_name='发起人',
            requirement={'game_type':'red_center_mahjong','stake':'1','start_time_kind':'asap_when_full'},
            known_players=[{'customer_id':'owner','display_name':'发起人','seat_count':3}],trace_id='setup')
    calls=[]
    original_search=type(store).search_current_games
    def tracked_search(self, requirement, **kwargs):
        """记录真实存储读取次数，仍执行原检索以验证证据复用。"""
        calls.append((requirement,kwargs))
        return original_search(self,requirement,**kwargs)
    monkeypatch.setattr(type(store),'search_current_games',tracked_search)
    result=search_current_games(store,ToolCall(name='search_current_games',arguments={'requirement':{}}),
        'query','customer-chat','customer','客户').result
    assert len(calls)==1
    assert result['customer_reply_contract']['evidence_scope']['unfiltered_available_pool_empty'] is (not populated)


def test_original_empty_result_failure_is_preserved_as_adversarial_evidence():
    """新增回归引用原失败原文和判定，不能用修复后的输出覆盖原始失败记录。"""
    root=Path(__file__).resolve().parents[1]
    handoff=json.loads((root/'eval/handoffs/20261030_full18_failures.json').read_text())
    original=next(case for case in handoff['cases'] if case['scenario_id']=='profile_range_empty_inquiry')
    cases=[json.loads(line) for line in (root/'eval/adversarial/agent_planning_discoveries.jsonl').read_text().splitlines() if line.strip()]
    recorded=next(case for case in cases if case['id']=='full18_empty_registered_scope_answer_20261030_001')
    assert recorded['source']['run_id']==handoff['run_id']
    assert recorded['input']['text']==original['input']
    assert recorded['input']['known_profile']==original['known_profile']
    assert recorded['observed']['final_reply']==original['actual_reply']
    assert recorded['observed']['query_results']==original['query_results']
    assert recorded['observed']['semantic_criteria']==original['semantic_criteria']
    assert any(not criterion['passed'] and criterion['id'].endswith('.answer_current_inquiry')
               for criterion in recorded['observed']['semantic_criteria'])


class InquiryClient:
    def complete(self, messages, **kwargs):
        payload=json.loads(messages[-1]['content'])
        previous=payload.get('previous_tool_results') or []
        if not previous:
            action=AgentAction(goal='查询局池',objective_status='needs_tool',reasoning_summary='先读局池',
                tool_calls=[ToolCall(name='search_current_games',arguments={'requirement':{}},reason='读局池',call_id='search')],
                stop_reason={'can_stop':False,'why':'查询','pending_work':['query'],'depends_on_tool_results':True})
        elif previous[-1]['name']=='customer_visible_content_review':
            action=AgentAction(goal='澄清未知需求',objective_status='waiting_user',reasoning_summary='局池空不代表已经知道客户需求',
                objective_state={'missing_facts':['玩法','档位']},reply_to_user='想打什么玩法，多少的？',
                stop_reason={'can_stop':True,'why':'等客户说明需求','pending_work':[],'depends_on_tool_results':False})
        else:
            action=AgentAction(goal='回答是否合适',objective_status='completed',reasoning_summary='此处故意复现被标注的错误结论',
                reply_to_user='现在还没有合适的局，想打什么玩法、几点，我帮你留意着。',
                stop_reason={'can_stop':True,'why':'错误原稿待审','pending_work':[],'depends_on_tool_results':True})
        return json.dumps(action.to_dict(),ensure_ascii=False)


class ReviewClient:
    def __init__(self):
        self.payloads=[]

    def complete(self,messages,**kwargs):
        payload=json.loads(messages[-1]['content']);self.payloads.append(payload)
        rejected=payload['review_items'][0]['text'].startswith('现在还没有合适')
        return json.dumps({'approved':not rejected,'needs_human':False,'reasoning_summary':'检查结论范围',
            'violations':['unsupported_conclusion'] if rejected else [],'item_reviews':[
                {'item_id':item['item_id'],'approved':not rejected,'suggested_safe_text':'想打什么玩法，多少的？' if rejected else item['text'],
                 'reasoning_summary':'无匹配依据时只能澄清','violations':['unsupported_conclusion'] if rejected else []}
                for item in payload['review_items']]},ensure_ascii=False)


def test_unsupported_fit_review_feedback_returns_to_main_agent_before_delivery(store):
    review=ReviewClient();runtime=AgentRuntime(store=store,llm_client=InquiryClient(),trace_recorder=InMemoryTraceRecorder(),
        reply_self_review_enabled=True,reply_self_review_client=review)
    result=runtime.handle_user_message(UserMessage(conversation_id='unknown-customer',sender_id='customer',sender_name='客户',text='现在有人吗',message_id='query'))
    assert result.status=='waiting_user' and result.final_reply=='想打什么玩法，多少的？'
    evidence=review.payloads[0]['turn_tool_evidence'][0]['result']['customer_reply_contract']['evidence_scope']
    assert evidence['kind']=='unfiltered_inventory'
    reviews=[tool for tool in result.tool_results if tool.name=='customer_visible_content_review']
    assert len(reviews)==2 and reviews[0].result['approved'] is False and reviews[1].result['approved'] is True
    assert not store.games and not store.waiting_demands and not store.scheduled_tasks
