"""局池事实与个人适合性结论必须有不同的证据范围。"""

import json

import pytest

from mahjong_agent_runtime import (
    AgentAction, AgentRuntime, InMemoryAgentStore, InMemoryTraceRecorder,
    SQLiteAgentStore, ToolCall, UserMessage,
)
from mahjong_agent_runtime.domains.tools.search_tools import search_current_games


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
