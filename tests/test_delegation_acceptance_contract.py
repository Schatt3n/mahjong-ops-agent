"""委托承接回归：离线检查共享合同、最小审查投影和独立真模型对照资产。

替身不判断自然语言意图；模型行为由同目录资产通过真实群入口另行回放。
"""

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest

from mahjong_agent_runtime import AgentAction, AgentContextBuilder, CustomerProfile, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.query_contract import game_query_strategy_contract
from mahjong_agent_runtime.models import now
from mahjong_agent_runtime.task_context import TaskContextManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from manual_scenario import validate_scenario
from run_inquiry_grounding_eval import _semantic_criteria_ids

ASSET = ROOT / 'eval/golden/group_delegation_acceptance_20261105.jsonl'


def review_payload(*, scope='reply_to_user', recipient='requester', context_update=None, request_scope=None):
    """提供真实输入及已由来源核验层生成的结构投影，不把分类标签当执行授权。"""
    message = UserMessage(conversation_id='private:requester', sender_id='requester', sender_name='周岚',
        text='老板，帮我找明天下午两点国标1块无烟的，我一个人。', message_id='delegate-source')
    context = {
        'task_context_window': {'task_context_id': 'own-task'},
        'group_topic_context': {
            'topic_id': 'public-topic', 'intent': 'delegate', 'goal_relation': 'continue',
            'related_game_ids': [],
            'own_condition_sources': [{'source_message_id': 'prior-query', 'sent_at': '2026-11-05T13:00:00+08:00',
                'constraints': {'game_type': 'national_standard_mahjong', 'stakes': ['1'], 'smoking': 'no_smoking',
                    'source_message_id': 'prior-query'}}],
            'internal_note': '其他客户的私密备注不得复制',
            'raw_group_history': '其他客户的群原文不得复制',
        },
        'turn_tool_evidence': [{'name': 'search_current_games', 'called': True, 'allowed': True, 'error': None,
            'result': {'matches': []}}],
    }
    context.update(context_update or {})
    action = AgentAction(goal='寻找符合条件的局', objective_status='waiting_user', reasoning_summary='仅检索后再次征询',
        reply_to_user='暂时没有现成的，要不要我帮你组一个？', objective_state={'missing_facts': []})
    if request_scope is not None:
        action.objective_state['request_scope'] = request_scope
    items = [{'item_id': 'reply_to_user', 'recipient_id': recipient, 'text': action.reply_to_user}]
    original = deepcopy(context)
    payload = build_reply_self_review_payload(message=message, action=action, review_items=items,
        context_payload=context, review_scope=scope)
    assert context == original
    return message, payload


def test_delegation_scope_contract_is_shared_and_does_not_require_specific_invitation_path():
    """主规划和审查共享授权边界，空库存的明确委托必须有持久承接而非重新征询。"""
    strategy = game_query_strategy_contract()
    delegation = strategy.get('delegation_acceptance')
    assert isinstance(delegation, dict), '缺少区分查询、持续委托和具体参加确认的共享合同'
    required = {
        'complete_delegation_requires_durable_responsibility',
        'same_scope_must_not_reask_consent',
        'query_does_not_authorize_recruitment',
        'explicit_inventory_only_limits_delegation',
        'concrete_game_join_requires_confirmation',
        'clarify_only_missing_execution_facts',
        'board_can_recruit_without_private_invites',
        'classification_is_not_authorization',
    }
    assert all(delegation.get(key) is True for key in required)
    message, review = review_payload()
    store = InMemoryAgentStore()
    built = AgentContextBuilder(store, ToolGateway(store)).build(message, trace_id='delegation-contract')
    assert built.payload['planning_contract']['game_query_strategy_contract']['delegation_acceptance'] == delegation
    assert review['game_query_strategy_contract']['delegation_acceptance'] == delegation
    service = review['customer_service_review_contract']
    assert service.get('existing_delegation_must_not_be_reset_to_proposal') is True
    assert service.get('missing_execution_requires_replan_not_textual_acknowledgement') is True


def test_empty_inventory_strategy_checks_user_scope_before_offering_a_new_service():
    """查空只更新库存，不允许借旧默认把同一已明确委托改写为尚未授权。"""
    strategy = game_query_strategy_contract()
    scope = strategy.get('request_scope_contract')
    assert isinstance(scope, dict), '委托、限定查询、缺事实的适用顺序未明确'
    assert scope['authority'] == 'model_proposal_only'
    assert scope['decision_order'] == ['apply_explicit_limits', 'retain_same_goal_delegation', 'resolve_execution_gaps', 'choose_inventory_response']
    assert scope['inventory_result_changes_scope'] is False
    assert scope['model_plan_can_revoke_user_delegation'] is False
    assert scope['channel_changes_scope'] is False
    assert scope['empty_query_offer_scope'] == 'query_only'
    assert scope['missing_execution_facts_allow_clarification'] is True
    assert scope['explicit_inventory_only_allows_recruitment'] is False
    assert scope['scope_field'] == 'objective_state.request_scope'
    assert 'request_scope_contract' in strategy['empty_personal_matching_default']


def test_review_verifies_execution_stage_before_classifying_an_additional_service_offer():
    """已有执行进展与后续通知分开核验，不把未配置通知倒推成从未承接原委托。"""
    delegation = game_query_strategy_contract()['delegation_acceptance']
    progress = delegation.get('progress_review')
    assert isinstance(progress, dict), '缺少先核同目标执行进展、再审提议及承诺的阶段合同'
    assert progress['review_order'] == ['verify_original_scope', 'verify_same_goal_execution',
        'classify_remaining_communication', 'verify_accepted_obligations']
    assert progress['delegation_not_advanced_requires_missing_substantive_execution'] is True
    assert progress['durable_game_and_candidate_search_are_substantive_execution'] is True
    assert progress['proposal_after_execution_is_not_reset'] is True
    assert progress['conditional_additional_service_requires_prior_schedule'] is False
    assert progress['substantive_execution_does_not_prove_follow_up'] is True
    _, review = review_payload()
    assert review['game_query_strategy_contract']['delegation_acceptance']['progress_review'] == progress
    rule = next(rule for rule in review['semantic_fidelity_contract']['rules'] if 'delegation_not_advanced' in rule)
    assert 'progress_review' in rule and '已建立同目标待组局' in rule


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
@pytest.mark.parametrize('stage,text', [
    ('query_only', '暂时没有现成的，要不要我帮你组一个？'),
    ('created_and_searched', '要我接着帮你留意着，有合适的叫你吗？'),
    ('created_and_searched', '行，我帮你留意着，有合适的叫你。'),
])
def test_review_sees_real_execution_progress_separately_from_missing_future_notification(tmp_path, backend, stage, text):
    """双存储真实工具证明推进阶段；同一无通知状态下，提议与承诺留给模型按完整句意审查。"""
    store = InMemoryAgentStore() if backend == 'memory' else SQLiteAgentStore(tmp_path / 'progress.sqlite3')
    try:
        store.upsert_customer(CustomerProfile(customer_id='requester', display_name='客户'))
        message = UserMessage(conversation_id='progress-conversation', sender_id='requester', sender_name='客户',
            text='帮我组一局本店玩法，1块底256封顶，烟都行，人齐开，我一个人。', message_id='progress-source')
        TaskContextManager(store).prepare(message, trace_id='progress')
        store.append_user_turn(message, 'progress')
        gateway = ToolGateway(store)
        requirement = {'game_type': 'club_custom_mahjong', 'stake': '1', 'cap_score': 256,
            'smoke_preference': 'any', 'start_time_kind': 'asap_when_full'}
        tools = []
        def execute(name, arguments):
            outcome = gateway.execute(ToolCall(name, arguments), trace_id='progress', step_index=len(tools) + 1,
                conversation_id=message.conversation_id, sender_id=message.sender_id, sender_name=message.sender_name,
                source_message_id=message.message_id)
            assert outcome.called and outcome.allowed and not outcome.error
            tools.append(outcome)
            store.append_tool_turn(message.conversation_id, json.dumps([outcome.to_dict()]), 'progress')
            return outcome
        execute('search_current_games', {'requirement': requirement})
        if stage == 'created_and_searched':
            created = execute('create_game', {'organizer_id': message.sender_id, 'organizer_name': message.sender_name,
                'requirement': requirement, 'known_players': []})
            game_id = created.result['game']['game_id']
            searched = execute('search_customers', {'game_id': game_id, 'requirement': requirement,
                'exclude_customer_ids': [message.sender_id]})
            assert searched.result['candidates'] == []
            assert searched.result['continuation']['can_stop'] is True
            assert store.require_game(game_id).organizer_id == message.sender_id
        context = AgentContextBuilder(store, gateway).build(message, trace_id='progress',
            previous_tool_results=tools[-1:], turn_tool_evidence=tools).payload
        action = AgentAction(goal='承接本次组局', objective_status='waiting_user' if text.endswith('？') else 'completed',
            reasoning_summary='固定两种沟通作用域只验证事实输入', reply_to_user=text)
        before = [game.to_dict() for game in store.games.values()]
        payload = build_reply_self_review_payload(message=message, action=action,
            review_items=[{'item_id': 'reply_to_user', 'recipient_id': message.sender_id, 'text': text}],
            context_payload=context, review_scope='reply_to_user')
        assert payload['current_message'] == message.to_dict()
        assert [tool['name'] for tool in payload['turn_tool_evidence']] == [tool.name for tool in tools]
        assert len(payload['active_game_visible_summaries']) == (1 if stage == 'created_and_searched' else 0)
        assert payload['active_follow_up_summaries'] == [] and not store.scheduled_tasks
        assert payload['customer_service_review_contract']['commitment_evidence_requires_accepted_responsibility'] is True
        assert payload['game_query_strategy_contract']['delegation_acceptance']['progress_review'][
            'conditional_additional_service_requires_prior_schedule'] is False
        assert before == [game.to_dict() for game in store.games.values()]
        assert not store.invite_drafts and not store.outbound_message_drafts
    finally:
        if backend == 'sqlite': store._connection.close()


@pytest.mark.parametrize('scope,text,limits,missing', [
    ('delegated_matching', '老板，帮我找明晚九点半川麻1块无烟的，我一个人。', [], []),
    ('query_only', '明晚九点半川麻1块无烟有局吗？我一个人，先看看。', [], []),
    ('query_only', '帮我找明晚九点半川麻1块无烟的，我一个人，只找现成，没有就算了。', ['只找现成'], []),
    ('delegated_matching', '老板，帮我找明晚九点半川麻1块无烟的。', [], ['己方人数']),
])
def test_group_empty_lookup_retains_source_backed_scope_as_proposal_without_forcing_writes(scope, text, limits, missing):
    """固定模型提案只验上下文连续性；非好友群路径和空工具结果都不改变用户范围。"""
    store = InMemoryAgentStore()
    gateway = ToolGateway(store)
    message = UserMessage(conversation_id='group:room:customer:nonfriend',sender_id='nonfriend',sender_name='群友',
        text=text,message_id='scope-source',metadata={'source':'group'})
    proposed = {'kind':scope,'source_message_ids':[message.message_id],'explicit_limits':limits}
    action = AgentAction(goal='完成本人找局目标',objective_status='needs_tool',reasoning_summary='固定提案只验证投影',
        objective_state={'request_scope':proposed,'missing_facts':missing})
    result = gateway.execute(ToolCall('search_current_games',{'requirement':{}},'读取真实空库存'),
        trace_id='scope-read',step_index=1,conversation_id=message.conversation_id,
        sender_id=message.sender_id,sender_name=message.sender_name)
    built = AgentContextBuilder(store,gateway).build(message,trace_id='scope-after-read',
        previous_action=action,previous_tool_results=[result],turn_tool_evidence=[result])
    previous = built.payload['previous_decision']
    assert previous['authority'] == 'model_proposal_only'
    assert previous['objective_state']['request_scope'] == proposed
    assert previous['objective_state']['missing_facts'] == missing
    assert built.payload['current_message']['text'] == text
    assert result.called and result.allowed and result.result['matches'] == []
    assert 'continuation' not in result.result, '库存工具不能自行决定用户目标可停或必须建局'
    assert not store.games and not store.scheduled_tasks
    contract = built.payload['output_contract']['objective_state_contract']
    assert 'request_scope' in contract


def test_review_receives_scope_proposal_with_sources_but_not_as_authoritative_permission():
    """审查比对提案、本人原话和事实，不把新计划解释变成撤销旧委托的权限。"""
    scope = {'kind':'delegated_matching','source_message_ids':['delegate-source'],
        'explicit_limits':[], 'private_notes':'不是审查所需字段'}
    _, payload = review_payload(request_scope=scope)
    projected = payload['action_boundary'].get('request_scope')
    assert projected == {'authority':'model_proposal_only','kind':'delegated_matching',
        'source_message_ids':['delegate-source'],'explicit_limits':[]}
    assert payload['game_query_strategy_contract']['request_scope_contract']['model_plan_can_revoke_user_delegation'] is False


def test_review_drops_unverified_scope_sources_and_source_free_kind_grants_no_authority():
    """模型编造的来源不能被包装成已核验本人授权，空来源也仍是低可信解释。"""
    _, payload = review_payload(request_scope={'kind':'delegated_matching',
        'source_message_ids':['other-customer-message'],'explicit_limits':[]})
    scope = payload['action_boundary']['request_scope']
    assert scope['authority'] == 'model_proposal_only' and scope['source_message_ids'] == []


def test_current_customer_review_keeps_verified_topic_conditions_without_group_raw_history():
    """群转私聊审查能核对同目标来源，不因只看当前短句丢失授权范围或本人事实。"""
    message, payload = review_payload()
    projected = payload.get('group_topic_context')
    assert isinstance(projected, dict), '主模型已有的本人话题条件未投影给审查器'
    assert projected['intent'] == 'delegate' and projected['goal_relation'] == 'continue'
    assert projected['own_condition_sources'][0]['source_message_id'] == 'prior-query'
    assert projected['own_condition_sources'][0]['constraints']['game_type'] == 'national_standard_mahjong'
    assert set(projected) <= {'topic_id', 'intent', 'goal_relation', 'related_game_ids', 'own_condition_sources'}
    assert '其他客户' not in json.dumps(payload, ensure_ascii=False)
    assert payload['current_message']['text'] == message.text
    assert payload['turn_tool_evidence'][0]['result']['matches'] == []


@pytest.mark.parametrize('scope,recipient', [('tool_calls', 'candidate'), ('reply_to_user', 'candidate')])
def test_other_recipient_review_does_not_receive_requesters_topic_history(scope, recipient):
    """给候选人审查邀约不能沿用发起人的个人话题历史。"""
    _, payload = review_payload(scope=scope, recipient=recipient)
    assert not payload.get('group_topic_context')
    assert 'prior-query' not in json.dumps(payload, ensure_ascii=False)


def deadline_task():
    """构造真实调度模型形状的本目标截止证据，未知私有载荷不得交给审查器。"""
    due = (now() + timedelta(hours=3)).isoformat()
    return {'task_id': 'deadline-1', 'task_type': 'participation_deadline', 'status': 'pending',
        'conversation_id': 'private:requester', 'subject_id': 'requester', 'due_at': due,
        'aggregate_type': 'game_participant', 'aggregate_id': 'game-1:requester',
        'payload': {'game_id': 'game-1', 'customer_id': 'requester', 'task_context_id': 'own-task',
            'deadline_at': due, 'condition': 'game_not_ready', 'effect': 'withdraw_self',
            'source_message_id': 'deadline-source', 'private_note': '其他客户的截止安排'}}


def test_deadline_review_evidence_keeps_only_effective_own_goal_task():
    """记忆或招募触发不能替代到点退出，只有对应本人目标的截止任务可证明承接。"""
    task = deadline_task()
    _, payload = review_payload(context_update={'active_participation_deadlines': [task]})
    summaries = payload.get('active_participation_deadline_summaries')
    assert summaries and len(summaries) == 1
    summary = summaries[0]
    assert summary['task_id'] == task['task_id'] and summary['deadline_at'] == task['due_at']
    assert summary['game_id'] == 'game-1' and summary['task_context_id'] == 'own-task'
    assert summary['condition'] == 'game_not_ready' and summary['effect'] == 'withdraw_self'
    assert '其他客户' not in json.dumps(payload, ensure_ascii=False)
    contract = payload['customer_service_review_contract']
    assert contract.get('waiting_memory_is_not_executable_deadline') is True
    assert contract.get('recruitment_window_is_not_participation_deadline') is True


@pytest.mark.parametrize('field,value', [
    ('subject_id', 'other'), ('conversation_id', 'other-private'), ('status', 'completed'),
    ('task_type', 'recruitment_window'), ('aggregate_type', 'game'),
    ('payload.customer_id', 'other'), ('payload.task_context_id', 'other-goal'),
    ('payload.effect', 'cancel_game'), ('payload.condition', 'always'),
    ('payload.execution_outcome', 'withdraw_self'), ('payload.execution_outcome', 'game_ready_or_closed'),
    ('due_at', '2026-01-01T00:00:00+08:00'), ('payload.deadline_at', 'not-time'),
])
def test_unrelated_or_unexecutable_deadline_cannot_prove_current_commitment(field, value):
    """其他人、旧目标、已完成或载荷不一致的任务均不能给当前承诺作证。"""
    task = deadline_task()
    if field.startswith('payload.'):
        task['payload'][field.partition('.')[2]] = value
    else:
        task[field] = value
    _, payload = review_payload(context_update={'active_participation_deadlines': [task]})
    assert not payload.get('active_participation_deadline_summaries')


def test_delegation_live_assets_are_real_group_scenarios_with_semantic_not_literal_expectations():
    """新资产可直接导入现有模拟入口，每项独立审查，固定业务边界而非老板原句。"""
    rows = [json.loads(line) for line in ASSET.read_text().splitlines() if line.strip()]
    assert len(rows) == len({row['case_id'] for row in rows}) == 8
    by_id = {row['case_id']: row for row in rows}
    for row in rows:
        validated = validate_scenario(row['scenario'])
        assert validated['messages']
        assert _semantic_criteria_ids(row['expectation'])
        assert not {'required_reply_contains', 'required_reply_any', 'reply_reference'} & set(row['expectation'])
        target = row['assertions']['target_message_index']
        assert validated['messages'][target]['channel'] == 'group'
        assert row['source']['kind'] in {'observed_failure_regression', 'adjacent_contrast'}
    for case_id in ('delegation_complete_empty', 'delegation_no_matching_friends_board',
                    'delegation_group_to_private_continuation', 'delegation_equivalent_wording'):
        checks = by_id[case_id]['assertions']
        assert checks['require_persisted_recruitment'] and checks['require_board_responsibility']
        assert 'create_invite_drafts' not in checks['required_tools']
    for case_id in ('delegation_query_contrast', 'delegation_inventory_only',
                    'delegation_existing_option_needs_choice', 'delegation_missing_party'):
        assert by_id[case_id]['assertions']['no_business_writes']
        assert by_id[case_id]['assertions']['requester_not_joined']
    continuation = by_id['delegation_group_to_private_continuation']
    assert len(continuation['scenario']['messages']) == 2
    assert continuation['assertions']['require_same_goal_continuation']
    assert by_id['delegation_existing_option_needs_choice']['setup']['games']
    assert by_id['delegation_complete_empty']['source']['trace_id'] == 'manual_group_08312f11fcfdca6b'
