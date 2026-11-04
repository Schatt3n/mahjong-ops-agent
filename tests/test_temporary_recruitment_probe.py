"""先验证探针自身：双Store夹具、逻辑时间、互斥状态及无模型预写动作。"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_temporary_recruitment_probe as probe
from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.models import now


@pytest.mark.parametrize('kind',['memory','sqlite'])
@pytest.mark.parametrize('stage',sorted(probe.STAGES))
def test_real_fixture_states_use_business_clock_and_actual_store_transactions(tmp_path,kind,stage):
    source=probe.load_dataset();store=InMemoryAgentStore() if kind=='memory' else SQLiteAgentStore(tmp_path/(stage+'.sqlite3'))
    try:
        with probe.BusinessClock(source['logical_date']) as clock:
            ids=probe.setup_fixture(store,source,stage,clock);state=probe.snapshot(store)
            assert now().isoformat()=='2026-11-03T18:50:00+08:00'
            original=store.require_game(ids['original'])
            if stage=='original_ready':assert original.status=='ready' and original.remaining_seats()==0
            elif stage=='after_d_cancelled':assert original.status=='forming' and original.remaining_seats()==1
            elif stage=='parallel_forming':
                assert original.status=='forming' and store.require_game(ids['late']).status=='forming'
                assert probe.members(store.require_game(ids['early']).to_dict())==sorted(probe.cid(a) for a in 'ABC')
                assert probe.members(store.require_game(ids['late']).to_dict())==sorted(probe.cid(a) for a in 'ABE')
            else:
                checks=probe.evaluate_state(state,{'plans':[{'clock':'19:30','planned_start_at':'2026-11-03T19:30:00+08:00','planned_end_at':'2026-11-03T23:30:00+08:00','members':list('ABCF'),'allowed_statuses':['ready'],'remaining':0},
                    {'clock':'20:00','planned_start_at':'2026-11-03T20:00:00+08:00','planned_end_at':'2026-11-04T00:00:00+08:00','members':['E'],'allowed_statuses':['forming','inviting'],'remaining':3}]})
                assert all(c['passed'] for c in checks),checks
            assert len(state['games'])==(2 if stage in {'parallel_forming','early_committed'} else 1)
    finally:
        if kind=='sqlite':store._connection.close()
    assert abs((now()-datetime.now().astimezone()).total_seconds())<5


def test_source_has_ten_fixed_cases_and_only_recruitment_inputs_not_scoring_labels():
    source=probe.load_dataset();assert sum(len(c['turns']) for c in source['cases'])==12
    for case in source['cases']:
        assert case['expected_final'] and case['required_meaning']
        for turn in case['turns']:
            assert set(turn)=={'actor','text','at'}
    assert source['execution_boundary']['runtime_changes_allowed'] is False


def test_oracle_does_not_accept_candidate_as_confirmed_or_double_ready_commitment():
    state={'games':[{'game_id':'g','status':'forming','planned_start_at':'2026-11-03T19:30:00+08:00','planned_end_at':'2026-11-03T23:30:00+08:00','requirement':{},
        'participants':[{'customer_id':probe.cid(a),'status':'joined','seat_count':1} for a in 'ABC']+
            [{'customer_id':probe.cid('F'),'status':'negotiating','seat_count':1}],'remaining_seats':1}]}
    assert all(r['passed'] for r in probe.evaluate_state(state,{'not_counted':['F']}))
    assert not all(r['passed'] for r in probe.evaluate_state(state,{'plans':[{'clock':'19:30','planned_start_at':'2026-11-03T19:30:00+08:00','planned_end_at':'2026-11-03T23:30:00+08:00','members':list('ABCF'),'allowed_statuses':['ready'],'remaining':0}]}))
    other=deepcopy(state['games'][0]);other['game_id']='other';other['status']='ready';state['games'][0]['status']='ready';state['games'].append(other)
    assert any(not r['passed'] for r in probe.evaluate_state(state,{}))


@pytest.mark.parametrize('kind',['memory','sqlite'])
@pytest.mark.parametrize('actor,stage',[('E','after_d_cancelled'),('F','parallel_forming'),('G','early_committed')])
def test_fixture_discovery_uses_real_gateway_and_correct_task_context_signature(tmp_path,kind,actor,stage):
    source=probe.load_dataset();store=InMemoryAgentStore() if kind=='memory' else SQLiteAgentStore(tmp_path/(actor+'.sqlite3'))
    try:
        with probe.BusinessClock(source['logical_date']) as clock:
            probe.setup_fixture(store,source,stage,clock);trace=probe.InMemoryTraceRecorder()
            before=deepcopy([g.to_dict() for g in store.games.values()])
            probe.discover_for_actor(store,actor,clock,trace)
            after=[g.to_dict() for g in store.games.values()]
            # Memory在读取时可把READY局的招募派生状态刷新为completed；不允许改实际参与、时刻或生命周期。
            for old,new in zip(before,after):
                if old['recruitment_status']!=new['recruitment_status']:
                    assert old['status']==new['status']=='ready' and new['recruitment_status']=='completed'
                    old['recruitment_status']=new['recruitment_status']
                    old['requirement']['recruitment_status']=new['requirement']['recruitment_status']
            assert after==before
            assert store.current_task_context(probe.conversation(actor),probe.cid(actor)) is not None
    finally:
        if kind=='sqlite':store._connection.close()


def test_recruiting_phase_does_not_confuse_inviting_with_ready():
    state={'games':[{'game_id':'g','status':'inviting','planned_start_at':'2026-11-03T19:30:00+08:00','planned_end_at':'2026-11-03T23:30:00+08:00','requirement':{},
        'participants':[{'customer_id':probe.cid(a),'status':'joined','seat_count':1} for a in 'ABC'],'remaining_seats':1}]}
    checks=probe.evaluate_state(state,{'plans':[{'clock':'19:30','planned_start_at':'2026-11-03T19:30:00+08:00','planned_end_at':'2026-11-03T23:30:00+08:00','members':list('ABC'),'allowed_statuses':['forming','inviting'],'remaining':1}]})
    assert all(c['passed'] for c in checks)
    assert not all(c['passed'] for c in probe.evaluate_state(state,{'plans':[{'clock':'19:30','planned_start_at':'2026-11-03T19:30:00+08:00','planned_end_at':'2026-11-03T23:30:00+08:00','members':list('ABCF'),'allowed_statuses':['ready'],'remaining':0}]}))


def test_board_obligation_must_cover_each_active_option():
    state={'games':[{'game_id':identity,'status':'forming','participants':[]} for identity in ['early','late']],
           'board_arrangements':{'early':{'status':'scheduled','pending_task_count':1,'outbound_authorized_room_count':1}}}
    assert not all(c['passed'] for c in probe.evaluate_state(state,{'recruitment_progress':True}))
    state['board_arrangements']['late']=deepcopy(state['board_arrangements']['early'])
    assert all(c['passed'] for c in probe.evaluate_state(state,{'recruitment_progress':True}))
