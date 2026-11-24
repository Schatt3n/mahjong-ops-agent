"""以确定性超时和回放替身验证完整门禁的同轮补试与证据边界。"""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_evals as gate
import eval_retry


FAILURE = {'stage': 'agent', 'category': 'unattributed_timeout',
           'retryable': True, 'reason': 'timeout_reading_response_body'}


@pytest.fixture
def bound(tmp_path, monkeypatch):
    """无等待地执行沙盒补试，保留每次实际目录。"""
    scope = tmp_path / 'checkpoint'
    scope.mkdir()
    monkeypatch.setenv(eval_retry.SCOPE_ENV, str(scope))
    monkeypatch.setenv(eval_retry.WINDOW_ENV, '0')
    monkeypatch.setattr(eval_retry, 'sleep', lambda _: None)
    return scope


def test_case_retries_at_most_three_times_and_keeps_attempts(bound, tmp_path, monkeypatch):
    paths = []

    def run(path):
        assert not path.exists()
        paths.append(path)
        return {'status': 'blocked', 'eval_inconclusive': FAILURE}

    assert eval_retry.checkpointed_case('case', run, case_dir=tmp_path)['status'] == 'blocked'
    assert len(set(paths)) == 4
    assert all((p / 'eval_result.json').is_file() for p in paths)
    assert eval_retry.checkpointed_case('case', run, case_dir=tmp_path)['status'] == 'blocked'
    assert len(paths) == 4
    monkeypatch.setenv(eval_retry.WINDOW_ENV, '1')
    assert eval_retry.checkpointed_case('case', lambda _: {'status': 'passed'}, case_dir=tmp_path)['status'] == 'passed'
    assert eval_retry.checkpointed_case('case', lambda _: pytest.fail('replayed a passed case'), case_dir=tmp_path)['status'] == 'passed'


def test_budget_uses_one_deadline_for_judge_and_agent(bound, monkeypatch):
    clock = [50.0]
    monkeypatch.setattr(eval_retry, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(eval_retry, 'sleep', lambda duration: clock.__setitem__(0, clock[0] + duration))
    budget = eval_retry.CaseBudget(bound, window=0)
    token = eval_retry.external_call_deadline.set(None)
    try:
        assert budget.begin_retry(FAILURE, stage='agent')
        assert eval_retry.external_call_deadline.get() == 410
        assert budget.limit_timeout(999) == 359
        clock[0] = 408
        assert not budget.begin_retry(FAILURE, stage='judge')  # 退避耗尽同一个窗口。
        assert budget.remaining_seconds() == 0
        assert not budget.begin_retry(FAILURE, stage='agent')
    finally:
        eval_retry.external_call_deadline.reset(token)


def test_valid_failure_is_frozen_after_eligible_retry(bound, tmp_path):
    results = iter([{'status': 'blocked', 'eval_inconclusive': FAILURE}, {'status': 'failed', 'checks': {'write': False}}])
    result = eval_retry.checkpointed_case('case', lambda _: next(results), case_dir=tmp_path)
    assert result['status'] == 'failed'
    assert eval_retry.checkpointed_case('case', lambda _: pytest.fail('replayed definitive failure'), case_dir=tmp_path) == result


def test_completed_registered_read_before_transport_failure_can_retry():
    boundary = {'tool_name': 'search_current_games', 'step_index': 1, 'idempotency_key': 'read-1'}
    events = [
        {'step': 'tool_gateway_received', 'content': boundary},
        {'step': 'tool_gateway_completed', 'content': {**boundary, 'called': True, 'allowed': True}},
        {'step': 'tool_result', 'content': {'name': 'search_current_games', 'idempotency_key': 'read-1',
                                          'called': True, 'allowed': True}},
        {'step': 'llm_error', 'content': {'completion_diagnostics': {'schema_version': 1, 'route': 'primary',
            'error_type': 'RuntimeError', 'attempts': [{'failure_reason': 'timeout_reading_response_body'}]}}},
    ]
    assert eval_retry.latest_agent_transport_failure(events, read_only_tools={'search_current_games'}) == FAILURE
    assert eval_retry.latest_agent_transport_failure(events) is None
    assert eval_retry.latest_agent_transport_failure([events[0], events[-1]], read_only_tools={'search_current_games'}) is None
    assert eval_retry.agent_failure_diagnosis([events[0], events[-1]])['category'] == 'local_tool'


def test_judge_only_continuation_does_not_repeat_agent(bound, tmp_path, monkeypatch):
    result = {'status': 'blocked', 'eval_inconclusive': {**FAILURE, 'stage': 'judge'}, 'frozen': {'reply': '原回复'}}
    eval_retry.checkpointed_case('case', lambda _: result, case_dir=tmp_path)
    monkeypatch.setenv(eval_retry.WINDOW_ENV, '1')

    def resume(prior, path):
        assert not path.exists()
        assert prior['frozen'] == result['frozen']
        assert eval_retry.current_budget().begin_retry(FAILURE, stage='judge')
        return {'status': 'passed'}

    assert eval_retry.checkpointed_case('case', lambda _: pytest.fail('restarted Agent'), case_dir=tmp_path,
                                      resume_judge=resume)['status'] == 'passed'


def test_owner_money_continuation_preserves_valid_semantic_and_original_trace(bound, tmp_path, monkeypatch):
    import run_real_owner_chat_live_eval as owner

    failure = {**FAILURE, 'stage': 'judge'}
    source = tmp_path / 'source.trace.json'
    price = {'blocks': True, 'status': 'error', 'scope_assessment': {'eval_inconclusive': failure}}
    semantic = {'passed': True, 'error': None, 'evidence': {'reply': '一块'}}
    source.write_text(json.dumps({'events': [{'step': 'reply_money_fact_check', 'content': price}]}))
    raw = source.read_bytes()
    report = {'trace_id': 'owner-case', 'runtime_status': 'completed', 'final_reply': '一块',
              'raw_trace_path': str(source), 'semantic_assessment': semantic, 'money_fact_check': price,
              'checks': [{'name': name, 'passed': False} for name in ('reply_should_match_expected_meaning',
                  'action_contract_errors_should_be_safely_recovered', 'reply_money_fact_check_should_match_original_evidence',
                  'reply_money_facts_should_not_conflict')]}
    valid_price = {'blocks': False, 'status': 'verified', 'coverage': 'matched'}

    def resume_price(client, *, reply, tool_results, trace_id, timeout_seconds, frozen_check):
        assert reply == '一块' and frozen_check == price
        assert eval_retry.current_budget().begin_retry(failure, stage='judge')
        return valid_price

    monkeypatch.setattr(owner, 'check_reply_money_facts', resume_price)
    monkeypatch.setattr(owner, 'validate_money_fact_check', lambda *args, **kwargs: True)
    monkeypatch.setattr(owner, 'resume_reply_semantics', lambda *args: pytest.fail('repeated the valid semantic verdict'))
    attempt = eval_retry.checkpointed_case('owner-case', lambda _: owner.owner_judge_disposition(report, [], semantic), case_dir=tmp_path)
    assert attempt['status'] == 'blocked'
    monkeypatch.setenv(eval_retry.WINDOW_ENV, '1')
    final = eval_retry.checkpointed_case('owner-case', lambda _: pytest.fail('repeated Agent'), case_dir=tmp_path,
        resume_judge=lambda prior, _: owner.resume_owner_judge(None, prior))
    assert final['status'] == 'passed' and final['semantic_assessment'] == semantic
    assert source.read_bytes() == raw
    assert json.loads(Path(final['raw_trace_path']).read_text())['events'][0]['content'] == valid_price


@pytest.fixture
def isolated_gate(tmp_path, monkeypatch):
    """小型完整门禁保留真实同轮报告和用例账本，用替身隔离业务。"""
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    asset = tmp_path / 'eval' / 'source.json'
    asset.parent.mkdir()
    asset.write_text('{}')
    scope = gate.LIVE_SCOPES[0]
    preliminary = 'runtime_boundary'
    final = 'later_scope'
    monkeypatch.setattr(gate, 'build_eval_plan', lambda args, folder: [
        {'scope': name, 'command': [name, str(args.live_case_workers), str(args.collect_all)] if name == scope else [name],
         'status': 'pending'} for name in (preliminary, scope, final)])
    monkeypatch.setattr(eval_retry, 'sleep', lambda _: None)
    invoked = []

    def run(command, *, log_path, retry_scope, retry_window):
        invoked.append((command[0], retry_window))
        log_path.write_text(f'window {retry_window}\n')
        if retry_scope is None:
            return
        with monkeypatch.context() as env:
            env.setenv(eval_retry.SCOPE_ENV, str(retry_scope))
            env.setenv(eval_retry.WINDOW_ENV, str(retry_window))
            result = eval_retry.checkpointed_case('test', lambda _: {'status': 'passed'} if retry_window else
                {'status': 'blocked', 'eval_inconclusive': FAILURE}, case_dir=tmp_path)
            if result['status'] == 'blocked':
                raise subprocess.CalledProcessError(eval_retry.BLOCKED_EXIT_CODE, command)

    monkeypatch.setattr(gate, 'run_command', run)
    monkeypatch.setattr(sys, 'argv', ['run_evals.py', '--full', '--live-case-workers', '3', '--collect-all'])
    assert gate.main() == 75
    report_file, = tmp_path.glob('runtime_data/eval_runs/*/gate_report.json')
    return report_file, asset, invoked


def test_resume_reuses_passed_scopes_and_keeps_original_report(isolated_gate, monkeypatch):
    report_file, _, invoked = isolated_gate
    initial = json.loads(report_file.read_text())
    assert initial['status'] == 'blocked' and not initial['failed_scopes']
    monkeypatch.setattr(sys, 'argv', ['run_evals.py', '--resume', initial['run_id']])
    assert gate.main() == 0
    final = json.loads(report_file.read_text())
    assert final['run_id'] == initial['run_id'] and final['status'] == 'passed'
    assert final['live_case_workers'] == initial['live_case_workers'] == 3
    assert final['failure_policy'] == initial['failure_policy'] == 'collect_all'
    assert len([step for step, _ in invoked if step == 'runtime_boundary']) == 1
    assert json.loads(report_file.with_name('gate_report.window_000.json').read_text()) == initial
    assert Path(initial['scopes'][1]['log_path']).read_text() == 'window 0\n'


@pytest.mark.parametrize('tamper', ['source', 'configuration', 'evidence'])
def test_resume_rejects_changed_candidate_or_evidence(isolated_gate, monkeypatch, tamper):
    report_file, asset, invoked = isolated_gate
    initial = json.loads(report_file.read_text())
    if tamper == 'source':
        asset.write_text('{"changed": true}')
    elif tamper == 'configuration':
        monkeypatch.setenv('MAHJONG_LLM_MODEL', 'different-model')
    else:
        Path(initial['scopes'][0]['log_path']).write_text('rewritten')
    monkeypatch.setattr(sys, 'argv', ['run_evals.py', '--resume', initial['run_id']])
    assert gate.main() == 1
    assert len(invoked) == 2
    assert json.loads(report_file.read_text()) == initial
