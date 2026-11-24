"""验证话题评测只并发独立会话，并按资产顺序汇总原有门禁结果。"""
import json
from pathlib import Path
import sys
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_conversation_topic_eval as runner


@pytest.mark.parametrize('second_status', ['passed', 'failed'])
def test_parallel_cases_keep_source_order_and_status(second_status, monkeypatch, tmp_path):
    cases = runner.load_cases()[:3]
    case_ids = [case['case_id'] for case in cases]
    work_dir = tmp_path / 'run'
    first_two_started = Barrier(2)
    third_started = Event()
    completion_lock = Lock()
    completion_order = []
    validated = []
    review_configs = []
    main_client = SimpleNamespace(config=object())
    review_client = object()

    def make_review_client(config):
        review_configs.append(config)
        return review_client

    def execute(case, actual_client, runtime_args, folder):
        case_id = case['case_id']
        index = case_ids.index(case_id)
        assert actual_client is main_client
        assert runtime_args.review_client is review_client
        assert folder == work_dir / case_id
        if index < 2:
            first_two_started.wait(timeout=5)
        if index == 0:
            assert third_started.wait(timeout=5)
        elif index == 2:
            third_started.set()
        report = {'case_id': case_id, 'status': second_status if index == 1 else 'passed'}
        folder.mkdir()
        (folder / 'report.json').write_text(json.dumps(report))
        with completion_lock:
            completion_order.append(case_id)
        return report

    monkeypatch.setattr(runner, 'load_cases', lambda: cases)
    monkeypatch.setattr(runner, 'load_dotenv_defaults', lambda path: None)
    monkeypatch.setattr(runner.OpenAICompatibleAgentClient, 'from_env', lambda: main_client)
    monkeypatch.setattr(runner, 'build_reply_review_client', make_review_client)
    monkeypatch.setattr(runner, 'run_case', execute)
    monkeypatch.setattr(runner, 'validate_scope', lambda payload, path: validated.append((payload, path)))
    monkeypatch.setattr(sys, 'argv', ['runner', '--work-dir', str(work_dir), '--workers', '2'])

    assert runner.main() == (0 if second_status == 'passed' else 1)
    assert completion_order[0] == case_ids[1]
    assert review_configs == [main_client.config]
    report_path = work_dir / 'report.json'
    payload = json.loads(report_path.read_text())
    assert [report['case_id'] for report in payload['reports']] == case_ids
    assert payload['status'] == second_status
    assert [json.loads((work_dir / case_id / 'report.json').read_text()) for case_id in case_ids] == payload['reports']
    assert validated == ([(payload, report_path)] if second_status == 'passed' else [])


@pytest.mark.parametrize('workers', ['0', '5'])
def test_workers_must_be_bounded(workers, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['runner', '--work-dir', str(tmp_path), '--workers', workers])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_validate_only_with_workers_does_not_initialize_model(monkeypatch, tmp_path, capsys):
    def unexpected_model():
        pytest.fail('validate-only initialized the model')

    work_dir = tmp_path / 'unused'
    monkeypatch.setattr(runner.OpenAICompatibleAgentClient, 'from_env', unexpected_model)
    monkeypatch.setattr(sys, 'argv', ['runner', '--work-dir', str(work_dir), '--workers', '4', '--validate-only'])
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out) == {'case_count': 9, 'turn_count': 29}
    assert not work_dir.exists()
