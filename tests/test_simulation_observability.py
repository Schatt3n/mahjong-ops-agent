from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "simulation_observability.py"


def load_module():
    spec = importlib.util.spec_from_file_location("simulation_observability_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _turn(sequence: int, *, channel: str = "group") -> dict:
    return {
        "observed_at": f"2026-07-21T10:00:0{sequence}+00:00",
        "sequence": sequence,
        "conversation_id": "sim:group:g1" if channel == "group" else "sim:private:u1",
        "channel": channel,
        "event_type": "text",
        "user": {
            "customer_id": "u1",
            "display_name": "王哥",
            "text": "今晚有局吗",
            "generation": {
                "source": "glm",
                "model": "glm-4.7-flash",
                "trace_id": f"trace_gen_{sequence}",
                "latency_ms": 1200,
                "error": None,
            },
        },
        "agent": {
            "reply": "现在没有，要组吗？",
            "trace_id": f"trace_agent_{sequence}",
            "objective_status": "waiting_user",
        },
        "tool_calls": ["search_current_games"],
        "latency_ms": 2300,
        "status_code": 200,
        "error": "",
    }


def test_dashboard_payload_groups_conversations_and_never_exposes_api_key(tmp_path) -> None:
    module = load_module()
    runtime_dir = tmp_path / "periodic"
    run_dir = runtime_dir / "runs" / "20260721T100000Z_abcd1234"
    run_dir.mkdir(parents=True)
    transcript = [_turn(1), _turn(2, channel="private")]
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": "completed",
                "quality_status": "passed",
                "total_messages": 2,
                "group_messages": 1,
                "private_messages": 1,
                "message_generation_mode": "glm",
                "message_generation_model": "glm-4.7-flash",
                "transcript": transcript,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (runtime_dir / "state.json").write_text(
        json.dumps({"last_run_id": run_dir.name}),
        encoding="utf-8",
    )

    payload = module.simulation_observability_payload(runtime_dir=runtime_dir)

    assert payload["selected_run"]["turn_count"] == 2
    assert len(payload["selected_run"]["conversations"]) == 2
    assert payload["runs"][0]["message_generation_model"] == "glm-4.7-flash"
    assert payload["runs"][0]["quality_status"] == "passed"
    assert payload["selected_run"]["quality_status"] == "passed"
    assert "API_KEY" not in json.dumps(payload, ensure_ascii=False)
    assert payload["generator"]["api_key_exposed"] is False


def test_dashboard_reads_live_turns_before_report_is_finished(tmp_path) -> None:
    module = load_module()
    runtime_dir = tmp_path / "periodic"
    run_id = "20260721T100000Z_live1234"
    run_dir = runtime_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "live_events.jsonl").write_text(
        json.dumps({"event": "chat_turn", "run_id": run_id, **_turn(1)}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "state.json").write_text(
        json.dumps({"current_run_id": run_id, "status": "running"}),
        encoding="utf-8",
    )

    payload = module.simulation_observability_payload(runtime_dir=runtime_dir)

    assert payload["selected_run"]["status"] == "running"
    assert payload["selected_run"]["transcript"][0]["user"]["text"] == "今晚有局吗"


def test_control_only_accepts_allowlisted_actions(tmp_path) -> None:
    module = load_module()

    try:
        module.update_simulation_control("rm -rf /", runtime_dir=tmp_path)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("arbitrary controls must be rejected")

    result = module.update_simulation_control("pause", runtime_dir=tmp_path)
    assert result["ok"] is True
    assert result["control"]["enabled"] is False


def test_start_control_uses_real_agent_mode_from_private_env(tmp_path, monkeypatch) -> None:
    module = load_module()
    env_file = tmp_path / ".env.simulation.local"
    env_file.write_text(
        "MAHJONG_PERIODIC_SIM_LLM_MODE=real\n"
        "MAHJONG_PERIODIC_SIM_MESSAGE_MODE=glm\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout='{"ok": true, "accepted": true}', stderr="")

    monkeypatch.setattr(module, "DEFAULT_ENV_FILE", env_file)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.update_simulation_control("start", runtime_dir=tmp_path / "runtime")

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--mode") + 1] == "real"
    assert command[command.index("--message-mode") + 1] == "glm"
    assert command[command.index("--workers") + 1] == "1"
    assert result["ok"] is True


def test_run_now_dispatches_executable_work_instead_of_only_persisting_signal(tmp_path, monkeypatch) -> None:
    module = load_module()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='{"ok": true, "accepted": true, "execution_mode": "once"}', stderr='')

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    monkeypatch.setattr(module, 'DEFAULT_ENV_FILE', tmp_path / 'no-secrets')
    result = module.update_simulation_control('run_now', runtime_dir=tmp_path)
    assert result['ok'] is True
    assert result['execution_mode'] == 'once'
    assert calls[0][2] == 'run-now'
    assert calls[0][calls[0].index('--mode') + 1] == 'real'
    assert calls[0][calls[0].index('--message-mode') + 1] == 'model'
    assert not (tmp_path / 'control.json').exists()


def test_control_does_not_claim_success_without_worker_acknowledgment(tmp_path, monkeypatch) -> None:
    module = load_module()
    for stdout, code in [('{}', 0), ('not-json', 0), ('{"ok": false, "error": "启动失败"}', 1)]:
        monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=code, stdout=stdout, stderr=''))
        result = module.update_simulation_control('run_now', runtime_dir=tmp_path)
        assert result['ok'] is False


def _record_review_run(runtime_dir: Path) -> str:
    run_id = '20261025T120000Z_review1234'
    folder = runtime_dir / 'runs' / run_id
    folder.mkdir(parents=True)
    turns = []
    for sequence, sender, topic in ((1, 'u1', 't1'), (2, 'u2', 't2'), (3, 'u1', 't1'), (4, 'u2', 't2'), (5, 'u1', 't2')):
        turn = _turn(sequence)
        turn['thread_id'] = topic
        turn['user'].update(customer_id=sender, display_name=sender, text=f'客户第{sequence}句\n保留换行')
        turn['user']['generation']['context'] = {'scenario_text': '两场局交错询问', 'last_agent_reply': '几点能到？'}
        turn['agent']['reply'] = f'实际回复{sequence}'
        turn['agent']['actions'] = [{'goal': '确认空位', 'tool_calls': [{'name': 'search_current_games', 'arguments': {'limit': 2}}]}]
        turn['tool_results'] = [{'name': 'search_current_games', 'result': {'count': 2, 'api_key': 'must-not-export'}}]
        turn['state_transitions'] = [{'entity_id': 'test-game', 'from_status': 'forming', 'to_status': 'ready'}]
        turns.append(turn)
    (folder / 'report.json').write_text(json.dumps({'llm_mode': 'real', 'quality_status': 'passed', 'transcript': turns}, ensure_ascii=False), encoding='utf-8')
    return run_id


def test_topic_views_keep_interleaved_threads_and_cross_topic_users_separate(tmp_path) -> None:
    module = load_module()
    run_id = _record_review_run(tmp_path)
    selected = module.simulation_observability_payload(runtime_dir=tmp_path, selected_run_id=run_id)['selected_run']
    topics = {topic['thread_id']: topic for topic in selected['topics']}
    assert [turn['sequence'] for turn in topics['t1']['turns']] == [1, 3]
    assert [turn['sequence'] for turn in topics['t2']['turns']] == [2, 4, 5]
    assert {person['customer_id'] for person in topics['t2']['participants']} == {'u1', 'u2'}
    assert all(topic['label_source'] == 'simulator' for topic in selected['topics'])
    assert selected['review_summary']['topic_evaluation'] == 'not_evaluated'
    assert selected['review_summary']['business_evaluation'] == 'not_evaluated'
    users = {person['customer_id']: person for person in selected['participants']}
    assert [turn['sequence'] for turn in users['u1']['turns']] == [1, 3, 5]


def test_full_transcript_exports_preserve_every_turn_and_execution_evidence(tmp_path) -> None:
    module = load_module()
    run_id = _record_review_run(tmp_path)
    markdown, media_type, filename = module.simulation_transcript_export(run_id, runtime_dir=tmp_path)
    content = markdown.decode('utf-8')
    assert filename.endswith('.md') and media_type.startswith('text/markdown')
    for number in range(1, 6):
        assert f'客户第{number}句\n保留换行' in content
        assert f'实际回复{number}' in content
    assert '两场局交错询问' in content
    assert 'search_current_games' in content and 'from_status' in content
    assert 'must-not-export' not in content
    assert '待人工验收' in content
    data, _, filename = module.simulation_transcript_export(run_id, format='jsonl', runtime_dir=tmp_path)
    rows = [json.loads(line) for line in data.decode().splitlines()]
    assert [row['sequence'] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[-1]['agent']['actions'][0]['tool_calls'][0]['arguments'] == {'limit': 2}
    assert rows[-1]['tool_results'][0]['result']['api_key'] == '[REDACTED]'
    assert rows[-1]['simulation_run']['run_id'] == run_id
    assert rows[-1]['simulation_run']['llm_mode'] == 'real'
    assert rows[-1]['simulation_run']['topic_label_source'] == 'simulator'
    assert rows[-1]['simulation_run']['business_evaluation'] == 'not_evaluated'
    assert content.index('实际回复5') < content.index('## 逐轮执行证据')
    assert filename.endswith('.jsonl')


def test_human_reviews_are_append_only_and_cannot_invent_a_turn(tmp_path) -> None:
    module = load_module()
    run_id = _record_review_run(tmp_path)
    original = (tmp_path / 'runs' / run_id / 'report.json').read_bytes()
    module.save_simulation_review(run_id, 3, 'incorrect', '这条回复串到了另一场局', runtime_dir=tmp_path)
    module.save_simulation_review(run_id, 3, 'uncertain', '需要核对引用', runtime_dir=tmp_path)
    payload = module.simulation_observability_payload(runtime_dir=tmp_path, selected_run_id=run_id)
    assert payload['selected_run']['reviews']['3']['verdict'] == 'uncertain'
    assert payload['selected_run']['review_summary']['reviewed_turns'] == 1
    assert (tmp_path / 'runs' / run_id / 'report.json').read_bytes() == original
    assert len((tmp_path / 'runs' / run_id / 'reviews.jsonl').read_text().splitlines()) == 2
    import pytest
    with pytest.raises(ValueError, match='does not exist'):
        module.save_simulation_review(run_id, 99, 'incorrect', runtime_dir=tmp_path)
    with pytest.raises(ValueError):
        module.simulation_transcript_export('../report', runtime_dir=tmp_path)
    with pytest.raises(ValueError):
        module.simulation_transcript_export(run_id, format='html', runtime_dir=tmp_path)
