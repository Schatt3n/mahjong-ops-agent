"""校准命令入口必须显式加载项目环境；离线命令连线不冒充真实模型判定。"""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_reply_scope_calibration as scope


def test_cli_loads_project_dotenv_and_writes_report_without_calling_real_provider(tmp_path, monkeypatch):
    """覆盖真实入口及环境函数签名，远端调用另由受控校准和完整门禁验证。"""
    (tmp_path / '.env').write_text('MAHJONG_LLM_MODEL=scope-cli-test-model\n')
    monkeypatch.setattr(scope, 'ROOT', tmp_path)
    monkeypatch.delenv('MAHJONG_LLM_MODEL', raising=False)
    report = tmp_path / 'out' / 'report.json'
    monkeypatch.setattr('sys.argv', ['run_reply_scope_calibration.py', '--strict', '--report-path', str(report)])
    client = SimpleNamespace()

    def configured_client():
        assert os.environ['MAHJONG_LLM_MODEL'] == 'scope-cli-test-model'
        return client

    def controlled_run(received_client, args, work_dir):
        assert received_client is client and args.strict is True
        assert work_dir.parent == report.parent
        return {'status': 'passed', 'main_agent_executed': False}

    monkeypatch.setattr(scope.OpenAICompatibleAgentClient, 'from_env', staticmethod(configured_client))
    monkeypatch.setattr(scope, 'run_calibration', controlled_run)
    assert scope.main() == 0
    assert json.loads(report.read_text()) == {'status': 'passed', 'main_agent_executed': False}


@pytest.mark.parametrize('inherited,explicit,expected',[(None,None,75),('90',None,90),('90','12',12)])
def test_cli_timeout_uses_shared_environment_with_explicit_argument_precedence(tmp_path,monkeypatch,inherited,explicit,expected):
    """完整门禁继承的90秒和项目默认均可生效，显式调用参数仍优先，不调用模型。"""
    (tmp_path/'.env').write_text('MAHJONG_AGENT_LLM_TIMEOUT_SECONDS=75\n')
    monkeypatch.setattr(scope,'ROOT',tmp_path)
    monkeypatch.delenv('MAHJONG_AGENT_LLM_TIMEOUT_SECONDS',raising=False)
    if inherited is not None:monkeypatch.setenv('MAHJONG_AGENT_LLM_TIMEOUT_SECONDS',inherited)
    report=tmp_path/'report.json';argv=['scope','--report-path',str(report)]
    if explicit is not None:argv+=['--timeout-seconds',explicit]
    monkeypatch.setattr('sys.argv',argv)
    monkeypatch.setattr(scope.OpenAICompatibleAgentClient,'from_env',staticmethod(lambda:SimpleNamespace()))
    def controlled(client,args,work_dir):
        assert args.timeout_seconds==expected
        return {'status':'passed'}
    monkeypatch.setattr(scope,'run_calibration',controlled)
    assert scope.main()==0
