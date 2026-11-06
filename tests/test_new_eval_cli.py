"""两个新增付费入口的离线启动合同：校验env路径、配置装配及完整来源循环。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_conversation_topic_eval as topic
import run_empty_pool_calibration as empty


@pytest.mark.parametrize('runner,count',[(topic,9),(empty,6)])
def test_non_validate_cli_loads_explicit_environment_and_executes_all_declared_sources(runner,count,monkeypatch,tmp_path):
    """只替换已另测的执行边界；main本身读取全部真实资产、建目录、保存报告并调用验真。"""
    seen=[];checked=[];env=[];client=object()
    monkeypatch.setattr(runner,'load_dotenv_defaults',lambda path:env.append(path))
    monkeypatch.setattr(runner.OpenAICompatibleAgentClient,'from_env',lambda:client)
    def execute(case, actual_client, *args):
        assert actual_client is client;seen.append(case['case_id'])
        folder=args[-1];assert folder.parent==tmp_path/'run'
        return {'case_id':case['case_id'],'status':'passed'}
    monkeypatch.setattr(runner,'run_case',execute)
    monkeypatch.setattr(runner,'validate_scope',lambda payload,path:checked.append((payload,path)))
    monkeypatch.setattr(sys,'argv',['runner','--work-dir',str(tmp_path/'run')])
    assert runner.main()==0
    assert env==[runner.ROOT/'.env'] and len(seen)==count
    assert seen==[case['case_id'] for case in runner.load_cases()]
    assert len(checked)==1
    payload=json.loads((tmp_path/'run/report.json').read_text())
    assert [case['case_id'] for case in payload['reports']]==seen


@pytest.mark.parametrize('runner',[topic,empty])
def test_non_validate_cli_rejects_missing_model_configuration_before_execution(runner,monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'load_dotenv_defaults',lambda path:None)
    monkeypatch.setattr(runner.OpenAICompatibleAgentClient,'from_env',lambda:None)
    monkeypatch.setattr(sys,'argv',['runner','--work-dir',str(tmp_path/'run')])
    with pytest.raises(ValueError,match='configuration'):runner.main()
    assert not (tmp_path/'run').exists()
