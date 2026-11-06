from mahjong_agent_runtime import SQLiteAgentStore, UserMessage
from mahjong_agent_runtime.tracing import JsonlTraceRecorder
from tests.simulation import sim_adapter


class BrokenProvider:
    def complete(self, messages, **kwargs):
        raise RuntimeError('synthetic provider timeout for evidence test')


def test_real_simulation_saves_provider_failure_outside_ui_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(sim_adapter.OpenAICompatibleAgentClient, 'from_env', lambda:BrokenProvider())
    store=SQLiteAgentStore(tmp_path/'simulation.db')
    runtime,_=sim_adapter.build_runtime(store,'real')
    result=runtime.handle_user_message(UserMessage(conversation_id='c',sender_id='a',sender_name='A',text='有局吗'),trace_id='provider-failure')
    assert result.status=='needs_human'
    persisted=JsonlTraceRecorder(tmp_path/'agent_trace.log').get_trace('provider-failure')
    assert any(event.step=='llm_error' and 'synthetic provider timeout' in str(event.content) for event in persisted)
    observation=sim_adapter.simulation_execution_observation(runtime,'provider-failure')
    assert observation['execution_failures']==[{'step':'llm_error','error_type':'RuntimeError'}]
    assert 'synthetic provider timeout' not in str(observation)
