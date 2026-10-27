"""固定回归须按真实前置条件推进，客户续聊只依据已看见的消息。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, ToolResult


def _runner():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'run_operational_chat_eval.py'
    spec = importlib.util.spec_from_file_location('operational_scenarios_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_participation_case


def _result(reply, tools=()):
    return SimpleNamespace(final_reply=reply, status='waiting_user', tool_results=list(tools))


@pytest.mark.parametrize('problem', ['empty_search', 'different_game', 'failed_search', 'silent_reply'])
def test_missing_offer_stops_before_customer_confirmation(problem):
    store = InMemoryAgentStore()
    sent, generated, checks, events = [], [], {}, []
    def send(case, text, **kwargs):
        sent.append(text)
        game = next(iter(store.games.values()))
        matches = [] if problem=='empty_search' else [{'game':{'game_id':'unrelated' if problem=='different_game' else game.game_id}}]
        return _result('' if problem=='silent_reply' else '现在没有合适的，帮你留意着？',
                       [ToolResult(name='search_current_games', called=True, allowed=problem!='failed_search', result={'matches':matches})])
    def generate(intent, history):
        generated.append(intent)
        raise AssertionError('前提失败不应进入客户确认阶段')
    _runner()(store, send, generate, checks, events)
    assert len(sent) == 1 and generated == []
    assert checks['cross_conversation_offer_ready'] is False
    assert checks['cross_conversation_join_persisted'] is False
    assert events[0]['step'] == 'confirm_join' and events[0]['status'] == 'blocked'
    assert len(next(iter(store.games.values())).participants) == 1


def test_customer_answers_actual_question_then_confirms_and_withdraws():
    store = InMemoryAgentStore()
    sent, generated, checks, events = [], [], {}, []
    replies = ['有一桌国标1块三缺一，你大概多久到？', '那这桌给你留一个？', '好，给你留好了', '行']
    utterances = ['半小时能到', '可以，给我留一个', '临时有事，今天不打了']
    def send(case, text, generation=None):
        index = len(sent)
        sent.append(text)
        game = next(iter(store.games.values()))
        if index == 0:
            return _result(replies[index], [ToolResult(name='search_current_games', called=True, allowed=True,
                result={'matches':[{'game':{'game_id':game.game_id}}]})])
        assert generation['source'] == 'model'
        if index == 2:
            store.join_game(game_id=game.game_id,customer_id='operational_customer',display_name='客户',trace_id='join')
            return _result(replies[index],[ToolResult(name='join_game',called=True,allowed=True)])
        if index == 3:
            store.record_candidate_reply(game_id=game.game_id,customer_id='operational_customer',display_name='客户',status='declined',trace_id='withdraw')
        return _result(replies[index])
    def generate(intent, history):
        generated.append((intent, history))
        return {'text':utterances[len(generated)-1],'source':'model','wants_to_continue':True}
    _runner()(store, send, generate, checks, events)
    assert all(checks.values()) and events == []
    assert [intent for intent,_ in generated] == ['join','join','withdraw']
    assert [history[-1]['text'] for _,history in generated] == replies[:3]
    assert all(set(turn) == {'role','text'} for _,history in generated for turn in history)
    assert sent[1:] == utterances


@pytest.mark.parametrize('reason', ['generation_error','customer_stops','turn_limit'])
def test_unconfirmed_customer_never_receives_forced_withdrawal_step(reason):
    store = InMemoryAgentStore()
    generated, sent, checks, events = [], [], {}, []
    def send(case, text, **kwargs):
        sent.append(text)
        game = next(iter(store.games.values()))
        return _result('有这桌，你再想想？',[ToolResult(name='search_current_games',called=True,allowed=True,
            result={'matches':[{'game':{'game_id':game.game_id}}]})])
    def generate(intent, history):
        generated.append(intent)
        return {'text':'' if reason=='generation_error' else '我再想一下','source':'model',
                'error':'timeout' if reason=='generation_error' else None,'wants_to_continue':reason!='customer_stops'}
    _runner()(store, send, generate, checks, events)
    assert 'withdraw' not in generated
    assert len(generated) == (3 if reason=='turn_limit' else 1)
    assert checks['cross_conversation_join_persisted'] is False
    assert events and events[-1]['status']=='blocked'
