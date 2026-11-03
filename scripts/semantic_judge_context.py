"""评测审查器的最小证据投影；只合并严格一致的冗余，完整原证据独立保留。"""
from copy import deepcopy
import json
from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract

PROJECTION_VERSION = "compact_v1_20261103"

# 只省略当前后端标准说明，非标准或有新限制的文字原样保留。
_STANDARD_CONTRACTS = [current_game_search_reply_contract(req, matches, alternatives)
    for req, matches, alternatives in (({}, [], []), ({"stake":"1"}, [], []),
        ({"stake":"1"}, [{"game":{}}], []), ({"stake":"1"}, [], [{"game":{}}]))]
_STANDARD_EXPLANATIONS = {key:{item[section][key] for item in _STANDARD_CONTRACTS}
    for section,key in (("evidence_scope","claim_boundary"),("search_result_semantics","instruction"))}


def same(a,b):
    """按精确JSON类型比较，布尔值不能冒充人数或金额。"""
    return json.dumps(a,sort_keys=True,ensure_ascii=False)==json.dumps(b,sort_keys=True,ensure_ascii=False)


def compact_requirement(requirement):
    """只合并与己方单一party完全相同的重复seat_claims，不猜人数或身份。"""
    result=deepcopy(requirement)
    if isinstance(result,dict) and 'requesting_party' in result and same(result.get('seat_claims'),[result['requesting_party']]):
        result.pop('seat_claims')
    return result


def compact_game(game):
    """保留局状态、时间、人数和人员；仅去掉与parties严格相同的占座副本。"""
    result=deepcopy(game)
    if not isinstance(result,dict):return result
    parties=result.get('parties')
    if parties is not None and same(result.get('seat_claims'),parties):result.pop('seat_claims',None)
    req=result.get('requirement')
    if isinstance(req,dict):
        if parties is not None and same(req.get('seat_claims'),parties):req.pop('seat_claims',None)
        if isinstance(parties,list) and len(parties)==1 and same(req.get('requesting_party'),parties[0]):req.pop('requesting_party',None)
        result['requirement']=compact_requirement(req)
    return result


def project(original):
    """只读投影，不生成结论、不放宽规则；完整输入由判定报告独立保留。"""
    result=deepcopy(original)
    # 请求保留身份、引用及metadata中的渠道/授权线索，不把审计裁剪扩展为语义删减。
    if isinstance(result.get('recent_conversation'),list):
        result['recent_conversation']=[{k:v for k,v in row.items() if k!='trace_id'}
            if isinstance(row,dict) else row for row in result['recent_conversation']]
    for tool in result.get('tool_results',[]):
        if not isinstance(tool,dict):continue
        tool.pop('idempotency_key',None)
        if 'requirement' in tool:tool['requirement']=compact_requirement(tool['requirement'])
        if 'game' in tool:tool['game']=compact_game(tool['game'])
        contract=tool.get('customer_reply_contract')
        if isinstance(contract,dict):
            # 只移除解释性长文；实际范围、匹配数量、空池及授权标志原样保留。
            for value in contract.values():
                if isinstance(value,dict):
                    for key, standards in _STANDARD_EXPLANATIONS.items():
                        if isinstance(value.get(key),str) and value[key] in standards:value.pop(key)
    state=result.get('persisted_state')
    if isinstance(state,dict):
        games=state.get('games',[]);related=state.get('related_active_games',[])
        if ('games' in state and 'related_active_games' in state and isinstance(games,list)
                and isinstance(related,list) and all(isinstance(r,dict) and isinstance(r.get('game_id'),str)
                and any(same(g,r) for g in games) for r in related)):
            state['related_active_game_ids']=[r['game_id'] for r in related]
            state.pop('related_active_games',None)
        if 'games' in state and isinstance(games,list):state['games']=[compact_game(g) for g in games]
        if 'related_active_games' in state:state['related_active_games']=[compact_game(g) for g in state['related_active_games']]
    return result
