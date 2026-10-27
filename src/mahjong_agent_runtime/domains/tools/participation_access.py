"""以服务端业务关系和查询证据授权本人参与，绝不共享其他会话原文。"""

from __future__ import annotations

import json

from ...models import Game
from ...stores import AgentStore


def participation_access_source(store: AgentStore, game: Game, *, conversation_id: str, sender_id: str) -> str | None:
    """返回本人参与权限的依据；此权限不允许改局条件、取消局或代别人确认。"""
    if game.conversation_id == conversation_id:
        return "originating_conversation"
    if game.organizer_id == sender_id or any(item.customer_id == sender_id for item in game.participants):
        return "own_participation"
    if any(draft.game_id == game.game_id and draft.customer_id == sender_id for draft in store.invite_drafts.values()):
        return "own_invitation"
    if any(draft.recipient_id == sender_id and draft.purpose == "waiting_match_notification"
           and draft.metadata.get("game_id") == game.game_id for draft in store.outbound_message_drafts.values()):
        return "own_waiting_notification"
    if _discovered_in_current_task(store, game.game_id, conversation_id=conversation_id, sender_id=sender_id):
        return "current_task_search_result"
    return None


def _discovered_in_current_task(store: AgentStore, game_id: str, *, conversation_id: str, sender_id: str) -> bool:
    """只信本人当前任务里成功查询的持久化工具结果，用户文本和摘要不是授权凭证。

    查询轮次即服务端可审计的发现证据；使用原始工具轮而非模型 checkpoint，因此重启或
    摘要压缩不会凭空授予权限。窗口外或已撤回的证据须重新查询，不能靠猜 game_id 加入。
    """
    context = store.current_task_context(conversation_id, sender_id)
    turns = (store.task_context_turns(conversation_id, context.task_context_id, limit=60)
             if context else store.recent_turns(conversation_id, limit=60))
    own_traces = {turn.trace_id for turn in turns if turn.role.value == "user" and turn.sender_id == sender_id
                  and not turn.metadata.get("context_invalidated")}
    for turn in turns:
        if turn.role.value != "tool" or turn.trace_id not in own_traces or turn.metadata.get("context_invalidated"):
            continue
        try:
            results = json.loads(turn.content)
        except (TypeError, ValueError):
            continue
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict) or result.get("name") != "search_current_games" or result.get("called") is not True or result.get("allowed") is not True or result.get("error"):
                continue
            payload = result.get("result")
            if not isinstance(payload, dict):
                continue
            for field in ("matches", "alternatives"):
                matches = payload.get(field)
                if isinstance(matches, list) and any(isinstance(match, dict) and isinstance(match.get("game"), dict)
                                                    and match["game"].get("game_id") == game_id for match in matches):
                    return True
    return False
