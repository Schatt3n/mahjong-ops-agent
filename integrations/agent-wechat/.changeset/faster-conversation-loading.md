---
"@agent-wechat/cli": patch
"@agent-wechat/desktop": patch
---

Start loading recent messages immediately when opening a listed conversation. Cancel reads that are still waiting when switching chats, reduce repeated work when paging through history, and combine conversation list refreshes during bursts of incoming messages.

After sign-in, show the last open conversation from the hub’s saved messages before loading the full conversation list.
