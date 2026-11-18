---
"@agent-wechat/cli": patch
---

Reduce Hub catch-up stalls by reusing owned database connections when checking for absent contacts in managed profiles.
Reuse the open contact database when applying ordinary contact updates, avoiding a second directory scan and database open per write.
