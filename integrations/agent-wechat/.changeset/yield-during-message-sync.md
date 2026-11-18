---
"@agent-wechat/cli": patch
"@agent-wechat/desktop": patch
---

Let queued chat reads and sends proceed between small groups of incoming sync commands. Keep the received batch saved so an interrupted sync can resume without replaying completed work.
