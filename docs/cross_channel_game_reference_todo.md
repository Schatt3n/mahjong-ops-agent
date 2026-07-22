# 群聊看板、私聊承接与跨通道局关联

## 状态

**核心领域能力已经实现；真实微信自动外发仍处于白名单灰度，默认关闭。**

当前实现覆盖：托管群三层分流、群友明确报局导入、不可变看板快照、编号认领、好友私聊优先、非好友最小群内通知、私聊切换上下文、原子占位、并发抢位和局状态驱动的看板刷新。

此外，仓库已增加一条尚未接入真实群入口的 Session 化候选链路，用于验证同群多话题并发、多人续接和条件结晶。它将 `BoardState` 持久化到 SQLite，但 `ChatSession` 暂存内存；目前只运行确定性测试与真实 DeepSeek 模拟，不修改只读观察群状态，也不自动外发。

它解决的是“业务引用与状态如何正确关联”，不代表个人微信协议、自动拉群或好友申请已经具备正式生产 SLA。

## 业务原则

1. **群聊是公告板，不是多人共享的私聊会话。** 老板主要在群里维护当前缺人局，复杂确认优先转到私聊。
2. **通道由后端选择。** 模型决定业务动作和说什么，后端根据好友关系、房间策略和外发权限决定在哪说。
3. **Game 是事实，Board 是投影。** 看板编号可以变化，`game_id` 不变；编号必须通过具体看板版本解析。
4. **跨通道共享业务标识，不共享原始会话。** 群聊与私聊通过 `customer_id + game_id + ChannelSwitch/GameConversationLink` 关联，不能互相复制原始聊天。
5. **认领必须确认且原子落库。** 看到局、表达兴趣与已占座是不同状态；最后一个座位不能靠模型判断并发结果。
6. **公开信息最小化。** 群聊和跨客户通知不得带微信备注、其他参与者私聊、画像、关系冲突原文或内部字段。

## 三层入口

```text
WeChaty room message
  -> managed room allowlist
  -> L1RuleEngine
       explicit game post -> import Game + schedule board
       numbered claim     -> ClaimHandler
       trivial/self       -> ignore
       otherwise          -> L2IntentRouter
  -> L2IntentRouter
       simple query       -> isolated group AgentLoop + short constraints
       complex + friend   -> private switch
       complex + no DM    -> isolated group AgentLoop
       irrelevant         -> ignore
```

### L1 的边界

L1 只处理稳定协议，不承担开放式语义理解：

- 明确报局，例如 `14:00 0.5 无烟 371`。
- 明确编号认领，例如 `2来`、`2我打`。
- 机器人自己的消息和纯噪声。

“川麻那个我可以”“刚才无烟那个算我一个”等没有唯一引用的表达不靠新正则硬猜。它们需要进入主 Agent，基于当前公开看板召回候选局；存在多个候选时追问。

## 身份、会话与局的建模

| 对象 | 作用 |
| --- | --- |
| `ChannelIdentity` | 将 `(channel, external_user_id)` 映射到稳定 `customer_id`、公开昵称、私聊会话和好友能力 |
| `conversation_id` | 消息顺序与隐私边界；群内每位客户也有独立会话 |
| `game_id` | 组局聚合标识，可以被多个隔离会话共同引用 |
| `GameConversationLink` | 只记录某个 Game 与房间/私聊任务的业务关联，不保存跨会话原文 |
| `ChannelSwitch` | 10 分钟有效的群转私聊指针，用于续接当前用户自己的请求 |

群内 L3 会话使用：

```text
group:<room_id>:customer:<customer_id>
```

这意味着同一房间内 A 的最近对话不会进入 B 的上下文。B 能看到的局信息来自 `public_group_game_summary()` 公开投影，而不是 A 的原始消息。

## 看板是不可变版本映射

`BoardEngine` 根据与房间关联的活跃、未满 Game 生成新看板：

```text
当前缺人局：
1、13:00 杭麻 0.5 无烟 272
2、人齐开 川麻 1-32 有烟 371

回复编号即可认领，如"2来"
```

每次成功发送后持久化：

```text
BoardSnapshot(snapshot_id, room_id, external_message_id, rendered_text, created_at)
BoardItem(snapshot_id, item_no, game_id, rendered_text)
```

解析策略：

1. 有引用消息 ID：精确读取被引用的 `BoardSnapshot`。
2. 无引用：只读取该群最新版快照。
3. 引用不存在、编号失效、局已满/取消/过期：拒绝认领。

设计明确不对“无引用旧编号”回退上一版。看板重新排序后，旧版 2 号可能已经变成另一局；猜测比追问危险。

## 看板刷新

Game 状态是权威事实源。以下两条链路触发派生看板：

- 群友明确报局：`BoardEngine.import_game_from_post()` 创建 Game、写 `GameConversationLink`、排队刷新。
- 主 Agent 工具修改 Game：`GroupBoardTrigger` 监听 `after_tool_execute` Hook，成功后查找房间关联并创建持久化 `publish_group_board` 任务。

普通创建/条件修改在默认 30 秒窗口内合并，避免连续刷屏；认领、释放和状态变化立即调度。任务使用现有 `ScheduledAgentTask` lease/retry/restart recovery 机制，通道暂时失败时不会把 Game 状态回滚成旧值。

## 群转私聊

当群友提出复杂请求且 `ChannelIdentity.can_private_message=true` 时：

1. 群内只回复“私聊回你”等最短确认。
2. 写入 `ChannelSwitch`，记录该客户、来源房间、来源消息和私聊会话。
3. 构造 `PrivateSwitchContext`：只含该用户原文、已解析需求、画像摘要、缺失字段和回复约束。
4. 通过 `handle_system_trigger()` 在其私聊 `conversation_id` 运行同一个主 Agent。
5. 若私聊中创建 Game，`GroupBoardTrigger` 会依据活动 `ChannelSwitch` 建立房间关联，后续自动进入看板。

群里其他人的消息不会被复制到私聊。用户在短时间内把业务补充继续发到群里时，只提醒其查看私聊，避免双通道同时推进同一个任务。

## 原子认领

编号认领流程：

```text
resolve board version
  -> resolve game_id
  -> identity / game state / duplicate / time conflict validation
  -> atomic_claim_seat()
       write participant state
       write GameClaim(source_conversation_id, source_message_id)
       commit once
  -> private notification for friend, minimal group @ for nonfriend
  -> urgent board refresh
```

SQLite 使用 `BEGIN IMMEDIATE` 写事务。同一来源消息通过唯一认领记录幂等；两名用户并发抢最后一位时，第二个事务会在重新读取权威状态后失败，不会超卖。

`371` 表示当前三座、还差一座，不代表只有三个已知微信联系人。若报局者一人代表三座，领域模型保留一名已知联系人及匿名席位数量；对外人数以权威席位计数为准。

## 公开投影与隐私

所有群聊域通知只能接收公开字段允许列表：

- 玩法/变体。
- 档位或底注/封顶。
- 时间或人齐开。
- 烟况。
- 总座位、已占座、剩余座位。
- 对当前接收者必要的状态。

默认排除：

- 参与者姓名、微信备注、外部账号 ID。
- 其他客户原始消息和私聊内容。
- 画像、关系冲突原文、候选排序理由。
- `game_id`、内部枚举、trace、工具和审批细节。

客户可见文本仍经过生成/审查/确定性末端过滤；通道层还要通过房间外发策略和发送幂等。

## 持久化结构

SQLite 新增以下领域表：

| 表 | 作用 |
| --- | --- |
| `runtime_channel_identities` | 渠道账号与客户/私聊能力映射 |
| `runtime_group_room_policies` | 房间是否托管、看板/外发开关、合并窗口 |
| `runtime_game_conversation_links` | Game 与房间/私聊任务的关联 |
| `runtime_group_board_snapshots` | 每次已发送看板的不可变版本 |
| `runtime_group_board_items` | 某版编号到 `game_id` 的映射 |
| `runtime_group_game_claims` | 来源消息幂等认领记录 |
| `runtime_channel_switches` | 短期群转私聊续接指针 |

内存 Store 与 SQLite Store 具有同一接口，专项测试同时覆盖逻辑与重启恢复。

## 权限与配置

房间分三类：

- 普通房间：不进入业务入口。
- 只读观察房间：允许语义观测，但禁止状态修改和外发。
- 托管房间：允许三层群聊域处理；看板与外发再由独立开关控制。

托管房间必须通过 `MAHJONG_WECHATY_MANAGED_ROOM_IDS/TOPICS` 或本地忽略文件显式授权。`MAHJONG_WECHATY_MANAGED_ROOM_OUTBOUND_ENABLED` 默认 `false`。

## 已有测试证据

专项测试覆盖：

- 报局/认领协议及数字文本误判。
- 简单查询、复杂请求、好友/非好友路由。
- 看板合并、紧急刷新、重排、空看板和 SQLite 重启恢复。
- 引用快照精确解析与无引用只认最新版。
- 好友私聊、非好友群内最小通知。
- 重复消息幂等、两人并发抢最后座、时间冲突、局满通知。
- 群转私聊上下文最小化和群内续接检测。
- 群内公开投影不包含参与者身份与内部席位结构。
- 主 Agent 工具修改 Game 后自动建立房间关联并排队刷新。
- WeChaty 托管入口、房间投递和外发关闭策略。
- 群聊 Session 的碎片聚合、引用/发送者/特征归属、多人续接、冲突阻止合并、范围时间细化、一次性结晶和跨 Session 原文隔离。

## 仍需完成

- 在目标微信版本与 puppet 上持续验证引用消息 ID、群消息回显和重启后的映射恢复。
- 实现自动创建局群、拉人入群及其高风险人工审批。
- 实现非好友的好友申请、申请状态持久化和通过后的任务恢复。
- 为无编号自然语言引用建立真实样本评测；多个候选时必须澄清，不能牺牲正确性追求自动化。
- 增加房间级频控、看板撤回/替换策略和长期灰度指标。
- 将 Session 化候选链路接入统一 trace 与 ActionRouter，并在房间级功能开关下替换现有 L3；接入前不得对真实群改状态或外发。

这些限制必须继续写入 README 和发布说明，不能把非官方个人微信桥包装成稳定官方生产通道。
