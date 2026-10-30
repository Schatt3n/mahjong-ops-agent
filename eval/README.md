# 评估集、Badcase 和 Golden Dataset

这个目录只保留当前 `mahjong_agent_runtime` 主链路相关的质量资产。

## 目录分工

- `badcases/badcases.jsonl`：当前 Agent Runtime 在测试、试用或真实微信灰度中发现的失败样本和边界样本。
- `regression/agent_runtime_regression.jsonl`：当前主链路专项回归集，验证“模型负责目标理解和工具规划，后端负责 schema、权限、状态、幂等和审计”。
- `golden/real_owner_chat_golden.jsonl`：真实老板聊天转写出的长对话 golden dataset，用于验证闲聊和业务组局穿插时上下文仍能接回。
- `golden/real_owner_chat_transcript_20260704.md`：真实聊天截图的可读转写。
- `golden/real_owner_chat_timeline_20260705_20260718.json`：12 张真实老板聊天截图清洗出的多日机器可读时间线，保留 86 条文本、7 个业务片段、时间、终态、业务轮次、事实、评测 case 和回放脚本；表情包及图片已排除。
- `golden/real_owner_chat_transcript_20260705_20260718.md`：上述多日时间线的可读转写，说明简单/复杂场景所需轮次、取消边界和推荐测试输入。
- `golden/real_owner_chat_timeline_20260627_20260719_owner_b.json`：第二组 11 张真实老板聊天截图清洗出的机器可读时间线，保留 80 条文本、7 个业务片段和 8 个评测 case，重点覆盖代约角色纠正、取消恢复、档位冲突及条件认领。
- `golden/real_owner_chat_transcript_20260627_20260719_owner_b.md`：第二组时间线的可读转写；第三方昵称已匿名，表情包、图片和不可恢复的撤回内容不进入消息正文。
- `golden/fragmented_input_golden.jsonl`：碎片化输入边界样本，验证等待、超时重触发、发送者隔离和聚合后一次处理。
- `golden/real_group_chat_20260722.jsonl`：真实微信群观察数据清洗出的高置信群聊 Gold，覆盖看板解析、状态演进、碎片聚合、引用更新、撤回和确定性噪声过滤。
- `adversarial/privacy_isolation.jsonl`：跨会话隐私对抗样本，一条一个攻击 case，与执行脚本解耦。
- `adversarial/real_group_chat_20260722.jsonl`：真实群聊中的指代、跨角色多轮和麻将黑话歧义样本；在领域问题确认前只能作为待复核对抗集，不能作为硬真值。
- `adversarial/agent_planning_discoveries.jsonl`：真实模型模拟中发现、尚未闭环的规划失败样本；保留输入、工具轨迹、来源 trace 和可接受结果，供后续规划/无进展评测复现。
- `few_shot_examples.jsonl`：老板认可的话术样例，用于改善后续回复风格。
- `tests/test_context_summary_quality.py`：上下文压缩决策一致性评测，比较完整历史与 checkpoint 路径的状态、工具调用和回复。

## 写入规则

- 发现当前主链路回复、工具调用或状态推进有问题，先写入 `badcases/badcases.jsonl`。
- 修复 badcase 后，必须补 `regression_refs`，指向 `agent_runtime_regression`、`real_owner_chat_golden`、`live_eval` 或具体 `pytest` 用例。
- 老板确认某句回复“像我会说的话”，写入 `few_shot_examples.jsonl`。
- 真实长对话样本优先沉淀到 `real_owner_chat_golden.jsonl`，用于验证上下文、闲聊分流和多轮恢复。
- 多日真实私聊必须按独立业务目标切分为 episode，并保存原始时间顺序。业务轮次按“一方围绕同一动作的连续消息 + 另一方连续回应”计数，不能把老板连续发送的多个短气泡误算为多个完整轮次。
- 轮次统计只能描述当前样本，不能从少量成功片段推导稳定成局率或平均耗时；自动回归应优先断言少追问、状态正确、取消生效、参与者不混淆和工具闭环。
- 真实群聊数据必须先匿名化：房间和发送者使用别名，源消息仅保留截断 SHA-256；禁止提交微信 ID、昵称、头像、签名、完整 payload 和现有分类器标签。
- 每日监听资产使用 `real_group_chat_YYYYMMDD.jsonl` 命名；校验器会自动发现所有日期文件，并校验文件名日期与 `source.capture_date` 一致。
- 只有业务含义及期望动作都明确的群聊样本才能进入 Gold；“那个一”“3块是否等于368”等仍需老板确认的样本进入 Adversarial，并保留 `open_questions`。
- 用户把一句需求拆成多条发送时，写入 `fragmented_input_golden.jsonl`；不能靠新增麻将关键词 `if-else` 修复，应由输入边界模型和通用并发合同解决。
- Agent 重复调用同一工具、在短周期动作间来回切换或连续没有状态/信息进展时，必须补 `ProgressMonitor` 回归；检测器只比较动作、结果和状态变化，不写麻将业务 `if-else`。
- 真实模拟中首次发现但尚未稳定修复的规划问题先写入 `agent_planning_discoveries.jsonl`，状态保持 `open`；只有重复复现通过并建立自动回归后，才迁入 fixed badcase 闭环。
- 修改摘要提示词、checkpoint 合同或上下文裁剪逻辑时，必须运行压缩决策一致性评测；不能只断言摘要已生成。
- 越权询问、提示词注入、格式诱导和间接暗示等攻击必须写入 `adversarial/`；仅使用合成身份和 canary，不提交真实客户私聊。

## 运行评估

运行当前主链路默认评估：

```bash
PYTHONPATH=src python scripts/run_evals.py
```

默认命令验证格式、执行机制与确定性回归，不代表真实模型的回复语义已通过。有密钥时，每次代码迭代提交前使用完整门禁：

```bash
PYTHONPATH=src python scripts/run_evals.py --full
```

门禁保留全量 pytest、全部可执行真实老板场景、六类咨询、群聊、隐私对抗和并发集合。真实老板回放动态读取 `real_owner_chat_golden.jsonl` 中的旧期望及本次新增多轮样本，并复用已有咨询评测的独立语义判定器；按应表达/禁止表达的含义逐项判定，不按参考回答逐字比对，也不以关键词命中替代语义。原始输入、参考回复、实际回复、工具和持久化证据、逐项判定与错误都会保存；语义判定未运行或返回无效结果时门禁失败。

按用户确认的验收口径，`action_contract_error` 不再只因出现过就阻断：后端仍严格拒绝非法动作，只有错误步骤无业务执行、状态修改和客户外发，完整拒绝反馈回灌后有合法动作、真实工具反馈与正确终态，并通过业务硬断言及独立语义，才计为安全恢复。逐例 `recovery_metrics` 区分合同错误、安全恢复、未恢复、不安全拒绝步骤和其他未处置工具/权限错误；顶层 `recovery_summary` 加总质量次数。`trace_steps` 和决策轨迹保留错误，`raw_trace_path` 指向该次运行独立保存的完整事件，统一门禁重新计算指标核验。缺少证据、未恢复、语义、权限、数据错误及其他既有硬失败仍阻断，不能用最终成功或模型自述抹去中途错误。

报告按集合区分执行范围。两份历史时间线当前仍由专用回归测试验证，未被接入真实老板回放的部分不能计为真实模型语义已通过。新增用例应继续沉淀到现有资产并绑定可执行回放，不能只放在脚本注释或一条通过日志中。本次样本 `owner_chat_flexible_time_and_help_offer_20261029_001` 保留“下午有人吗 → 追问 → 一块、时间都行”的原始对话，并将晚间有局、无局提供问人服务和纯盘点等派生条件分开标注。

旧 `group_duration_mismatch_records_exit` 将拒绝五小时和后续明确四小时上限合在一个期望中。真实回放分别在拒绝回合与原始第56轮明确四小时的回合检查这两部分；只有两个片段都执行才算该案例完整覆盖，不能提前向拒绝回合注入未来信息。旧原始消息和期望均保留。好话术样例仅供参考，坏样例归入禁止含义，不能误当必须输出的内容。

时间放宽的断言检查是否仍残留下午、固定开场或人齐开等限制，不要求查询 JSON 采用唯一写法。完整回放曾使用 `start_time_kind=flexible` 找到晚间局并正确征询；现有开放 schema 允许该值，排序器在固定时间和人齐开目标上与省略字段的得分及理由相同。因此评测允许这一等价表示，同时保留旧时间、擅自收窄、错误匹配和未授权写操作的负例；原始失败报告及人工期望保留。

通用回复标准衡量是否推进当前目标：缺少真正必要的条件时，直接询问缺项已经有效，不额外要求先说“好/收到”或承诺帮忙。群聊明确金额沿用现有完整数值语法比较“一块”和“1”等价；区间、封顶和未知表达保持原值，不能为了匹配期望截取一个数字。

当前 34 个真实调用场景均有明确的输入、关键上下文和期望回复含义，对应 35 个黄金案例。新增 22 个参考案例明确标记为合成回归夹具，不冒充原始人类聊天，也不计为真人话术风格证据。逐项判定依据这些显式标准，不再由一个泛化的目标问题临时猜测该场景必须说什么；工具、状态、隐私及禁止虚构动作的门禁独立保留。

独立判定器必须拿到足以核对回复的真实结果。房态投影保留配置状态、查询时间区间和库存数量，预定结果另保留真实预定记录；不能只给工具名，也不能将查询有房当作已经预定。复查承诺另读取当前客户、会话及目标的持久化任务末态，保留观测时间、状态、期限和剩余次数所需字段，避免已取消或到期的安排被历史工具成功覆盖。2026-10-30 的 `custom_profile_game_preserved` 曾因投影遗漏真实可用房间数而被判定器拒绝，原始失败报告保留；修复补充证据，不修改参考回复或验收含义。对应 Memory/SQLite 回归见 `test_semantic_room_evidence.py` 与 `test_follow_up_semantic_evidence.py`。

当前老板场景集尚不覆盖等待需求登记后实际通知、邀约审批后真实外发成功的完整链路。待审批邀约草稿和有限复查的通过，不能被表述为这些外发链路已经通过真实模型验收。

缺少关键底注的个人咨询允许先问真实缺项，若实际查询仍须核验玩法等参数；独立的条件充分案例继续强制执行查询，不能用澄清代替工具覆盖。全店盘点和显式限定玩法的盘点另有混合库存对照，验证画像不能擅改请求范围。原始输入、人工语义与参考回复保持，结构断言的修订理由和旧值单独保留，失败报告不会被追改为通过。

己方人数与所找牌局的当前人数分开验收；三人同行的容量对照验证请求方座位、可入局结果及加入后的真实投影。旧结构断言通过追加 `hard_assertion_revisions` 保存完整前后快照、理由和来源；校验器核对完整修订链及当前代码，不能忽略旧快照不一致，也不改写原人工语义。

`golden/active_input_revision.jsonl` 独立保存运行中改目标的双输入与语义期望。完整并发门禁在旧轮模型调用阶段接收新目标，验证原文先落库、旧轮无动作/回复、新轮采用最新条件、既有状态不被擅自修改，并保留独立语义判定；这一项失败也会阻断整套并发门禁。Memory/SQLite 的确定性 Event 测试另覆盖三条并发输入、重复投递、工具/审查/摘要边界及已交付历史。

完整并发门禁还包含两份资产的 14 条受控真实审查：`golden/customer_visible_review_contract.jsonl` 保留原 8 条空库查局对照，`golden/follow_up_commitment_review.jsonl` 新增 6 条开放局、真实有限复查、取消复查及虚构外发对照。前置状态通过真实 Tool Gateway 建立，再用生产审查入口核验服务提议、当前工作的短承接、持续反馈责任和执行声明。开放局本身不代表未来会再查或通知；真实任务也只支持其对象、期限和能力范围内的责任。期望标签、案例 ID 和夹具标签不进入模型请求，每例只调用一次；这项不是主 Agent 规划回放的替代品。完整门禁必须覆盖全部 14 个来源，核验原始请求与响应、真实工具轨迹、持久化任务身份和状态、审查前后业务状态及生产解析后的批准结果；只跑旧 8 条不能充当全量通过。隐私运行器也保留包括供应商错误详情的完整独立 trace；复跑不会覆盖旧轨迹。

单独校验真实群聊数据的 JSON 合同、匿名化和分层状态：

```bash
python scripts/validate_real_group_chat_dataset.py
PYTHONPATH=src python -m pytest -q tests/test_real_group_chat_dataset.py
```

校验真实老板私聊的基础 JSONL 与多日时间线：

```bash
PYTHONPATH=src python scripts/validate_real_owner_chat_golden.py
PYTHONPATH=src python -m pytest -q \
  tests/test_real_owner_chat_golden.py \
  tests/test_real_owner_chat_golden_validator.py \
  tests/test_real_owner_chat_timeline.py \
  tests/test_real_owner_chat_timeline_owner_b.py
```

校验器只负责数据格式与匿名化，不能证明 Agent 能正确处理样本。真实群聊 Gold 还必须回放到实际的 `OwnerMessageParser`、`QuickFilter`、`MessageAccumulator`、`SessionRouter` 和 `GroupSessionClassifier`：

```bash
# 不调用模型：看板解析、状态演进、引用、撤回和噪声过滤
PYTHONPATH=src python scripts/run_real_group_chat_flow_eval.py --strict

# 调用当前配置的模型，并把语义样本继续送入主 Agent 验证工具选择
PYTHONPATH=src python scripts/run_real_group_chat_flow_eval.py --live --strict \
  --report-path runtime_data/real_group_chat_flow_eval_live.json
```

报告逐 case 保存 `expected`、`actual`、差异、模型调用次数、模型耗时、主 Agent 工具调用和 trace 步骤。`cq` 规范化为“杭麻/财敲”，“川麻换三”规范化为“川麻/换三张”；`568/368/132` 等三位码在场馆确认规则前仅保存在 `rule_code`，不得推断成底注、封顶或番数。

单独运行上下文压缩质量评测：

```bash
PYTHONPATH=src python -m pytest -q tests/test_context_summary_quality.py
```

该评测覆盖失败历史、临时约束、子任务续接和工具结果依赖，并包含摘要漏约束的反向对照。二进制核心指标为 `DecisionConsistencyReport.consistent`；差异会写入 `context_summary_decision_consistency` trace，便于定位状态、工具参数或回复在哪一项发生漂移。

运行真实模型 live 评估：

```bash
MAHJONG_LLM_PROVIDER=deepseek MAHJONG_LLM_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=*** PYTHONPATH=src python scripts/run_real_owner_chat_live_eval.py --strict
```

运行跨会话隐私对抗评测：

```bash
PYTHONPATH=src python scripts/run_privacy_isolation_live_eval.py --strict
```

评测器默认读取 `adversarial/privacy_isolation.jsonl`；可以用 `--case direct_reason` 单独调试某个 case，或用 `--cases-path` 切换到其他对抗数据集。

也可以把 live 评估接到主链路总评估里：

```bash
MAHJONG_LLM_PROVIDER=deepseek MAHJONG_LLM_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=*** PYTHONPATH=src python scripts/run_evals.py --live-real-owner
```

追加 `--live-real-group` 可把真实群聊语义样本也纳入总评估。

`scripts/check_badcase_regression_coverage.py` 会审计所有 fixed badcase 是否已经闭环到当前回归资产，避免问题只停留在日志里。
