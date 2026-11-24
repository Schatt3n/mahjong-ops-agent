# 500 人营业仿真接续记录

工作分支为 `codex/venue-simulation`。架构、最终能力和分阶段验收见 [实施路径与验收](venue_simulation.md)。每个阶段先跑专项验收，再提交和推送。主 Agent 在沙盒中使用独立实例、业务库、档案和模型客户端。优先在副本中复现时间问题，先改场景时间或仿真时钟，再判断是否有可独立复现的领域缺陷。

## 阶段进度

| 阶段 | 当前状态 | 接续要点 |
| --- | --- | --- |
| 1. 独立沙盒与模型连接 | 已推送 `83aae14` | 500 人建群、角色真值隔离、聊天归档、实际开局与释放均已离线验证。本机 OpenCode Go 最小请求返回 200。完整真实对话的结果另记。 |
| 2. 差异化角色与并发合同 | 已推送 `d3a0788` | 固定种子生成 500 套表达方式和 500 套行为倾向。验证群聊与私聊实际并发、同会话顺序、单次投递及最后一座原子确认。 |
| 3. 开放式多轮与独立评测 | 框架和 500 人离线专项已验收；真实语义验收待通过 | 下一轮由本人实际收到的回复触发。核对实际交付、业务结果及独立模型采样。真实试跑已取得有因果链的回复，仍出现主模型正文截断。 |
| 4.1 OpenCode 截断诊断 | 已复现并修复沙盒请求 | 同模型、同输入、同输出预算对照确认缺省推理耗尽正文预算。 |
| 4.2 独立评测超时 | 实施中 | 评测证据缩减后用同一份历史档案重放，核对是否能返回可引用的判定。 |
| 4.3 小批真实成局和未成局 | 待通过 | 两种真实用例均通过因果、状态与语义验收后再扩大。 |
| 4.4 场景与规模 | 待实施和验收 | 把取消、改期、关系冲突、临时关房、多个局和跨营业时间编排成可重放场景。扩大真实样本，验证角色表达差异以及真实模型的吞吐和语义质量。 |

## 已验收证据

- 同一群预装 500 个公开身份后，离线场景以 500 次背景唤醒留下 498 条聊天事件。20 个检查全部通过。四人确认后在 19:00 实际开打，22:59 仍占用，23:00 释放，23:02 同房间成功复订。运行编号为 `2026-11-25-mock-4f4eebaed0`。
- OpenCode Go 真请求使用本机模型配置的 `deepseek-v4.1-flash` 和 Responses 接口。最小 JSON 请求返回 HTTP 200。配置阶段 13 项专项通过，真实开局受控场景的角色请求也返回了 200。
- 同一个可接受邀约输入到 500 位角色时，离线合同桩输出 500 段不同发言，区别来自称呼、信息顺序、时钟表达、句式和标点。角色配置另有承诺倾向、协商倾向、反应习惯等可影响行为的字段。
- 并发专项覆盖 32 条同时提交的群聊与私聊、不同顾客实际重叠、同一会话有序且旧回复被撤销、重复入站收敛到同一份回执，以及 15 个顾客争同一末席只写入一个赢家。
- 第二阶段运行下列四组专项，共 52 项通过。

```bash
.venv/bin/python -m pytest -q tests/simulation/test_venue_concurrency.py tests/simulation/test_venue_actors.py tests/simulation/test_venue_channel.py tests/simulation/test_group_transport.py
```

## 第三阶段实验

| 运行 | 结果与证据 |
| --- | --- |
| `2026-11-25-mock-13bca8e734` | 处理 6 位活跃客户和群内背景事件。每名客户均有实际交付的因果续轮。峰值 3 个交付在途，6 人均经正式搜索确认无匹配。 |
| `2026-11-25-mock-8c9e2d1057` | 目标 500 人，16 个工作线程，每人执行两轮。500 人均有两轮实际交付的私聊，共归档 2,000 条入站和出站消息。实测峰值 16 个交付同时在途，500 条会话在归档中的首尾跨度有重叠。正式搜索返回 500 个有证据的无匹配。所有硬检查通过。角色与主模型使用离线合同桩，渠道、业务工具和 SQLite 为实际执行。 |
| `2026-11-25-real-d022fdcab1` | 3 人、2 并发、最多 3 轮，归档 14 条消息。两位客户有因果续轮，两例落为等待。一位客户的第二轮实际创建了待组局、预订了房间并写入邀约草稿，第三轮也收到回复。另有一位客户第一轮查询后遇到 HTTP 200、4,000 输出 token 用尽。三次独立评测有两次超时，一次有效失败指出故障兜底没有承接请求。原报告因模型故障汇总为 `inconclusive`。现在已证实的失败优先计入总结果。 |
| `2026-11-25-real-890f5a121b` | 1 人、2 轮、主模型本轮超时预算 180 秒。两条客户来话和两条 Agent 回复均已归档，第二条顾客来话对应首条回复。第二轮主模型 HTTP 200、7,200 输出 token 用尽，顾客收到错误兜底。独立评测原文判顾客意向及说话方式通过、Agent 第二轮响应失败。修复回显案例编号的解析规则后，重放该判定原文得到有效 `fail`。原运行报告保留当时的 `inconclusive`。 |

第三阶段 9 组专项共 81 项通过。覆盖角色合同、模型设置、群和私聊通道、并发、独立验收、开放对话、受控场景及实际开打散场。中文文档字符串与差异检查通过。

```bash
.venv/bin/python -m pytest -q tests/simulation/test_venue_dialogues.py tests/simulation/test_venue_acceptance.py tests/simulation/test_venue_scenario.py tests/simulation/test_venue_models.py tests/simulation/test_venue_actors.py tests/simulation/test_venue_concurrency.py tests/simulation/test_venue_channel.py tests/simulation/test_group_transport.py tests/test_actual_game_play.py
.venv/bin/python scripts/run_venue_simulation.py --mode mock --scenario dialogues --active-users 500 --workers 16 --dialogue-turns 2 --background-events 0
.venv/bin/python scripts/run_venue_simulation.py --mode real --model-source opencode-go --scenario dialogues --active-users 3 --workers 2 --dialogue-turns 3 --judge-users 3 --background-events 0
MAHJONG_AGENT_LLM_TIMEOUT_SECONDS=180 .venv/bin/python scripts/run_venue_simulation.py --mode real --model-source opencode-go --scenario dialogues --active-users 1 --workers 1 --dialogue-turns 2 --judge-users 1 --background-events 0
```

## 问题、尝试与处理

| 问题 | 尝试、证据与处理 |
| --- | --- |
| 项目旧配置的直连模型返回 HTTP 402 | 首次用项目 `.env` 走 DeepSeek 直连。角色生成失败，保存原始 HTTP 状态，主 Agent 及语义评测没有取得有效对话。之后核对本机配置，确认当前执行进程没有继承 OpenCode 环境变量，但已配置读取本地 OpenCode Go 凭证。运行器通过显式或自动来源选择复用本机配置，密钥留在本机。 |
| 最小请求成功，主 Agent 长请求读超时 | 首次受控真实试跑 `2026-11-25-real-9333090be3` 中，OpenCode 的角色文本已返回，长上下文请求触发原有 45 秒预算。独立仿真实例对 OpenCode 默认使用 120 秒，可用 `MAHJONG_AGENT_LLM_TIMEOUT_SECONDS` 覆盖。新试跑中已有可用的真实主模型响应。 |
| 已收到 HTTP 200，仍拿不到主模型正文 | 完整 Responses HTTP 报文中的 `incomplete_reason=max_output_tokens`，与等待 HTTP 响应时的超时分开记录。默认主模型输出限额由 4,000 调到 7,200，可用 `OPENCODE_GO_MAX_OUTPUT_TOKENS` 覆盖。单次把运行时限增至 180 秒的尝试仍遇输出截断，扩大规模前需要先验证小批真实质量。 |
| 独立判定返回了额外的案例编号 | 明确约束输出键，兼容回显当前案例的编号。证据引用、逐项判定和汇总一致性继续校验。用该轮原始判定重放，确认有效失败会保留。 |
| 单局闭环、并发容量和语义质量需要各自的证据 | 初版 500 次背景唤醒顺序处理，角色口吻只有少量样式。增加可审查的多维角色差异，补并发、重投、末席竞争和跨私聊隔离测试。再按实际送达的因果链计算多轮，按真实落库和独立判定计算业务质量。 |
| 旧十轮测试在本机短时限下提前结束 | 原始版本也因 3 秒上限结束，只完成部分轮次。复用了已有 15 秒测试保护时限，继续要求十轮、十次 HTTP 成功以及按消息数结束。涉及本机回环 HTTP 的专项测试用会话临时的 no-proxy 配置，鉴权规则保留。 |

## OpenCode 截断对照

对同一个简单 JSON 请求，使用本机已配置的 `deepseek-v4.1-flash`、48 个输出 token、20 秒总期限和单次 HTTP 尝试。下列为 2026-11-25 沙盒探针，未改变线上配置。

| 请求 | HTTP | 耗时 | 输出和结果 |
| --- | --- | --- | --- |
| Responses，省略 `reasoning` | 200 | 2.803 秒 | `incomplete/max_output_tokens`，48 个输出 token 全用于推理，正文为空。 |
| Responses，显式 `reasoning.effort=none` | 200 | 2.084 秒 | 8 个输出 token，0 推理 token，有效 JSON。 |
| Chat Completions，显式 `thinking.type=disabled` | 200 | 2.568 秒 | 8 个输出 token，0 推理 token，有效 JSON。仅作控制。 |

DeepSeek Responses 把推理计入输出，缺省启用推理。仿真配置原先把内部的 disabled 映射为空缺字段，现在仅给沙盒的 DeepSeek Responses 显式传 `none`，报告记录实际的协议和推理参数。用 `OPENCODE_GO_RESPONSE_REASONING=provider-default` 可复现旧请求，`low` 可做另一组对照。保留独立回复审查已有的显式低强度配置。 [DeepSeek Responses 说明](https://api-docs.deepseek.com/api/create-response/) · [OpenCode Go 接口](https://opencode.ai/docs/console/go/)

模型配置和开放对话专项共 17 项通过。记录了发送到线上的推理参数，审查回归和供应商对照可各自复现。

旧的独立评测超时样本携带约 63 KB 与 80 KB 的证据，45 秒总预算分给两次请求，每次约 22 秒。两样本均在收到 HTTP 状态前结束。约 21 KB 的样本在 8.87 秒返回 HTTP 200。现有日志能确认本地尝试期限和证据体积，供应商内部排队和处理时间仍需通过同档案重放进一步区分。

## 下一阶段

1. 在同一日期、种子和客户数下比较小批真实运行。优先查清第二轮输出耗尽和评测超时，逐一比对返回的 HTTP 状态、正文和实际工具证据。测试参数先加在独立副本。
2. 将离线验收里的取消、到期和查无匹配等口径用于场景矩阵。补临时改期、主动退出、已经告知的避桌关系、同群多局和关房后复订。每个场景至少保存一次成功路径和一次边界路径的证据。
3. 在小批真实质量通过后，按同一业务场景扩大活跃用户和工作线程数。记录不同种子的因果续轮分布、合理终止、各模型超时与输出截断、语义判定样本及资源用量。500 人真实长对话及所有角色在真实模型上的表达差异仍待验收。

运行数据库、原始聊天、请求轨迹和凭证保留在运行端的沙盒。跨端接续以本文件、验收口径和代码为准。接手后可以按运行编号在原机定位完整证据，或用相同种子重跑初始条件。
