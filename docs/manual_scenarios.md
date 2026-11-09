# JSON场景与真实群聊模拟

打开本地`/simulation`，选择“导入 JSON 场景”，粘贴[示例JSON](examples/group_scenario.json)。导入会创建独立测试库、保存画像与待发消息，不调用模型或自动发送队列。原有运行和失败记录保留。

多话题测试可使用[七人群聊JSON](examples/seven_customer_group_scenario.json)：7位客户、23条群聊输入，交错覆盖杭麻、下午国标和川麻晚场。先看[画像表与逐条话题索引](examples/seven_customer_group_topics.md)，再按实际消息逐条回放；人工话题标签不输入主Agent。

导入后可以点“执行下一条”，检查本轮路由、实际输出、工具和状态，再决定是否继续；也可以连续执行并暂停。每次单步只消费一条队列消息。私聊输入应明确设置`channel: "private"`，群聊输入设置`channel: "group"`。

点页面顶部的“清理模拟记录”并确认，可以清空模拟列表。报告、对话、审阅和数据库会随整批记录移入页面提示的本机归档目录。处理中的模拟任务会阻止归档，结束后可以重试。

## 场景格式

- 根字段：`version: 1`、`actors`、`messages`。
- 角色：`id`为本场景内唯一别名；`profile`保存老板已掌握的画像；`is_friend`为布尔值，控制沙盒中的好友及私聊可达关系。它属于通道身份，不是客户自然语言偏好。
- 消息：`actor_id`引用角色别名，`channel`只接受`group`或`private`，`text`保留待发送原话。
- 画像常用字段：`display_name`、`preferred_games`、`preferred_stakes`、`preferred_time_tags`、`smoke_preference`、`profile_facts`、`notes`、`no_contact`。列表字段使用字符串数组；烟况为`no_smoke`、`smoking`、`any`或`null`。
- 最多20个角色、每批30条消息、每条4000字。未知字段、非法类型、重复角色和不存在的消息角色会被拒绝，不能只导入前半部分。

角色可以在页面继续编辑画像和好友关系。画像更新作用于后续输入，历史输入时的画像证据保持不变。

## 群聊如何执行

群输入进入生产`GroupMessageHandler`，由同一套编号认领协议、通道身份和语义路由决定忽略、公开群答或私聊续接。模拟器负责构造受控身份与沙盒收件箱，不新增一套自然语言路由规则。普通私聊进入同一个主Agent Runtime。

群输入不意味着回复也必须发群。以沙盒Messenger记录的实际输出为准：群消息进入群收件箱，私聊仅进入对应客户收件箱，忽略就是没有输出。转私聊后的内容在相应私聊视图中查看；群视图中的审阅证据不能被当作对所有群成员可见的消息。

旧自动模拟记录没有正式群路由证据，页面标明“历史直连模拟”。这些记录原样保留，不能改写成已通过新群入口的结果。

这是本地通道测试。输出只进入沙盒，不连接真实微信；真实模式会调用已配置的模型。语义路由判断为闲聊时可以不调用主 Agent，这不算主模型空响应。手动面板中的话题证据来自正式入口的同一次语义分类及持久收据，不再额外调用观察分类器。离线模式只用显式分类桩验证接线、通道与持久化，不能替代真实模型的意图验收。

## API单步控制

所有操作使用受鉴权保护的`POST /api/simulations/manual`：

1. `{"action":"import_scenario","request_id":"import-001","scenario":{...}}`：返回新`run_id`及角色映射。同一请求重试不重复创建；同一请求号换场景内容会被拒绝。
2. `{"action":"batch_next","request_id":"step-001","run_id":"返回的运行编号"}`：执行一条。请求号相同表示重试当前操作，执行下一条必须换新请求号。

先保存场景，再单步执行，可以分别检验画像是否进入上下文、消息是否经过群入口、最终回复究竟发往哪里。

群友自招缺档位时，可切换另一位角色私聊查局。安排核实后，询局人先看到现有局和待核实条件；发局人的问句单独进入其私聊收件箱。可切换为发局人，使用私聊回复本桌档位。档位写入原局后，仍有空位且未过期时，后台将档位另行回告原询局人。双方各看自己的私聊，核实和回告都不会占座。测试收件箱不向真实微信发消息。

核实软偏好的对照：A 群发“七点财敲371，无烟，最好再来个女生”，B 的测试画像为男，B 私聊问晚上有没有局。A 的私聊应同时问档位和是否接受这位匿名男生；A 只回“一块的”时，B 的私聊必须实际收到价格，参与资格仍待确认。B 的姓名、画像和原话不带给 A，A 的同桌偏好也不带给 B。把“最好”改为明确的硬限制，或把 B 的性别设为未知，核实消息不能再宣称有一位男生可补。

## 话题存储与个人目标

语义分类使用当前消息、最多5个公开话题卡片、本人少量目标与来源、真实已发布看板。分类不选择通道，也不直接写牌局。群友发帖自招时，发帖人明确表达的条件随原消息静默登记；玩法和自带人数足够时，同时安排下一次看板发布。右侧证据显示登记结果及看板任务。合并窗口到期后处理下一条群消息时，模拟器会发布该群到期看板。明确委托可以接续原局，原群短答和私聊独立判断。

SQLite保存原始来源、分类证据、话题版本、每条消息与本人任务的绑定。相同话题可以有多个人、多个独立目标；共同卡片不覆盖个人条件。群内委托转私聊后，后续补充以真实用户输入进入所选个人任务，推进会话版本并经过既有工具、状态、审查及外发边界。只读询局和目标不确定的输入还受到工具作用域限制，可查询及记录本人上下文，不能直接建局、占座或邀约。

可安装可选依赖并配置独立缓存：

```bash
.venv/bin/python -m pip install 'redis>=5.0'
export MAHJONG_TOPIC_REDIS_URL=redis://127.0.0.1:6387/0
```

Redis仅保存规范公开卡片和有界群索引，默认最多保留30分钟；不保存个人画像、原始消息、私人任务或群成员身份。缓存失效、旧版本回写或Redis断线时，从Store读取并核验。话题过期只影响活跃检索，不能取消用户任务、牌局或删除证据；本人冷话题与明确引用仍可恢复。当前仍会用Store核验候选全集，尚未以此减少SQLite查询量。

当前运行库schema为4。开发环境不迁移旧运行库，应保留原始测试、日志和失败证据，使用新的运行目录创建数据库。旧记录在面板继续保留，但不能拿旧版本的通过结果替代新语义链路验收。

## 个人等待截止

“最多等到十点，没成就不等了”需要保存本人参与的条件任务，不能只写画像或安排一次普通提醒。
主Agent可调用 `set_participation_deadline`，后端核验本人真实输入、独立席位、对应牌局和有效时间；
截止可以超过普通复查的24小时上限，但不能超过该局已知有效期。撤销条件使用 `clear_participation_deadline`。

到期由现有Scheduler重新检查权威人数：尚未成局时只撤回本人的独立席位，并安排看板刷新；
已经成局或结束时不再撤人。包含同行的多席报名需要先澄清，不能连带退出同行或取消整局。
改期、撤销、重复调度和崩溃重试均保留原始任务及执行证据；话题缓存过期不会删除截止责任。
面板的“推进待办”可在沙盒触发这类任务，输出仍只进入沙盒，不代表真实微信发送。

## 群入口定向回归

`scripts/run_group_topic_regression_eval.py` 复用上述正式群入口、SQLite和沙盒发送器。
人工期望不进入分流器或主Agent，回复由既有独立语义判定器验收，业务状态另外执行硬断言。
每次运行创建新目录并保存原始trace；默认首例失败即停，不覆盖或重试有效失败结论。

```bash
PYTHONPATH=src .venv/bin/python scripts/run_group_topic_regression_eval.py --validate-only
PYTHONPATH=src .venv/bin/python scripts/run_group_topic_regression_eval.py --case delegation_complete_empty
PYTHONPATH=src .venv/bin/python scripts/run_group_topic_regression_eval.py --dataset eval/golden/group_topic_recovery_20261105.jsonl
```

委托对照集覆盖仅查询、明确帮找、只找现成、缺必要条件、现成局待确认、没有合适好友和群转私聊。
恢复集保留原时间变更、个人等待截止及两人只询局的输入。先运行受影响案例，再执行未筛选回归。
这些定向报告与完整发布门禁分开记录，不能替代 `scripts/run_evals.py --full`。

`group_delegation_scope_drift_20261105.jsonl` 另保留非好友群场景的范围漂移反例：
已有明确委托，查空后却再次征询同一服务。共享合同要求先核对本人原话、限制和执行缺项，再选择库存后的动作；
`objective_state.request_scope` 只保存有来源的模型解释，不授予工具权限，不能由新计划自行撤销用户委托。

`group_query_history_join_20261105.jsonl` 验证找到现成局后下一轮确认参加：必须真正加入原有局。
历史选择按去除精确重复静态合同后的传输大小计预算，避免整条查询候选被重复文本挤掉；
完整历史与原始审计仍保留，版本不同的合同不能借用当前合同缩减。

## 低成本分流旁路实验

`scripts/run_group_decision_model_eval.py` 使用固定人工标注的33条输入，分别评估意图、公共话题和本人目标。
输入只含当前原话、公开卡片及本人少量历史；输出不触发业务工具。标签不传给模型，首轮错误保留，
不同样本使用固定正确前态，避免把前一次误判当成下一次的事实。个人目标选项通过ID对应输入中的摘要，
因此本实验包含跨字段关联，不能当作单句分类能力的全面评价。

```bash
# 在独立虚拟环境安装固定版本，不修改主Agent环境。
.venv/bin/python -m venv runtime_data/decision_model_trial/venv
runtime_data/decision_model_trial/venv/bin/python -m pip install laya==0.4.22
runtime_data/decision_model_trial/venv/bin/python scripts/run_group_decision_model_eval.py \
  --backend laya --cache-root runtime_data/decision_model_trial \
  --output runtime_data/decision_model_trial/laya_first
.venv/bin/python scripts/run_group_decision_model_eval.py \
  --backend localjev --endpoint http://127.0.0.1:8080 \
  --output runtime_data/decision_model_trial/localjev_first
```

Laya试验固定多语言模型revision和MPS设备，预检状态、指令和选项未被截断，分别记录加载、首次预测及后续延迟。
置信度仅作观测，不提供动作授权，也不代表已校准的正确率。
[LocalJev](https://github.com/githubnext/localjev)需要另行运行本地推理服务和API桥接；它的概率由底层生成模型输出，
不等同于直接读取模型logits。没有就绪的本地端点时明确记录 `not_run`，不悄悄换成远端模型。

2026-11-05首轮Laya旁路结果：意图14/33、公共话题8/33、本人目标17/33，三项同时正确0/33；
有18条至少一项高置信误判（阈值0.85）。后续请求中位数约103ms、P95约333ms。
人工抽查未发现标签映射颠倒或输入截断；这仅说明当前模型与输入合同尚未通过本项目验收，
不能概括该模型在其他任务上的效果。LocalJev本轮未运行。生产分流未接入这两个实验后端。
