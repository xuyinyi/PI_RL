# DAPiGen / LLM-SciCF 开发交接文档

更新日期：2026-09-08，Asia/Shanghai。面向接手开发者及后续 AI 编程助手。

本交接基于本地仓库、已归档实验报告，以及当日约 09:11–09:14 对 n001 的只读核验。
交接前最新代码/证据提交为 `4d35d3e3bfb3b615928e63fe3c624bf92af192d6`；本文件自身属于其后的文档提交。
查看最终文档版本用 `git log -1`，不要将文档提交误认为实验执行版本。

## 1. 接手时首先要知道的结论

当前主线是用户选择的 **LLM-SciCF 架构优先路线**：先跑通 PPO、LLM 候选选择、AFP 反事实验证和辅助更新，再逐模块完善和验证效果。

已经完成一轮真实六迭代 SciCF 工程运行，以及同种子、同 PPO 训练预算的六迭代 PPO-only 对照。基础工程链路可用，但有效性证据仍处于单种子探索阶段。

- SciCF：n001 Slurm `4748`，六轮、768 transitions、48 个 PPO 优化步、6 个辅助优化步。
- PPO-only：n001 Slurm `4751`，六轮、768 transitions、48 个 PPO 优化步、零辅助优化步。
- 初始策略、第一轮 PPO 更新后的策略、rollout、GAE、critic returns 精确匹配。
- 可评估终止结果：158 对 146；每 transition 的终止产出率为 20.57% 对 19.01%，差 1.5625 个百分点。
- SciCF 实际 evaluator 请求为 299，对照为 146。两者实际总成本不相同。
- 第一至第五轮累计 SciCF 少 3 个终止结果，第六轮多 15 个；不能据此宣称稳定收益。
- 没有共同的完整奖励分布或最终策略固定预算评估，尚不能判断奖励提升。
- 两个单次真实运行授权已消耗，无自动重跑或续训安排。
- 当日核验时 `squeue -u wch` 为空；这是快照，接手后应重新查询。

三个容易混淆的状态：**程序成功执行、算法效果得到验证、科学结论成立**，目前完成的是第一项及一个有限对照。

## 2. 项目目标、路线与历史关系

研究对象是基于片段生成聚酰亚胺的 DAPiGen。完整分子才具有该任务定义下的终端奖励；AFP 是重建的 QSPR 代理 evaluator，不是实验测得的科学真值。

项目有四条需要区分的线：

| 路线 | 作用 | 当前定位 |
|---|---|---|
| 原始 RLlib PPO compatibility reproduction | 修复公开代码、重建运行资产、冻结可运行参照 | 已完成并保留；不是作者原权重/论文结果复现 |
| Stage 0 + 共享原生 PPO 引擎 | 稳定状态、动作、mask、回放和 evaluator 预算接口 | 当前在线工作的基础设施 |
| 离线 Gate 1–1B.3 | 检查候选排序/描述符可学习性、结构隔离泛化 | 实验已完成但 efficacy gate 未通过；历史诊断 |
| 在线 LLM-SciCF | PPO 主更新后做 LLM 选候选和 Oracle 软偏好辅助更新 | 当前主线，已完成六轮工程对照 |

另有 2026-09-04 的 Policy-CC / MCC-PPO 参考路线：它计划通过反事实估计器改变 actor credit，包含 completion-value ensemble、DR correction、selection adjustment 等。这些不能算作当前 SciCF 的已实现模块；参考文件的十个 toy tests 也不是集成验收。

2026-09-06 用户选择在线架构优先，改变了工程顺序。离线 no-go 仍成立，但不再阻止已单独授权的在线工程 smoke。不要重新把这些历史 gate 设为默认开发阻塞，也不要把后续工程通过回填成离线 gate 通过。

原始设计来源记录在 `scicf/specification-source.json`；MCC 参考材料在 `reference-intake-20260904.md`。文件名包含 OpenSpec 不代表完整生产实现已经完成验收；本次没有找到仓库内可作为当前已验收实现清单的 OpenSpec change tree。

## 3. 项目位置与版本

除特别说明，后文相对路径均以 `dapigen-reproduction/` 为根。

### 3.1 本地

- 工作区：`/Users/steveo/Documents/ChatGPT/LLM_RL_高分子`
- 开发仓库：`/Users/steveo/Documents/ChatGPT/LLM_RL_高分子/dapigen-reproduction`
- 当前开发分支：`codex/scicf-soft-pair-resilience`
- 最新总进度入口：`reproduction/PROJECT_PLAN.md`
- 最新比较报告：`reproduction/results/ppo-only-matched-short-horizon-20260908/README.md`
- 冻结 compatibility tag：`dapigen-ppo-compat-baseline-v1`
- 上游原始提交：`5f692946cbe0d15eede882dfe4cff7fb26eb7d8c`

用户要求：**本地只编辑、阅读、Git 管理和证据文件传输；实验代码、测试、训练、评估只在服务器上执行，并通过 Slurm 提交。**

### 3.2 n001

SSH alias：`n001`；真实 hostname：`yanlih100n1`；用户：`wch`；Slurm partition：`compute`。

| 用途 | 绝对路径 |
|---|---|
| 远端项目根 | `/home/wch/workspaces/DAPiGen-reproduction` |
| 上游 Git 对象库/主工作树 | `/home/wch/workspaces/DAPiGen-reproduction/upstream` |
| 旧 compatibility 工作树 | `/home/wch/workspaces/DAPiGen-reproduction/worktree` |
| 当前 PPO-only 运行工作树 | `/home/wch/workspaces/DAPiGen-reproduction/ppo-only-control-a1059ca` |
| SciCF 六轮运行工作树 | `/home/wch/workspaces/DAPiGen-reproduction/scicf-short-horizon-5f2a6a1` |
| 验证过的 Python | `/home/wch/workspaces/DAPiGen-reproduction/stage0-v23-mask-dev-20260904/.test-venv/bin/python` |
| 依赖附加根 | `/home/wch/workspaces/DAPiGen-reproduction/dependencies/p2-gymnasium-0.29.1-py38-363fec3` |
| 基础 Conda 环境 | `/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38` |
| polyBERT 资产 | `/home/wch/workspaces/DAPiGen-reproduction/stage0-v23-mask-dev-20260904/RL_PPO/models` |
| AFP 资产 | `/home/wch/workspaces/DAPiGen-reproduction/stage0-v23-mask-dev-20260904/RL_PPO/GNN/model` |
| 运行结果 | `/home/wch/workspaces/DAPiGen-reproduction/runs` |
| Slurm 日志 | `/home/wch/workspaces/DAPiGen-reproduction/slurm-logs` |
| Git bundle / 证据传输 | 项目根下 `bundles/`、`evidence-transfer/` |

基础 `dapigen-py38/bin/python` 没有 pytest，导致 preflight 4749 在测试启动前失败。已核验的 `.test-venv/bin/python` 使用该基础环境及测试依赖；后续必须使用上述真实路径，不能根据环境名猜测。

远端 `upstream/main` 仍是原始提交；不要在它上面直接续写开发。执行工作树 `a1059ca` 刻意保持 clean/detached，证据提交不应强行移动它。最新文档和证据以本地开发分支为准；Git 对象已传到服务器不等于远端工作树已经切到最新提交。

### 3.3 实验版本索引

| 提交 | 含义 |
|---|---|
| `373b329` | Stage 0 v2.3 / P1 正式 compatibility 验收 |
| `6a4bd2c` | 共享原生 PPO/GAE 引擎 smoke |
| `0c33bce` | P4-A 标准五步 baseline |
| `39cd7fb` | 显式 AFP evaluator 路由和绑定 |
| `93501f3` | 单迭代 v3 optional-LLM / K=5 runner |
| `5f2a6a1` | SciCF 六轮 runner；Job 4748 实际执行版本 |
| `8bbc5bb` | Job 4748 结果证据归档 |
| `8bd5423` | PPO-only 及分析协议冻结 |
| `a1059ca` | PPO-only runner / analyzer；Jobs 4750、4751 实际执行版本 |
| `a497db3` | PPO-only 一次性授权记录 |
| `4d35d3e` | PPO-only 及比较结果证据归档 |

## 4. 已完成事项与证据

### 4.1 资产与原始 compatibility baseline

1. Linux/H100、DGL、旧 RLlib/Gym 兼容环境已经打通。
2. polyBERT 用户指定镜像 `xushijie/polyBERT@e7dce434fb3eff37905dc114008660e5479ca9a8` 已下载、验证 600-D embedding；主仓库 identity 未闭合。
3. 重建 PPO 实际加载的四个 AFP 成员：transmittance(400) 43、cte 56、strength 84、tg 64。
4. “12 个 AFP 文件”指四组 weights/scaler/settings，不是已训练十二个独立奖励模型；当前完整 evaluator 指纹还包含 `fpscores.pkl.gz`，共 13 个文件。
5. 全量 AFP 训练数组 Job 4551 和 runtime validation 4558 完成，详见 `reproduction/results/afp-full-20260830/`。
6. RLlib compatibility PPO Job 4562 完成 100 iterations、99,000 steps、11 个 checkpoint、110,000 条生成评估样本。
7. baseline 已冻结。最终 10,000 样本有效率 1.0、平均 reward 0.551809，但只有两个 unique valid PI，存在明显模式坍塌。

证据入口：`reproduction/baseline-freeze.md`、`reproduction/results/ppo-compat-20260830/`。
这些结果不能和后来标准五步、1,246-D Stage 0 任务直接混为同一 baseline。

### 4.2 通用框架、Stage 0、共享 PPO

- `AlgorithmAdapter + CommonEvaluator + Slurm runner` 已实现，Jobs 4563/4565 完成工程验证和 common metrics 验证。
- 旧框架任务为 1,200-D polyBERT 观测；当前原生引擎使用 1,246-D augmented_v3。接口和 evaluation schema 可以参考，不能直接声称已经接通当前 SciCF checkpoint。
- Stage 0 v2.3 在 Jobs 4659–4661 接受：custom BRICS、polyBERT/evaluator parity、精确 mask、1,000-transition replay/restore、Ray 全局账本。
- P2-A mask profile Job 4663 与真实完整 stack profile Job 4665 完成。
- P2 原生 PPO/GAE Job 4671 通过 26 项测试，运行两个 64-transition 迭代，checkpoint/resume 对齐 13 项身份。
- P2 仅 PPO/GAE 子范围完成；Policy-CC、MCC-PPO 和 I02–I04 集成仍未完成。

### 4.3 离线 Gate 历史

| 阶段 | 完成状态 | 结论与限制 |
|---|---|---|
| Gate 1 本地 Qwen | 已执行、no-go | 0/3 阶段通过；历史使用本地 LLM，不代表当前还用本地 LLM |
| Gate 1 DeepSeek | 已执行、no-go | 24 请求；prompt-starved-v2 负向诊断，信息不足，不能推翻完整 SciCF 假设 |
| Gate 1A headroom | 审计完成 | 存在 Oracle-minus-Random 空间，late 稀疏/饱和 |
| Gate 1B 描述符学习 | 已执行、no-go | early rescue 有信号，middle 排序未达标准 |
| Gate 1B.1 | 开发集 no-go | train/dev 结构隔离；middle 通过，early 和 late calibration 未通过 |
| Gate 1B.2 | 开发集 no-go | RF validity 改善 early；middle NDCG 捕获 23.50%，低于 25%；late 仍不可校准 |
| Gate 1B.3 fresh-dev | 采集与评估均完成、no-go | 32 条 fresh trajectories、1,491 Oracle calls；early 捕获 2.63%、middle BestGain 4.57%、NDCG 劣于 Random |

Gate 1B.3 collection Jobs 4635/4637、evaluation 4640。所有历史失败结果保留；sealed test 没有被该路线解封。当前 descriptor router 没有被接入在线训练作为可靠决策器。

结果目录：`reproduction/results/scicf-gate1*/`；历史清单：`reproduction/scicf/task-status.md`。

### 4.4 在线 SciCF 开发和修复

| 阶段/Job | 状态 | 交接要点 |
|---|---|---|
| 4684 architecture smoke | 通过 | K=1 完整链路跑通 |
| 4685 acquisition/verifier | 工程通过 | 同池四策略比较、96 候选 K=2 验证、无策略更新 |
| 4686 pairwise stability | 通过 | 12 独立一步 probes；不等同于长期训练稳定性 |
| 4688 单迭代 v1 | no-go | DeepSeek 返回 out-of-pool ID `cf-281`，失败证据保留 |
| schema robustness development | 通过 | invalid raw response 先存盘；有上限的修复/重试 |
| 4693 v2 real | 失败 | polyBERT 指向源码包而非模型资产 |
| 4694 model preflight | 通过 | 14 文件完整性和模型指纹绑定 |
| 4695 model-bound real | 失败 | clean worktree 缺失 ignored AFP 资产 |
| 4697 full-runtime preflight | 通过 | 显式 AFP 路由、13 文件绑定 |
| 4713 schema-3 real | 执行完成、准入 no-go | 8 候选 K=2 全部含零增益或符号冲突，零 accepted pair |
| 4714 soft-pair component / 4715 v3 preflight | 通过 | K=5、软权重、可降级 LLM；v3 完整 runtime 无凭据验证 |
| 4736 schema-4 real | 通过 | 128 transitions、8 候选、4 非零软权重、1 辅助更新 |
| 4737 short-horizon preflight | 通过 | breaker/checkpoint/时间上限/KL 数值边界 |
| 4748 schema-5 real | 通过 | 六轮 SciCF；12 pool decisions、13 HTTP transmissions、47 K=5 候选、20 非零权重 |
| 4750 / 4751 PPO-only control | 通过 | 38 tests；六轮真实对照及分析完成 |

Job 4748 中一条 schema-invalid 响应在一次修复后恢复；未发生实际 provider failure，因此真实任务未触发 circuit-open。熔断冷却分支有 synthetic preflight 证据，不能写成“在真实 API 故障中已观测”。

## 5. 当前在线算法到底如何工作

```text
冻结本轮行为策略 pi_old
  -> 收集 128 个真实 on-policy transitions
  -> 环境 reward 计算 GAE / critic returns
  -> 标准 PPO 更新，先保存 primary checkpoint
  -> 从完整轨迹构造两个盲化候选池，各 24 个候选
  -> DeepSeek 每池最多选 4 个，可少选或 abstain
  -> 只对选中候选做 5 次 matched factual/counterfactual continuation
  -> AFP 终端差值聚合为经验软权重
  -> 若软权重条件满足，最多一次 SciCF pairwise 辅助更新
  -> KL / value-drift 等检查，必要时辅助回滚
  -> 保存引擎、控制状态、预算、RNG 和 hash chain
```

LLM 不预测可信 reward，不作为 critic，不提供 loss 的置信权重。训练权重来自 AFP factual/counterfactual 结果。反事实动作不混入标准 PPO clipping 样本。

K=5 指同一已选干预的五次匹配 continuation 验证，不是让 LLM 对同一个问题回答五遍后投票。软权重结合符号证据、非零比例和增益量级；它允许不确定性降低贡献，而非把每个候选硬判为可用/不可用。

可恢复的超时、限流、transport/schema 耗尽、abstain、软质量不足或辅助 KL 回滚，会记录明确的 `ppo_only_degraded` 状态并保留 PPO 主更新。资产错误、预算违规、信息泄漏、非法 ID 等完整性错误仍然应停止。

| 当前冻结参数 | 值 |
|---|---:|
| PPO seed / environment base seed | 20260907 |
| 六轮 smoke 的 rollout_steps | 128 / 轮 |
| PPO epochs / minibatch | 2 / 32 |
| PPO learning rate / target KL | 3e-4 / 0.02 |
| PPO hidden_sizes | [128, 128] |
| gamma / GAE lambda | 1.0 / 0.95 |
| 候选池 | 2×24 / 轮 |
| 选中候选上限 | 8 / 轮 |
| matched replicates | K=5 |
| verification branches 上限 | 80 / 轮 |
| practical delta tolerance | 0.005 |
| Beta prior alpha / magnitude scale | 0.5 / 0.05 |
| minimum/maximum training weight | 0.05 / 2.0 |
| minimum effective mass / weighted count | 0.5 / 2 |
| 单 pair 最大质量比例 | 0.75 |
| 辅助更新 learning rate / 步数 | 1e-5 / 最多 1 |
| LLM wall time | 每轮 60 秒、总计 180 秒 |
| circuit breaker | 连续失败 2 次，冷却 3 轮 |
| 微负 KL | [-1e-6, 0) 记录为 0；低于 -1e-6 视为错误 |
| evaluator requested/unique 上限 | 1248 / 六轮 |

这些是工程 smoke 参数，不代表已调优的最终算法参数。正式比较需要按问题重新冻结预算和终点，不能随意扩展已有一次性授权。

## 6. 开发者代码导航

| 要改/理解的模块 | 入口 | 注意事项 |
|---|---|---|
| 状态、动作、化学转换、mask | `RL_PPO/envs/`，`RL_PPO/moldr/env_stage0.py` | 当前任务唯一实现来源，不另造 PIState/PIAction |
| PPO / checkpoint / RNG | `reproduction/p2/engine.py` | 对照需共享该引擎，不为对齐结果更改优化器 |
| GAE / 预算 / contracts | `reproduction/p2/gae.py`、`budget.py`、`contracts.py` | actor 与 critic 的目标来源分别审计 |
| 当前候选池、matched verifier | `reproduction/scicf/online/pipeline.py` | 行为策略冻结、匹配随机流、合法候选池 |
| 在线 prompt / 响应 schema | `reproduction/scicf/online/prompt.py`、`response_guard.py` | request-time 信息与 Oracle 真值分离 |
| DeepSeek transport | `reproduction/scicf/llm/api_client.py` | 官方兼容 API；不把 key 放入参数或日志 |
| K=5 软权重与辅助更新 | `reproduction/scicf/online/soft_pair.py` | 权重来自验证数据，非 LLM confidence |
| 熔断与时间预算 | `reproduction/scicf/online/resilience.py` | checkpoint 后预算不能重置 |
| 控制 checkpoint / 协议 | `reproduction/scicf/online/short_horizon.py` | 明确 resume identity，不能自动续跑 |
| SciCF 六轮 runner | `reproduction/scicf/online/run_short_horizon_multi_iteration.py` | Job 4748 的入口 |
| PPO-only runtime/guard | `reproduction/scicf/online/ppo_only_control.py` | 共享声明含 evaluation；实际 wrapper 禁止非 on-policy 调用 |
| PPO-only runner / analyzer | `run_ppo_only_control.py`、`analyze_ppo_only_control.py`（同目录） | 首轮等价、单次授权、逐轮证据、固定分母 |
| 模型完整性与路由 | `reproduction/scicf/online/model_asset.py`、`evaluator_asset.py` | 模型必须显式指向资产目录 |
| 通用旧框架 | `reproduction/framework/` | `contracts.py` / `evaluator.py`；注意 1,200-D 旧任务 |
| P4-A 旧标准 baseline | `reproduction/p4/` | audit/aggregate 脚本已存在，完整结案未完成 |

当前代码没有已接通的统一在线训练可视化平台证据。原生 runner 的权威日志为 JSON/JSONL 和 Slurm stdout/stderr；不要假设已有 SwanLab/W&B/TensorBoard 仪表盘。

## 7. 结果、日志与检查点

### 7.1 关键结果目录

| 结果 | 本地目录（相对仓库根） | 远端目录（项目根下） |
|---|---|---|
| 原始兼容 baseline | `reproduction/results/ppo-compat-20260830/` | `runs/ppo-compat/full-20260830-v1` |
| P1 | `reproduction/results/stage0-v23-p1-373b329-20260904/` | 具体坐标见其 manifest |
| SciCF 六轮 | `reproduction/results/scicf-short-horizon-real-schema5-20260907/` | `runs/scicf-short-horizon-real-schema5-5f2a6a1-20260907` |
| PPO-only 六轮及比较 | `reproduction/results/ppo-only-matched-short-horizon-20260908/` | `runs/ppo-only-matched-short-horizon-v1-seed20260907-20260908` |
| PPO-only preflight | 上述本地目录的 `preflight/` | `runs/ppo-only-preflight-envfix-a1059ca-20260908` |
| P4-A formal | 尚未完整本地同步 | `runs/p4a-native-ppo-0c33bce/formal/seed-20260911` 至 `seed-20260915` |

SciCF 输出目录名中的 `20260907` 是冻结日期；实际 Job 4748 在 2026-09-08 执行，不能仅凭目录名判运行日期。

### 7.2 当前记录内容

- SciCF 总报告：`short-horizon-report.json`；PPO-only 总报告：`control-report.json`。
- 每轮：`iterations/iteration-XX/iteration-report.json`，包括 transition/terminal 数、PPO loss/entropy/KL、GAE/returns hash、policy hash、evaluator ledger。
- SciCF：primary-before-LLM checkpoint、final engine checkpoint、control checkpoint、acquisition/repair 原始记录、soft verification。
- PPO-only：`primary-ppo.pt`、RNG 摘要、previous report hash；最终 analyzer 重算 checkpoint/report chain。
- 比较：`analysis/comparison.json`，固定 SciCF-minus-PPO 方向，不选择最好轮次。
- Slurm：分别保存任务 stdout、stderr、accounting 和授权消耗记录。
- checkpoint `.pt` 和部分原始盲化请求/响应留在 n001；Git 保存报告、精选记录和完整文件哈希。

原始请求/响应可能含科研分子与轨迹数据，应保留在原授权范围内。交接不需要读取 API 密钥或重新发送数据。

### 7.3 关键身份锚点

```text
polyBERT fingerprint:
6bdd24f951dd90d3031e749ef0130752811bfefe6c850af82b805cf015ea195f
AFP fingerprint:
0bdcea6155f5a53acd94afd322d408fe3e31c02cf13778a0c65f5f0938096ab4
SciCF Job 4748 report SHA-256:
07ffac03f444a46bbb7e412550d1919b384fd9753777b466478ae08c0e3c1e11
PPO-only Job 4751 report SHA-256:
e70115954580fc43765a99b8404bf194f425797d8f44200e2d04a1b94c58a534
comparison SHA-256:
0696b0626ef8dc6f994effc724389c564cfa1b2097fa032eccb11d94598ca65c
```

资产 binding JSON 另有自身哈希；不能用单个 `.pt` 哈希代替完整目录指纹。准确字段见运行授权与 `model_asset.py` / `evaluator_asset.py`。

## 8. P4-A 遗留状态：已停止，不是仍在运行

当日实查数组 `4676`：

| Seed | Job | report status | 需要处理 |
|---:|---|---|---|
| 20260911 | 4676_0 | failed_gate | evaluation_metrics_complete=false |
| 20260912 | 4676_1 | passed | 仍需完整汇总归档 |
| 20260913 | 4676_2 | passed | 仍需完整汇总归档 |
| 20260914 | 4676_3 | failed_gate | evaluation_metrics_complete=false |
| 20260915 | 4676_4 | passed | 仍需完整汇总归档 |

两个失败报告的其他已显示 acceptance checks 为 true。记录中的预算 stop_reason 与失败的 evaluation gate 是两个概念；不能把正常预算停止直接写成训练崩溃原因。本次仅核实终态/报告标志/哈希，未运行完整 per-seed audit 或重算指标。

原始报告绝对路径和 SHA-256 已记在 `reproduction/results/handoff-status-20260908/REMOTE_STATUS.md`。三成功两失败不能择优删掉失败种子，再声称五种子 baseline 完成。

P4 完整验收还缺 `legacy_effective` 六步语义对照等；现有原始 RLlib compatibility 记录需要正式语义对齐。它们属于遗留 baseline 审计，不等于要重跑当前 SciCF。

## 9. 未完成任务清单与建议顺序

下列项目是待办建议，不是自动运行授权。优先级以当前 SciCF 主线为准。

| ID | 优先级 | 任务 | 已有基础 | 完成标准 |
|---|---|---|---|---|
| H01 | P0 | 统一原生引擎的 on-policy reward/terminal-event 日志 | 当前只有摘要和 checkpoint | 两组同 schema；记录 reward、终止原因、有效分子身份、明确统计分母，测试通过 |
| H02 | P0 | 共同 final-checkpoint evaluator | 旧 CommonEvaluator 和 Stage 0 evaluator | 正确支持 1,246-D 当前策略；固定 evaluation seed/预算；训练与评估账本隔离 |
| H03 | P0 | 冻结正式比较口径 | 当前 interaction-matched 单种子协议 | 明确 primary endpoint、步数/Oracle 实花成本、invalid/重复样本、missing-data、停止规则 |
| H04 | P1 | 区分 LLM 的增量贡献 | 4685 小规模同池四策略诊断 | PPO-only、Random/heuristic+同 K=5 辅助路径、LLM+同路径；预算一致 |
| H05 | P1 | 分离辅助流程 RNG 消耗 | 当前已记录 RNG hash | 测试证明辅助开关不额外扰动下一轮 sampling RNG；任何新设计版本化 |
| H06 | P1 | P4-A 五种子结案 | 5 个原始报告与已存在 audit/aggregate 脚本 | 全部种子纳入、失败评估点定位、哈希归档、清晰终态；不放宽旧规则 |
| H07 | P1 | 多种子、较长训练范围 | 六轮工程框架可用 | 独立种子重复、预先预算、完整结果；不能把 episode 当独立 run |
| H08 | P2 | 长时间可用性/恢复验证 | synthetic breaker 与 checkpoint 已验证 | 外部故障、恢复、总预算持久化的独立工程测试；resume 不重置预算 |
| H09 | P2 | 候选覆盖与 late abstain 改进 | 跨 timestep 构池已实现 | 评价信息量、机会识别和漏选；late 校准需要正/负两类样本 |
| H10 | P2 | 原始资产 identity 和代理模型适用域 | compatibility 指纹已冻结 | 作者权重/主模型可核验，或明确保留 compatibility 并做独立鲁棒性评估 |
| H11 | P2 | 结构泛化、独立 evaluator、第二环境 | 离线结构隔离和通用 domain 接口 | 新协议/数据边界/预算，通过新环境工程验收后再比较 |
| H12 | 可选分支 | Policy-CC/MCC-PPO 集成与 P3 | 参考 intake、共享 engine seam | I02–I04、估计器正确性、pending-label 边界等完成后独立验收 |

建议下一步先解决 H01–H03。当前六轮 pair 已按批准范围完成，无需为了继续开发而重复此前的单次 smoke。若要评估已有 checkpoint，应先区分“新增固定预算评估”与“重新训练”，分别记录资源范围。

## 10. 开发与测试工作流

### 10.1 接手第一轮只读检查

以下命令用于查看已有状态，不运行实验代码：

```bash
git -C /Users/steveo/Documents/ChatGPT/LLM_RL_高分子/dapigen-reproduction status --short
git -C /Users/steveo/Documents/ChatGPT/LLM_RL_高分子/dapigen-reproduction log -8 --oneline
ssh n001 'hostname; squeue -u wch'
ssh n001 'sacct -j 4748,4750,4751 --format=JobID,State,ExitCode,Elapsed --parsable2'
ssh n001 'git -C /home/wch/workspaces/DAPiGen-reproduction/upstream worktree list'
```

Slurm 很快可能不再保留 `scontrol show job` 的记录；历史结果优先查 `sacct` 和归档 accounting/report，不能仅凭 scontrol 找不到任务断言任务不存在。

### 10.2 新代码验证

1. 本地读协议和实现、编辑代码；保留用户已有改动。
2. 将源代码提交到版本库；使用 Git bundle/SSH 传输到 n001 新的隔离 clean worktree。
3. 验证真实 Python、package root、资产目录，不能把模型代码目录当权重目录。
4. 通过 Slurm 测试，使用已验证环境。需要资产的 runtime preflight 与 synthetic tests 分清。
5. 真实运行绑定协议/分析/实现/preflight/资产指纹、种子、预算及唯一输出目录。
6. 核验报告结论、账本和 checkpoint hash，再归档到 `reproduction/results/`。

当前 Slurm 入口：

- PPO-only：`reproduction/slurm/run_ppo_only_matched_control.sbatch`
- SciCF preflight：`reproduction/slurm/run_scicf_short_horizon_preflight.sbatch`
- SciCF real：`reproduction/slurm/run_scicf_short_horizon_multi_iteration.sbatch`

PPO-only 入口参数为 `MODE REPO PYTHON PACKAGE_ROOT POLYBERT AFP OUTPUT [AUTHORIZATION PREFLIGHT_REPORT]`，`MODE` 是 `preflight` 或 `real`。这是接口说明；不要复用已经存在的输出目录或已消耗授权直接提交。

标准环境变量保存在各 sbatch：`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`DGLBACKEND=pytorch`、单线程 BLAS/OMP、明确 `PYTHONPATH`、`PYTHONDONTWRITEBYTECODE=1`、关闭 tokenizer 并行。

### 10.3 测试覆盖入口

- 通用 PPO/GAE/预算：`reproduction/p2/tests/`。
- 当前对照与分析：`reproduction/tests/test_ppo_only_control.py`。
- online contracts/API/schema：`test_scicf_online.py`、`test_scicf_api.py`、`test_scicf_schema_robustness.py`。
- K=5/v3/六轮：`test_scicf_soft_pair_resilience.py`、`test_scicf_single_iteration_v3.py`、`test_scicf_short_horizon_multi_iteration.py`。
- 环境或模型改动：追加 `reproduction/stage0/tests/` 相应验收；不要把 stub 测试当真实 chemistry/AFP 验收。

以上 pytest 入口都应在 n001 Slurm 内运行。最近 38 tests 只代表 PPO-only 相关套件，不代表整个仓库所有历史测试都在此次重跑。

## 11. API、私有文件与授权

当前使用用户自己的 DeepSeek 官方 API，历史运行记录模型为 `deepseek-v4-flash`，snapshot marker 为 `DeepSeek-V4-Flash-0731-api-snapshot-2026-09-03`。模型名是 rolling alias；这里是实验记录，不是对当前线上版本的实时核验。

服务器凭据位置：`/home/wch/.config/scicf/deepseek-v4-flash-20260903.env`，应保持 `0600`。交接只提供 locator，不含密钥。不得读取并打印 key、将 key 放入 Git/聊天/命令行参数，或为测试随手发出 API 请求。

此前授权的数据范围为盲化轨迹/候选干预与分子描述，不包含 API key、Oracle reward 真值或未盲化分数。信息边界由 request schema 和构建代码共同控制。

PPO-only 不需要凭据，也不需要 LLM。用户已消耗的授权只覆盖已完成的 Job 4748/4751 等特定任务；新预算、新 seed、新真实训练或新的外发范围不能借用旧授权。已经批准的开发任务应持续完成，不应反复询问同一权限。

Mihomo 下载辅助配置历史位于服务器项目 `tools/mihomo/`，监听 localhost。当前活性未检查，训练资产已本地化；无需为了文档交接重新安装代理或同步订阅。

## 12. 已知坑与结果解释风险

1. **原始 README 的直接启动不是可靠入口。** 旧上游含 import/config 错误；按已修复的 Slurm runner 和 source-audit 进入。
2. **clean worktree 不含 ignored 模型文件。** 用显式资产路由，不能临时随机初始化或静默 fallback。
3. **同 seed 不等于后续每条轨迹完全一样。** 初始/首轮等价已验证，辅助更新和 RNG 消耗之后可以分叉。
4. **最后 rollout 不评估最后 checkpoint。** 第六轮 rollout 先于第六轮 PPO/辅助更新，当前统计不代表最终策略质量。
5. **高有效率不等于好生成器。** compatibility baseline 两个分子的模式坍塌是明确反例。
6. **`COMPLETED` 不等于 gate 通过。** 读 report 的 status/decision/完整性字段；P4-A 要保留失败种子。
7. **微小 KL 不证明策略改进。** 它主要描述辅助更新幅度和数值稳定性。
8. **K=5 不消除代理偏差。** 它刻画匹配 continuation 的经验不确定性；AFP 本身仍需独立验证。
9. **不要按模型名称保证可复现。** 固定 source、asset fingerprint、prompt/response、配置和报告哈希。
10. **旧描述符、MCC 估计器和在线 SciCF 不是一个模型。** 文档里的相似词不能替代实现和实验身份。

## 13. 本次交接文档更新范围

- 更新 `PROJECT_PLAN.md` 的当前完成/待办看板和下一步任务；将旧 MCC/P4 work order 明确标为历史。
- 更新 `reproduction/README.md`、`scicf/README.md`、`p4/README.md` 导航与状态。
- 为 `scicf/task-status.md` 添加当前导航，保留原始离线快照及 no-go 文字。
- 新建本交接文档和 `results/handoff-status-20260908/REMOTE_STATUS.md`，记录远端终态和 P4-A 五报告哈希。
- 未更改冻结实验协议、模型资产、代码或历史结果；未启动任何新训练/评估/API 调用。

## 14. 可直接交给下一位开发者的任务说明

> 项目是 DAPiGen / LLM-SciCF，当前分支 `codex/scicf-soft-pair-resilience`。
> 先阅读本交接文档、PROJECT_PLAN.md 和 Job 4751 比较报告。用户要求所有测试、实验代码、训练和评估在 n001 Slurm 内执行，本地仅做编辑/Git/证据传输。
> Stage 0/PPO 基础及六轮 SciCF、PPO-only 对照已完成。首轮等价精确通过；158 vs 146 只是一种子下的可评估终止产出差异，不是奖励提升证据。下一步优先准备共同 reward/terminal-event 日志与当前 1,246-D native checkpoint evaluator，明确 evaluation seeds、预算、invalid/重复处理后再做新评估。
> 保留历史 offline no-go、compatibility baseline、P4-A 失败种子和已消耗单次授权。不要重新训练 AFP、改 mask/reward、重用旧输出目录、自动续跑六轮任务或开启 sealed test。Policy-CC/MCC-PPO 是未集成的可选路线，不是当前 SciCF 的已实现部分。
> 先报告实际代码差距和可审阅实现计划；已授权的开发范围内直接推进，新的真实运行范围应具体绑定版本和预算。
