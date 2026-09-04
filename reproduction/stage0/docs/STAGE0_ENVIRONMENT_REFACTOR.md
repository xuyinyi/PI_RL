# DAPiGen Stage 0：统一、可分叉、终局可评价环境规范

## 1. 目标与边界

Stage 0 只固定 PPO、Policy-CC 和 MCC-PPO 共同使用的任务层，不在这一阶段实现或比较信用分配算法。共同任务层包括：

1. 分子状态与动作空间；
2. BRICS 组装和 PI 终局反应；
3. 动作合法性、失败和终止规则；
4. observation encoder；
5. 完整 PI 的终局 QSPR evaluator；
6. evaluator cache 与 oracle budget；
7. 随机性、状态恢复和事实/反事实分叉协议；
8. 环境、模型和预算的版本化契约。

严格禁止：

- 给未完成二酐、二胺或部分 PI 定义最终物理性质；
- 使用终局 reward 决定化学转移或产品选择；
- 让不同算法修改动作掩码、终止规则或 evaluator；
- 在多个 worker 上复制独立预算账本后再汇总为一个实验；
- 混用不同 `task_contract_id` 的结果。

---

## 2. 原始环境中需要纠正的任务定义问题

### 2.1 中间 reward 为零不是错误

原环境只在二酐和二胺均完成、成功生成完整 PI 后计算透光率、CTE、强度、玻璃化温度、SA score 和综合得分。未完成状态没有这些终局物理量，因此 Stage 0 保留：

```text
未完成结构                  reward = 0，不调用 evaluator
成功生成完整 PI             调用统一 terminal evaluator
无效动作/无产物/超原子数     固定 failure_reward，默认 0
设计 horizon 耗尽            固定失败或 time-limit truncation，由配置明确指定
```

后续 MCC-PPO 改进的是 actor 使用的 credit/advantage，而不是环境 reward。

### 2.2 全局随机数破坏可重放性

原环境的中间候选选择使用 `np.random.choice`，终局 PI 生成还使用 Python `random.choice`。事实和反事实分支即使从同一前缀开始，也会因调用顺序不同而消耗不同随机数。

Stage 0 将随机性改为无状态命名流：

```text
derive_seed(run_seed, episode, step)
       ├── dianhydride_product
       ├── diamine_product
       └── terminal_polyimide_product
```

相同 `(state, action, transition_seed)` 必须逐字段重放一致。不同命名流互不移动随机数游标。

### 2.3 终局 `argmax` 构成 oracle 泄漏

原环境枚举多个 PI 候选后，先对全部候选调用性质模型，再选择 reward 最大者。这使 terminal evaluator 成为 transition kernel 的一部分，相当于环境替 agent 做了一次隐含搜索。

Stage 0 的顺序固定为：

```text
枚举并 canonicalize 全部合法终局 PI
                 ↓
按 canonical_first 或 seeded_uniform 选一个
                 ↓
只对已选 PI 调用 evaluator
```

产品选择与任何性质或 reward 无关。

### 2.4 完整 building block 擦除前缀

原环境只要动作块不含 `*`，就直接用该块替换当前侧结构。因此后期动作可以把此前所有生长历史抹掉，使早期动作信用不再有稳定含义。

主配置采用：

```text
complete_block_policy = pristine_only
```

完整单体只可在对应侧尚未生长时选择。长时域配置可使用 `never`，完全移除直接完成动作。`legacy_replace` 仅用于回归诊断。

### 2.5 已完成侧存在大量动作别名

原环境在一侧完成后仍接收该侧任意 action，但全部静默忽略。这会造成许多环境等价动作、浪费策略概率质量，并使反事实干预含义模糊。

Stage 0 在每侧动作目录末尾增加唯一 NOOP：

```text
dianhydride_noop_id = number_of_dianhydride_blocks
diamine_noop_id     = number_of_diamine_blocks
```

未完成侧不能使用 NOOP；已完成侧只能使用 NOOP。

### 2.6 horizon 存在一步偏移

原代码在 `env_step > step_length` 时结束，因此配置 5 实际最多执行 6 步。Stage 0 在完成第 `max_steps` 个动作后立即结束。标准配置使用 5 步；legacy-effective 回归配置使用 6 步。

### 2.7 原状态恢复接口不包含化学状态

原 `set_state()` 没有恢复二酐/二胺 SMILES、完成标志、逐侧生长次数和步骤索引，无法支持严格的事实/反事实分叉。

Stage 0 使用不可变 `DAPiGenState`，成功终局 snapshot 还保存被选中的 `terminal_smiles` 和候选集合。

---

## 3. 架构分层

```text
Algorithm layer
PPO / Policy-CC / MCC-PPO
           │ action
           ▼
Stateful wrapper or controller
Gym 0.19 / Gymnasium / direct controller
           │ (state, action, seed)
           ▼
BranchableDAPiGenCore
pure chemistry state machine; no reward model
           │ CoreTransition
           ▼
TerminalRewardAdapter
intermediate=0; failed=fixed; success=evaluator score
           │ complete PI only
           ▼
BudgetedCachingTerminalEvaluator
canonicalization, validation, cache, requested/unique budget
           │
           ▼
PersistentDAPiGenBenchmarkEvaluator
four frozen QSPR networks + SA + released objective formula
```

该分层确保：环境转移永远不能访问终局属性，算法也不能绕过统一预算账本。

---

## 4. 状态、动作和 observation

### 4.1 不可变状态

```python
DAPiGenState(
    dianhydride_smiles: str,
    diamine_smiles: str,
    environment_id: str,
    dianhydride_complete: bool,
    diamine_complete: bool,
    dianhydride_growth_steps: int,
    diamine_growth_steps: int,
    step_index: int,
    max_steps: int,
    terminated: bool,
    truncated: bool,
    termination_reason: Optional[str],
    schema_version: int = 3,
)
```

`state_id` 是上述字段（包括 `environment_id`）稳定 JSON 的 SHA-256 前 24 位。它用于：

- 检查事实与反事实是否来自同一个前缀；
- 轨迹与异常审计；
- checkpoint 恢复；
- 测试确定性。

终局候选产物及已选 `terminal_smiles` 属于 `CoreTransition`，不属于
`state_id`。终局评价/cache 去重必须使用 canonical `terminal_smiles`，不能
只使用 `state_id`。恢复成功终局 snapshot 时会重算化学候选集以拒绝
损坏或不一致的候选产物记录，但不调用 QSPR evaluator。

旧环境 snapshot 缺少逐侧生长计数，不能无损恢复；新实现会明确拒绝，而不是猜测。

### 4.2 动作

```python
DAPiGenAction(dianhydride_id, diamine_id)
```

动作 ID 必须是整数值；`1.5` 和 Boolean 不会被静默转换为整数。

### 4.3 动作掩码

主配置使用 `closure_exact_cached`：

1. 用 BRICS attachment label 做可扩展的 side-level 预筛；
2. 若一次连接后仍有 attachment，保留标签兼容动作；
3. 若一次连接将消耗双方最后的 attachment，只执行 label-pair 匹配的 released reverse-BRICS reaction，并要求产物是完整 side-specific monomer；
4. 实际选中动作的 transition 仍运行完整 BRICS 候选枚举。

其他模式：

| 模式 | 含义 | 使用场景 |
|---|---|---|
| `closure_exact_cached` | 标签预筛 + 最后 attachment 的定向 exact 闭合检查 | 正式大动作空间候选 |
| `compatibility` | 仅标签预筛，选中后精确验证 | 历史对照/误报基线 |
| `exact_cached` | 构建 mask 时对每个动作枚举产物 | 小动作空间、掩码审计 |
| `all` | 除明显不可用项外不做反应兼容预筛 | legacy 回归 |

注意：掩码是两个 factorized side masks。它不能预先排除“两个单体各自合法、但二者终局 PI 反应失败”的联合动作；该情形按 `polyimide_reaction_failed` 终止。

### 4.4 Exact state and learned observation

主配置 `augmented_v3`：

```text
polyBERT(dianhydride partial structure)
polyBERT(diamine partial structure)
12-dimensional metadata
two attachment-label histograms for BRICS labels 0..16
```

metadata：

```text
两侧完成标志                       2
两侧是否仍为 pristine              2
step_index/max_steps               1
remaining_steps/max_steps          1
两侧 growth_steps/max_steps        2
两侧 atom_count/max_atoms          2
两侧 attachment_count/4            2
```

不可变 `DAPiGenState` 是转移所需的 exact Markov state。polyBERT 向量是有损的 learned observation，不宣称为单射或严格 Markov 编码。`augmented_v3` 显式补入转移依赖的 attachment-label 计数；`markov_v2` 和 `legacy_polybert` 仅用于回归。

polyBERT checkpoint 以内容 hash 标识，而不是本地绝对路径。外部声明的 fingerprint 必须与本地 checkpoint 内容重算值相同。

---

## 5. 纯化学状态机

核心接口：

```python
initial = core.initial(seed)
mask = core.valid_action_mask(state)
transition = core.transition(state, action, seed)
observation = core.observe(state)
restored = core.restore_transition(snapshot)
```

### 5.1 确定性不变量

```python
core.transition(s, a, z) == core.transition(s, a, z)
```

候选集合先 canonicalize、去重、排序，再使用命名随机分位数选择。因此集合枚举顺序不会改变结果。

### 5.2 正式运行禁止静默使用 stock RDKit fallback

DAPiGen 自带修改后的 `RL_PPO.moldr.utils.BRICSBuild`。正式 factory 默认要求成功导入该实现；如果导入失败会停止，而不会静默切换为 stock RDKit BRICS。stock fallback 只允许无仓库模型的 isolated smoke test，并且会写入不同的 `chemistry_backend` 与 `environment_id`。

### 5.3 终止原因

```text
success
invalid_action
no_reaction_product
no_valid_continuation
atom_limit_exceeded
design_horizon_exhausted
polyimide_reaction_failed
```

每个终局状态必须有唯一 reason；未结束状态不得携带 reason。

---

## 6. 终局 evaluator 与 reward 边界

### 6.1 TerminalRewardAdapter 是唯一 reward 入口

```python
result = reward_adapter.apply(core_transition, source="ppo/on_policy")
```

规则：

| transition | evaluator call | exposed reward |
|---|---:|---:|
| 中间状态 | 0 | 0 |
| 成功完整 PI | 1 logical request | QSPR objective |
| 化学失败 | 0 | `failure_reward` |
| 外部 time-limit truncation | 0 | 0，critic 可 bootstrap |

任何含 `*` 的结构都会被 evaluator service 拒绝。

### 6.2 持久化 QSPR evaluator

`PersistentDAPiGenBenchmarkEvaluator` 一次加载：

- transmittance AFP；
- CTE AFP；
- strength AFP；
- Tg AFP；
- 四个 scaler；
- SA fragment table。

它保留 released `Benchmark` 的属性四舍五入和综合目标：

\[
R = \frac{T}{100}
\frac{1 + S_{CTE} + S_{strength} + S_{T_g} + S_{SA}}{5}.
\]

这是论文主文 Equation (1) 与已归档 bytecode 采用的加权算术形式，
不是 public-source geometric variant。两者不能在同一任务契约中混用。

正式训练前必须运行 `scripts/compare_evaluators.py`，在固定 PI 面板上验证 persistent 与 released evaluator 在四舍五入后的所有输出完全一致。未通过时不能开始算法比较。

### 6.3 evaluator version

QSPR model、scaler、settings 和 SA table 的 SHA-256 共同构成 `evaluator_version`。替换任一 checkpoint 都会改变任务契约。

---

## 7. Oracle budget 协议

账本同时记录：

```text
requested_calls      逻辑上请求评价的完整 PI 数，包括 cache hit
unique_calls         当前 ledger 首次出现的 canonical PI 数
backend_calls        cache miss 后提交到 QSPR 服务边界的分子数，包括驱逐后重算/重试
cache_hits           重复结构复用次数
invalid_results      evaluator 失败或拒绝的结果数
requested_by_source  各算法阶段的逻辑请求
unique_by_source     各算法阶段的新结构请求
backend_by_source    各算法阶段的 cache-miss 服务边界提交
```

正式公平比较建议：

1. 每个算法/seed 使用全新的 evaluator service、cache 和 ledger；
2. 三种算法具有相同 `maximum_requested_calls`；
3. 训练在 requested budget 耗尽时停止，而不是固定 PPO iteration；
4. 同时报告 requested 和 unique calls；
5. 若还约束真实新 QSPR 成本，再设置相同 `maximum_unique_calls`；
6. evaluation/holdout 调用使用独立账本，不消耗训练预算。

推荐来源标签：

```text
ppo/on_policy
policy_cc/on_policy
policy_cc/factual
policy_cc/counterfactual
mcc_ppo/on_policy
mcc_ppo/factual
mcc_ppo/counterfactual
evaluation
environment_regression
```

### 7.1 多 worker

不能让每个 RLlib worker 各自持有 `BudgetedCachingTerminalEvaluator`。正式多 worker 运行应创建一个：

```python
actor = create_stage0_evaluator_actor(...)
client = RayTerminalEvaluatorClient(actor)
```

再把同一个 client 作为 `evaluator_service` 传给所有环境。Ray actor 串行维护全局 cache 和预算事务。

---

## 8. 环境和实验契约

`write_environment_manifest()` 生成：

```text
environment_id            化学规则、配置、动作目录、初始结构和 encoder identity
evaluator_version         QSPR/scaler/SA artifacts
task_contract_id          环境 + evaluator + reward adapter + 完整 Stage-0/QSPR 源码树 + runtime
budget_contract_id        task contract + requested/unique budget + cache scope
implementation_sha256     core/chemistry/encoder/evaluator/reward 文件 hash
runtime                    Python/RDKit/NumPy/Torch/Transformers/Gym/Ray 版本
```

`write_action_catalogs()` 生成两个固定 CSV，包含：

- action ID；
- 原 CSV 行号；
- canonical SMILES；
- fragment / complete_monomer / unusable / noop；
- attachment labels；
- attachment count；
- atom count。

PPO、Policy-CC 和 MCC-PPO 只有在 `task_contract_id` 完全相同的条件下才能进入同一主比较。

---

## 9. 配置剖面

### 9.1 `stage0_environment.json`

标准、最小改动的 DAPiGen Stage-0 任务：

- `max_steps=5`，修复 off-by-one；
- `closure_exact_cached` mask；
- complete block 只在 pristine side 可用；
- reward-independent seeded terminal-product selection；
- `augmented_v3` observation。

### 9.2 `stage0_long_horizon_example.json`

用于证明 delayed terminal reward 问题，而不是替代标准结果：

- `max_steps=12`；
- 两侧至少各执行 3 次生长；
- 禁止直接 complete block；
- `max_atoms=100`。

正式论文还应冻结 8/12/16 步版本，并分别生成 contract。

### 9.3 `stage0_exact_mask_audit.json`

用于小规模精确掩码审计，不建议直接作为大规模默认训练配置。

### 9.4 `stage0_legacy_effective_regression.json`

近似原始有效 6-step 行为，仅用于定位重构造成的结果变化。它仍然不会恢复 reward-dependent terminal argmax，因为该机制本身违反环境—evaluator 分离。

---

## 10. Stage-0 验收门槛

环境进入算法开发前，以下门槛必须全部通过：

- [ ] 全目录 audit 完成，无未解释 invalid row；
- [ ] action ID CSV 已固化并纳入实验 artifact；
- [ ] custom DAPiGen BRICS backend 被实际加载，非 fallback；
- [ ] 100 个固定 PI 上 persistent evaluator 与 released Benchmark 完全一致；
- [ ] 相同 state/action/seed 重放一致；
- [ ] snapshot/restore 后下一步逐字段一致；
- [ ] 中间和失败状态 evaluator call 为零；
- [ ] terminal product 在替换 evaluator 后不改变；
- [ ] 单 worker 与共享 Ray actor 的 reward/ledger 一致；
- [ ] 三个算法的 `task_contract_id` 相同；
- [ ] 三个算法按相同 requested-call budget 停止；
- [ ] 所有 run 保存 manifest、action catalogs、ledger 和 audit events。

只有此后才进入 Stage 1：原始 PPO 在统一环境上的复现与性能基线。
