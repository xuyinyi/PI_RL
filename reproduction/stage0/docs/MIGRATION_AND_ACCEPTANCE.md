# DAPiGen Stage-0 迁移与验收流程

## A. 保留原始环境作为回归基线

先固定当前 DAPiGen commit，并保存：

```bash
git rev-parse HEAD
git status --short
```

不要直接改写 `RL_PPO/moldr/env.py`。Stage-0 overlay 使用独立入口 `RL_PPO/moldr/env_stage0.py`。

## B. 当前 v2.3 安装边界

v2.3 实现已位于仓库根目录 `RL_PPO/envs/`，验收材料位于
`reproduction/stage0/`。用户提供的 v2.1 ZIP 只是 hash 绑定的输入来源；
不要再次运行其 `apply_overlay.sh`覆盖 v2.3。v2.2 验收记录是历史证据，不应被改写。

## C. 无 QSPR 模型的本地验证

```bash
pytest -q reproduction/stage0/tests
python reproduction/stage0/scripts/validate_core_without_models.py
```

这一步验证核心接口和真实 RDKit 反应链，但 stock RDKit fallback 只用于 smoke test，不代表正式 DAPiGen chemistry parity。

## D. Building-block audit

```bash
python reproduction/stage0/scripts/audit_stage0_catalogs.py \
  --dapigen-root /path/to/DAPiGen \
  --mask-mode closure_exact_cached \
  --output stage0_audit/catalogs.json
```

检查：

- invalid rows；
- canonical duplicate rows；
- complete monomers；
- fragments；
- 无 attachment 且不是合法单体的 unusable actions；
- 初始合法动作数；
- 实际 chemistry backend 必须含 `dapigen_custom`。

脚本必须在确定性 rollout 达到的状态上同时构建 label compatibility、`closure_exact_cached` 和独立 full exact mask，报告改进 mask 的假阳性、假阴性、相对 label baseline 的误报削减与构建成本。任何假阴性直接失败；假阳性和性能也必须经验收后才能冻结正式 mask 模式。

## E. QSPR evaluator parity

```bash
python reproduction/stage0/scripts/compare_evaluators.py \
  --dapigen-root /path/to/DAPiGen \
  --input-csv /path/to/DAPiGen/raw_data/PI.csv \
  --sample-size 100 \
  --seed 20260903 \
  --device cpu \
  --tolerance 0 \
  --output-dir stage0_audit/evaluator_parity
```

验收标准：

- validity 完全一致；
- transmittance、CTE、strength、Tg、SA 在 released 四舍五入后完全一致；
- objective 在 4 位小数后完全一致；
- mismatch count = 0。

CPU parity 通过后再验证目标 GPU。任何差异都应先归因到 checkpoint、DGL/PyTorch 或 scaler，而不是放宽阈值。

## F. 固化运行契约

```bash
python reproduction/stage0/scripts/create_stage0_manifest.py \
  --dapigen-root /path/to/DAPiGen \
  --polybert-path /path/to/polyBERT \
  --config configs/stage0_environment.json \
  --device cuda:0 \
  --maximum-requested-calls 100000 \
  --maximum-unique-calls 100000 \
  --output-dir contracts/ppo_seed_0 \
  --run-name ppo_seed_0
```

对 Policy-CC 与 MCC-PPO 重复。`run_name` 只是运行实例的 provenance
标签；默认 `cache_scope=per_run` 描述三者共享的“每次运行独立空
cache”生命周期。三者的：

```text
task_contract_id
```

必须相同；`budget_contract_id` 也必须相同。不要把方法名或 seed
写入 `cache_scope`，否则会伪造预算契约差异。运行目录必须保留 action
catalogs。

## G. P1 后、算法效果实验前的回归矩阵

| 检查 | 配置 | seed 数 | 验收 |
|---|---|---:|---|
| original env 原始 PPO | released | 3 | 仅用于历史参考 |
| Stage-0 legacy-effective PPO | 6 step | 3 | 定位重构差异 |
| Stage-0 standard PPO | 5 step | 5 | 建立正式基线 |
| deterministic replay | standard | ≥1,000 transitions | 逐字段一致 |
| snapshot/restore | standard | ≥1,000 states | 下一步一致 |
| local vs shared Ray evaluator | standard | ≥100 PI | reward 与 ledger 一致 |

不要把 original-env 结果与 Stage-0 standard 结果直接归因于算法差异；两者任务语义不同。

## H. 训练运行必须保存

```text
environment_manifest.json
action_catalogs/*.csv
oracle_ledger.json
oracle_audit.json
resolved_config.json
random seeds
DAPiGen git SHA
Stage-0 source-tree SHA256
polyBERT fingerprint
QSPR evaluator version
checkpoint 中的 task_contract_id、environment_id 和 evaluator state
```

## I. 完成条件

当且仅当：

1. evaluator parity 为零差异；
2. custom BRICS backend 被确认；
3. replay、restore、terminal-only 和 budget tests 全部通过；
4. 开发坐标下的 runtime 和 task manifest 已生成，正式验收仍等待
   clean Git 坐标。

上述 Stage 0 工程验收通过后才能进入 P2 统一训练器开发。标准
PPO 的单 worker/共享 evaluator actor smoke test 属于 P2/P4，不得作为
P1 的前置条件；PPO、Policy-CC 和 MCC-PPO 的三方契约一致性也在
P2 实现后验收。

## J. polyBERT parity

```bash
python reproduction/stage0/scripts/compare_polybert.py \
  --polybert-path /path/to/polyBERT \
  --device cpu \
  --atol 1e-6 \
  --rtol 1e-6 \
  --output stage0_audit/polybert_parity.json
```

所有测试结构必须同时满足：

1. released helper 重复调用一致；
2. persistent encoder cache 重放逐元素一致；
3. released 与 persistent 输出在预注册容差内一致；
4. checkpoint 内容 hash 已写入环境契约。

## K. Ray 全局预算并发验收

```bash
python reproduction/stage0/scripts/validate_ray_evaluator.py \
  --dapigen-root /path/to/DAPiGen \
  --evaluator-mode persistent \
  --device cpu \
  --duplicates 8
```

P1 通过且 P2 的统一 PPOEngine 完成后，再至少用 2、4、8 个
rollout worker 运行 PPO smoke test。所有 worker 必须指向同一个
evaluator actor；每个算法/seed 使用新的 actor、空 cache 和零 ledger。

## L. 三种算法的任务与预算契约一致性

在每种算法正式运行前分别生成 manifest，然后执行：

```bash
python reproduction/stage0/scripts/compare_stage0_contracts.py \
  contracts/ppo/environment_manifest.json \
  contracts/policy_cc/environment_manifest.json \
  contracts/mcc_ppo/environment_manifest.json \
  --output stage0_audit/contract_comparison.json
```

以下字段必须逐字符一致：`task_contract_id`、`budget_contract_id`、
`runtime_contract_id`、`environment_id`、`evaluator_version`、`objective_contract`、
requested/unique 上限和预算协议版本。
