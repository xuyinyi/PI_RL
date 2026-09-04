# Stage-0 Environment API Reference

## 1. `DAPiGenEnvConfig`

```python
from RL_PPO.envs import DAPiGenEnvConfig

config = DAPiGenEnvConfig(
    max_atoms=60,
    max_steps=5,
    mask_mode="closure_exact_cached",
    complete_block_policy="pristine_only",
    product_selection="seeded_uniform",
    observation_mode="augmented_v3",
    horizon_semantics="failure_terminal",
    invalid_action_handling="raise",
    failure_reward=0.0,
)
```

配置是 frozen dataclass。`config.fingerprint` 可用于配置一致性检查。

## 2. Factory

```python
from RL_PPO.envs import build_stage0_components

components = build_stage0_components(
    dapigen_root="/path/to/DAPiGen",
    polybert_path="/path/to/polyBERT",
    config=config,
    device="cuda:0",
    maximum_requested_calls=100_000,
    maximum_unique_calls=100_000,
    allowed_evaluator_sources=("ppo/on_policy",),
    cache_scope="per_run",
)

core = components.core
evaluator = components.evaluator
reward_adapter = components.reward_adapter
```

正式 factory 默认要求 DAPiGen 自带 custom `BRICSBuild`，不会静默 fallback。

### 共享 Ray evaluator

```python
from RL_PPO.envs import (
    RayTerminalEvaluatorClient,
    create_stage0_evaluator_actor,
)

actor = create_stage0_evaluator_actor(
    dapigen_root="/path/to/DAPiGen",
    evaluator_mode="persistent",
    device="cuda:0",
    maximum_requested_calls=100_000,
    allowed_sources=("ppo/on_policy",),
    cache_scope="per_run",
    num_gpus=1,
)
client = RayTerminalEvaluatorClient(actor)

components = build_stage0_components(
    dapigen_root="/path/to/DAPiGen",
    polybert_path="/path/to/polyBERT",
    config=config,
    evaluator_service=client,
)
```

`evaluator_service` 已经负责全局 budget；factory 不会再次包一层本地 ledger。

## 3. Pure core

### Reset

```python
initial: CoreTransition = core.initial(seed=0)
state = initial.state
observation = initial.observation
action_mask = initial.action_mask
```

### Transition

```python
from RL_PPO.envs import DAPiGenAction

out = core.transition(
    state=state,
    action=DAPiGenAction(dianhydride_id=12, diamine_id=27),
    seed=transition_seed,
)
```

`CoreTransition` 不包含 reward，只有结构变化及可选的完整 `terminal_smiles`。
`DAPiGenState` schema 3 必须带有产生它的 `environment_id`；其他
环境实例不能接收该状态。

### Branch

```python
factual = core.transition(s_t, factual_action, seed=z)
counterfactual = core.transition(s_t, counterfactual_action, seed=z)
```

两者使用同一个不可变 `s_t` 和相同的命名随机分位数。

### Snapshot

```python
snapshot = out.to_snapshot()
restored = core.restore_transition(snapshot)
```

转移 snapshot schema 2 包含 `environment_id`、严格状态、已选终局分子和候选集。
不要直接 deep-copy Gym environment；分叉单位是 `DAPiGenState` / `CoreTransition`。

## 4. Terminal reward

```python
evaluated = reward_adapter.apply(
    out,
    source="mcc_ppo/counterfactual",
)
reward = evaluated.reward
terminal_properties = evaluated.evaluation
```

只有 `out.terminal_smiles` 存在时 evaluator 才被调用。

## 5. Direct controller

```python
from RL_PPO.envs import DAPiGenEpisodeController

controller = DAPiGenEpisodeController(core, reward_adapter, run_seed=0)
initial = controller.reset()
result = controller.step(action, source="ppo/on_policy")
checkpoint = controller.snapshot()
controller.restore(checkpoint)
branch = controller.branch(s_t, alternative_action, z, source="policy_cc/counterfactual")
```

`branch()` 不改变主 episode 的 current state。checkpoint schema 2 同时
保存环境绑定的 transition 和 evaluator cache/ledger；缺少任一部分
都不可声称 exact resume。

## 6. Gym 0.19 / RLlib

```python
from RL_PPO.envs import DAPiGenGymEnv

env = DAPiGenGymEnv(
    core,
    reward_adapter,
    source="ppo/on_policy",
    return_masked_dict_observation=True,
    seed=0,
)
```

Observation dict：

```python
{
    "observations": float32_vector,
    "action_mask": concatenated_int8_mask,
    "dianhydride_mask": int8_mask,
    "diamine_mask": int8_mask,
}
```

RLlib 1.13 可注册：

```python
from RL_PPO.envs.rllib_masked_model import register_stage0_masked_model
model_name = register_stage0_masked_model()
```

然后把 `custom_model=model_name` 写入 PPO config。

## 7. Budget ledger

```python
ledger = evaluator.ledger()
```

关键字段：

```python
ledger["requested_calls"]
ledger["unique_calls"]
ledger["backend_calls"]
ledger["cache_hits"]
ledger["requested_by_source"]
ledger["unique_by_source"]
ledger["backend_by_source"]
ledger["remaining_requested_calls"]
```

- `requested_calls`：所有逻辑终局请求，包括 cache hit；
- `unique_calls`：本 run ledger 中第一次出现的 canonical PI；
- `backend_calls`：由 cache miss 促成、被提交到 QSPR 服务边界的分子数，
  LRU 驱逐或失败重试后可大于 `unique_calls`。

```python
evaluator.write_audit("oracle_audit.json")
```

每个正式算法/seed 必须从空 cache 和零 ledger 开始。禁止保留 cache
同时清零 ledger，因为这会破坏 run 级预算溯源。

## 8. Manifest

```python
from RL_PPO.envs import write_action_catalogs, write_environment_manifest

write_environment_manifest("run/environment_manifest.json", components)
write_action_catalogs("run/action_catalogs", components)
```

训练入口应在启动时读取并打印 `task_contract_id`、
`runtime_contract_id` 和 `objective_contract`，checkpoint 也应保存这些 ID。
正式 manifest 默认拒绝 dirty checkout；`--allow-dirty` 只用于开发验证。

## 9. 直接作为 RLlib EnvContext 环境

正式的 Ray 1.13 训练不需要在 driver 中先构造 `core`。使用只接收
`EnvContext` 的入口：

```python
from RL_PPO.envs import DAPiGenRLlibEnv

trainer = PPOTrainer(
    env=DAPiGenRLlibEnv,
    config={
        "num_workers": 4,
        "env_config": {
            "dapigen_root": "/path/to/DAPiGen",
            "polybert_path": "/path/to/polyBERT",
            "task_config": config.to_dict(),
            "evaluator_actor": shared_actor,
            "source": "ppo/on_policy",
            "seed": 0,
            "encoder_device": "cpu",
        },
    },
)
```

`DAPiGenRLlibEnv` 根据 `worker_index` 与 `vector_index` 派生互不重叠的
worker seed。若 `num_workers > 0` 而未提供 `evaluator_actor`，默认直接报错，
防止每个 worker 无意间获得独立 cache 与独立 oracle budget。

完整入口见：

```bash
python examples/rllib_ppo_stage0.py \
  --dapigen-root /path/to/DAPiGen \
  --polybert-path /path/to/polyBERT \
  --task-config configs/stage0_environment.json
```

## 10. polyBERT 数值回归

持久化 encoder 必须与 released helper 使用相同的 attachment-label 清理、
tokenization、mean pooling 与 checkpoint：

```bash
python scripts/compare_polybert.py \
  --polybert-path /path/to/polyBERT \
  --device cpu \
  --output stage0_audit/polybert_parity.json
```

脚本分别检查 released helper 自身的重复调用一致性、persistent encoder 的
cache 重放一致性，以及二者的数值差异。正式实验中必须保存报告和
`encoder_version`。

## 11. Ray evaluator 并发与全局预算

```bash
python scripts/validate_ray_evaluator.py \
  --dapigen-root /path/to/DAPiGen \
  --evaluator-mode persistent \
  --duplicates 8 \
  --output stage0_audit/ray_evaluator.json
```

首批 8 个并发请求应满足：

```text
requested_calls = 8
unique_calls    = 1
backend_calls   = 1
cache_hits      = 7
```

随后脚本会保存 evaluator state，清空服务，恢复 checkpoint，再请求
同一分子。最终应为 requested=9、unique=1、backend=1、cache-hit=8。
这同时证明并发 rollout worker 共用一个事务性账本，且 checkpoint
恢复不会重复调用 QSPR。

## 12. Ledger 读取频率

`DAPiGenGymEnv` 默认不把全量 ledger 塞入每一步 `info`。对共享 Ray actor 来说，
每步读取 ledger 会形成同步 RPC，并显著拖慢 rollout。应在 iteration、checkpoint
或预算停止边界显式读取：

```python
ledger = env.oracle_ledger()
# 或在 driver 中：ray.get(evaluator_actor.ledger.remote())
```

只有诊断运行才设置：

```python
include_ledger_in_step_info=True
```
