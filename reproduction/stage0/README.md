# DAPiGen Stage-0 Unified Environment v2.3

本目录保存 Stage-0 的配置、测试、验收脚本和设计记录；实际实现已安装在仓库根目录的 `RL_PPO/envs/`，兼容入口为 `RL_PPO/moldr/env_stage0.py`。v2.3 在 v2.2 上将正式 mask 改为 `closure_exact_cached`：先做 label compatibility，仅在动作将消耗最后两个 attachment 时执行定向 legacy BRICS 闭合检查。它已在 clean Git commit `373b3291` 上通过 P1 工程验收：同一 100 状态审计中两侧均为 0 假阳性/0 假阴性，正式计时比独立全量 exact 快 128.834 倍；1,000 步 replay/restore、模型 parity 与正式 contract 也通过。验收范围是 reconstructed AFP compatibility，不代表作者原始模型身份或科学结论。

这是用于 **PPO、Policy-CC 与 MCC-PPO 公平比较** 的 DAPiGen 环境重构包。它不修改原始 `RL_PPO/moldr/env.py`，而是在 `RL_PPO/envs/` 下新增一个可分叉、可重放、终局评价与预算可审计的任务层。

## 已解决的核心问题

- 环境状态和 evaluator 账本/cache 可作为环境绑定 checkpoint 保存、恢复和分叉；
- 相同 `(state, action, seed)` 产生相同化学转移；
- 不再使用 Python/NumPy 全局随机数；
- 中间结构始终没有伪造物理 reward；
- 多个终局 PI 先按与 reward 无关的规则选定，再调用 QSPR；
- 每侧完成后只能使用显式 NOOP，消除被静默忽略的动作别名；
- 完整 building block 默认不能擦除已生长前缀；
- 严格修复 horizon 的一步偏移；
- polyBERT 与 QSPR 模型支持一次加载和缓存；
- evaluator 独立于状态机，并分别记录 requested/unique/backend/cache-hit 预算；
- 多 RLlib worker 可共享同一个 Ray evaluator actor；
- 每次正式运行生成环境、动作目录、模型、完整 Stage-0 源码树、关键 DAPiGen/QSPR 源码树、runtime 和预算契约 ID。

## 目录

```text
../../RL_PPO/envs/          新环境实现
../../RL_PPO/moldr/         向后兼容入口
configs/                    标准、长时域、精确审计和 legacy 回归配置
scripts/                    环境验收与 evaluator 回归脚本
tests/                      无外部模型即可运行的单元/集成测试
docs/                       设计规范与迁移流程
```

## 先运行本地无模型验证

```bash
cd /path/to/DAPiGen
pytest -q reproduction/stage0/tests
python reproduction/stage0/scripts/validate_core_without_models.py
```

原始 `RL_PPO/moldr/env.py` 保持不变。不要再次从 v2.1 ZIP 覆盖 `RL_PPO/envs/`。

## 正式训练前的强制验收

```bash
# 1. 检查 building-block 目录及初始动作掩码
python reproduction/stage0/scripts/audit_stage0_catalogs.py \
  --dapigen-root /path/to/DAPiGen \
  --output stage0_audit/catalogs.json

# 2. 要求持久化 evaluator 与原始 Benchmark 在固定 PI 面板上数值一致
python reproduction/stage0/scripts/compare_evaluators.py \
  --dapigen-root /path/to/DAPiGen \
  --sample-size 100 \
  --device cpu \
  --output-dir stage0_audit/evaluator_parity

# 3. 检查 persistent polyBERT 与 released helper
python reproduction/stage0/scripts/compare_polybert.py \
  --polybert-path /path/to/polyBERT \
  --output stage0_audit/polybert_parity.json

# 4. 检查多 worker 共用一个 evaluator ledger
python reproduction/stage0/scripts/validate_ray_evaluator.py \
  --dapigen-root /path/to/DAPiGen \
  --duplicates 8 \
  --output stage0_audit/ray_evaluator.json

# 5. 固化环境、动作映射、QSPR/polyBERT 与预算契约
python reproduction/stage0/scripts/create_stage0_manifest.py \
  --dapigen-root /path/to/DAPiGen \
  --polybert-path /path/to/polyBERT \
  --config reproduction/stage0/configs/stage0_environment.json \
  --maximum-requested-calls 100000 \
  --output-dir stage0_contract \
  --run-name ppo_seed_0
```

只有三个算法的 `task_contract_id` 完全相同，且按同一 `maximum_requested_calls` 停止，结果才可直接比较。

## 文档

- [完整环境设计规范](docs/STAGE0_ENVIRONMENT_REFACTOR.md)
- [接口参考](docs/API_REFERENCE.md)
- [迁移与验收流程](docs/MIGRATION_AND_ACCEPTANCE.md)
- [v2.3 clean-coordinate P1 验收记录](VALIDATION_V2.3_P1.md)
- [v2.3 mask refinement 验收状态](VALIDATION_V2.3.md)
- [v2.2 验收状态](VALIDATION_V2.2.md)
- [v2.1 历史验证记录](VALIDATION.md)
