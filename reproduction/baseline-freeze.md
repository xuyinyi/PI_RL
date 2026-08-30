# Reconstructed AFP compatibility PPO baseline freeze

Freeze date: 2026-08-30

## Identity

- Frozen tag: `dapigen-ppo-compat-baseline-v1`
- Upstream repository: `https://github.com/xuyinyi/DAPiGen.git`
- Upstream commit: `5f692946cbe0d15eede882dfe4cff7fb26eb7d8c`
- Required label: `reconstructed AFP compatibility baseline`
- `original_asset_reproduction`: `false`
- Asset mode: `reconstructed-afp-compatibility`

The tag freezes the compatibility repairs, paper Equation (1), asset gates,
Slurm runner, and evidence records used for the completed baseline. It does not
change reconstructed AFP weights or the user-selected polyBERT mirror into
author-supplied assets.

## Completed execution

- Host/partition: `n001` / `compute`
- Slurm job: `4562`
- Slurm state/exit: `COMPLETED` / `0:0`
- Elapsed allocation: `01:53:20`
- Resources: 20 CPUs, 256 GB RAM, four GPUs
- Training iterations: 100
- Environment timesteps: 99,000
- Checkpoints: 11 (`0,10,...,100`)
- Evaluations: 11 x 10,000 generated samples
- PPO seed: unspecified (`null`), matching the public source
- Remote artifact root: `/home/wch/workspaces/DAPiGen-reproduction/runs/ppo-compat/full-20260830-v1`

## Archived terminal evidence

| File | SHA-256 |
|---|---|
| `results/ppo-compat-20260830/full-4562-run-manifest.json` | `5253510bf0af2cf53a0f78e31450cf44dff865b8deaa41a413f8888d2fbba6c9` |
| `results/ppo-compat-20260830/full-4562-training-metrics.jsonl` | `961d62ffc527a9112cefc9dc3f03552a4ac3bfb3e153eddb1ec2af95e162da98` |
| `results/ppo-compat-20260830/full-4562-config-summary.json` | `b63ad215043c3d0b99f8d3679f840300814af02234722cd37aa9f094aac91fca` |

Large checkpoints, Ray logs, Slurm logs, and evaluation CSVs remain under the
remote artifact root. They are not duplicated in Git.

## Baseline endpoint and boundary

The final evaluation has validity `1.0`, mean reward `0.551809`, and only two
unique valid PI strings among 10,000 samples. The baseline therefore exhibits
strong terminal mode collapse.

This freeze authorizes engineering comparisons against the compatibility PPO.
It does not establish paper-result reproduction because the author AFP assets,
primary polyBERT identity, paper-level metric closure, deterministic evaluation
semantics, and multi-seed uncertainty remain unresolved.
