# AFP reconstructed compatibility baseline (2026-08-30)

This result reconstructs the four AFP members loaded by DAPiGen PPO. It is not
an identity reproduction of the authors' serialized reward-model weights.

## Fixed coordinates

- DAPiGen source: `xuyinyi/DAPiGen@5f692946cbe0d15eede882dfe4cff7fb26eb7d8c`
- Server: `n001` / `yanlih100n1`, NVIDIA H100 80 GB
- Training root:
  `/home/wch/workspaces/DAPiGen-reproduction/runs/afp-reconstruction/full-20260830-v1`
- Promoted runtime root:
  `/home/wch/workspaces/DAPiGen-reproduction/worktree/RL_PPO/GNN/model`
- Label: `reconstructed AFP compatibility baseline`

## Slurm record

| Purpose | Job | Result |
|---|---:|---|
| Initial pilot | 4543 | Failed before training: optional `prettytable` import |
| Corrected two-epoch pilot | 4547 | Four array tasks completed, exit 0 |
| Full selected-member training | 4551 | Four array tasks completed, exit 0 |
| Runtime validation attempts | 4555-4557 | Exposed fixed-action/Kekulize completion semantics |
| Final runtime validation | 4558 | Completed, exit 0 |

All pilot and full training tasks were submitted through Slurm. Each property
used one GPU, four CPUs, and 32 GB RAM.

## Full-training results

| Property | Model id | Init seed | Best epoch | Stop | Test R2 | Test RMSE |
|---|---:|---:|---:|---|---:|---:|
| transmittance(400) | 43 | 922 | 372 | early stopping | 0.803635 | 9.441306 |
| cte | 56 | 125 | 321 | early stopping | 0.826821 | 8.872693 |
| strength | 84 | 734 | 136 | early stopping | 0.748338 | 28.052548 |
| tg | 64 | 109 | 982 | max epoch | 0.903172 | 18.503042 |

The staged and promoted copies of all four scalers, four weights, and four
settings files match the SHA-256 values recorded in the property manifests and
the aggregate validation report.

## Runtime gate

Slurm job 4558 loaded the promoted assets through the real PPO `Benchmark`
interface, checked finite/plausible predictions, loaded the user-selected
polyBERT mirror, reset the environment, and exercised a terminal PI action.

- Fixed benchmark reward: `0.3159`
- Environment action: `(47, 0)`
- Terminal PI reward: `0.3356`
- `ppo_runtime_files_ready`: `true`
- Aggregate validation-report SHA-256:
  `7ea185632dbe0e9368e6bd1a435f489b223f86cadc2bf83617460e2f1e4f7bfe`

`ppo_assets_ready` intentionally remains `false`: `xushijie/polyBERT` is a
user-selected mirror whose identity has not been verified against the expected
`kuelumbus/polyBERT` revision, and the AFP weights are reconstructions rather
than author-supplied binaries. Runtime readiness must not be promoted to an
original-asset scientific reproduction claim.

## Evidence files

- `validation-report.json`: aggregate asset hashes, model metrics, Benchmark
  smoke, environment smoke, and identity boundary.
- `<property>-run-manifest.json`: source-data hash, seeds, parameters,
  environment, split sizes, metrics, stop reason, and three asset hashes for
  each trained property.
