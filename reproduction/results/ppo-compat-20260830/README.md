# DAPiGen PPO reconstructed compatibility run (2026-08-30)

This execution uses the explicit `reconstructed-afp-compatibility` asset mode
and must be labeled `reconstructed AFP compatibility baseline`. It is not an
original-author-asset run and its successful completion is not evidence of
paper-result reproduction.

## Frozen coordinates

- DAPiGen commit: `5f692946cbe0d15eede882dfe4cff7fb26eb7d8c`
- Host/partition: `n001` / `compute`
- Full-run root: `/home/wch/workspaces/DAPiGen-reproduction/runs/ppo-compat/full-20260830-v1`
- Full-run Slurm job: `4562`
- Allocation: 20 CPU, 256 GB RAM, four H100 GPUs, 72-hour limit
- PPO seed: unspecified, matching the public source

## Attempt ledger

| Job | Purpose | State | Interpretation |
|---:|---|---|---|
| 4559 | One-GPU functional smoke | Failed before trainer construction | Ray Unix-socket path exceeded the platform limit; no training occurred |
| 4560 | One-GPU functional smoke after short Ray path repair | Completed, exit 0 | Untrained and one-iteration evaluation/checkpoint paths passed |
| 4561 | 15-worker, four-GPU topology smoke | Completed, exit 0 | Slurm topology, remote rollout workers, reward inference, training, evaluation, and checkpoints passed |
| 4562 | Full compatibility baseline | Completed, exit `0:0` | 100 iterations, 99,000 environment steps, 11 checkpoints, and 11 evaluations completed |

## Full-run contract

- 100 PPO training iterations
- checkpoints and 10,000-sample evaluations every 10 iterations
- an additional 10,000-sample untrained evaluation at epoch 0
- 15 rollout workers and four trainer GPUs
- training batch size 500; SGD minibatch 128; 30 SGD passes
- FC hidden layers 256/128/128 with ReLU
- polymer length 60 and step length 5
- paper Equation (1) weighted-average reward

RLlib 1.13 reconciles the requested rollout fragment length of 200 to an
effective length of 33 for 15 workers and a batch size of 500. This is a logged
framework behavior, not a silent configuration edit.

## Completed-run evidence

The epoch-0 evaluation completed 10,000 samples in 414.00 seconds and wrote a
10,001-line CSV (header plus samples). It reported 1,679 valid samples,
validity 0.1679, 1,672 unique valid PIs, mean reward 0.04650, and maximum reward
0.5429. The archived CSV SHA-256 is:

```text
b7f06de0900d56d872d5ae40f2bb8dfba8d1e6985f50a40a4f8f025e1b0ece7f
```

The first PPO iteration completed with all 15 rollout workers healthy, 990 total
timesteps, and mean episode reward 0.03872. Iteration 100 completed with 99,000
total timesteps, 92,201 episodes, mean training reward 0.55174, and all 15
workers healthy. The terminal manifest and all 100 compact metric rows are
archived as `full-4562-run-manifest.json` and
`full-4562-training-metrics.jsonl`.

At the final 10,000-sample evaluation, validity reached 1.0 and mean reward
reached 0.55181, but only two unique valid PI strings remained. This is strong
mode-collapse evidence and must be retained in comparisons with new algorithms.

The epoch-0 validity is below the paper's reported untrained target of 0.4626.
That discrepancy is recorded immediately and is not normalized away or treated
as a successful match.

The terminal evidence SHA-256 values are:

```text
5253510bf0af2cf53a0f78e31450cf44dff865b8deaa41a413f8888d2fbba6c9  full-4562-run-manifest.json
961d62ffc527a9112cefc9dc3f03552a4ac3bfb3e153eddb1ec2af95e162da98  full-4562-training-metrics.jsonl
b63ad215043c3d0b99f8d3679f840300814af02234722cd37aa9f094aac91fca  full-4562-config-summary.json
```

## Scientific boundary

The asset gate verifies the archived AFP validation report, 12 reconstructed
reward assets, and 14 files in the selected polyBERT mirror by SHA-256. It also
records `original_asset_reproduction=false`. The primary polyBERT identity and
the authors' original AFP weights remain unverified, so a successful terminal
run will establish only the reconstructed compatibility baseline. Checkpoint
metrics must still be compared against the paper with matched metric definitions
and sample counts.
