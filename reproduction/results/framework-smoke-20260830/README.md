# Common algorithm framework validation

Validation date: 2026-08-30

## Job 4563: adapter/runner smoke

- State/exit: `COMPLETED` / `0:0`
- Elapsed: 16 seconds
- Allocation: 4 CPUs, 64 GB RAM, one GPU
- Source: `95d3a7178cf62d4b9621d6a5157ec6b6de7baa39`, clean worktree
- Adapter: `RLLibPPOAdapter`, Ray 1.13.0
- Seed: 2023
- Budget: exactly 20 environment steps
- Checkpoints/evaluations: steps 0 and 20
- Evaluation: 10 samples per checkpoint, `explore=false`, seeds 2023 and 2043
- Result: complete common artifact tree and terminal manifest

The tiny untrained/trained evaluations produced no valid PI in ten samples.
That is acceptable for an interface smoke and is not an algorithm-performance
result.

## Jobs 4564 and 4565: paper-metric path smoke

- State/exit: both `COMPLETED` / `0:0`
- Elapsed: 3 seconds each
- Allocation: 4 CPUs, 32 GB RAM
- Job 4564 source: `9ae3afd8936e42bcd85ec17123f9698431562a64`, clean worktree
- Formal Job 4565 source: `dd0b9bfa408e06eb1771c53709ada99fb2706cf4`, clean worktree
- Protocol: `dapigen-common-v1`
- Exercised metrics: validity, canonical uniqueness, novelty, diversity, Frag, SNN
- Input: deterministic four-row interface fixture, not a scientific PI result

Job 4565 additionally verified that the formal evaluator fails closed unless it
has both a Slurm job id and a clean Git worktree; its summary records
`formal=true`.

The fixture produced validity 0.75, uniqueness 1.0, novelty 0.3333,
diversity 0.8571, Frag 0.3333, and SNN 0.5671. These values only prove that the
metric pipeline executes and records its definitions; they are not baseline
performance claims.

## Archived evidence hashes

| File | SHA-256 |
|---|---|
| `job-4563-run-manifest.json` | `535e65953ef2fb23c865f9b68d24a3ff31f97c6c1d5d29e97ae3dda76c753bd1` |
| `job-4563-training-metrics.jsonl` | `a01c6cd0d4bcb6b522d2f8a039331acf36822eb1180ed39a8a56bc56bf5f4db5` |
| `job-4563-evaluation-metrics.jsonl` | `3c0f07632651698917ebaf1e98890ededbd4bc77d729895a10707744c749d9d6` |
| `job-4564-evaluator-summary.json` | `1a7e3b492334547470667e5fb822b5d87cd4a60bc7e096f47efe05da588ed878` |
| `job-4565-formal-evaluator-summary.json` | `a0563ceb697ea8cac2fc0e625bcc8362344a5642d14fb6079aa88858b1bd2d98` |

Full checkpoints, native Ray logs, generation CSVs, and Slurm logs remain under
`/home/wch/workspaces/DAPiGen-reproduction/runs/algorithms/` and
`/home/wch/workspaces/DAPiGen-reproduction/logs/` on n001.
