# SciCF-PPO task status

Status date: 2026-08-30

## Baseline and framework

| Spec task | Status | Evidence or boundary |
|---|---|---|
| 1.1 upstream revision | complete | `5f692946cbe0d15eede882dfe4cff7fb26eb7d8c` in the frozen baseline record |
| 1.2 isolated legacy runtime | complete | `reproduction/environment-linux-h100.yml` and the n001 runtime |
| 1.3 compatibility assets and environment cycle | complete for compatibility assets | Author AFP assets remain unavailable; this is not original-asset reproduction. |
| 1.4 baseline PPO run | complete for compatibility baseline | Slurm job 4562; not a paper-result reproduction claim. |
| 1.5 wrapper regression | complete | n001 Slurm job 4567, `COMPLETED 0:0`; direct and wrapped scientific outputs match. |
| 2.1-2.4 scaffold, records, adapter, config/manifest | implemented | `reproduction/scicf` and `scicf-gate1-dev-v1.json` |

## Trajectory, oracle, and acquisition foundation

| Spec task | Status | Evidence or boundary |
|---|---|---|
| 3.1-3.4 serialization and snapshot/restore | implemented | DAPiGen is wrapped; the chemistry stack is not rewritten. |
| 3.5 deterministic replay | complete for adapter runtime | n001 Slurm job 4567 restored the same state with matched continuation randomness. |
| 3.6 identity intervention | complete for adapter runtime | n001 Slurm job 4567 produced `delta = 0.0` and equal terminal objects. |
| 4.1-4.3 atomic oracle accounting | implemented and unit-tested | Factual, counterfactual, and evaluation calls are independent. |
| 4.4 all run summaries | partial | SciCF manifest supports counters; the offline Gate 1 runner is not implemented yet. |
| 5.1-5.7 fixed-pool non-LLM acquisition | implemented and unit-tested | Random, policy-probability, and Morgan-distance chemistry strategies share one pool. |

## Gates and blocked work

- Offline Gate 1 experiments are not yet run.
- The LLM model/service and versioned prompt schema are not yet selected.
- Pairwise refinement and PPO integration are intentionally not implemented.
- Gate 1 status remains `pending`; configuration rejects early refinement.
- No SciCF performance, oracle-efficiency, or generality claim is currently authorized.
