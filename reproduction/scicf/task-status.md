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
| 2.1-2.4 scaffold, records, adapter, config/manifest | implemented and server-tested | n001 Slurm job 4568; `reproduction/scicf` and `scicf-gate1-dev-v1.json` |

## Trajectory, oracle, and acquisition foundation

| Spec task | Status | Evidence or boundary |
|---|---|---|
| 3.1-3.4 serialization and snapshot/restore | implemented | DAPiGen is wrapped; the chemistry stack is not rewritten. |
| 3.5 deterministic replay | complete for adapter runtime | n001 Slurm job 4567 restored the same state with matched continuation randomness. |
| 3.6 identity intervention | complete for adapter runtime | n001 Slurm job 4567 produced `delta = 0.0` and equal terminal objects. |
| 4.1-4.3 atomic oracle accounting | implemented and unit-tested | Factual, counterfactual, and evaluation calls are independent. |
| 4.4 all run summaries | complete for offline Gate 1 | Per-stage reports, LLM manifest, aggregate decision, and compact archive record exact identities and atomic oracle accounting. |
| 5.1-5.7 fixed-pool non-LLM acquisition | implemented and server-tested | n001 Slurm job 4568; Random, policy-probability, and Morgan-distance chemistry strategies share one pool. |

## Gates and blocked work

- Offline Gate 1 is implemented and formally run; valid jobs are 4573,
  4583-4586.
- Gate 1 status is `failed`: 0/3 stages simultaneously beat Random and the
  chemistry heuristic on the pre-declared paired-bootstrap NDCG@4 rule.
- The frozen acquisition model is `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, with prompt
  `scicf-dapigen-acquisition-blinded-v2` and deterministic candidate-order
  blinding.
- Pairwise refinement and PPO integration are intentionally not implemented.
- Gate 1 status is `failed`; configuration continues to reject refinement.
- Jobs 4581-4582 are explicitly excluded because the unblinded pool order made
  23/24 LLM selections equal the policy-near first-four block.
- A private OpenAI-compatible API acquisition path is implemented for the next
  Gate 1 run. It is pending server tests and the user's endpoint, API key,
  model ID, and immutable model/deployment revision; no formal API request has
  been made yet.
- No SciCF performance, oracle-efficiency, or generality claim is currently authorized.
