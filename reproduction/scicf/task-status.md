# SciCF-PPO task status

Status date: 2026-09-03

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
- The private API path was extended for official DeepSeek Flash and passed
  23/23 compatibility/security tests in Slurm job 4601 plus shell validation
  in job 4602. It disables provider thinking, omits unsupported `seed`, enables
  JSON-object mode, and keeps the API key and credential path out of manifests.
- The user-authorized DeepSeek API run completed in jobs 4603-4606. The
  one-request smoke and all 24 formal requests passed schema validation. The
  resulting 0/3 no-go is now classified as the prompt-starved-v2 negative
  diagnostic, not a test of the fully specified SciCF acquisition hypothesis.
  Pairwise refinement remains unauthorized. See
  `reproduction/results/scicf-gate1-deepseek-flash-20260903`.
- Gate 1A headroom audit is complete. Slurm job 4607 passed 27/27 tests and job
  4608 computed exact expected Random top-4 performance without new LLM,
  oracle, or PPO calls. Statistical headroom exists in all three stages, making
  Gate 1B cheap-information learnability diagnosis eligible. Late-stage
  headroom is sparse: 5/8 trajectories have no positive candidate. See
  `reproduction/results/scicf-gate1a-headroom-20260903`.
- No SciCF performance, oracle-efficiency, or generality claim is currently authorized.
