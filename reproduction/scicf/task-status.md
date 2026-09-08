# SciCF-PPO task status

> **Navigation updated 2026-09-08:** the dated text below is the offline
> acquisition-track snapshot. It does not describe the later online programme.
> Online PPO + K=5 SciCF and the matched PPO-only control have now completed
> six iterations each (Jobs 4748/4751). Current status and unfinished work:
> [project plan](../PROJECT_PLAN.md) and
> [handoff](../DEVELOPMENT_HANDOFF_20260908.md). Historical no-go decisions below
> are retained unchanged; their old “not implemented” statements apply only to
> that snapshot.

Status date: 2026-09-03

> **Historical track notice (2026-09-04):** This status is preserved as the
> immutable record of the SciCF acquisition programme and its no-go decisions.
> It is no longer the active implementation roadmap. The separately gated
> Stage 0 / PPO / Policy-CC / MCC-PPO route is defined in
> `reproduction/PROJECT_PLAN.md`. Nothing in the new plan promotes, reopens, or
> relabels the SciCF sealed-test, pairwise-refinement, or PPO-integration gates.

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
  oracle, or PPO calls. Post-archive Slurm job 4609 passed 28/28 tests,
  including exhaustive ordered-enumeration validation of expected Random
  NDCG. Statistical headroom exists in all three stages, making Gate 1B
  cheap-information learnability diagnosis eligible. Late-stage headroom is
  sparse: 5/8 trajectories have no positive candidate. See
  `reproduction/results/scicf-gate1a-headroom-20260903`.
- Gate 1B deterministic-descriptor learnability audit is complete with a formal
  no-go. Slurm job 4611 passed 32/32 tests and job 4612 completed the frozen
  Gate1-dev audit. Early-invalid rescue passed, capturing 95.51% of available
  Oracle-minus-Random headroom. Middle-valid BestGain@4 and NDCG@4 both failed
  their pre-declared confidence, permutation, and 25%-headroom criteria. Late
  remains a non-gating saturation diagnostic. Gate 1C is not eligible; LLM
  ranking, pairwise refinement, and PPO integration remain closed. See
  `reproduction/results/scicf-gate1b-learnability-20260903`.
- Gate 1B.1 structure-isolated two-stage acquisition stopped at a development-
  entry no-go. Train/dev structure overlap is zero and early trajectories now
  contribute candidates from both t=0 and t=1. The policy-conditioned middle
  ranker passed both development entry metrics, but early rescue captured only
  14.58% of headroom versus the required 25%, and all 8 dev-late pools lacked a
  positive candidate, making the abstention threshold non-calibratable. The
  frozen model manifest therefore has `test_collection_authorized=false`; no
  sealed test labels were collected. See
  `reproduction/results/scicf-gate1b1-structure-split-20260903`.
- Gate 1B.2 nonlinear-validity development evaluation also stopped with a
  no-go while preserving the sealed test. A single fixed random-forest validity
  filter raised early rescue headroom capture to 68.75%, and the unchanged
  policy-conditioned Ridge gain ranker retained 85.33% BestGain@4 headroom
  capture. The validity-first joint ordering captured only 23.50% of middle
  NDCG@4 headroom, below the pre-declared 25% threshold. All 32 combined
  late-dev trajectories lacked a positive candidate, so the zero-to-four
  abstention rule remained non-calibratable. Slurm jobs 4625-4628 completed;
  `sealed_test_eligible=false`, and no test label was collected, accessed, or
  evaluated. See `reproduction/results/scicf-gate1b2-development-20260903`.
- Gate 1B.3 stage-routed acquisition is implementation-complete and its
  fresh-development dataset has been acquired and evaluated. Early routes to the frozen Gate
  1B.2 nonlinear validity model; middle routes to the frozen Gate 1B.1 linear
  validity and policy-conditioned gain models; uncalibrated late defaults to
  abstention and is non-gating. A future dev-only collector uses new seeds and
  excludes all 200 previously observed dev structure keys. Slurm array 4635
  collected 16 early and 16 middle trajectories; job 4637 validated 85 fresh
  keys with zero train/prior-dev overlap and 1,491 atomic Oracle calls. After
  separate user authorization, Slurm job 4640 completed the checksum-bound
  fresh-dev evaluation and returned a no-go. Early rescue captured 2.63% of
  Oracle headroom, middle BestGain@4 captured 4.57%, and middle NDCG@4 was
  worse than Random; all three missed the required 25%. Cross-timestep
  coverage passed at 50%. `sealed_test_eligible=false`; sealed test, Gate 1C,
  pairwise refinement, and PPO integration remain unauthorized. See
  `reproduction/results/scicf-gate1b3-fresh-dev-evaluation-20260903`.
- No SciCF performance, oracle-efficiency, or generality claim is currently authorized.
