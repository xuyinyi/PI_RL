# SciCF acquisition and verifier development evidence

## Decision

- Protocol: `dapigen-scicf-acquisition-verifier-dev-v1`
- Code coordinate: clean commit
  `d58b3f9d2d5eaa6060e39c93d3149f84e559120e`
- Execution: n001 (`yanlih100n1`) Slurm Job 4685, `COMPLETED 0:0`
- Server tests: 35 passed
- Engineering integrity: passed
- Pre-declared module decision: `go_pairwise_stability_development`
- Formal training, sealed-test access, algorithm-effectiveness claims, and
  scientific claims: not authorized

## Frozen execution facts

- One frozen, initially random PPO policy collected 128 on-policy transitions;
  zero optimizer steps were taken.
- Four complete cross-timestep episodes were selected by a deterministic shuffle
  of eligible episode IDs without outcome filtering.
- Each episode produced one fixed pool of 24 opaque candidates.
- DeepSeek, Random, frozen-policy probability, and the Morgan-distance chemistry
  heuristic shared the same pool and maximum `B=4` acquisition budget.
- Four official DeepSeek Flash requests all returned four valid IDs and did not
  abstain. The provider reported 30,838 prompt tokens, 506 completion tokens,
  and 31,344 tokens total. No API key or credential path is archived.
- All 96 candidates were executed with two matched factual/counterfactual
  continuation replicates. These exhaustive labels are evaluation-only and were
  not used to update the policy.
- The 384 branch executions resulted in 91 terminal scientific-evaluator
  requests because invalid branches stop before scientific scoring. The
  verification ledger recorded 74 backend calls and 17 cache hits with zero
  invalid evaluator results.
- The verifier accepted 25 sign-consistent candidate pairs: 13 positive and 12
  negative. The initial, behavior, and final policy hashes are identical.

## Development metrics

| Strategy | HitRate@4 | Effective BestGain@4 | Regret@4 | NDCG@4 | Stable positive rate@4 |
|---|---:|---:|---:|---:|---:|
| DeepSeek LLM | 0.3750 | 0.2256 | 0.0860 | 0.2777 | 0.3125 |
| Random | 0.1875 | 0.0764 | 0.2352 | 0.0912 | 0.1875 |
| Policy probability | 0.1250 | 0.0498 | 0.2617 | 0.0578 | 0.0625 |
| Chemistry heuristic | 0.3750 | 0.2106 | 0.1009 | 0.3423 | 0.3125 |

DeepSeek minus Random was `+0.18644` mean NDCG and `+0.14923` mean
effective-BestGain, with 3/4 paired NDCG non-losses. These values satisfy the
frozen entry rule. They do not support a superiority claim: the chemistry
heuristic had higher mean NDCG, the sample has only four pools, no confidence
interval was predeclared, and there is no held-out-structure evaluation.

## Evidence identities

- `module-report.json`:
  `f052a6343a275230bde3045717a6f5cadc6f0c3dbb33827a9595c830a0380d7d`
- `matched-acquisition-evaluation.json`:
  `039e933f4844fc3aa17ffce7f8859aaf51cb5bd3b9558f336bc8260e1e7e5e23`
- `oracle-verification.json`:
  `c20ef67372bbd0d8511390d963bbf38d2ceb1b1bfef9755e125fcf7eeeddb862`
- `run-intent.json`:
  `7ff668ca68a10c5e9e53155425d8783f4a572c5f6dace61c1d5466a38b991be0`

The four blinded request files, four validated response files, four pool audits,
and complete Slurm stdout/stderr are retained alongside these summary artifacts.
