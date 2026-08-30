# SciCF-PPO Offline Acquisition Gate 1

Status: **failed / no-go**

The valid blinded run did not satisfy the pre-declared decision rule. No stage
simultaneously beat both Random and the chemistry heuristic, so pairwise
refinement remains blocked.

## Required evidence

- Fixed, versioned trajectory sets from early, middle, and late PPO checkpoints
- Identical legal candidate pools for every acquisition strategy
- Exhaustive or near-exhaustive paired scientific-oracle gains
- Matched acquisition budget for Random, Policy Probability, chemistry
  heuristic, and LLM acquisition
- HitRate@B, BestGain@B, Regret@B, and NDCG@B
- Paired comparisons and bootstrap confidence intervals across declared seeds
- Exact environment, checkpoint, oracle, prompt, model, and code identities
- Atomic oracle-call and environment-transition accounting

## Pre-declared decision rule

The primary metric is NDCG@4. For each of the early, middle, and late PPO
stages, success requires the lower bound of the paired bootstrap 95% confidence
interval for `LLM minus comparator` NDCG@4 to be greater than zero against both
Random and the Morgan-distance chemistry heuristic. Gate 1 passes only if this
condition holds in at least two stages. Otherwise online pairwise refinement
remains blocked and the failure must be analyzed first.

The fixed pool has 24 unique atomic interventions: eight policy-near, eight
random-legal, and eight structural candidates after de-duplication. Every
strategy selects exactly four IDs from the identical pool. Ground-truth gains
use two paired continuations with common random numbers. Eight fixed episode
seeds are evaluated at each stage. Bootstrap aggregation uses 10,000 resamples.

The LLM acquisition model is `Qwen/Qwen2.5-7B-Instruct` at immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`, with greedy decoding, prompt
version `scicf-dapigen-acquisition-blinded-v2`, and response schema
`scicf-ranked-interventions-v1`. Generation protocol
`greedy-attention-mask-exact-budget-v2` uses an explicit all-ones attention
mask, unsets sampling-only temperature, top-p, and top-k values, and requires
exactly four unique IDs rather than a full-pool ranking. The model receives no
verified gains. Before prompting, candidate presentation order is blinded with
`sha256-shuffle-v1`, seeded only by request and pool identity. This removes the
policy-near/random/structural source-block ordering while preserving the exact
same candidate set used by every acquisition strategy.

## Decision

The valid result is `failed`: zero of the required two stages succeeded.

| Stage | LLM NDCG@4 | Random NDCG@4 | Chemistry NDCG@4 | LLM-Random 95% CI | LLM-Chemistry 95% CI | Stage success |
|---|---:|---:|---:|---:|---:|---|
| early | 0.3057 | 0.3105 | 0.5845 | [-0.1886, 0.1672] | [-0.4026, -0.1339] | no |
| middle | 0.1278 | 0.2913 | 0.0000 | [-0.3695, 0.0318] | [0.0373, 0.2285] | no |
| late | 0.0817 | 0.0839 | 0.0000 | [-0.1333, 0.1267] | [0.0000, 0.2283] | no |

The middle stage beat the chemistry heuristic but not Random. The late-stage
chemistry comparison touched zero at the lower confidence bound and therefore
failed the strict `ci_lower > 0` rule. Early failed both comparisons.

## Valid evidence

- Formal fixed-pool collection: Slurm array job 4573, tasks 0-2, all
  `COMPLETED 0:0`; source commit
  `ad3889d2e6db365e93c6d42daee1fa4cc4e6c3ee`.
- Blinding regression tests: job 4583, 19/19 passed.
- Deterministic request blinding: job 4584, 24/24 requests rebuilt.
- LLM acquisition: job 4585, `COMPLETED 0:0`, 24/24 validated responses,
  model revision fixed above.
- Final aggregation: job 4586, `COMPLETED 0:0`; source commit
  `6f1cbc8ce0d5cbbeaa6c4a9ef2d6e91b7a9fa1af`.
- Formal decision artifact on n001:
  `/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1-formal-ad3889d-v1/gate1-decision-blinded-v2.json`.
- Compact repository artifact:
  `reproduction/results/scicf-gate1-20260830/decision-summary.json`.

All 24 candidate presentations were shuffled, every final response selected
exactly four valid IDs, zero selections reproduced the original pool's first
four IDs, and four reproduced the independently shuffled presentation's first
four. Two responses needed a schema-repair retry after inventing an ID; both
final repaired responses passed the exact-budget and membership checks.

## Invalid diagnostics excluded from the decision

Jobs 4581 and 4582 completed before presentation blinding. In 23/24 requests,
the LLM selected the original pool's first four IDs, which were also the
policy-near block. Those outputs are order-confounded and are not Gate 1
evidence. Job 4576 was cancelled during decoding-protocol correction, and job
4578 failed strict response validation; neither partial output is used.

## Claim boundary

This no-go result applies to the frozen DAPiGen compatibility baseline, the
declared candidate generator, the fixed Qwen checkpoint and prompt, eight
seeds per stage, and the specified oracle protocol. It is not evidence that all
LLMs or all scientific counterfactual acquisition methods fail, and it makes no
claim about online PPO improvement, independent-oracle robustness,
cross-domain generality, or wet-lab validity.
