# SciCF-PPO Offline Acquisition Gate 1

Status: **pending**

No go/no-go decision has been made. This file is a decision record template,
not evidence that Gate 1 passed.

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
version `scicf-dapigen-acquisition-v1`, and response schema
`scicf-ranked-interventions-v1`. The model receives no verified gains.

## Decision

Not evaluated.
