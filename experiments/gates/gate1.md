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

Pairwise PPO refinement may proceed only if LLM acquisition improves over
Random and at least one strong non-LLM heuristic across more than one PPO stage
under matched candidate pools and atomic oracle budgets. Otherwise the online
refinement work remains blocked and the failure must be analyzed first.

## Decision

Not evaluated.
