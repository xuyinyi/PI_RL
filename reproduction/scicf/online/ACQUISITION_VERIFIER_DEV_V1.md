# LLM-SciCF acquisition and verifier development gate v1

Status: **pre-declared; development-only; no formal training or claim authority**.

## Question

On four outcome-blind complete PPO trajectories, can DeepSeek select useful
counterfactual interventions from the same 24-candidate pool at maximum budget
`B=4`, and does paired verification at `K=2` yield a sufficiently stable corpus
for the next pairwise-optimizer development module?

## Frozen protocol

- Run only through Slurm on `yanlih100n1` against the accepted Stage-0 binding.
- Collect 128 transitions from one frozen, untrained behavior policy. No PPO or
  pairwise optimizer step is allowed in this gate.
- Deterministically shuffle eligible complete cross-timestep episode IDs without
  reading terminal success or reward, then use the first four episodes.
- Construct 24 opaque legal candidates per episode, spanning at least two
  timesteps.
- Compare `deepseek_llm`, `random`, `policy_probability`, and
  `chemistry_heuristic` on exactly the same candidate IDs with maximum `B=4`.
  Explicit LLM abstention is preserved and never padded.
- Issue exactly four blinded DeepSeek API requests before generating any new
  counterfactual labels. Reward truth and policy scores are excluded from every
  prompt.
- Exhaustively evaluate all 96 development candidates with two matched
  factual/counterfactual continuation replicates. This full-pool label budget is
  evaluation-only; it is distinct from each selector's `B=4` acquisition budget
  and cannot update the policy.
- Report HitRate@4, effective BestGain@4, Regret@4, NDCG@4, stable-positive
  coverage, abstention and exact evaluator ledgers.

The chemistry baseline ranks Morgan-fingerprint Tanimoto distance. A legal NOOP
has score `-1` because it has no molecular fingerprint and is placed after all
chemically defined alternatives.

## Pre-declared decision

Integrity passes only if all four pools, candidate identities, prompts, API
responses, `K=2` records, evaluator sources, and unchanged policy hash close.

Verifier corpus readiness requires at least 16 sign-consistent pairs, including
at least four positive and four negative pairs. LLM acquisition readiness
requires at least two of four paired NDCG non-losses versus Random, non-negative
mean NDCG margin, and non-negative mean effective-BestGain margin.

Only if both readiness checks pass may a separately specified pairwise-stability
development job be prepared. A failed efficacy threshold produces a valid
`no_go` result rather than converting the Slurm execution into an infrastructure
failure.

## Stop and claim boundary

Stop on a dirty or changed binding, non-Slurm/wrong host, candidate-pool mismatch,
budget overflow, response-schema failure, reward/policy leakage, mismatched
replicates, unexpected evaluator source, or any policy mutation. The sealed test
remains closed, Gate 1B.3 remains failed, the compatibility baseline is unchanged,
and no result from four development pools establishes algorithm effectiveness or
a scientific claim.
