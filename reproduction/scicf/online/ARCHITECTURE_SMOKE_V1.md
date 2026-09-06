# LLM-SciCF online architecture smoke v1

Status: **passed in n001 Slurm Job 4684 at commit `d05a05c`; claim-ineligible**.

## Purpose

This protocol implements the user's architecture-first route. It asks only
whether the end-to-end control and data path can run on the accepted DAPiGen
Stage-0 stack:

```text
frozen behavior policy -> on-policy rollout -> standard PPO update
  -> legal cross-timestep pool -> one blinded DeepSeek request
  -> selected-only matched Oracle verification -> verified pair records
  -> one component-local pairwise step -> KL accept or exact rollback
```

It does not test whether LLM acquisition beats Random, whether online learning
improves PPO, or whether any generated polymer has independently validated
properties.

## Frozen limits

- Host: `yanlih100n1`, executed only through Slurm.
- One PPO iteration of 64 transitions.
- One external DeepSeek API request using the private server-side credential
  file; no local model.
- One deterministic pool of 24 legal, opaque-ID candidates spanning at least
  two timesteps.
- The factual source episode is the first complete multi-step episode,
  preferring a successful one only to maximize architecture-path reachability;
  this smoke selection rule is not valid for estimating algorithm performance.
- Zero to four LLM selections. Abstention is represented explicitly and never
  padded with arbitrary candidates.
- Only selected candidates are verified; development uses one matched
  factual/counterfactual continuation pair per selection.
- At least one non-zero sign-consistent verified pair is required to exercise
  the refinement path.
- Exactly one attempted component-local pairwise optimizer step. It is rolled
  back if mean joint KL exceeds `0.01`.
- Pair weights come only from verified Oracle deltas. LLM confidence and
  reasoning never enter a loss.
- Counterfactual actions never enter PPO clipping.

## Stop conditions

The smoke fails closed if the Stage-0 binding changed, the Git worktree is
dirty, Slurm/host identity is wrong, the API response violates schema or
budget, the prompt exposes reward truth or policy scores, any unselected
candidate is verified, evaluator-source accounting is invalid, no verified
pair is available, the pairwise step is rolled back, or checkpointing fails.

## Evidence boundary

Gate 1B.3 remains a failed, non-blocking deterministic-descriptor diagnostic.
This protocol neither relabels it nor opens its sealed test. A successful smoke
authorizes only module-by-module hardening. Formal training, matched baselines,
multi-seed comparisons, performance claims and scientific claims require new
protocols and explicit gates.

The successful run used 24 candidates across timesteps 0, 1 and 2, one
DeepSeek request, four selected candidates, four non-zero `K=1` verified pairs,
and one accepted pairwise update with mean joint KL
`5.792708179797046e-07`. These are path-execution facts, not an acquisition or
learning-effect result. Evidence is archived under
`reproduction/results/scicf-online-architecture-smoke-20260906/`.
