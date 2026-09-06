# LLM-SciCF pairwise stability development gate v1

Status: **frozen and authorized for one development execution; no formal or
multi-iteration PPO training**.

## Question

Does the already implemented one-step, Oracle-weighted pairwise policy update
behave reproducibly and conservatively on the 25 stable `K=2` pairs admitted by
Job 4685?

## Frozen inputs and execution

- Bind the three Job 4685 input artifacts by SHA-256, its exact initial policy
  hash, rollout digest, 96 candidate IDs, and 25 accepted pairs.
- Reconstruct the 128-transition frozen-policy rollout and four 24-candidate
  pools on n001. Reconstruction may use only the on-policy evaluator source.
- Reuse the 25 accepted labels (13 positive, 12 negative); make zero DeepSeek
  calls and zero new factual/counterfactual Oracle calls.
- Use the unchanged pairwise configuration: learning rate `1e-5`, maximum
  gradient norm `0.5`, Oracle-delta weight cap `2.0`, and KL target `0.01`.
- Build five deterministic sign-stratified folds. Run one full-corpus scenario
  and five train/holdout scenarios, each twice from the exact same initial
  policy and optimizer state: 12 independent one-step probes in total.
- Restore the initial policy and optimizer after every probe. These probes are
  not sequential iterations and do not include a PPO update.

## Pass criteria

- Both replicas of every scenario must produce identical post-update policy
  hashes and complete metric signatures.
- Every one-step update must apply exactly once, change the policy, improve its
  training-pair mean signed margin, and leave value-head parameters unchanged.
- Across all 96 candidate states, maximum joint KL must be at most `0.01`,
  maximum non-target-factor KL at most `0.005`, and maximum absolute critic-value
  drift at most `0.01`.
- The full-corpus update must improve mean signed margin without reducing pair
  preference accuracy.
- Mean holdout signed-margin improvement across five folds must be non-negative,
  with at least three non-negative folds.

## Stop and claim boundary

Stop on any input/reconstruction mismatch, dirty worktree, wrong host, non-Slurm
execution, unexpected evaluator source, new API/counterfactual query, non-finite
update, KL rollback, value-head mutation, replica mismatch, or failed stability
threshold. Passing opens only a separately frozen single-iteration integration
smoke. It does not authorize formal training, multi-iteration PPO, a sealed test,
algorithm-effectiveness claims, or scientific claims.
