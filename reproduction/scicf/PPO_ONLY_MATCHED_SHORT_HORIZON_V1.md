# PPO-only matched short-horizon control and analysis v1

Frozen on 2026-09-08, after SciCF Job 4748 was observed and before this PPO-only
control runs. This is a prospective control protocol with a retrospective SciCF
reference, not a preregistration of both arms. Execution is pending approval.
The machine-readable contract is
`online/configs/ppo_only_matched_short_horizon_v1.json`.

## Question and comparison unit

With the same initial policy, task, seed, PPO hyperparameters and 768 on-policy
transitions, how do the six-iteration training trace and cost of PPO-only differ
from Job 4748? The independent unit is one training run (one seed per method).
Iterations and terminal episodes are correlated observations, not independent
replicates. No p-values, seed-level confidence intervals, superiority threshold,
or generalization conclusion will be produced from this pair.

This control uses the shared P2 PPO engine and environment-return GAE. It is
the PPO-only ablation of the current compatibility implementation; it is not a
new reproduction of the original authors' RLlib baseline or original AFP weights.

## Fixed training and cost budgets

Use seed 20260907, six consecutive iterations of 128 transitions, two PPO epochs
and minibatches of 32, with all other PPO values copied exactly from the hashed
v2 base protocol. Preserve target-KL early stopping: eight steps per iteration
is a ceiling, not a requirement to bypass that stop. Start from the same seeded
initialization, never from a trained SciCF checkpoint. Preserve the environment,
action masks, rewards, fragment data and complete accepted model fingerprints.

Both arms have a 1,248 requested/unique evaluator-call ceiling and the same
Slurm allocation (n001, one H100, eight CPUs, 96G, two hours). Each arm starts
with its own empty evaluator cache. PPO-only issues on-policy calls only; it
does not consume spare budget through extra transitions or dummy evaluations.
Requested, unique, backend and cache-hit counts must be reported separately.

This matches the PPO interaction budget and allowed resource ceilings. Actual
Oracle calls, optimizer steps including auxiliaries, GPU time, and LLM cost are
not matched. Job 4748 used 299 evaluator requests and six additional auxiliary
steps. Report this overhead explicitly. Do not call this an equal-total-cost or
equal-Oracle-spend experiment. Such a comparison needs a later protocol.

## Implementation contract for the control runner

Reuse `PPOEngine`, `GAECreditEstimator`, the accepted Stage-0 task and the explicit
AFP/polyBERT routing and validation. Set method to `ppo` and evaluator source to
`ppo/on_policy`; reject factual, counterfactual and evaluation calls. Use the
same engine `run_iteration(query_requested_calls=0)` semantics, including its
default query-seed RNG draw. Remove candidate acquisition, response handling,
replicate rollouts and auxiliary updates entirely. No credentials argument or
provider initialization belongs in this runner.

Do not edit the shared engine, environment, GAE, model, reward or accepted
baseline to obtain agreement. Any required change to these invalidates v1 and
must be documented before results are compared. Additive control runner,
configuration, tests, Slurm wrapper and analysis code are allowed implementation
changes. Record their clean Git commit and source diff against `5f2a6a1`.

Use the original seed once, with the existing engine RNG and episode-seed
derivation. Do not reseed every iteration or reset RNG to hide divergence after
SciCF intervention. Same seed does not guarantee identical later trajectories;
auxiliary work may also affect random-stream consumption. Record RNG-state
digests at iteration boundaries and disclose any remaining coupling. This
comparison measures the complete optional SciCF path, not LLM selection alone.

Before iteration 1, match the initial policy-state hash in the JSON contract.
After iteration 1, compare the post-PPO policy hash, GAE and critic-return hashes
with Job 4748 before allowing iteration 2. A mismatch ends the run with
`comparison_ineligible_initial_equivalence_failed`; retain all evidence and do
not rerun automatically. Serialized checkpoint bytes and method-labelled batch
digests are not expected to match across methods. The policy tensor hash is the
comparison anchor. Check actor credit equals GAE and critic targets use only
environment rewards throughout.

Write a primary checkpoint, policy hash, RNG-state digest, PPO receipt and
evaluator ledger after every iteration. Chain checkpoint/report hashes and
retain a terminal report on failure. Validate source/asset bindings and unused
authorization before initialization; consume a one-run authorization before
training. Reject existing output directories, resume, job arrays and retries.

## Frozen analysis and missing-data rules

Analyze every completed iteration, including unfavorable outcomes. Define all
differences as SciCF minus PPO-only. Align iterations at cumulative transitions
128, 256, 384, 512, 640 and 768. Do not pick a best checkpoint after seeing the
control. Iteration 1 is an equivalence check; SciCF's first auxiliary update can
affect on-policy observations from iteration 2 onward.

The primary descriptive endpoint is the total number of transitions containing
`terminal_evaluation`, divided by 768. This is evaluated-terminal yield per
transition, not the fraction of all completed episodes that are valid. The
SciCF reference numerator is 158. Report the count, the yield, and their absolute
between-method difference. Also report the six per-iteration numerators over
128 and the iteration-2-to-6 subtotal over 640 as a declared secondary summary.
These endpoints measure successful evaluable generation, not reward improvement.

Secondary engineering tables contain transition count, actual PPO steps,
target-KL stop status, approximate KL, entropy, actor/value loss, requested,
unique, backend and cache-hit evaluator calls, auxiliary steps, API attempts,
tokens, LLM time and Slurm duration. Report both arms' values and differences;
do not interpret actor/value loss as a common policy-quality measure. Total job
duration includes setup/tests and cannot isolate training speed.

Reward/return, validity per completed episode, uniqueness, diversity and held-out
policy quality are unavailable as primary matched outcomes in the archived
Job 4748 summary. Mark them `not_available_in_matched_evidence`. Do not substitute
LLM-selected counterfactual rewards, auxiliary training pairs, or PPO value loss.
Any future extraction of comparable raw on-policy rewards must be separately
specified and labelled exploratory; no new evaluator calls are included here.

If fewer than six iterations finish, report the partial trace and failure; the
fixed-768 endpoint is unavailable, with no zero imputation or horizon extension.
If initial equivalence or asset checks fail, no eligible matched comparison is
declared. Missing metrics stay unavailable. No tuning, seed replacement,
automatic retries, sealed-test access, or fresh SciCF execution is part of v1.

## Reviewable execution proposal

After approval, implement the additive runner and analyzer against this frozen
contract. Validate them only through n001 Slurm: synthetic tests for scope,
budgets, hash mismatches and analysis denominators, then a no-credential full
runtime preflight with zero PPO updates, zero AFP property calls and zero API
calls. Require exact protocol/assets, clean source, and a passing report.
Preflight is distinct from the one real training submission.

Bind the resulting implementation commit, protocol and analysis-file hashes,
passing preflight-report hash, accepted model fingerprints, seed, six-iteration
ceiling and unique output directory into a new single-use execution record.
This protocol JSON is not itself an execution authorization. Submit exactly one
real PPO-only job after those checks pass. Stop on a failed preflight instead
of submitting training. No DeepSeek data transfer or SciCF rerun is needed.

The final analysis must link both run reports, authorization, configuration,
source commits, Slurm accounting and asset/checkpoint manifests. Its conclusion
is a one-seed exploratory comparison, with the post-SciCF protocol timing and
unequal actual auxiliary cost stated alongside the results.
