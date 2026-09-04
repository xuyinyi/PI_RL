# P2 single-engine / credit-estimator contract v1

Status: **frozen implementation contract; integrated engine acceptance pending**

## Scope

PPO, Policy-CC and MCC-PPO must use one native PyTorch PPO engine. They may
differ only through an object implementing `CreditEstimator`. The accepted
Stage-0 `DAPiGenState`, `DAPiGenAction`, masks, transition kernel, observation
encoder, terminal reward adapter and evaluator service remain the only task
source of truth.

The downloaded MCC files are design references only. Their duplicate
`PIState`/`PIAction` and environment adapter cannot enter the integrated path.

## Frozen interfaces

Executable definitions live in `contracts.py`:

- `PPOEngine.freeze_policy()` returns one immutable `FrozenPolicyHandle`.
- `PPOEngine.collect_rollout()` returns a method-independent `RolloutBatch`.
- `CreditEstimator.estimate()` receives a `CreditRequest` bound to that batch,
  frozen policy, frozen pre-query credit-model version and pre-reserved query
  budget.
- `PPOEngine.update()` receives the original rollout plus one validated
  `CreditEstimate` and returns a `PPOUpdateReceipt`.
- `CreditEstimator.commit_after_update()` may consume staged labels only after
  that receipt passes `validate_update_receipt()` and
  `validate_pending_label_commit()`.

Contract identifiers:

```text
PPO engine       dapigen-p2-single-ppo-engine-v1
credit estimator dapigen-p2-credit-estimator-v1
```

## Required iteration order

1. Verify the immutable environment, task, budget, evaluator and engine
   identities.
2. Freeze `pi_old` and record its version and state hash.
3. Reserve the worst-case requested-call allowance needed to finish the
   on-policy batch; fail before rollout if it is unavailable.
4. Collect one method-independent rollout using `pi_old`.
5. Compute critic returns and GAE inside the common engine.
6. Freeze the completion-value model and reserve requested counterfactual calls
   before sampling active queries.
7. Let the method-specific estimator return actor advantages and an optional
   pending-label batch. New labels remain invisible to the current estimator.
8. Validate method, batch, policy, source-ledger, budget and no-leakage fields.
9. Run the common PPO update: actor loss uses the validated actor advantages;
   critic loss uses only the rollout's environment-return targets.
10. Validate the update receipt, advance the policy version exactly once, then
    commit pending labels for later iterations.
11. Atomically checkpoint engine, optimizers, policy/credit versions, pending
    and committed replay, evaluator state, RNG states and all contract IDs.

## Method-specific behavior

| Method | Actor credit | Query sources | Pending paired labels |
|---|---|---|---|
| PPO | engine GAE exactly | none | forbidden |
| Policy-CC | validated counterfactual fusion | `policy_cc/factual`, `policy_cc/counterfactual` | allowed after update |
| MCC-PPO | validated mechanism-guided counterfactual fusion | `mcc_ppo/factual`, `mcc_ppo/counterfactual` | allowed after update |

On-policy and evaluation requests retain their method-specific source tags from
`RL_PPO.envs.sources`. Requested terminal calls are the optimization budget;
unique calls, backend calls and cache hits are reported but cannot substitute
for requested calls.

## Fail-closed conditions

Comparative training remains unauthorized if any of the following occurs:

- task, evaluator, engine or PPO hyperparameter identity differs by method;
- a state/action type or chemistry transition bypasses Stage 0;
- a credit estimator changes critic targets or optimizer behavior;
- the policy or completion model changes while estimating current credit;
- requested-call capacity is checked only after rollout/query execution;
- labels acquired in iteration `i` enter selection, control variates, fusion or
  the actor update in iteration `i`;
- an evaluator source is missing, unregistered or attributed to the wrong
  method;
- checkpoint/resume changes any policy, optimizer, replay, ledger, RNG or
  contract identity.

Passing the executable contract tests validates these guards, not the future
integrated PPO implementation or any optimization claim.
