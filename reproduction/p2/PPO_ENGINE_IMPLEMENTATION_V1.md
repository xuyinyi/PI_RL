# P2 native PPO / GAE implementation v1

Status: **PPO/GAE path implemented and governed smoke accepted at `6a4bd2c`**

## Implemented path

`engine.py` is the only native PyTorch PPO implementation for the active P2
route. It consumes the accepted Stage-0 masked dictionary observation, samples
the dianhydride and diamine factors independently from one frozen policy, and
performs one shared clipped-PPO actor/critic update. The environment, action
masks, chemistry, terminal reward and evaluator ledger remain owned by Stage 0.

`gae.py` provides environment-return GAE and `GAECreditEstimator`. For method
`ppo`, the provider returns the engine GAE byte-for-byte, makes no evaluator
request and cannot stage a pending label. The PPO critic target is always the
environment-return target computed before the credit seam.

`budget.py` pre-reserves worst-case requested terminal calls before rollout or
credit execution and reconciles the reservation against the global evaluator
ledger. Requested, unique, backend, cache and source counters remain distinct.

## Atomic iteration order

1. Freeze and hash `pi_old`.
2. Pre-reserve the worst-case on-policy terminal-call allowance.
3. Collect a method-independent rollout with legal factorized masks.
4. Compute GAE and critic returns from real environment rewards.
5. Pre-reserve the method-specific query allowance.
6. Validate the returned actor credit and its evaluator-ledger delta.
7. Run the common PPO update and advance the policy version once.
8. Commit pending labels, when a later provider supplies them, only after the
   update receipt passes.
9. Checkpoint policy, optimizer, credit-provider state, environment/evaluator,
   budget manager, RNGs, counters and contract identities at the iteration
   boundary.

## Governed smoke evidence

Slurm Job 4671 used Python 3.8.20, PyTorch 2.1.2, Gymnasium 0.29.1 and one H100.
It passed 26 tests and ran two distinct 64-transition PPO iterations on the
accepted real Stage-0 stack. The first iteration made 12 terminal evaluator
requests; the second made 24. Every request was attributed to `ppo/on_policy`.
The GAE provider made zero evaluator requests.

The checkpoint after iteration 1 was loaded into a newly constructed
environment/model stack. Re-executing iteration 2 reproduced its batch ID,
rollout, GAE, critic returns, actor credit, update receipt, update metrics,
policy state/version and evaluator ledger exactly. All 13 resume comparisons
passed.

The smoke accepts the PPO/GAE implementation path and the engine-side portions
of P2-I01, P2-I05 and P2-I06. It does not accept Policy-CC, MCC-PPO, I02-I04,
parallel training, a production budget, optimization performance or a
scientific claim.
