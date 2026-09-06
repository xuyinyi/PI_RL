# SciCF pairwise stability development evidence

## Decision

- Protocol: `dapigen-scicf-pairwise-stability-dev-v1`
- Code coordinate: clean commit
  `06ec14f3a6c136cdb9895ff31bcc1b475acc3b61`
- Execution: n001 (`yanlih100n1`) Slurm Job 4686, `COMPLETED 0:0`
- Server tests: 36 passed
- Integrity and pre-declared stability thresholds: passed
- Module decision: `go_single_iteration_integration_smoke`
- Formal training, multi-iteration PPO, sealed-test access, effectiveness
  claims, and scientific claims: not authorized

## Input closure

The run bound Job 4685's module report, Oracle record, and matched evaluation by
SHA-256. It reproduced the exact initial policy hash, 128-transition rollout
digest, four candidate-pool IDs, 96 candidate IDs, and 25 accepted `K=2` pairs
(13 positive and 12 negative).

Reconstruction made 25 on-policy terminal-evaluator calls. It made zero
DeepSeek/API calls and zero new factual/counterfactual Oracle calls. The pairwise
probe environment's final Oracle ledger remained exactly zero.

## Stability probes

- One full-corpus scenario plus five deterministic sign-stratified holdout
  scenarios were each repeated twice.
- All 12 probes were independent one-step updates from the same initial policy
  and optimizer state; the initial state was restored after each probe.
- All six replica groups produced identical post-update policy hashes and metric
  signatures.
- All 12 optimizer steps applied without KL rollback; every training subset's
  mean signed margin increased.
- All five held-out folds improved: `+0.0002442`, `+0.0002608`, `+0.0005714`,
  `+0.0007395`, and `+0.0000616`; mean `+0.0003755`.
- The full-corpus mean signed margin improved from `-0.0017728` to `-0.0005251`
  (`+0.0012477`), while preference accuracy rose from `0.40` to `0.48`.
- Across all scenarios and all 96 candidate states, maximum joint KL was
  `4.94e-7`, maximum non-target-factor KL was `3.56e-7`, and maximum absolute
  critic-value drift was `0.00412`.
- Value-head parameters did not change. The non-zero critic-output drift comes
  from the shared actor-critic trunk and remains a quantity to monitor during
  integration.
- The final stored policy was restored to the initial hash
  `38ee9386cfe69e424782e1ba0af8d78ec083f52cbb6a2a492d29c9c9ffa0f8c6`.

## Interpretation boundary

This gate establishes deterministic, bounded behavior for one isolated
pairwise optimizer step on a small development corpus. It does not establish
that the resulting policy is accurate: post-update full-corpus preference
accuracy is still `0.48`. It also does not test interaction with a PPO update,
sequential accumulation, return improvement, sample efficiency, or structural
generalization. Those questions remain closed.

## Evidence identities

- `stability-report.json`:
  `ae818935097b644f4c6421e99ff4eaa3cd98f2b9447253f321d2e8a08f99a0b1`
- `update-probes.json`:
  `0fe8001e4489b53dad061494b1658240d4a9b1700e5afb9e1be7c8e18a8c80a1`
- `reconstruction-audit.json`:
  `9b1183d3203de688c95fef87bf86f9e074c72cd41b7e94de39923375571db2b3`
- `run-intent.json`:
  `ec459290620ea62b7eef9d7e4c5dc4c132e5763960eac8e363b7e4921e7ab899`

Complete Slurm stdout and stderr are retained in the same directory.
