# SciCF Gate 1B.3 stage-routed implementation candidate

Gate 1B.3 is implemented and server-validated as a **code candidate only**.
No fresh development labels or sealed-test labels have been collected.

## Frozen architecture

- Early uses the exact Gate 1B.2 nonlinear validity filter artifact.
- Middle uses the exact Gate 1B.1 linear validity filter followed by the exact
  Gate 1B.1 policy-conditioned Ridge gain ranker.
- Late is a non-gating saturation diagnostic and returns zero candidates while
  its opportunity/abstention threshold remains uncalibrated.
- No model is refit and no model-family or hyperparameter search is permitted.

This routing is motivated by the Gate 1B.1/B.2 development diagnostics and is
therefore explicitly post-hoc. The old development results cannot be recombined
and presented as independent evidence for this architecture.

## Fresh-development confirmation contract

A future confirmation is restricted to `dev/early` and `dev/middle`, with new
trajectory seeds `4241` through `4256`. The collector requires the exact frozen
Gate 1B.2 development manifest as a structure-exclusion seal and removes all
200 structure keys previously observed in development. Candidate structures
must also remain in the original dev hash bucket, which keeps them disjoint
from the original train bucket and sealed test bucket.

The confirmation keeps the same 24-candidate pool, budget four, source quotas,
paired-replicate verifier, and 25% Oracle-headroom threshold. It requires early
rescue, middle BestGain@4, and middle NDCG@4 to pass. Late saturation cannot
change the entry decision.

## Server validation

- Implementation commit: `db9018a7fd5299f90f136f0fbaf596ce8d78d4c7`.
- Frozen-artifact preflight commit:
  `fb3979a216d95a4351fb509348a64a7ba51eb7e1`.
- Slurm 4631: 49/49 full SciCF tests passed.
- Slurm 4632: actual Gate 1B.1/B.2 manifest hashes, the serialized
  `RandomForestRegressor`, 87-dimensional validity features, 89-dimensional
  gain features, and the 200-key exclusion seal all validated.
- Gate 1B.3 implementation config SHA-256:
  `148734e658e7f52d6144f4e298beae9e9b93ad6d3f3f8e92b3b66ef11b51cffa`.
- Gate 1B.3 collection config SHA-256:
  `0a1c82f6f6d8c8d76a277c4090f26dd2e8d8ee02bdd346b7a5efb8c5e0776932`.

The preflight used zero Oracle calls. The Gate 1B.3 run root does not exist,
and the parent Gate 1B.1 test directory remains absent. Configuration records
`fresh_dev_collection_authorized=false`,
`fresh_dev_evaluation_authorized=false`,
`test_collection_authorized=false`, `test_evaluation_authorized=false`, and
`ppo_integration_authorized=false`.

The next operation, if separately authorized, is the fresh unseen-structure
development collection. It is not a sealed-test run and cannot itself authorize
Gate 1C or PPO integration.
