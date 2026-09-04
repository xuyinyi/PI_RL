# P2 unified-engine test matrix v1

Status: **frozen; contract-validator layer passed, integrated layer pending**

The machine-readable source is `configs/p2_contract_test_matrix_v1.json`.

| ID | Current layer | Acceptance target |
|---|---|---|
| P2-C01 | implemented | only the credit seam differs across method contracts |
| P2-C02 | implemented | PPO credit is exactly GAE and evaluator-free |
| P2-C03 | implemented | finite, batch-bound, frozen-policy credit |
| P2-C04 | implemented | no same-iteration completion-model update |
| P2-C05 | implemented | pre-reserved budget and exact source allowlist |
| P2-C06 | implemented | actor-credit / critic-return separation |
| P2-C07 | implemented | post-PPO pending-label commit only |
| P2-I01 | pending engine | identical task and PPO path across methods |
| P2-I02 | pending engine | no-intervention and `eta = 0` equivalence |
| P2-I03 | pending estimator | identity-intervention paired delta is zero |
| P2-I04 | pending estimator | no same-iteration target leakage |
| P2-I05 | pending engine | exact concurrent ledger at exhaustion |
| P2-I06 | pending engine | exact checkpoint/resume identity |

The C-series tests accept the interface guards only. Gate P2 remains open until
all I-series tests run against the integrated implementation on the governed
compute coordinate.

The complete P2 suite passed 15/15 tests before the governed full-stack profile
in Slurm Job 4665. This executes the C-series guards but does not change any
I-series item from pending.
