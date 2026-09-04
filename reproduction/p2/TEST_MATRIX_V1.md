# P2 unified-engine test matrix v1

Status: **frozen; PPO engine layer partially passed, estimator layer pending**

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
| P2-I01 | passed engine seam | identical task and PPO path for equal credit across all method identifiers |
| P2-I02 | pending Policy-CC | no-intervention and `eta = 0` equivalence |
| P2-I03 | pending estimator | identity-intervention paired delta is zero |
| P2-I04 | pending estimator | no same-iteration target leakage |
| P2-I05 | passed engine side | exact concurrent reservation plus Stage-0 ledger at exhaustion |
| P2-I06 | passed PPO/GAE | exact checkpoint/resume identity on the real accepted stack |

The C-series tests accept the interface guards only. P2-I01, I05 and I06 passed
their declared engine-side/PPO scope in Slurm Job 4671. Gate P2 remains open
until I02-I04 and the Policy-CC/MCC-PPO estimator paths pass on the governed
compute coordinate.

The pre-engine P2 suite passed 15/15 tests in Slurm Job 4665. The implemented
PPO/GAE suite passed 26/26 tests in Job 4671 before the real-stack smoke run.
