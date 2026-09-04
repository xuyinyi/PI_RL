# P4-A native PPO preflight evidence

- Protocol: `dapigen-p4a-native-ppo-standard-v1`
- Frozen source commit: `0c33bced38ae321247f952e0d36a52b830c1efa2`
- Protocol SHA-256: `7bdc2a4fc35aa14a6f2c52f3d34b6945718ef826d7145de030f729ac54685f33`
- Accepted P1 environment: `dapigen:8004c1fa2055a186b4a4f3ed`
- Valid preflight: Slurm Job 4675, seed `20260910`
- Decision: **passed; frozen five-seed formal array launched as Job 4676**

## Valid preflight

Job 4675 passed 30 tests before executing the real accepted Stage-0 stack on
one H100. Twelve PPO iterations produced 1,536 environment transitions and 390
requested / 390 unique / 390 backend training evaluator calls, all tagged
`ppo/on_policy`; there were zero cache hits and zero invalid evaluator results.
The frozen 512-call cap was not exceeded. Training stopped as declared when the
122 remaining calls could not cover another worst-case 128-call reservation.

Independent deterministic evaluations contained 128 episodes at target calls
0, 256 and 512. Their actual training-call coordinates were 0, 271 and 390.
All generated rows were valid; canonical uniqueness was 2/128, 2/128 and 4/128.
These values are recorded diagnostics, not a preflight performance criterion or
evidence of optimization superiority.

All acceptance checks passed: exact P1 source/environment/task/encoder/evaluator
binding, clean Slurm execution, source isolation, finite exact-GAE PPO updates,
changed policy, complete metrics, independent evaluation ledgers, checkpoint
round-trip, no sealed-test access and no external API call. The final policy is
`60157665c688a29722447d282b6eb178d9cf3894e5d9e2e356604ad26b44d1b1`.

The `valid/` directory is an exact copy of the remote run. Its internal
`ARTIFACT_SHA256SUMS.txt` hashes the three checkpoints, three generation CSVs,
run report and training metrics.

## Retained diagnostics

- Job 4672 failed at zero seconds because the submitted package-root argument
  did not contain Gymnasium. No test, PPO step, evaluator call or output
  directory was created.
- Job 4673 failed before training because the submitted Python environment did
  not contain pytest. No PPO step, evaluator call or output directory was
  created.
- Job 4674 ran the bounded workload but failed the environment-identity gate.
  The asset copy had omitted 29 Hugging Face metadata/lock files that are part
  of the current polyBERT directory fingerprint, producing `1d0e5a...` instead
  of accepted `6bdd24...`. The formal array remained closed. Its report and
  logs are retained as invalid-stack diagnostic evidence; its checkpoints are
  not promoted.
- The complete accepted 43-file polyBERT directory was then checksum-copied,
  restoring fingerprint
  `6bdd24f951dd90d3031e749ef0130752811bfefe6c850af82b805cf015ea195f`.
  No source, hyperparameter, gate, seed or protocol change was made before Job
  4675.

## Boundary

This evidence opens only the already frozen P4-A formal array. It is not the
complete P4 gate: original RLlib compatibility and `legacy_effective` six-step
semantic-control evidence remain separate. It does not authorize Policy-CC,
MCC-PPO, sealed-test access, an algorithm comparison or a scientific claim.

