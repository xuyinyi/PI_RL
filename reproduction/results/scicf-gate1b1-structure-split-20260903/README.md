# SciCF Gate 1B.1 structure-isolated two-stage acquisition

Gate 1B.1 stopped with a **development-entry no-go**. The sealed test set was
not collected or evaluated.

The protocol assigned candidate structures globally by the SHA-256 hash of
`(component, canonical alternative structure)`, with disjoint train, dev, and
test buckets and disjoint trajectory seeds. It collected every available
factual timestep before constructing one common 24-candidate pool. The fitted
architecture comprised a deterministic validity filter without policy score
and a policy-conditioned gain ranker. Middle selection used validity-first
lexicographic ranking; late selection was allowed to return zero to four
candidates after a dev-calibrated gain threshold.

Test collection was code-gated on a frozen model manifest containing
`dev_entry_passed=true`. The actual manifest contains `false`, so test
collection remains unauthorized.

## Development result

| Check | Result | Pre-declared requirement | Decision |
|---|---:|---:|---|
| train/dev structure overlap | 0 keys | 0 keys | pass |
| cross-timestep trajectories | 8/24 = 0.3333 | >= 0.25 | pass |
| early rescue headroom captured | 0.1458 | >= 0.25 and positive gain | fail |
| middle BestGain@4 headroom captured | 0.8533 | >= 0.25 and positive gain | pass |
| middle NDCG@4 headroom captured | 0.2930 | >= 0.25 and positive gain | pass |
| late abstention calibration | 0 positive / 8 no-positive trajectories | both classes required | fail |

The early validity filter achieved a mean RescueRate@4 improvement of
0.109375 over exact expected Random, but this captured only 14.58% of available
Oracle-minus-Random headroom. The fixed 25% threshold was not changed after the
result.

The policy-conditioned middle ranker improved BestGain@4 over exact Random by
0.015964 and captured 85.33% of available headroom. Its NDCG@4 improvement was
0.240215 and captured 29.30%. These positive development results do not rescue
the overall entry decision because every pre-declared condition was mandatory.

All eight dev-late pools lacked a positive-gain candidate. A threshold fitted
only to these examples would reward an always-abstain rule and could not test
opportunity response, so the calibration failed closed. No post-result
threshold was invented.

## Cross-timestep and structure coverage

Every early train/dev trajectory had two available timesteps and every pool
contained candidates from both `t=0` and `t=1`. The frozen middle and late
policies terminated in one step for every sampled trajectory, so no later
timestep existed in those strata. This protocol therefore implements
cross-timestep acquisition, but the current data do not support a claim of
cross-timestep benefit for middle or late policies.

The fitted data contained 160 unique train structure keys and 104 unique dev
keys, with zero overlap. The validity filter used 576 training rows; the gain
ranker used 270 fully valid middle/late training rows. Candidate ID, verified
gain, counterfactual outcome, and verification acceptance were excluded from
the predictors; policy probability was available only to the gain ranker.

## Provenance and boundary

- Frozen protocol and collector commit:
  `9663b2d9060da427fef71cf42a2d271d616e4c3c`.
- Coverage-denominator correction and fitting commit:
  `0204fd460d74691e6b1bfccb626797d94c552563`.
- Slurm 4614: initial 36/36 tests passed.
- Slurm 4615: entrypoint and sbatch preflight passed.
- Slurm arrays 4616 and 4617: train/dev collection completed.
- Slurm 4622: corrected suite passed 37/37 tests.
- Slurm 4623: dev fitting and frozen no-go manifest completed.
- Atomic Oracle calls: 1,457 train and 1,374 dev; zero test calls.
- No LLM call, PPO update, pairwise refinement, or local code execution.

Full server artifacts:

- `/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1b1-structure-split-20260903-v1/frozen-model-v1/dev-entry-report.json`
- `/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1b1-structure-split-20260903-v1/frozen-model-v1/frozen-model-manifest.json`

Their SHA-256 values are respectively
`c7c657c31ec842be55b07322114c127b5edb8f1b0b41540e7322c71d361d256c`
and `b5a5c400b0bc90decaffa333d60c0378fea10f07c2fb3dbc02cb1487d2cae47d`.

This is development evidence only. It does not establish sealed-test
generalization, LLM acquisition benefit, PPO improvement, independent-oracle
robustness, or wet-lab validity.
