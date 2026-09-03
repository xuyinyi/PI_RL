# SciCF Gate 1B.3 fresh-development evaluation

Gate 1B.3 stopped with a **fresh-development no-go**. The sealed test was not
collected, accessed, or evaluated.

The evaluation used the previously frozen stage router without fitting,
hyperparameter search, or model-family search: early used the Gate 1B.2
nonlinear validity filter, middle used the Gate 1B.1 linear validity filter and
policy-conditioned gain ranker, and late remained a non-gating always-abstain
diagnostic while uncalibrated. The 32 fresh trajectories used seeds 4241-4256
and excluded every structure key observed in original train and prior
development data.

## Fresh-development result

| Check | Result | Pre-declared requirement | Decision |
|---|---:|---:|---|
| train/fresh-dev structure overlap | 0 keys | 0 keys | pass |
| prior-dev/fresh-dev structure overlap | 0 keys | 0 keys | pass |
| cross-timestep trajectories | 0.5000 | >= 0.25 | pass |
| early rescue headroom captured | 0.0263 | positive improvement and >= 0.25 | fail |
| middle BestGain@4 headroom captured | 0.0457 | positive improvement and >= 0.25 | fail |
| middle NDCG@4 headroom captured | -0.1262 | positive improvement and >= 0.25 | fail |

Early rescue improved over exact expected Random by only `0.0143229167`,
capturing `2.6253%` of the available Oracle-minus-Random headroom. Middle
BestGain@4 improved by `0.0011462745`, capturing `4.5743%` of headroom. Middle
NDCG@4 decreased by `0.0350066433` relative to Random, corresponding to
`-12.6153%` of available headroom. Consequently, none of the three efficacy
entry checks passed.

This fresh, structure-isolated result does not reproduce the stronger
post-hoc development diagnostics that motivated stage routing. It is therefore
evidence against promoting the current frozen router to the sealed test, not a
reason to inspect test labels or tune against them.

## Authorization, provenance, and boundary

- Evaluation authorization/enforcement commit:
  `03809fdaa383adb3e697090584f9cfe643e1eb30`.
- Evaluation authorization receipt SHA-256:
  `071c24a118c7a6b7221a06ad06c96de82a54d5ebf3e12b3762b33ad4e64c780d`.
- Collection source commit:
  `bbac223d123588619675539e989e3010c079ddea`.
- Collection validation SHA-256:
  `6f260f26700f4fc5c8da65ab173f9920989f37fd372b1a8b5a44e25f9e1ba13a`.
- Slurm 4639: 52/52 regression tests passed before evaluation.
- Slurm 4640: fresh-dev evaluation completed with exit code 0.
- Development report SHA-256:
  `0c04a58d990aacbe5b0b4ce00252ca719a4056111c6173237439b2ed9893d612`.
- Frozen stage-routed manifest SHA-256:
  `330145c21c5bda3da3e4ee733f945bf7bb60d72cb4b262e6ace828d59377b20d`.

Full server artifacts are under:

`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1b3-stage-routed-20260903-v1/frozen-stage-routed-model-v1`

The resulting manifest records `dev_entry_passed=false`,
`sealed_test_eligible=false`, `test_collection_authorized=false`,
`test_evaluation_authorized=false`, `test_data_accessed=false`, and
`test_evaluations_completed=0`. Gate 1C, pairwise refinement, and PPO
integration remain unauthorized. No new Oracle, LLM, or PPO call occurred
during evaluation.
