# SciCF Gate 1B deterministic-descriptor learnability audit

Gate 1B is complete with a formal **no-go** decision on Gate1-dev. The frozen
request-time RDKit descriptor model passed the early-invalid validity-rescue
task, but it failed both pre-declared middle-valid gain-ranking metrics.
Consequently, Gate 1C is not eligible and LLM ranking, pairwise refinement, and
PPO integration remain closed.

The audit used 81 deterministic features computed only from the pre-decision
chemical context, factual building-block fragment, alternative fragment,
component identity, and Morgan distance. Terminal properties, terminal object,
verified gain, verification outcomes, candidate identity, and policy score were
excluded from the fitted model. The fixed model was ridge regression with
alpha 1.0, training-fold-only standardization, no hyperparameter search, and
leave-one-trajectory-out validation. Inference used 10,000 trajectory-level
bootstrap resamples and 10,000 within-trajectory label permutations.

## Pre-declared decision

Each required metric had to satisfy all four conditions: positive mean
descriptor-minus-exact-Random difference, a strictly positive lower 95%
trajectory-bootstrap bound, one-sided permutation p <= 0.05, and at least 25%
of mean Oracle-minus-Random headroom captured. Early rescue rate and both
middle BestGain@4 and NDCG@4 were required. Late did not contribute to the
decision.

| Stratum / metric | Descriptor | E[Random] | Difference (95% CI) | Permutation p | Headroom captured | Decision |
|---|---:|---:|---:|---:|---:|---|
| early-invalid rescue rate@4 | 0.968750 | 0.304688 | 0.664063 [0.606771, 0.708333] | 0.000100 | 95.51% | pass |
| middle-valid BestGain@4 | 0.006488 | 0.005964 | 0.000523 [-0.002229, 0.002447] | 0.165783 | 6.22% | fail |
| middle-valid NDCG@4 | 0.217577 | 0.191459 | 0.026118 [-0.046452, 0.090885] | 0.101390 | 3.23% | fail |

The early descriptor AUC was 0.936985 and all eight early trajectories placed
at least one rescue in the top four. This is evidence that the present cheap
features recover the development-set validity-rescue pattern. It is not an
independent-test or unseen-chemistry result.

The policy-probability comparator was materially stronger than the descriptor
model in the middle stratum (BestGain@4 0.010406 and NDCG@4 0.474616), but it was
not an allowed input to the pre-declared descriptor model and cannot rescue the
Gate 1B decision post hoc.

## Saturation and generalization diagnostics

Five of eight late trajectories had no positive candidate. The descriptor
model's late BestGain@4 was 0.000375 versus exact expected Random -0.000484,
while its NDCG@4 was 0.014338 versus Random 0.056796. These numbers remain a
saturation diagnostic and do not affect the gate.

A post-result, non-gating identity-overlap check found 80 unique
component-plus-alternative-structure keys among 192 candidate rows per stage.
Sixteen keys were repeated across trajectories and accounted for 128/192 rows.
The repeated early keys had no rescue-label conflicts. Therefore,
leave-one-trajectory-out validation does not establish generalization to unseen
candidate structures and may be optimistic for that stronger claim.

## Provenance and boundary

- Source commit: `ffae2415de83b79f95962baa430d42c6b5a7e18d`.
- Slurm 4611: 32/32 tests passed.
- Slurm 4612: audit completed with exit code 0.
- Full artifact SHA-256:
  `62ea1c6ca867563c014162d8909825b3cd7b81911e08ab449acb5dc7d545fe03`.
- No new LLM call, Oracle call, PPO update, or local code execution occurred.
- The independent Gate1-test split has not been created.

The full row-level artifact remains at:

`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1-formal-ad3889d-v1/gate1b-learnability-gate1-dev-cheap-descriptors-20260903-v1.json`
