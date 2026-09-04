# P2 unified-engine development

P2 builds one native PyTorch training path for PPO, Policy-CC and MCC-PPO on
the accepted Stage-0 environment. The methods may differ only through the
declared credit-provider seam.

## P2-A: accepted-mask throughput profiling

Before trainer implementation or budget selection, profile the accepted
`closure_exact_cached` mask with 1, 2, 4 and 8 independent rollout workers.
The frozen profile uses an observation stub because the timed region is the
mask calculation itself; state generation still executes the real custom
DAPiGen chemistry and accepted transition rules. The profiler binds itself to
the accepted P1 manifest and fails if Stage-0 source differs from the accepted
commit.

The functional gate requires:

- a clean Git checkout descended from the accepted P1 coordinate;
- unchanged Stage-0 source and standard configuration;
- the same custom chemistry backend and action catalogs as P1;
- complete work for every worker and repetition;
- deterministic state and mask digests for a given worker seed;
- no `no_reaction_product` during workload-state generation.

The throughput admission requires the median eight-worker speedup to be at
least 1.5x, median eight-worker efficiency to be at least 0.25, and every
adjacent worker-count throughput ratio to be at least 0.85. These thresholds
are engineering stop rules for the next design step, not scientific results or
training-performance claims.

The frozen configuration is
`configs/stage0_mask_throughput_v1.json`. No PPO, terminal evaluator or external
API is invoked by this profile.

### P2-A result (2026-09-04)

P2-A passed on Slurm Job 4663 at clean profile commit
`d06365e4485baddcdad65386c69cc47056bf94be`, bound to accepted P1 commit
`373b3291ac04dbf654c134bee2ca61a0c86d1a68`. Median aggregate throughput was
7.329, 14.738, 29.174 and 33.807 masks/s for 1, 2, 4 and 8 workers,
respectively. The eight-worker speedup was 4.613x, parallel efficiency was
0.577, and the minimum adjacent throughput ratio was 1.159. All functional,
determinism, specification and throughput failure lists were empty.

This admits the accepted mask to the next P2 design step only. It does not
accept the complete P2 engine or freeze a training budget: the timed operation
excluded polyBERT observation encoding, the terminal evaluator, PPO and the
credit estimator. The 4-to-8-worker increment was only 1.159x, so full-stack
profiling remains required before worker-count and budget selection.

The synchronized evidence, including the superseded failed Job 4662 diagnostic,
is under `../results/p2-mask-profile-d06365e-20260904/`.

## P2-B: single-engine contract freeze

The v1 interface freeze is defined by
`PPO_ENGINE_CREDIT_CONTRACT_V1.md`, executable guards in `contracts.py`, and the
frozen matrix in `TEST_MATRIX_V1.md` /
`configs/p2_contract_test_matrix_v1.json`.

The contract-validator layer fixes the single-PPO-engine seam, actor-credit /
critic-return separation, frozen-policy binding, pre-reserved query budget,
method-specific evaluator sources and post-update pending-label commit. These
guards do not implement or accept the PPO engine. The six integrated I-series
tests remain pending until the engine and estimators exist.
