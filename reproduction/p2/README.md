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
tests were initially pending until the engine and estimators existed; the
PPO/GAE implementation below now closes I01, I05 and I06 for their declared
engine-side scope.

The contract-validator tests passed locally and as part of the 15-test P2 suite
on Slurm Job 4665. This freezes the seam for implementation; it does not mark
the six integrated I-series tests as passed.

## P2-C: real-observation / evaluator-ledger profile

`FULL_STACK_PROFILE_V1.md` freezes a single-worker, two-pass characterization
using the accepted custom chemistry, real persistent polyBERT observations and
persistent QSPR evaluator ledger. The cache replay must reproduce trajectory,
observation and evaluation digests without another backend call.

### P2-C result (2026-09-04)

The governed characterization passed at clean commit
`7e3ec9d4436f94af40f6204788ac8dec9dd9466e` in Slurm Job 4665. Model/stack
construction took 4.535 s. The first 100-transition pass achieved 6.751
transitions/s; exact cache replay achieved 216.812 transitions/s. Both passes
produced 31 successful terminal molecules and identical trajectory, observation
and evaluation digests.

The final shared ledger recorded 124 requested calls, 31 unique/backend calls,
93 cache hits and zero invalid results. Replay made no backend call. The result
remains a single-worker characterization with no throughput acceptance
threshold; it cannot freeze parallelism or the training budget and invokes
neither PPO nor a credit estimator. Synchronized evidence is under
`../results/p2-full-stack-7e3ec9d-20260904/`.

## P2-D: native PPO / GAE implementation

`PPO_ENGINE_IMPLEMENTATION_V1.md` describes the implemented common engine,
environment-return GAE provider, requested-call reservation manager and atomic
checkpoint contents. The engine uses the accepted Stage-0 Gymnasium observation
and factorized masks; it does not duplicate chemistry, reward or evaluator
logic.

### P2-D result (2026-09-04)

The governed PPO/GAE smoke passed at clean commit
`6a4bd2c21b657c9f1b9c734028322cf66af6367f` in Slurm Job 4671. All 26 tests
passed before two distinct 64-transition PPO iterations ran on the real
polyBERT/QSPR stack. The first and second iterations recorded 12 and 24
successful terminal evaluator calls, respectively, all under `ppo/on_policy`.
The GAE credit provider made zero evaluator calls.

Restoring the iteration-1 checkpoint into a newly built stack reproduced the
second iteration's rollout, GAE, critic returns, actor credit, receipt, metrics,
policy and evaluator ledger exactly across all 13 checks. This accepts the
PPO/GAE path plus the engine-side portions of I01, I05 and I06. Policy-CC,
MCC-PPO, I02-I04, parallel training and production-budget selection remain
open. Evidence is under `../results/p2-native-ppo-6a4bd2c-20260904/`.
