# P2 full-stack performance profile v1

Status: **frozen protocol; governed characterization completed at `7e3ec9d`**

## Purpose

P2-A isolated accepted-mask scaling with a stub encoder. This profile measures
one complete pre-training Stage-0 path with:

- accepted custom DAPiGen chemistry and `closure_exact_cached` masks;
- real persistent polyBERT observations;
- persistent five-property QSPR terminal evaluation;
- one shared transactional requested/unique/backend/cache ledger;
- explicit `environment_regression` and `evaluation` source attribution;
- a deterministic first measured pass and exact cache replay.

The profiler performs no policy inference, PPO update, credit estimation,
counterfactual query, external API call or scientific validation.

## Frozen workload

The configuration is `configs/stage0_full_stack_profile_v1.json`:

- 100 deterministic transitions in each of two identical passes;
- at least 10 successful terminal molecules per pass;
- every successful terminal is first scored through the reward adapter and then
  requested again through the `evaluation` source to expose cache overhead;
- the second pass repeats the same states/actions/seeds and must use only cached
  evaluator results;
- the evaluator state/ledger must survive an exact state-dict round trip.

The first measured pass starts after model construction. `model_build_seconds`
is reported separately and includes persistent model loading and initial
Stage-0 construction. The core may already contain its constructor-time initial
state caches; the report therefore calls this `first_pass`, not a fully cold
start.

## Functional acceptance

The run fails closed on any accepted-environment/specification mismatch,
Stage-0 source change, dirty checkout, insufficient successful terminals,
`no_reaction_product`, invalid evaluator result, source-ledger mismatch, replay
backend call, deterministic digest mismatch or ledger checkpoint mismatch.

No minimum throughput is declared. The output is a characterization used to
design the unified engine and its later multi-worker benchmark. It cannot freeze
worker count or training budget because it profiles one worker and excludes PPO
and the credit estimator.

## Governed result

Slurm Job 4665 completed in 24 s after 15/15 P2 tests passed. Model/stack build
took 4.535 s. The first measured pass processed 100 transitions in 14.812 s
(6.751 transitions/s); exact cache replay took 0.461 s (216.812 transitions/s).
Both passes produced the same 31 terminal molecules and identical trajectory,
observation and evaluation digests.

The first pass recorded 62 requested, 31 unique, 31 backend and 31 cached calls.
Replay recorded 62 requested, zero unique, zero backend and 62 cached calls.
The final ledger contained 124 requested calls, 31 unique/backend calls and 93
cache hits, with zero invalid results. Full evidence is under
`../results/p2-full-stack-7e3ec9d-20260904/`.
