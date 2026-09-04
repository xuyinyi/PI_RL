# P2 full-stack performance profile v1

Status: **frozen profiling protocol; formal governed run pending**

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
