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
