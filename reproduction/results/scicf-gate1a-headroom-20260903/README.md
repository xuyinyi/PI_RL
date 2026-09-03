# SciCF Gate 1A candidate-pool headroom audit

This is a read-only diagnostic over the frozen Gate1-dev oracle tables. It made
no LLM call, no new scientific-oracle call, no PPO update, and no change to the
frozen compatibility baseline.

The audit computes the exact expected Random top-4 performance for each fixed
24-candidate pool, rather than relying only on the one previously sampled
Random ranking. Exact expected Random BestGain@4 is obtained from the
without-replacement order statistic; exact expected Random NDCG@4 is obtained
from the uniform ordered top-4 expectation. Stage summaries use the existing
10,000-resample 95% bootstrap convention across eight trajectories.

## Main result

| Stratum | N | Valid | P(delta > 0) | Oracle BestGain@4 | E[Random BestGain@4] | Gain headroom | E[Random NDCG@4] | NDCG headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early-invalid | 8 | 0 | 0.3125 | 0.418156 | 0.284272 | 0.133884 | 0.250975 | 0.749025 |
| middle-valid | 8 | 8 | 0.291667 | 0.014381 | 0.005964 | 0.008417 | 0.191459 | 0.808541 |
| late-valid | 8 | 8 | 0.067708 | 0.002100 | -0.000484 | 0.002584 | 0.056796 | 0.318204 |

`early-valid` was not observed. Five of eight late trajectories had no positive
candidate at all. The configured zero-only statistical rule detects nonzero
mean headroom in all three stages, so the existing fixed pools are eligible for
a Gate 1B cheap-information learnability probe. However, no minimum practical
effect size was pre-declared; the late-stage result is sparse and very small and
must not be reported as strong practical headroom.

## Interpretation boundary

- Early is a validity-rescue task, not property optimization.
- Middle has clear property-improvement headroom.
- Late is close to local saturation under the current t=0 intervention pool.
- This audit does not show that RDKit descriptors can predict gain and does not
  show that an LLM can exploit the headroom.
- It does not authorize LLM Gate 1C, pairwise refinement, or PPO integration.

The full row-level artifact remains at:

`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1-formal-ad3889d-v1/gate1a-headroom-prompt-starved-v2-20260903-v1.json`

Slurm job 4607 passed 27/27 tests before the audit. Job 4608 produced the full
headroom artifact from clean source commit
`f8e5e314d43cf250394d3d770df9bbf4a87cdbb0`.
After archiving the audit and adding an exhaustive ordered-enumeration check
for exact expected Random NDCG@4, Slurm job 4609 passed 28/28 tests from commit
`73f18ae34ccabf1aae206d3ed15aafeac53953e2`.
