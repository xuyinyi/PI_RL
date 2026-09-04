# P2-B/P2-C contract and full-stack profiling evidence

- Status: **contract validators and single-worker full-stack characterization passed**
- Date: 2026-09-04
- Formal Slurm job: 4665 (`COMPLETED`, exit `0:0`, elapsed `00:00:24`)
- Compute node: `yanlih100n1`, 8 allocated CPUs, 96 GiB requested memory, 1 GPU
- Profile commit: `7e3ec9d4436f94af40f6204788ac8dec9dd9466e` (clean)
- Accepted P1 commit: `373b3291ac04dbf654c134bee2ca61a0c86d1a68`
- Accepted environment: `dapigen:8004c1fa2055a186b4a4f3ed`

## Decision

The v1 `PPOEngine` / `CreditEstimator` interface, its 13-item test matrix and
the P2-C profiling protocol are frozen. Job 4665 passed the complete 15-test P2
suite before loading the real models. This accepts the executable contract
guards, not the future engine: all six integrated I-series tests remain pending.

The single-worker full-stack characterization passed with:

- unchanged Stage-0 source and a clean profile checkout;
- accepted custom DAPiGen chemistry and `closure_exact_cached` mask;
- real persistent polyBERT, 1,246-dimensional observations and the accepted
  checkpoint fingerprint;
- accepted persistent five-property QSPR evaluator and objective contract;
- exact requested/unique/backend/cache/source accounting;
- exact evaluator state/ledger round trip;
- zero `no_reaction_product`, invalid evaluator results or functional failures;
- no PPO, credit estimator, counterfactual query or external API invocation.

This result is pre-implementation engineering evidence. It is not P2 exit,
parallel-scaling admission, a training-budget freeze or scientific validation.

## Performance characterization

Model and stack construction took 4.535 s. The measured workload used two
identical 100-transition passes. Each pass covered 50 episodes and produced 31
successful terminals plus 18 atom-limit terminations; the final active episode
was incomplete at the fixed transition boundary.

| Phase | Elapsed | Transitions/s | Evaluator event time | Evaluator fraction |
|---|---:|---:|---:|---:|
| First pass | 14.812 s | 6.751 | 1.506 s | 10.17% |
| Exact cache replay | 0.461 s | 216.812 | 0.00042 s | 0.092% |

Replay was 32.114x faster than the first measured pass. This ratio combines
Stage-0 observation/mask caches with evaluator cache reuse; it is not a PPO
speedup and cannot be extrapolated to multi-worker training.

Peak process RSS was 1,318.40 MiB. Peak CUDA allocation/reservation was
151,284,736 / 157,286,400 bytes.

## Ledger evidence

| Phase | Requested | Unique | Backend | Cache hits | Invalid |
|---|---:|---:|---:|---:|---:|
| First pass | 62 | 31 | 31 | 31 | 0 |
| Exact cache replay | 62 | 0 | 0 | 62 | 0 |
| Final total | 124 | 31 | 31 | 93 | 0 |

Each successful terminal was attributed once to `environment_regression` and
once to `evaluation`. The first request reached the backend; the duplicate and
all replay requests were served from the shared per-run cache. Trajectory,
observation and terminal-evaluation digests matched exactly across passes.

## Preserved failed development run

Job 4664 ran the same 15 tests successfully but failed after four seconds at
commit `5efcf930ade731d698b503f2ac62f92210ed02fc`, before real model construction.
The profiler called `torch.cuda.reset_peak_memory_stats()` before CUDA context
initialization, producing `RuntimeError: Invalid device argument 0`. Commit
`7e3ec9d4436f94af40f6204788ac8dec9dd9466e` removed the unnecessary pre-reset.
Job 4665 then completed without a fatal stderr marker.

## Evidence layout

- `formal/profile/stage0_full_stack_profile.json`: complete report.
- `formal/logs/full-stack-4665.out`: tests and concise profiling summary.
- `formal/logs/full-stack-4665.err`: non-fatal tokenizer and DGL warnings.
- `superseded-failed-5efcf93/logs/`: immutable Job 4664 failure evidence.
- `SLURM_ACCOUNTING.txt`: captured accounting for Jobs 4664 and 4665.
- `SHA256SUMS.txt`: local evidence-file hashes.

Formal bundle: `dapigen-p2-full-stack-7e3ec9d.bundle`, SHA-256
`a11d460a22aae9e465387599a765f1643f87873d41d761b1f03f301f235eae08`.
The superseded bundle SHA-256 was
`48a86d5d56a20826c9a7f00c30d9e1f1249114c379236d58367e06ae9b327e8f`.
Both remote checkouts and run directories remain preserved under
`/home/wch/workspaces/DAPiGen-reproduction/`.
