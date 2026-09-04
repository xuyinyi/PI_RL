# P2-A accepted-mask throughput evidence

- Status: **passed for accepted-mask throughput scope only**
- Date: 2026-09-04
- Formal Slurm job: 4663 (`COMPLETED`, exit `0:0`, elapsed `00:17:08`)
- Compute node: `yanlih100n1`, 8 allocated CPUs
- Profile commit: `d06365e4485baddcdad65386c69cc47056bf94be` (clean)
- Accepted P1 commit: `373b3291ac04dbf654c134bee2ca61a0c86d1a68`
- Accepted environment: `dapigen:8004c1fa2055a186b4a4f3ed`

## Decision

The frozen P2-A throughput admission passed. The formal report records:

- unchanged Stage-0 source relative to the accepted P1 commit;
- matching accepted configuration, custom chemistry backend and action catalogs;
- five profiler tests passed before profiling;
- empty specification-mismatch, functional-failure, determinism-violation and
  throughput-failure lists;
- zero `no_reaction_product` workload-generation failures;
- no terminal-evaluator, PPO or external-API invocation.

This decision admits `closure_exact_cached` mask evaluation to the next P2
engineering step. It is not acceptance of the unified engine, a PPO result, a
training-budget freeze or scientific evidence.

## Frozen workload and thresholds

- Base seed: `20260904`.
- Worker counts: 1, 2, 4 and 8.
- Repetitions: 3 per worker count.
- Deterministic workload states: 250 per worker.
- Timed operation: `BranchableDAPiGenCore.raw_action_mask_for_mode`.
- Minimum eight-worker speedup: 1.5x.
- Minimum eight-worker efficiency: 0.25.
- Minimum adjacent throughput ratio: 0.85.

State generation used the real accepted transition rules and custom DAPiGen
chemistry. A deterministic zero encoder was used outside the timed operation so
this profile isolates mask cost.

## Formal result

| Workers | Median masks/s | Speedup vs. 1 | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 7.329 | 1.000x | 1.000 |
| 2 | 14.738 | 2.011x | 1.005 |
| 4 | 29.174 | 3.981x | 0.995 |
| 8 | 33.807 | 4.613x | 0.577 |

The adjacent throughput ratios were 2.011 (1-to-2), 1.980 (2-to-4), and 1.159
(4-to-8). Aggregate peak worker RSS at eight workers was 1,730.98-1,741.18 MiB
across repetitions. Maximum worker setup time increased from 38.81 s at one
worker to 63.80 s at eight workers. The maximum observed start spread was
0.0081 s.

The 8-worker result clears the predeclared threshold, but the 4-to-8 increment
shows saturation. A worker count and training budget must therefore remain
unfrozen until a full-stack profile includes real observation encoding,
evaluator-ledger overhead and the unified engine path.

## Preserved failed development run

Job 4662 at candidate commit `ef5d4950e785b137e7d723ea432e5ce5b3b76ab0`
failed in two seconds after four tests passed. Its worker process raised
`AttributeError: 'NoneType' object has no attribute 'initial'` because
`build_profile_core` did not return the constructed core after a patch-placement
error. Commit `d06365e4485baddcdad65386c69cc47056bf94be` restored that return and
added a test that constructs the accepted-mask core. The corrected remote suite
passed 5/5 tests before Job 4663 ran.

Job 4663 stderr contains expected RDKit valence diagnostics from rejected
candidate chemistry. A fatal-pattern scan found no traceback, exception, fatal,
segmentation-fault, OOM, killed or error record in the formal stderr.

## Evidence layout

- `formal/profile/stage0_mask_throughput.json`: complete machine-readable report.
- `formal/logs/profile-4663.out`: test and benchmark summary.
- `formal/logs/profile-4663.err`: RDKit diagnostic stream.
- `superseded-failed-ef5d495/logs/`: immutable Job 4662 failure evidence.
- `SLURM_ACCOUNTING.txt`: captured `sacct` status for Jobs 4662 and 4663.
- `SHA256SUMS.txt`: local evidence-file hashes.

The formal Git bundle was
`dapigen-p2-mask-profile-d06365e.bundle`, SHA-256
`081615af887603c957c1568228de11a0a54d6f64aa8da37ef1df97e63309d364`.
The remote clean checkout and run directory were preserved under
`/home/wch/workspaces/DAPiGen-reproduction/`.
