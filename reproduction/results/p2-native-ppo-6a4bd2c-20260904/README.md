# P2 native PPO / GAE governed smoke evidence

- Status: **passed for PPO/GAE engineering scope**
- Date: 2026-09-04
- Formal Slurm job: 4671 (`COMPLETED`, exit `0:0`, elapsed `00:00:45`)
- Implementation commit: `6a4bd2c21b657c9f1b9c734028322cf66af6367f` (clean)
- Accepted P1 commit: `373b3291ac04dbf654c134bee2ca61a0c86d1a68`
- Accepted environment: `dapigen:8004c1fa2055a186b4a4f3ed`
- Bundle SHA-256: `409a3d2a588860f903522b09e9dcbf39fd89aaa6cdecb19cb1f006b93987526f`

## Decision

The common native `PPOEngine` skeleton and evaluator-free
`GAECreditEstimator` pass their current governed acceptance scope. Job 4671
passed 26 tests before running the accepted custom chemistry, real 1,246-D
polyBERT observation, persistent five-property QSPR evaluator and factorized
masked PPO model on one H100.

This accepts the PPO/GAE path and the engine-side portions of P2-I01, P2-I05
and P2-I06. It does not accept Policy-CC, MCC-PPO, I02-I04, multi-worker
training, a production budget, optimization performance or scientific validity.

## PPO smoke

The formal workload contains two distinct PPO iterations of 64 transitions
each. A third execution re-runs iteration 2 from the checkpoint saved after
iteration 1; it is a determinism check, not an additional training result.

| Iteration | Successful terminal evaluations | Requested/unique/backend | Reward sum | Optimizer steps |
|---|---:|---:|---:|---:|
| 1 | 12 | 12 / 12 / 12 | 3.5523 | 4 |
| 2 | 24 | 24 / 24 / 24 | 5.0394 | 4 |

All terminal calls were source-tagged `ppo/on_policy`; both GAE credit deltas
were exactly zero. Actor-credit hashes equal the common-engine GAE hashes, while
critic receipts remain bound to environment-return hashes. These reward sums
are smoke diagnostics and carry no learning-performance interpretation.

## Exact checkpoint/resume

The 4.61 MB checkpoint records the policy, optimizer, GAE provider,
environment/evaluator state, requested-call manager, NumPy/Python/Torch RNGs,
counters and contract identities. Loading it into a newly constructed stack
reproduced all 13 declared comparisons: contract, Stage-0 specification,
checkpoint ledger, batch ID, rollout, GAE, critic returns, actor credit, update
receipt, update metrics, policy state/version and final evaluator ledger.

The final ledger contains 36 requested / 36 unique / 36 backend / 0 cache-hit
calls, with zero invalid results. The final policy SHA-256 is
`70fb8ee4d08cf719ebebdd761b92ca27de975892e6947e3ab8e5d1cd424d16cf`.

## Runtime and dependency evidence

The run used Python 3.8.20, PyTorch 2.1.2, Gymnasium 0.29.1 and CUDA device 0.
Model/stack construction took 4.029 s and the complete smoke script took 36.373
s. Peak RSS was 1,680.24 MiB; peak CUDA allocation/reservation was 190,295,040 /
197,132,288 bytes.

Gymnasium was absent from the immutable P1 Python environment. Job 4668
installed four pinned wheels into an isolated P2 directory without modifying
P1: Gymnasium 0.29.1, Farama-Notifications 0.0.4, importlib-metadata 8.5.0 and
zipp 3.20.2. The pip report contains their upstream wheel hashes, and the
installed-file manifest records 472 files.

## Development diagnostics

- Job 4666 failed at the dependency probe because Gymnasium was absent; PPO did
  not run.
- Job 4667 installed the first two wheels but failed its import probe because
  Python 3.8 lacked `importlib_metadata`.
- Job 4669 passed 25 tests and executed PPO, then failed restoring CUDA RNG
  state because `torch.load(map_location=cuda)` moved the ByteTensor to CUDA.
  Commit `c548cb4` restores saved CUDA RNG tensors on CPU and copies immutable
  NumPy targets before tensor conversion.
- Job 4670 passed the corrected 25-test smoke. It was superseded by Job 4671,
  which adds the real Stage-0 evaluator concurrency/exhaustion test and passes
  26 tests.

Formal evidence is under `formal/`; pinned dependency evidence is under
`dependency/`; diagnostic runs are retained under `superseded-failed/` and
`superseded-passed/`. `SHA256SUMS.txt` hashes every retained evidence file.
