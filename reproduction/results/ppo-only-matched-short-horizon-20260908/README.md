# PPO-only matched control: Job 4751

## Result

The one authorized real PPO-only job completed all six iterations on n001.
Job 4751 ran from 2026-09-08 09:05:04 to 09:07:43, exited `0:0`, and produced
`matched_ppo_only_six_iteration_engineering_complete`. The analyzer in the same
Slurm job declared `eligible_complete_comparison=true`.

At seed 20260907 and 768 on-policy transitions per arm, the frozen descriptive
endpoint was 158 evaluated terminals for SciCF Job 4748 versus 146 for PPO-only.
The evaluated-terminal yield was 20.5729% versus 19.0104%, a SciCF-minus-PPO
difference of 12 terminals or 1.5625 percentage points. These denominators count
transitions, not completed episodes. No matched reward distribution or
held-out evaluation was available under this protocol.

## Complete trace

Each iteration has 128 transitions and eight PPO optimizer steps in both arms.
Neither arm stopped early for target KL.

| Iteration | SciCF evaluated terminals | PPO-only evaluated terminals | SciCF minus PPO | SciCF PPO KL | PPO-only KL |
|---:|---:|---:|---:|---:|---:|
| 1 | 27 | 27 | 0 | 0.0084227864 | 0.0084227864 |
| 2 | 23 | 22 | +1 | 0.0033467095 | 0.0023053624 |
| 3 | 22 | 24 | -2 | 0.0027417671 | 0.0032824464 |
| 4 | 23 | 31 | -8 | 0.0034589376 | 0.0034808423 |
| 5 | 28 | 22 | +6 | 0.0055487771 | 0.0052064937 |
| 6 | 35 | 20 | +15 | 0.0049933009 | 0.0060391352 |
| Total | 158 | 146 | +12 | — | — |

The predeclared iteration-2-to-6 secondary endpoint is 131/640 for SciCF and
119/640 for PPO-only, a difference of 1.875 percentage points. Before iteration
6, the cumulative terminal-count difference was -3; the final iteration accounts
for the positive total. The full trace therefore does not show consistent
iteration-by-iteration superiority.

## Actual resource accounting

| Quantity | SciCF Job 4748 | PPO-only Job 4751 | SciCF minus PPO |
|---|---:|---:|---:|
| On-policy transitions | 768 | 768 | 0 |
| PPO optimizer steps | 48 | 48 | 0 |
| Auxiliary optimizer steps | 6 | 0 | +6 |
| Evaluator requested calls | 299 | 146 | +153 |
| Evaluator unique/backend calls | 255 | 146 | +109 |
| Evaluator cache hits | 44 | 0 | +44 |
| Invalid evaluator results | 0 | 0 | 0 |
| LLM pool decisions | 12 | 0 | +12 |
| LLM HTTP transmissions | 13 | 0 | +13 |
| LLM prompt tokens | 102845 | 0 | +102845 |
| LLM completion tokens | 1695 | 0 | +1695 |
| LLM seconds | 22.689 | 0 | +22.689 |
| Runner elapsed seconds | 285.876 | 155.365 | +130.511 |
| Slurm job elapsed seconds | 294 | 159 | +135 |

Both arms had the same evaluator ceiling (1248), eight CPUs, 96G and one H100,
with a two-hour Slurm time limit. Actual spending differs. The SciCF evaluator
ledger includes 158 on-policy, 69 factual and 72 counterfactual requests; PPO-only
has 146 on-policy requests and zero other requests. Runner and Slurm durations
include different setup stages; SciCF's real job also ran tests, whereas the
control tests ran in its preflight. These timings do not isolate training speed.

## Equivalence and integrity

The initial policy-state hash matched exactly:
`3218aeada82b9a49b306eaa345cde010693222d13952e1708f3767432c503b59`.
After the first PPO update, the policy-state hash also matched:
`ce518902a2321fa0968c200bc0850a6f753f60b4ec88075054582db22cd11b2f`.
The first rollout digest, GAE, actor-credit and critic-return hashes matched
Job 4748, and every first-iteration PPO metric had zero difference.

All six control iterations retained exact GAE actor credit, environment-return
critic targets, finite PPO diagnostics and zero credit-estimator Oracle calls.
The six saved checkpoint hashes and chained iteration-report hashes were
recomputed by the analyzer before it produced the comparison. An independent
post-job `sha256sum` inventory is archived as `remote-artifact-sha256.txt`.

The shared engine declares the standard PPO source set including `evaluation`.
An additive `OnPolicyOnlyEvaluator` wrapper rejects every non-`ppo/on_policy`
request before it reaches the cache/backend. Its rejection behavior was tested;
the actual ledger contains only `ppo/on_policy`. Shared engine/environment/GAE
sources were unchanged relative to the SciCF implementation.

Initial seed matching is established. Later random-stream coupling between
SciCF auxiliary work and sampling has not been isolated; per-iteration RNG
digests are saved. The comparison concerns the complete SciCF path, not the
causal contribution of LLM ranking alone.

## Preflight and authorization

- Implementation: `a1059ca9d3dcd2d9b111de2472f90966ebadb815`.
- Frozen protocol commit: `8bd5423`.
- Authorization record commit: `a497db3`.
- Job 4749 stopped before tests/runtime because the initially supplied base
  interpreter had no pytest. Both logs are retained. No PPO/AFP/API ran.
- The exact Job 4748 interpreter and dependency path were recovered from Slurm
  accounting, with no code or protocol change.
  Interpreter: `/home/wch/workspaces/DAPiGen-reproduction/stage0-v23-mask-dev-20260904/.test-venv/bin/python`;
  package root: `/home/wch/workspaces/DAPiGen-reproduction/dependencies/p2-gymnasium-0.29.1-py38-363fec3`.
  Clean runtime checkout: `/home/wch/workspaces/DAPiGen-reproduction/ppo-only-control-a1059ca`.
  Slurm entry: `reproduction/slurm/run_ppo_only_matched_control.sbatch`.
- Job 4750 passed 38 tests in 4.00 seconds and built the full runtime. Its
  preflight report records zero real PPO iterations, zero AFP calls and zero API
  requests, with exact initial policy and model fingerprints.
- User approval covered implementation, server verification and one real run
  after preflight. Job 4751 is the sole real submission. Its authorization has
  an exclusive consumption marker and a separate submission lock on n001.
- No local experiment code, new API request, real rerun, resume or sealed-test
  evaluation was executed.

## Provenance and scope

The frozen protocol and analysis remain byte-identical. The protocol was written
after observing SciCF and before PPO-only; this timing is retained explicitly.
The experiment has one independent training run per arm. No significance test,
multi-seed claim, final-policy quality claim or reward-improvement claim follows.
The sixth on-policy rollout precedes the sixth PPO/auxiliary update, so this
endpoint also does not evaluate the final saved policies.

Relevant files:

- `execution-authorization.json` and `authorization-consumption.json`;
- `preflight/preflight-report.json`;
- `run/control-report.json` and six per-iteration reports;
- `run/analysis/comparison.json` (all PPO diagnostics and per-iteration costs);
- `slurm-accounting.txt`, `logs/` and `remote-artifact-sha256.txt`.

Remote checkpoints remain under
`/home/wch/workspaces/DAPiGen-reproduction/runs/ppo-only-matched-short-horizon-v1-seed20260907-20260908`.
The repository copy excludes the six `.pt` files and retains their hashes.

SHA-256 anchors:

- protocol: `0a0f138f51862652ae4033700a25099b76b1f441d11d4acec0759d3ae473cc80`
- analysis protocol: `66c53241055aab8bf6a649c5271404731eece2e68fa099933f90c7498cc4029d`
- passing preflight: `e94d20b34dbf0c19ecc719b3aba7ed2650e9a5f878bf705406cd2bacc0652393`
- authorization: `4ec4c0f6f5c8e83e40f3d6d7e6cbfb30f9039ecda610027fb798c5ec253b1032`
- control report: `e70115954580fc43765a99b8404bf194f425797d8f44200e2d04a1b94c58a534`
- comparison: `0696b0626ef8dc6f994effc724389c564cfa1b2097fa032eccb11d94598ca65c`
- SciCF reference report: `07ffac03f444a46bbb7e412550d1919b384fd9753777b466478ae08c0e3c1e11`
- transferred evidence archive: `b806e16ba1a0b641636a9b4c55db6fb00f56dd7b3bf9f79d598162f0832dc616`

The authorized implementation, preflight, one real control and descriptive
comparison are complete. A useful next experiment would record comparable
on-policy rewards and evaluate both final checkpoints under a separately fixed
evaluation budget, followed by independent seeds. None was started here.
