# SciCF short-horizon multi-iteration runner preflight

## Decision

n001 Slurm Job 4737 passed the no-credential full-runtime preflight at clean
implementation commit `5f2a6a100edcfd0b15fe82228a0eef8e1c2c2904`:

```text
go_request_separate_short_horizon_real_execution_authorization
```

Twenty-nine server tests passed and all 30 report checks passed. This decision
permits only a request for a new, exact, one-submission schema-5 authorization.
It is not that authorization and does not permit credentials, API requests,
real PPO, AFP Oracle calls, automatic resume/rerun, formal training, sealed-test
access, an algorithm-effectiveness claim, or a scientific claim.

## What was exercised

- complete polyBERT and reconstructed AFP runtime construction on n001;
- a six-iteration synthetic controller sequence with a checkpoint after every
  iteration;
- two recoverable LLM failures, a three-iteration circuit-open cooldown, and an
  iteration-6 retry;
- exact restoration of the engine, circuit-breaker state, cumulative LLM
  wall-clock state, chained history digest, and prior control-checkpoint hash;
- a 60-second per-iteration and 180-second cumulative LLM budget contract, with
  each transport timeout bounded by the shared remaining deadline;
- exact schema-5 authorization rejection when all real operations are closed;
- the 1,248 requested-evaluator-call worst-case ceiling; and
- numerical KL handling that clamps values in `[-1e-6, 0)` to zero and rejects
  values below `-1e-6`.

The synthetic controller attempted LLM acquisition only in iterations 1, 2,
and 6. Its persisted cumulative times were 20 seconds after iteration 2,
20 seconds through the three circuit-open skips, and 25 seconds after the
iteration-6 recovery. No provider was contacted.

## Explicit zero-use boundary

- credentials loaded: no
- external API requests: 0
- PPO iterations: 0
- AFP Oracle calls: 0
- AFP property inferences: 0
- local LLM loaded: no
- sealed-test accesses: 0

Runtime construction performed the expected local polyBERT initial-embedding
inference. All checkpoint and acquisition outcomes in this preflight are
synthetic control-flow fixtures, not training evidence.

## Provenance

- Job / state / exit: `4737` / `COMPLETED` / `0:0`
- Elapsed: 10 seconds
- Allocation: 8 CPU, 96 GiB, one H100 GPU
- Tests: `29 passed in 1.54s`
- Implementation commit:
  `5f2a6a100edcfd0b15fe82228a0eef8e1c2c2904`
- Protocol SHA-256:
  `ec4e70980e34ae9f69e38b10fc485588cfcf56238883f7066bb627511a044254`
- Runner SHA-256:
  `60619f2185995421faf9bef9f4b088b61efc17b4772dd0731c7099b90555246c`
- Preflight runner SHA-256:
  `49ded95a78893c9a56e31ca88b05e51577df06d27b704376b595f9cd3c98c835`
- Report SHA-256:
  `a5505456a5e00d85a9ee4b8374007cbead80976c2d0a13f599eb13d612082237`
- polyBERT fingerprint:
  `6bdd24f951dd90d3031e749ef0130752811bfefe6c850af82b805cf015ea195f`
- AFP evaluator fingerprint:
  `0bdcea6155f5a53acd94afd322d408fe3e31c02cf13778a0c65f5f0938096ab4`

The 141-byte stderr file contains only the Hugging Face tokenizer warning that
no maximum length was defined, so truncation defaulted to disabled. It contains
no error and every Slurm step exited `0:0`.

Artifacts:

- `run/short-horizon-preflight-report.json`: authoritative report;
- `run/run-intent.json`: pre-execution zero-use scope and asset receipt;
- `run/closed-schema5-authorization.json`: deliberately closed authorization;
- `run/mock-control-sequence/`: synthetic checkpoint/restore evidence;
- `logs/`: Slurm stdout and stderr; and
- `slurm-accounting.txt`: job and step accounting.

## Next boundary

A real short-horizon engineering run would require new explicit user
authorization and a schema-5 file bound to commit `5f2a6a1`, this protocol
hash, both asset fingerprints, an exact iteration range, an exact previously
absent output directory, and optionally one exact resume checkpoint path and
hash. It may be submitted only once through n001 Slurm. This preflight does not
authorize creating that file or submitting the real run.
