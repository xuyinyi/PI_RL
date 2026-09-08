# SciCF short-horizon real engineering smoke: Job 4748

## Decision

The exact schema-5 authorization was consumed by one n001 Slurm submission.
Job 4748 completed all six iterations successfully at clean implementation
commit `5f2a6a100edcfd0b15fe82228a0eef8e1c2c2904`:

```text
short_horizon_engineering_run_complete_no_further_scope_authorized
```

All six iterations retained their primary PPO update and applied one
Oracle-weighted SciCF auxiliary update. There were no integrity failures,
budget overruns, provider failures, circuit-open iterations, candidate-ID
violations, or materially negative KL diagnostics. This is short-horizon
engineering evidence only. It does not establish improvement over PPO and does
not authorize an automatic rerun, automatic resume, formal training, matched
comparison, sealed-test access, algorithm-effectiveness claim, or scientific
claim.

## Six-iteration result

| Iteration | PPO transitions | PPO approximate KL | Selected / weighted | LLM seconds | Soft effective mass | Auxiliary maximum KL | Cumulative evaluator requests |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.008423 | 7 / 2 | 4.268 | 1.4733 | 1.23e-7 | 45 |
| 2 | 128 | 0.003347 | 8 / 2 | 3.053 | 1.2833 | 2.46e-7 | 85 |
| 3 | 128 | 0.002742 | 8 / 4 | 5.573 | 2.2106 | 0 | 130 |
| 4 | 128 | 0.003459 | 8 / 5 | 3.457 | 4.1667 | 1.59e-7 | 198 |
| 5 | 128 | 0.005549 | 8 / 2 | 3.253 | 0.8478 | 2.05e-7 | 234 |
| 6 | 128 | 0.004993 | 8 / 5 | 3.086 | 3.3967 | 3.72e-7 | 299 |

Each PPO update used eight optimizer steps. Each auxiliary update used exactly
one optimizer step. The primary checkpoint was written and independently hashed
before credential loading in every iteration.

## Aggregate accounting

- PPO transitions: 768
- successful terminal episodes: 158
- DeepSeek pool decisions: 12
- semantic attempts / HTTP transmissions: 13 / 13
- prompt / completion tokens: 102,845 / 1,695
- selected and `K=5` verified candidates: 47
- candidates with non-zero empirical training weight: 20
- SciCF auxiliary updates applied: 6
- cumulative LLM wall time: 22.689 seconds of the 180-second ceiling
- evaluator requested / unique / backend / cache-hit calls:
  299 / 255 / 255 / 44
- evaluator request ceiling: 1,248
- maximum auxiliary KL across all iterations: `3.7181e-7`
- final circuit state: closed, zero consecutive failures

The LLM selected candidates only; its reasoning and confidence did not enter
the loss. All auxiliary weights came from matched AFP factual/counterfactual
outcomes, and counterfactual actions did not enter PPO clipping.

## Schema repair and resilience

Eleven of the twelve pool decisions validated on their first response. The
second pool in iteration 3 returned one schema-invalid ranked response. That raw
response was persisted before validation, the bounded single repair succeeded,
and the candidate allowlist remained valid. Across the complete run there are
13 request, raw-response, and validation records on n001.

No live provider failure occurred, so the real run did not open the circuit.
Live persistence is evidenced by the six exact control-checkpoint hash links;
the two-failure cooldown path remains supported by the separate synthetic
preflight in Job 4737 rather than by this successful provider trace.

## Checkpoints and logs

All six primary PPO checkpoints, final engine checkpoints, and control
checkpoints remain on n001 in the output directory recorded by the report. The
six primary, engine, and control hashes were independently recomputed after the
job. Every value matched its iteration report, and every control checkpoint
contained the exact previous-control hash. The final control SHA-256 is
`5a8efe21eba0259eafb666b2a26cdf2cc0f663732e466ac40a427784fca5c80e`.

The stderr log contains a tokenizer warning, one DGL deprecation warning, and
RDKit explicit-valence diagnostics for invalid structures. It contains no
Python exception or Slurm failure. The terminal evaluator ledger reports zero
invalid evaluator results, all report integrity checks passed, and all Slurm
steps exited `0:0`.

## Provenance

- Job / state / exit: `4748` / `COMPLETED` / `0:0`
- Submit / start / end: `2026-09-08 08:35:56` / `08:35:56` / `08:40:50`
- Elapsed: 4 minutes 54 seconds
- Allocation: 8 CPU, 96 GiB, one H100 GPU
- Tests: `67 passed in 4.80s`
- Authorization SHA-256:
  `a57d9d42255a93b94be362109ca03b0188e79acf6c0d699edb01b480c863caad`
- Protocol SHA-256:
  `ec4e70980e34ae9f69e38b10fc485588cfcf56238883f7066bb627511a044254`
- Report SHA-256:
  `07ffac03f444a46bbb7e412550d1919b384fd9753777b466478ae08c0e3c1e11`
- Complete remote-artifact manifest SHA-256:
  `ffe7a5413d2482c190de99b545c7f929551394d827c74c476873c28e8f2f92bb`
- Final history digest:
  `1a2bdb16a01ba33178808ee2c04c0297a977e782a443b760752b3caee003a682`

The authorization and output identifier retain `20260907` because that unique
coordinate was frozen on the preceding day; the single authorized Slurm
submission occurred on 2026-09-08.

The repository copy excludes model checkpoint binaries, blinded request
payloads, and raw DeepSeek response content. They remain on n001, while
`run/artifact-sha256.txt` binds all 137 remote files. The archived guarded
outcomes and validation records retain the hashes needed to audit the omitted
raw responses without placing their content in Git.

## Next boundary

This authorization is consumed and cannot be reused. Before any comparison or
longer training, separately freeze a matched PPO-only short-horizon control and
an analysis contract that compares both methods at identical predeclared
budgets and seeds. Job 4748 alone is not evidence that SciCF improves PPO.
