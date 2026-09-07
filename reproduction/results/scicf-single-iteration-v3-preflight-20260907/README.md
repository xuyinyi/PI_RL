# SciCF single-iteration v3 full-runtime preflight

## Decision

n001 Slurm Job 4715 passed the no-credential full-runtime preflight at clean
commit `93501f33cf4594791b31a805b153bf1accc94538`:

```text
go_request_separate_real_single_iteration_v3_authorization
```

Twenty-five server tests passed and all 31 report checks passed. This result
validates the v3 runner wiring and permits only a request for a new, exact,
single-run schema-4 authorization. It is not that authorization and does not
permit an automatic rerun, multi-iteration training, formal training, sealed-
test access, an algorithm-effectiveness claim, or a scientific claim.

## What was exercised

- the accepted polyBERT and reconstructed AFP asset paths, inventories, full
  fingerprints, versions, and the routing-only evaluator source delta;
- construction of the complete Stage-0 and PPO runtime;
- the source ordering `assets + authorization -> one PPO -> durable primary
  checkpoint -> private settings -> optional LLM`;
- exact schema-4 single-run authorization matching and closed-scope rejection;
- all five recoverable LLM codes as explicit `ppo_only_degraded` receipts;
- valid abstention and insufficient soft-pair mass as non-blocking degraded
  outcomes, with no silent fallback;
- a two-failure circuit threshold, exactly three skipped iterations, and retry
  after cooldown; and
- exact `K=5` soft aggregation, empirical-confidence weight ordering, and the
  effective-mass gate.

The full runtime loaded the scientific polyBERT/AFP assets. Runtime construction
performed the expected polyBERT initial-embedding inference. It performed no AFP
property inference. No local LLM was loaded.

## Explicit zero-use boundary

- credentials loaded: no
- external API requests: 0
- PPO iterations: 0
- AFP Oracle calls: 0
- local LLM loaded: no
- sealed-test accesses: 0

The mock checkpoint and synthetic LLM/Oracle evidence exist only to exercise
control flow. They are not training evidence.

## Provenance

- Job / state / exit: `4715` / `COMPLETED` / `0:0`
- Elapsed: 10 seconds
- Allocation: 8 CPU, 96 GiB, one H100 GPU
- Tests: `25 passed in 1.75s`
- Protocol SHA-256:
  `57d7a06c8e519370b2cbb70330db4132bc08e7ff24e1fefa393ef13467b2d925`
- Runner SHA-256:
  `4b9bf0bb71ddcc8f05e3feb5c56248c592f42dac7f2060d6eb5e38d32c915c7a`
- Report SHA-256:
  `b075cbf98713edc37f9b2ea9c6011aa421cc2eb946de7fb0ae97b007c864d79e`
- Soft-pair config SHA-256:
  `97ed1bcf468293faa5c9e0eafb2dfde819061c201996ccbbcaf6b66bacce7173`
- polyBERT fingerprint:
  `6bdd24f951dd90d3031e749ef0130752811bfefe6c850af82b805cf015ea195f`
- AFP evaluator fingerprint:
  `0bdcea6155f5a53acd94afd322d408fe3e31c02cf13778a0c65f5f0938096ab4`

Artifacts:

- `run/integration-v3-preflight-report.json`: authoritative report;
- `run/run-intent.json`: pre-execution scope and asset receipt;
- `run/closed-authorization-fixture.json`: deliberately closed schema-4
  authorization used to prove rejection;
- `run/mock-primary-ppo-checkpoint.bin`: non-training checkpoint used only for
  checkpoint-before-LLM control testing;
- `logs/`: Slurm stdout and stderr; and
- `slurm-accounting.txt`: job and step accounting.

## Next boundary

A real v3 single iteration requires a newly created schema-4 authorization
bound to implementation commit `93501f3`, the exact protocol and soft-pair
hashes above, both asset fingerprints, and one previously absent output
directory. It may be submitted exactly once through n001 Slurm. Passing such a
run still would not automatically authorize multi-iteration training.

