# SciCF short-horizon multi-iteration runner v1

## Scope

This coordinate develops and preflights a maximum-six-iteration engineering
runner. It does not authorize credentials, API requests, PPO, AFP Oracle calls,
a real multi-iteration run, automatic resume/rerun, formal training, sealed-test
access, or any effectiveness or scientific claim.

The runner is based on the successful v3 single iteration from n001 Slurm Job
4736. It preserves standard PPO as the primary transaction and K=5 empirical
soft pairs as an optional auxiliary update.

## Non-blocking LLM contract

Every iteration performs and hashes the primary PPO checkpoint before reading
credentials or entering LLM acquisition. Provider, transport, schema,
abstention, time-budget, insufficient-mass, and pairwise-KL rollback outcomes
are explicit `ppo_only_degraded` iterations. They cannot erase the primary PPO
checkpoint. Silent fallback is forbidden.

The LLM wall-clock budget is shared across both pools, schema repair, and
transport retries:

- maximum 60 seconds in one iteration;
- maximum 180 seconds over the six-iteration coordinate; and
- every transport timeout is reduced to the remaining shared deadline.

The cumulative consumed time is stored in every control checkpoint. Restarting
from a checkpoint therefore cannot reset the total LLM allowance.

## Persistent circuit breaker

Two consecutive recoverable LLM failures open the circuit. Iterations 3-5 skip
LLM acquisition while PPO continues; iteration 6 may retry. The breaker state,
including failure count and `open_until_iteration`, is stored alongside the PPO
engine state and the cumulative LLM wall-time state after every iteration.

Each control checkpoint contains:

- exact protocol SHA-256;
- completed iteration;
- independently hashed PPO engine checkpoint;
- circuit-breaker state;
- cumulative LLM wall-time state;
- chained iteration-history digest;
- previous control-checkpoint hash; and
- explicit assertions that no credentials or API key were persisted.

Resume requires the exact control path and SHA-256 in a separate schema-5
authorization. Automatic resume is not allowed.

## KL numerical policy

Analytical KL is non-negative, but float32 summation produced
`-2.8936e-7` in Job 4736. Values in `[-1e-6, 0)` are now recorded as zero.
Values below `-1e-6` are a fatal contract violation rather than being hidden.
This policy is applied to both action factors before their joint KL is used for
the auxiliary rollback decision.

## Scientific and compute ceilings

- at most six PPO iterations;
- at most eight LLM-selected candidates per iteration;
- exactly five matched replicates per selected candidate;
- at most 80 verification branches per iteration;
- at most one auxiliary optimizer step per iteration; and
- at most 1,248 requested evaluator calls across the coordinate, equal to the
  six-iteration worst-case ceiling of 128 rollout plus 80 verification calls.

## Required preflight

The preflight must run on n001 through Slurm from a clean commit. It constructs
the complete polyBERT/AFP runtime, but runs zero PPO iterations and zero AFP
property evaluations. It accepts no credential argument and makes zero external
API calls. Synthetic state must prove two failures, three circuit-open skips,
iteration-6 recovery, exact checkpoint restoration, cumulative wall-time
restoration, and the KL numerical boundary.
