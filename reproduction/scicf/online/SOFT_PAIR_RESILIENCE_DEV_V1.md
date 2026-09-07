# SciCF K=5 soft-pair and optional-LLM resilience development v1

Status: **frozen for no-credential mock/preflight only; no real execution or
multi-iteration training is authorized**.

## Motivation

n001 Slurm Job 4713 completed the full integration-v2 path but produced zero
strict pairs. Twenty of 32 selected-only factual/counterfactual branches ended
at the atom limit, and every candidate's two deltas either contained zero or
changed sign. The strict K=2 rule therefore provided no pairwise signal.

This coordinate makes two additive engineering changes without rewriting the
archived v2 result:

1. estimate one candidate preference from exactly five matched continuations
   using a continuous Oracle-evidence weight; and
2. commit standard PPO before attempting LLM acquisition, treating bounded LLM
   availability failures as an explicitly degraded optional auxiliary stage.

## K=5 soft empirical confidence

For each candidate and matched replicate, define

```text
delta_k = counterfactual_return_k - factual_return_k
```

Values within `0.005` of zero enter a practical tie band. Counts above and below
that band receive a Jeffreys `Beta(0.5, 0.5)` shrinkage prior. Direction is the
majority of non-tie signs. Its training weight is

```text
2.0
* posterior sign confidence
* non-tie fraction
* min(abs(median(delta)) / 0.05, 1)
```

Weights below `0.05`, balanced directions, and all-tie records abstain. A 3:2
majority can contribute only a small weight; it is not promoted to the same
status as 5:0. LLM self-reported confidence never weights the loss.

The optional pairwise step requires at least two weighted candidates, at least
`0.5` normalized effective-pair mass, and no candidate above 75% of total mass.
Insufficient mass skips only the auxiliary update and records
`ppo_only_degraded`; it does not invalidate or roll back the standard PPO
transaction.

## LLM availability boundary

The training iteration is ordered as:

```text
standard PPO update
  -> durable primary PPO checkpoint
  -> bounded optional LLM acquisition
  -> optional K=5 Oracle verification
  -> optional single weighted pairwise step
  -> explicit applied/degraded receipt
```

Provider timeout, rate limiting, provider unavailability, exhausted transport
retries, exhausted schema repair, valid abstention, and insufficient soft-pair
mass are recoverable. They preserve the committed PPO iteration, emit
`ppo_only_degraded`, and never substitute a heuristic while retaining the SciCF
label.

Two consecutive provider/schema failures open a circuit breaker. It skips the
next three iterations without making LLM requests, then allows one new attempt.
A validated response closes the circuit. Circuit state is checkpointable.

Authorization mismatch, model/evaluator binding mismatch, budget overrun,
information leakage, unvalidated model output, and unexpected programming
errors remain fatal. Availability is fail-open for PPO; scientific integrity is
still fail-closed.

## Reporting required for future training

Every iteration must report one of:

- `ppo_plus_scicf_soft_pair`: the auxiliary update was actually applied;
- `ppo_only_degraded`: PPO committed, but the LLM or pairwise auxiliary stage
  was skipped;
- fatal integrity failure: the whole run stops without disguising the breach.

Aggregate reports must include LLM request availability, schema success,
abstention, circuit-open iterations, verification branch counts, effective pair
mass, auxiliary application rate, and separate performance for applied and
degraded iterations. A run with degraded iterations cannot be silently labelled
as fully applied SciCF.

## Current authorization boundary

The configuration and preflight accept no credential argument and authorize no
external API request, PPO iteration, Oracle call, local LLM, sealed-test access,
real single-iteration run, multi-iteration training, formal training, or
scientific claim. A server preflight can validate only synthetic soft-weight and
resilience scenarios. Real integration requires a separately implemented and
authorized runner with the existing polyBERT and AFP bindings.
