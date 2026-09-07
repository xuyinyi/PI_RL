# SciCF single-iteration integration v3

## Purpose

Version 3 replaces the archived v2 strict `K=2` admission rule with `K=5`
matched continuations and a continuous Oracle-evidence weight. It also makes
the LLM acquisition an explicitly optional auxiliary stage so an expected
provider or response failure cannot erase or stop the already completed PPO
iteration.

This is an engineering protocol. It does not establish algorithmic benefit and
does not authorize a real run, an automatic rerun, multi-iteration training,
sealed-test access, or a scientific claim.

## Primary transaction boundary

One standard PPO iteration is the primary transaction. The runner must:

1. validate the execution authorization and all frozen assets;
2. run exactly one standard PPO iteration;
3. write and hash a primary PPO checkpoint and receipt;
4. only then load private API settings or attempt an LLM request;
5. retain the primary checkpoint regardless of any recoverable LLM outcome.

The following outcomes are recoverable and are recorded as
`ppo_only_degraded`: provider timeout, rate limiting, provider unavailability,
bounded transport exhaustion, bounded schema-repair exhaustion, valid LLM
abstention, insufficient effective soft-pair mass, or pairwise KL rollback.
There is no silent heuristic selection and no cached-response substitution.

Authorization, asset/hash, budget, information-leakage, allowlist, or unexpected
programming failures remain fatal. An unvalidated LLM response never reaches the
AFP evaluator.

## K=5 empirical soft weight

For each LLM-selected intervention, the evaluator produces five matched
factual/counterfactual deltas. Deltas within `0.005` are practical ties. The
majority of the non-tie signs determines direction; a balanced sign vote or all
ties abstains.

The training weight is:

```text
2.0 * posterior_sign_confidence * nonzero_fraction
    * min(abs(median_delta) / 0.05, 1.0)
```

The posterior sign confidence uses a Jeffreys `Beta(0.5, 0.5)` prior. Weights
below `0.05` abstain. The optional update additionally requires at least two
weighted candidates, effective pair mass of at least `0.5`, and no single pair
holding more than `75%` of the total mass. At most one weighted pairwise
optimizer step is allowed. LLM self-reported confidence never enters the loss,
and counterfactual actions never enter PPO clipping.

## Availability control

After two consecutive recoverable LLM failures, the circuit opens for the next
three iterations and then permits a retry. Each skipped iteration remains a
normal PPO-only degraded iteration. The breaker state is serializable so a
future bounded multi-iteration runner can persist it across checkpoints.

The present v3 runner is single-iteration only; multi-iteration orchestration
requires a separate frozen protocol and authorization.

## Preflight boundary

The full-runtime preflight must run on n001 through Slurm. It loads and binds
polyBERT and the AFP evaluator runtime, but makes zero API calls, loads no API
credentials, runs zero PPO iterations, and makes zero AFP Oracle calls. It uses
only synthetic LLM outcomes and synthetic deltas to test degradation, circuit,
schema-4 authorization, K=5 weighting, and source-ordering controls.

