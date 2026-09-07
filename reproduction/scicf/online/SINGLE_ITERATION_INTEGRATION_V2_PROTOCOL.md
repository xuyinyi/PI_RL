# LLM-SciCF single-iteration integration-v2 protocol freeze

Status: **protocol frozen; model-asset-hardened runner passed mock-only
preflight, but no new real execution is authorized**.

## Purpose

Repeat the v1 single-iteration engineering integration with exactly one
scientific change: route both real DeepSeek acquisition decisions through the
already accepted bounded response-schema guard. The v1 PPO, candidate-pool,
Oracle-verification, pairwise-update, seed, and threshold contracts remain
unchanged.

This protocol is a response-boundary repair of the failed v1 integration. It is
not a new performance experiment and may not be used to tune the prompt,
candidate pool, PPO, verification budget, pairwise update, or pass thresholds.

## Frozen prerequisites

- v1 remains a no-go. Its corrected Job 4688 failed before Oracle verification
  after DeepSeek returned the out-of-pool identifier `cf-281`.
- The mock-only response-schema robustness gate passed in n001 Slurm Job 4689
  at clean commit `5387611e1ba088b8dafd0c50a80751059e8fdfed`.
- The exact prerequisite reports, source files, Stage-0 manifest, and their
  SHA-256 values are bound in
  `configs/single_iteration_integration_v2_protocol.json`.
- An execution implementation must validate every bound artifact before loading
  credentials, constructing the environment, running PPO, or calling the API.

## Frozen execution order

If separately implemented and authorized, one run must execute in this order:

```text
write immutable run intent and validate all bindings
  -> freeze pi_old
  -> collect 128 on-policy transitions
  -> exactly one standard PPO update
  -> build two outcome-blind 24-candidate cross-timestep pools
  -> finish both guarded DeepSeek decisions, maximum B=4 per pool
  -> only if both guard outcomes are validated, verify selected candidates at K=2
  -> at most one Oracle-delta-weighted pairwise update
  -> write terminal report and final checkpoint
```

All work must run through Slurm on n001 from a clean Git coordinate. The two
candidate pools and their original blinded messages must be written before the
first provider call. No Oracle call may begin until both acquisition outcomes
are valid, including a valid explicit abstention.

## Bounded LLM response contract

Each of the two pool decisions uses the frozen response guard:

- one initial schema attempt plus at most one semantic repair;
- at most one transport retry per logical attempt;
- at most four HTTP transmissions per pool and eight for the complete run;
- at most 512 completion tokens per semantic attempt;
- the original candidate-ID allowlist and `B=4` budget are unchanged during
  repair;
- every request receipt is persisted before its provider call;
- every returned model-content string is persisted before extraction or schema
  validation;
- invalid raw text is retained but is not replayed to the model;
- no reward truth, policy score, API key, or credential path is persisted or
  added to a repair request; and
- no heuristic fallback, padding, cached-response substitution, or unvalidated
  ID may authorize Oracle selection.

Accounting must distinguish two pool decisions, up to four semantic attempts,
and up to eight HTTP transmissions. Token accounting aggregates every completed
attempt, including invalid attempts and repairs; missing provider usage is
reported as incomplete rather than imputed.

## Fail-closed behavior

Schema exhaustion or transport exhaustion for either pool terminates the run
before all Oracle verification and pairwise refinement. The runner must still
write a terminal failure report containing the completed PPO receipt, pool and
request identities, guard outcome, attempt counts, retained artifact hashes,
and zero post-failure Oracle/pairwise counts. A partial valid selection from the
other pool is not verified and cannot be cached for a later run.

A valid abstention is not a schema failure. It contributes zero selected
candidates and proceeds to the unchanged integration thresholds; insufficient
accepted pairs therefore remains a declared no-go rather than being filled by a
fallback selection.

Any preflight, binding, schema, leakage, budget, ordering, ledger, drift,
checkpoint, or threshold failure returns a no-go. Slurm `COMPLETED` by itself is
not a passing gate; only the frozen terminal decision is authoritative.

## Post-Job-4693 model-asset hardening

The first authorized v2 attempt failed before PPO because its submitted
`QSPR/polyBERT` path was a source package without checkpoint files. The
scientific protocol remains unchanged, but the execution boundary now requires
the complete accepted polyBERT asset to be validated before credential loading.

The exact resolved model path, repository asset-binding SHA-256, 14 core-file
hashes, and the existing full-checkpoint fingerprint are recorded in the run
intent and must also be present in authorization schema version 2. The full
fingerprint uses the existing Stage0 checkpoint algorithm, including download
metadata and lock files and excluding only `.git` and `__pycache__`. The runner
also passes this fingerprint to the Stage0 factory for a second check during
model construction.

CPU-only n001 Slurm Job 4694 passed the resulting mock/preflight at clean commit
`46aef57c693a4644a52bdfcd337cbc90184ac31e`. This permits only requesting a
fresh exact one-run authorization. It did not load credentials, call the API,
run PPO or Oracle verification, or create an execution authorization.

## Immutable numerical and scientific boundary

The v1 values remain fixed: seed `20260907`, one 128-transition PPO iteration,
two pools of 24 candidates, maximum `B=4`, selected-only matched verification at
`K=2`, minimum two accepted pairs, and at most one pairwise optimizer step. PPO
and pairwise hyperparameters, evaluator budget, Stage-0 binding, and all drift
and preference thresholds are copied exactly into the protocol JSON.

Counterfactual actions never enter PPO clipping. The PPO actor uses exact
environment-return GAE, the critic uses environment-derived returns, and only
sign-consistent Oracle-verified pairs may supervise refinement. LLM reasoning
and confidence remain advisory and enter neither loss.

## Authorization and stop boundary

This freeze authorizes no execution. It does not authorize credentials loading,
a DeepSeek request, PPO, Oracle verification, a rerun, multi-iteration training,
sealed-test access, baseline mutation, or any effectiveness or scientific
claim. Implementation and server-only mock/preflight verification require a
separate development authorization. The real one-run Slurm execution then
requires another explicit authorization and a unique execution manifest bound
to this frozen protocol and its implementation commit.

Even a passing integration-v2 run would open only a separately frozen bounded
short-horizon multi-iteration engineering protocol. It would not authorize that
protocol's execution or formal training.
