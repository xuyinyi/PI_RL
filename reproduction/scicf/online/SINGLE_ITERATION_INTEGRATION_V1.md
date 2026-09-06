# LLM-SciCF single-iteration integration smoke v1

Status: **executed with a no-go; no rerun, formal training, or multi-iteration
training is authorized**.

## Question

Can the accepted native PPO update and the hardened SciCF acquisition,
verification, and pairwise update execute once in their intended order without
credit leakage, optimizer failure, uncontrolled drift, or evaluator-budget
ambiguity?

## Frozen execution

```text
freeze pi_old
  -> collect 128 on-policy transitions
  -> standard PPO update with exact environment-return GAE actor credit
  -> two outcome-blind 24-candidate trajectory pools
  -> two blinded DeepSeek requests, maximum B=4 per pool
  -> selected-only matched K=2 factual/counterfactual verification
  -> at most one Oracle-delta-weighted pairwise update
  -> checkpoint
```

- Run only through Slurm on n001 against the accepted Stage-0 binding.
- Bind the passing pairwise-stability report by SHA-256 and required decision.
- Use one new development seed. Episode selection is a deterministic shuffle of
  all complete cross-timestep episodes and cannot inspect terminal outcome.
- Complete both LLM requests before generating any new counterfactual labels.
- LLM reasoning and confidence remain advisory and never enter either loss.
- The PPO actor receives exact GAE and the critic receives environment-derived
  returns. Counterfactual actions never enter PPO clipping.
- Pairwise supervision comes only from sign-consistent `K=2` Oracle pairs and is
  applied after the standard PPO update.

## Pass criteria

- Exactly one standard PPO update changes the policy with exact GAE actor credit
  and zero credit-phase evaluator calls.
- Both fixed pools contain 24 legal opaque candidates and span at least two
  timesteps; both API responses respect the maximum budget or explicit
  abstention schema with no reward/policy leakage.
- At least two selected candidates yield sign-consistent verified pairs.
- Exactly one pairwise update applies, improves mean signed preference margin,
  does not reduce pairwise accuracy, and changes the post-PPO policy.
- Across all 48 candidate states, maximum pairwise-update joint KL is at most
  `0.01`, non-target-factor KL at most `0.005`, and absolute critic-output drift
  at most `0.01`; value-head parameters remain unchanged.
- Evaluator sources, requested/unique/backend/cache counters, policy versions,
  source hashes, and the final checkpoint all close. Every matched environment
  branch must have a terminal record; only branches with a terminal molecular
  evaluation are expected to increment the QSPR evaluator ledger, because
  structurally invalid branches terminate before that evaluator is called.

## Stop and claim boundary

A schema, binding, candidate, policy, matched-seed, credit, ledger, drift, or
checkpoint failure stops the integration route. A threshold no-go is retained
as evidence and is not tuned away in this run. Passing opens only a separately
frozen short-horizon multi-iteration engineering smoke; it does not authorize
formal training, sealed-test access, algorithm-effectiveness claims, or
scientific claims.

## Execution outcome

Slurm Job 4687 completed the numerical path but exposed an erroneous ledger
assertion that treated structurally invalid, pre-QSPR terminal branches as
missing evaluator calls. The corrected clean coordinate
`b61d9e5c81dbef5108a2d796e0acd7ecf3a7640b` was executed in Slurm Job 4688.
That job failed closed on the first DeepSeek response because it invented the
out-of-pool candidate ID `cf-281`. The gate is therefore a no-go, and the next
scope remains closed. Evidence is retained under
`reproduction/results/scicf-single-iteration-integration-20260906/`.
