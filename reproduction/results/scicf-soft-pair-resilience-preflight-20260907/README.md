# SciCF K=5 soft-pair and optional-LLM resilience preflight

## Decision

n001 Slurm Job 4714 passed the mock-only component preflight at clean commit
`7d005b07df0cb3022ead92efedd13d3aec04fc3c`:

```text
pass_components_only_no_real_run_authorized
```

Eighteen server tests passed and all 20 report checks passed. The job accepted
no credential-file argument and used zero external API requests, PPO
iterations, Oracle calls, local models, or sealed-test accesses.

This result validates the K=5 soft aggregation and optional-LLM control
components only. It does not authorize a real v3 run, multi-iteration training,
formal training, or scientific claims.

## Frozen configuration

- Protocol: `dapigen-scicf-soft-pair-resilience-dev-v1`
- Config SHA-256:
  `97ed1bcf468293faa5c9e0eafb2dfde819061c201996ccbbcaf6b66bacce7173`
- Exactly five matched continuations per candidate
- Practical tie band: `abs(delta) <= 0.005`
- Jeffreys prior: `Beta(0.5, 0.5)`
- Magnitude scale: `0.05`
- Maximum / minimum non-zero training weight: `2.0` / `0.05`
- Minimum normalized effective-pair mass: `0.5`
- Minimum weighted candidates: 2
- Maximum single-candidate share of total mass: 75%
- At most one auxiliary pairwise optimizer step per iteration

The weight uses only matched Oracle evidence: posterior sign confidence,
non-tie fraction, and robust median delta magnitude. LLM self-reported
confidence does not enter the loss.

## Soft-weight checks

The synthetic matrix verified that:

- 5:0 sign support receives more weight than 4:1;
- 4:1 receives more weight than 3:2;
- 3:2 remains a small non-zero contribution rather than a strong label;
- three same-sign deltas plus two ties can retain reduced weight;
- a balanced direction and five ties both abstain;
- adequate distributed mass permits the optional update; and
- zero mass skips the auxiliary stage while keeping primary training alive.

The passing gate example contained four weighted candidates with total weight
`2.893333333333333`, normalized effective mass `1.4466666666666665`, and
maximum single-candidate mass fraction `0.576036866359447`. The zero-mass
example correctly remained ineligible while reporting
`continue_primary_training: true`.

The server tests also exercised a real CPU tensor update on a small policy:
exactly one weighted pairwise step changed the policy, left value-head
parameters unchanged, used no LLM confidence, and kept counterfactual actions
out of PPO clipping.

## LLM non-blocking checks

The primary transaction requires an existing PPO checkpoint whose SHA-256 is
recomputed successfully before any LLM call. After that durable boundary:

- provider timeout -> `ppo_only_degraded`, training continues;
- schema exhaustion -> `ppo_only_degraded`, training continues;
- two consecutive recoverable LLM failures -> circuit opens;
- the next three iterations skip LLM calls without stopping PPO;
- the following validated response closes the circuit; and
- insufficient soft-pair mass after verification -> `ppo_only_degraded`,
  training continues.

Every degraded path records `silent_fallback_used: false`. A heuristic or
policy-only fallback may not be silently presented as successful SciCF.
Authorization errors, binding errors, budget overruns, information leakage,
unvalidated LLM output, and unexpected programming errors remain fail-closed.

## Slurm and evidence

- Job: 4714
- State / exit: `COMPLETED` / `0:0`
- Elapsed: 7 seconds
- Allocation: 4 CPU, 16 GiB, no GPU
- Tests: `18 passed in 3.76s`
- Report SHA-256:
  `bd95fbe56f2fb47fdadb1b736ba3388cfb5755f9777d8916efcf2d29699400f8`
- Mock checkpoint SHA-256:
  `69cad890df6e2b733479a62e0f3da2b613e91580ec22033a2804735570b558b9`

Artifacts:

- `run/soft-pair-resilience-preflight-report.json`: authoritative component
  report and every synthetic receipt;
- `run/mock-primary-ppo-checkpoint.bin`: explicit mock checkpoint used to prove
  the pre-LLM existence/hash boundary;
- `logs/`: Slurm stdout and stderr; and
- `slurm-accounting.txt`: job and step accounting.

## Next boundary

The next permissible engineering step is to implement a separately gated v3
single-iteration runner that connects these passing components to the existing
polyBERT/AFP-bound runtime. That runner still requires a no-credential full
runtime preflight before any request for a real execution. This component pass
does not itself authorize that runner, a DeepSeek call, PPO, Oracle evaluation,
or multi-iteration training.
