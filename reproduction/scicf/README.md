# SciCF-PPO development boundary

Current handoff (2026-09-08): online K=5 SciCF and its PPO-only control completed
six iterations each in Jobs 4748/4751. See
[development handoff](../DEVELOPMENT_HANDOFF_20260908.md) for implemented modules,
open tasks and exact runtime paths. The chronological paragraph below preserves
the scope of each earlier authorization; it is not a statement that later
separately authorized work remains unimplemented.

> **Architecture-first amendment (updated 2026-09-07):** Historical SciCF evidence and
> no-go decisions remain unchanged. The online architecture smoke and the
> acquisition/verifier and pairwise-stability development gates passed under
> `reproduction/scicf/online/`, but the subsequent single-iteration
> integration smoke returned a no-go after a DeepSeek response invented an
> out-of-pool candidate ID. The subsequent mock-only response-schema robustness
> development gate passed.
> These are engineering and development evidence only; they do not reopen the
> sealed test or authorize performance claims. Only a separate integration-v2
> protocol is frozen and its authorization-gated runner passed a mock-only n001
> Slurm preflight. The separately authorized real attempt was submitted once as
> Job 4693 but failed before PPO because its selected polyBERT path was a source
> package without model configuration or weights. No API transmission, PPO
> iteration, Oracle call, or pairwise update occurred, and the consumed
> authorization was not reused. A new asset-completeness preflight and fresh
> exact-scope authorization are required before any further real attempt. That
> asset hardening passed CPU-only n001 Slurm Job 4694 at clean commit `46aef57`:
> the exact model path, 14 required-file hashes, and full accepted checkpoint
> fingerprint are now validated before credentials and bound into future
> authorization schema version 2. This admits only requesting a fresh one-run
> authorization. The schema-v2 attempt was subsequently submitted once as Job
> 4695 and passed the polyBERT boundary, but failed before PPO because the clean
> implementation worktree lacked the ignored reconstructed AFP evaluator
> assets. Credentials configuration and local polyBERT were loaded, but no
> DeepSeek transmission, PPO, Oracle, or pairwise update occurred. The consumed
> authorization was not reused. The runner now explicitly routes and hash-binds
> all 13 evaluator assets, and n001 Slurm Job 4697 passed a no-credential
> complete-runtime preflight at clean commit `39cd7fb`. A fresh schema-3
> authorization was then bound to that implementation, both accepted asset
> fingerprints, and one unique output directory, and was consumed by exactly
> one n001 submission: Job 4713. The full runner completed one PPO iteration,
> two valid DeepSeek acquisitions, and all selected-only `K=2` verification,
> but all eight selected candidates had a zero or sign-inconsistent replicate
> delta. Zero pairs were accepted, pairwise refinement was skipped, and the
> authoritative decision was `no_go_multi_iteration_protocol_freeze`. No
> automatic rerun was submitted; multi-iteration scope remains closed. See
> `reproduction/PROJECT_PLAN.md`,
> `reproduction/scicf/online/ARCHITECTURE_SMOKE_V1.md`, and
> `reproduction/scicf/online/ACQUISITION_VERIFIER_DEV_V1.md`, and
> `reproduction/scicf/online/PAIRWISE_STABILITY_DEV_V1.md`, and
> `reproduction/scicf/online/SINGLE_ITERATION_INTEGRATION_V1.md`, and
> `reproduction/scicf/online/SCHEMA_ROBUSTNESS_DEV_V1.md`, and
> `reproduction/scicf/online/SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL.md`, and
> `reproduction/results/scicf-single-iteration-integration-v2-real-20260907/`,
> and
> `reproduction/results/scicf-single-iteration-integration-v2-model-asset-preflight-20260907/`,
> and
> `reproduction/results/scicf-single-iteration-integration-v2-real-modelbound-20260907/`,
> and
> `reproduction/results/scicf-integration-v2-evaluator-asset-preflight-20260907/`,
> and
> `reproduction/results/scicf-single-iteration-integration-v2-real-schema3-job4713-20260907/`.
> A follow-up K=5 soft-pair and optional-LLM resilience coordinate then passed
> its mock-only component preflight in n001 Slurm Job 4714 at clean commit
> `7d005b0`. Future candidate evidence receives a continuous Oracle-derived
> weight rather than an all-or-nothing K=2 label. Standard PPO must be durably
> checkpointed before the optional LLM stage; bounded provider/schema failures,
> abstention, circuit cooldown, and insufficient pair mass become explicit
> `ppo_only_degraded` iterations instead of terminating training. Integrity
> errors remain fatal and silent fallback under the SciCF label is forbidden.
> This component pass authorizes no real v3 execution or multi-iteration
> training. See
> `reproduction/scicf/online/SOFT_PAIR_RESILIENCE_DEV_V1.md` and
> `reproduction/results/scicf-soft-pair-resilience-preflight-20260907/`.
> The v3 runner was then implemented at clean commit `93501f3`. n001 Slurm Job
> 4715 passed 25 server tests and all 31 no-credential full-runtime checks. It
> verified that the primary PPO checkpoint precedes credentials/LLM, every
> bounded LLM availability failure remains an explicit non-blocking degraded
> path, and integrity failures remain fatal. The job made zero API requests,
> ran zero PPO iterations and zero AFP Oracle calls, and loaded no local LLM.
> Its decision is only
> `go_request_separate_real_single_iteration_v3_authorization`; no real v3 run
> or multi-iteration training is authorized. See
> `reproduction/scicf/SINGLE_ITERATION_INTEGRATION_V3_PROTOCOL.md` and
> `reproduction/results/scicf-single-iteration-v3-preflight-20260907/`.
> The new schema-4 authorization was then consumed exactly once by n001 Slurm
> Job 4736. The job passed 69 tests, committed one 128-transition PPO update
> before LLM access, validated two DeepSeek decisions, completed selected-only
> `K=5` verification for eight candidates, retained four non-zero soft weights,
> and applied one KL-bounded auxiliary step. The authoritative result is
> `primary_ppo_committed_auxiliary_applied` with no integrity failures. This is
> single-iteration engineering evidence only: the authorization cannot be
> reused, and no automatic rerun, multi-iteration training, effectiveness
> claim, or scientific claim is authorized. See
> `reproduction/results/scicf-single-iteration-v3-real-schema4-job4736-20260907/`.
> A six-iteration maximum short-horizon controller was next implemented at clean
> commit `5f2a6a1`. It persists the circuit-breaker, cumulative LLM wall time,
> engine state, prior control hash, and chained history digest after every
> iteration. LLM work is capped at 60 seconds per iteration and 180 seconds
> total, and every provider timeout is reduced to the remaining shared deadline.
> Tiny negative KL diagnostics in `[-1e-6, 0)` are clamped to zero; materially
> negative KL remains fatal. n001 Slurm Job 4737 passed 29 tests and all 30
> no-credential full-runtime checks. Its synthetic sequence proved the expected
> two-failure/three-skip/iteration-6-retry circuit behavior and cumulative time
> restoration. It made zero API requests and ran zero PPO or AFP Oracle calls.
> The decision only permits requesting a separately bound schema-5 authorization;
> it does not authorize a real short-horizon run, automatic resume/rerun, formal
> training, effectiveness claims, or scientific claims. See
> `reproduction/scicf/SHORT_HORIZON_MULTI_ITERATION_V1.md` and
> `reproduction/results/scicf-short-horizon-preflight-20260907/`.
> The exact schema-5 authorization was subsequently consumed by one n001 Slurm
> submission, Job 4748. Sixty-seven tests and all six iterations passed at clean
> implementation commit `5f2a6a1`. The run completed 768 PPO transitions, 12
> DeepSeek pool decisions, selected and verified 47 candidates with `K=5`,
> retained 20 non-zero empirical weights, and applied six one-step auxiliary
> updates. One invalid response schema in iteration 3 was durably retained and
> repaired within the single-repair bound. LLM time was 22.689 of 180 seconds,
> evaluator use was 299 of 1,248 requested calls, and maximum auxiliary KL was
> `3.72e-7`. All six control checkpoint hashes and previous-hash links matched.
> No live provider failure occurred, so the circuit did not open. The result is
> a short-horizon engineering pass, not evidence that SciCF improves PPO. It
> authorizes no rerun, longer/formal training, matched comparison, effectiveness
> claim, or scientific claim. See
> `reproduction/results/scicf-short-horizon-real-schema5-20260907/`.

The separately authorized PPO-only control completed in n001 Job 4751 after
38 tests and the full-runtime preflight in Job 4750. It matched the initial and
first-post-PPO policy hashes, first rollout and GAE exactly. At seed 20260907,
both methods used six iterations and 768 transitions: SciCF yielded 158
evaluated terminals and PPO-only 146, with 299 versus 146 evaluator requests.
The positive aggregate depended on the last iteration. This one-seed exploratory
comparison does not establish reward or final-policy improvement. See
`reproduction/results/ppo-only-matched-short-horizon-20260908/README.md`.

SciCF-PPO is being added as a feature-gated extension of the frozen
`dapigen-ppo-compat-baseline-v1` baseline. The upstream PPO clipped objective,
DAPiGen observation/action spaces, and scientific reward implementation remain
unchanged for genuine on-policy rollout samples.

## Scientific invariant

1. The LLM may rank enumerated scientific counterfactual interventions.
2. The LLM is not a reward model, critic, value estimator, or trusted labeler.
3. Factual and counterfactual branches start from the same restored state and
   use the same frozen continuation-policy version and matched randomness.
4. Only scientific-oracle-verified outcomes may create pairwise supervision.
5. Counterfactual actions never enter PPO clipping as behavior-policy samples.
6. Atomic scientific-object scoring operations are counted independently from
   environment transitions.

## Development gates

Offline Gate 1 is implemented and has been run against early, middle, and late
checkpoints. It includes typed records, a domain-adapter contract, DAPiGen
snapshot/replay support, fail-closed oracle accounting, fixed candidate pools,
Random, policy-probability, chemistry-heuristic, and blinded LLM acquisition.

Gate 1 must compare Random, Policy Probability, a chemistry heuristic, and LLM
ranking on identical fixed pools and matched oracle budgets at early, middle,
and late PPO checkpoints. HitRate@B, BestGain@B, Regret@B, and NDCG@B must be
reported with paired/bootstrap uncertainty. The valid 2026-08-30 run failed the
pre-declared rule, so pairwise PPO refinement remains blocked by configuration.
See `experiments/gates/gate1.md` and
`reproduction/results/scicf-gate1-20260830/decision-summary.json`.

Future LLM acquisition runs use the private OpenAI-compatible HTTP client in
`reproduction/scicf/llm/rank_requests_api.py`. This is a separate experiment
path with a private credential file, immutable API model/deployment identity,
pre-request run intent, response cache, token accounting, and CPU-only Slurm
runner. See `reproduction/scicf/llm/API.md`. The original local-Qwen runner is
retained only to reproduce the archived 2026-08-30 decision.

## Source boundary

The algorithm definition was supplied in `SCICF_OPENSPEC_COMBINED.md`. Its
normative scientific requirements were used as the design contract. Embedded
installation, copying, and OpenSpec CLI commands were not executed and are not
treated as user authorization. The exact source hashes are recorded in
`specification-source.json` and the SciCF configuration.

All code execution, tests, runtime-dependent validation, and experiments for
this project are run on the server through Slurm. Local work is restricted to
source inspection, editing, provenance, and version control.
