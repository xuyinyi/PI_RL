# SciCF-PPO development boundary

> **Architecture-first amendment (updated 2026-09-07):** Historical SciCF evidence and
> no-go decisions remain unchanged. The online architecture smoke and the
> acquisition/verifier and pairwise-stability development gates passed under
> `reproduction/scicf/online/`, but the subsequent single-iteration
> integration smoke returned a no-go after a DeepSeek response invented an
> out-of-pool candidate ID. The subsequent mock-only response-schema robustness
> development gate passed.
> These are engineering and development evidence only; they do not reopen the
> sealed test or authorize performance claims. Only a separate integration-v2
> protocol freeze is admitted; no real API/PPO rerun or multi-iteration scope is
> authorized. See `reproduction/PROJECT_PLAN.md`,
> `reproduction/scicf/online/ARCHITECTURE_SMOKE_V1.md`, and
> `reproduction/scicf/online/ACQUISITION_VERIFIER_DEV_V1.md`, and
> `reproduction/scicf/online/PAIRWISE_STABILITY_DEV_V1.md`, and
> `reproduction/scicf/online/SINGLE_ITERATION_INTEGRATION_V1.md`, and
> `reproduction/scicf/online/SCHEMA_ROBUSTNESS_DEV_V1.md`.

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
