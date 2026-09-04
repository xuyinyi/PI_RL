# Active project plan: terminal-evaluable counterfactual credit for DAPiGen

Status date: 2026-09-04

Plan status: **active planning baseline; implementation and scientific gates remain closed unless explicitly opened below**

## 1. Authority and scope

This document is the authoritative forward plan for the DAPiGen reinforcement-learning project. The active route is:

```text
Stage 0 environment acceptance
        -> one PPO engine and one evaluator/budget contract
        -> PPO / Policy-CC / MCC-PPO
        -> matched-budget pilot
        -> preregistered multi-seed comparison
        -> independent scientific validation
```

The earlier SciCF acquisition programme is retained as immutable negative and engineering evidence under `reproduction/scicf/`, `experiments/gates/`, and `reproduction/results/`. Its failed gates, sealed-test decisions, and claim boundaries are not overwritten. It is no longer the active implementation roadmap. MCC-PPO is a separately gated method hypothesis and must not be presented as a continuation that passed the earlier SciCF gates.

Reference files returned by the planning conversation are recorded in `reference-intake-20260904.md`. They are design inputs, not accepted production code or experimental evidence.

## 2. Scientific problem

DAPiGen receives a trustworthy task reward only after a complete polyimide has been generated. Partial dianhydride, diamine, and intermediate structures do not have reliable physical-property labels. The project therefore asks whether terminal outcomes can be converted into better step-level policy credit without inventing intermediate physical rewards.

The active research questions are:

1. Does policy-conditioned counterfactual completion improve sample efficiency over standard PPO under the same requested terminal-evaluator budget?
2. Does an LLM-guided, mechanism-matched proposal improve on policy-only counterfactual completion under the same task, PPO engine, and evaluator budget?
3. Are any gains stable across seeds, held-out structural regions, evaluator perturbations, and independent higher-fidelity validation?

The two primary matched method contrasts are:

```text
PPO       vs Policy-CC : incremental value of counterfactual credit
Policy-CC vs MCC-PPO   : incremental value of mechanism-matched LLM guidance
```

PPO versus MCC-PPO is useful as an end-to-end comparison but cannot by itself identify the LLM contribution.

## 3. Non-negotiable scientific invariants

All methods and all formal runs must satisfy the following invariants.

1. Only complete PI molecules may enter the terminal evaluator.
2. The LLM may propose a mechanism and a legal counterfactual-action distribution; it is not a reward model, critic, value estimator, or trusted labeler.
3. Factual and counterfactual branches start from the same immutable state and use the same frozen continuation-policy version and matched random streams.
4. Chemistry transitions and product selection are independent of predicted reward.
5. PPO, Policy-CC, and MCC-PPO share one state/action definition, one PPO implementation, one terminal evaluator, one action-mask implementation, and one budget service.
6. The actor may use counterfactual credit. The critic continues to use returns derived from the real environment reward.
7. Requested terminal-evaluator calls are the primary optimization budget. Unique calls, cache hits, environment transitions, wall time, and compute are reported separately.
8. Newly queried outcomes cannot affect the same iteration's query selection, control variate, confidence gate, or actor update through model retraining.
9. Every formal result is bound to code, environment, data/model assets, task contract, algorithm configuration, seeds, and output hashes.
10. Engineering completion, model readiness, optimization performance, independent validation, and scientific claims remain separate states.

## 4. Methods under comparison

### 4.1 PPO

- Standard on-policy PPO on the accepted Stage 0 environment.
- Actor advantage: environment-return GAE only.
- No counterfactual branches or auxiliary completion-value model in the decision path.

### 4.2 Policy-CC

- Uses paired factual and policy-proposed counterfactual completions.
- Counterfactual proposal is the frozen policy, equivalent to `eta = 0` in the support-preserving mixture.
- Tests counterfactual credit without an LLM mechanism prior.

### 4.3 MCC-PPO

- The LLM compiles scientific mechanism constraints into a legal action proposal.
- The proposal preserves policy support:

  ```text
  q_eta(b | s, a) = (1 - eta) pi_old(b | s) + eta q_mech(b | s, a)
  ```

- Paired terminal completions, the completion-value ensemble, doubly robust residual correction, randomized active querying, selection adjustment, and conservative GAE fusion produce actor credit.
- The terminal reward and critic target are unchanged.

The initial reference value `eta = 0.25` is a hypothesis to validate, not a frozen formal setting.

## 5. Stage gates

### Gate P0: preserve the historical coordinate

Required evidence:

- Keep upstream commit `5f692946cbe0d15eede882dfe4cff7fb26eb7d8c` and the reconstructed AFP compatibility baseline immutable.
- Preserve all SciCF no-go artifacts and sealed-test boundaries.
- Record every newly received reference artifact by filename and SHA-256.

Exit criterion: the new route can be developed without rewriting or relabelling historical evidence.

Current status: **passed for planning; continuing provenance checks required**.

### Gate P1: Stage 0 environment acceptance

Required evidence on a newly resolved governed Slurm checkout:

- DAPiGen custom BRICS is used; an RDKit fallback is not acceptable for formal acceptance.
- Persistent evaluator matches the frozen legacy evaluator on the declared sample with `mismatch_count = 0`.
- Persistent polyBERT matches the frozen encoder within declared tolerances.
- State/action/seed replay and snapshot restoration pass on real chemistry cases.
- Reward-independent terminal-product selection and explicit NOOP semantics pass regression tests.
- Shared Ray evaluator produces one global transactional requested/unique/cache ledger.
- The formal Stage 0 configuration is fixed before baseline training.
- `legacy_effective` regression quantifies changes caused by task-semantic repair separately from algorithm changes.
- Environment, budget, action-catalog, model-asset, and source hashes are emitted in a contract manifest.

Stop condition: any parity, custom-chemistry, replay, ledger, or contract mismatch keeps all algorithm-training gates closed.

Current status: **not passed, with mask-refinement development acceptance
complete**. Stage-0 v2.3 passed 44 tests, custom BRICS, polyBERT parity,
100-PI evaluator parity, 1,000-transition replay/restore and Ray evaluator
checkpoint accounting on n001. On the same 100-state sequence used for v2.2,
`closure_exact_cached` removed all 3,742/11,878 label-mask false positives with
zero false negatives and was 129.5x faster than the independent full-exact
reference. The only active P1 blocker is now regeneration from a clean immutable
Git coordinate. No PPO was launched. The separate `ssh cpu` alias is a different
host and is not n001 evidence.

### Gate P2: unified algorithm implementation

Required implementation:

- Stage 0 `DAPiGenState`, `DAPiGenAction`, masks, transition, and evaluator interfaces are the only environment source of truth.
- Remove or quarantine the earlier MCC reference adapter and its duplicate `PIState` / `PIAction` definitions.
- PPO, Policy-CC, and MCC-PPO use one native PyTorch `PPOEngine`; RLlib remains only for frozen original-baseline reproduction.
- Use one supported Python/Gymnasium/PyTorch environment for all three new-method variants.
- Source-tag every evaluator request as on-policy, factual completion, counterfactual completion, or evaluation.
- Enforce remaining requested-call budget before rollout and before counterfactual query sampling.
- Buffer new paired labels and commit them only after the current actor update.
- Freeze and record `pi_old` for rollout, proposal probabilities, factual completions, counterfactual completions, and model policy-value sampling.

Required tests:

- Identical-task and identical-PPO-path tests across all three methods.
- No-intervention and `eta = 0` equivalence tests.
- Identity intervention gives zero paired difference under matched replay.
- No same-iteration target leakage.
- Exact requested/unique/cache/source accounting under concurrency and budget exhaustion.
- Checkpoint/resume preserves policy, optimizer, replay, evaluator ledger, RNG, and contract identity.

Stop condition: if methods differ outside the declared credit-provider seam, comparative training is not authorized.

Current status: **not started**. The received MCC files are a reference implementation and are not integrated with Stage 0.

### Gate P3: estimator and active-query validation

Required evidence before chemical pilot training:

- Monte Carlo recovery of the frozen-policy advantage under a deliberately misspecified completion model.
- Correct proposal probabilities for factorized masked joint actions.
- Verified importance-weight bound for the support-preserving mixture.
- Randomized inclusion probabilities are positive, calibrated, and reproduce the target expected query budget.
- Selection-adjusted estimates recover the target expectation without post-query outcome-dependent gating.
- Shared-randomness pairing reduces variance without changing the estimand.
- Completion-value uncertainty is calibrated enough for its declared use in active querying and fusion.
- Sensitivity analysis covers `eta`, query budget, `M`, `K`, policy-value samples, policy lag, and maximum fusion weight.

Stop condition: estimator correctness failure blocks the use of counterfactual credit in PPO even if toy training appears to improve.

Current status: **partial reference evidence only**. The supplied validation reports ten toy tests, not acceptance of the integrated implementation.

### Gate P4: frozen Stage 0 PPO baseline

Required evidence:

- Accepted P1/P2 contract and clean immutable run coordinate.
- Multiple declared seeds and matched evaluator checkpoints.
- Learning curves versus requested evaluator calls and environment transitions.
- Final validity, objective, uniqueness, novelty, diversity, Frag, and SNN metrics under a common evaluation protocol.
- Failure-mode and molecule-duplication audits.

This baseline is distinct from both the original reconstructed compatibility run and paper-result reproduction.

Current status: **not started**.

### Gate P5: matched-budget pilot

Run PPO, Policy-CC, and MCC-PPO with the same frozen P1/P2 contract, small predeclared budgets, and paired seeds.

Pilot questions:

- Do all methods execute without budget, replay, or mask violations?
- Does Policy-CC produce finite and stable credit beyond GAE?
- Does MCC-PPO change proposal quality relative to Policy-CC without collapsing support?
- Are gains visible against evaluator calls rather than only iterations or wall time?
- Are effect directions consistent enough to justify a formal experiment?

Pilot outputs are diagnostic and cannot support the main scientific claim.

Stop condition: material instability, contract drift, ineffective proposal, or no plausible information-efficiency signal triggers diagnosis or method revision before formal runs.

Current status: **closed pending P1-P4**.

### Gate P6: preregistered formal comparison

Freeze before execution:

- primary and secondary endpoints;
- requested-evaluator budgets and reporting checkpoints;
- seeds, paired-seed mapping, and stopping rules;
- hyperparameters and any tuning allocation;
- molecule deduplication and invalid-result handling;
- statistical models, confidence intervals, and multiplicity handling;
- success, failure, and claim-promotion rules.

Recommended primary endpoint:

- area under the best-so-far terminal objective curve versus requested terminal-evaluator calls, or another single budget-indexed endpoint selected and frozen before looking at formal outcomes.

Required secondary reporting:

- terminal objective distribution and success rate;
- validity, uniqueness, novelty, diversity, Frag, and SNN;
- requested and unique evaluator calls, cache hits, transitions, wall time, CPU/GPU/API usage;
- credit variance, effective sample size, inclusion probabilities, importance weights, and fusion weights;
- paired uncertainty across seeds, not molecule-level pseudoreplication.

Current status: **closed pending a successful P5**.

### Gate P7: independent scientific validation

Progressively evaluate a frozen shortlist using:

1. alternative or held-out QSPR models and domain-of-applicability checks;
2. higher-fidelity computational validation such as MD/DFT where scientifically appropriate;
3. expert chemical review, synthesizability assessment, and, if available, experimental validation.

QSPR-optimized performance alone supports an optimization result relative to that evaluator. It does not establish physical-property improvement or experimental validity.

Current status: **not started**.

## 6. LLM and private API boundary

- Formal LLM mechanism compilation will use the user's private OpenAI-compatible API, not a local model on n001.
- Credentials remain in a server-side file with mode `0600`; keys never enter chat, Git, command arguments, manifests, or logs.
- Provider, endpoint class, model/deployment identity, immutable revision where available, prompt/schema versions, response hashes, token accounting, retries, and cache identity must be recorded without recording secrets.
- The earlier SciCF ranking prompt and negative results are historical evidence. Reuse of API transport code does not make MCC-PPO the same experiment.
- No external API call is authorized by this planning document alone.

## 7. Execution and repository policy

- Author and review code locally.
- Run dependency-sensitive tests, environment acceptance, pilots, and formal experiments on the governed compute coordinate through Slurm.
- Resolve and record the live hostname, checkout, environment and asset paths before execution; historical n001 paths are not current evidence.
- Keep upstream source, active worktree, governed model/data assets, and run outputs separate.
- Do not copy the downloaded reference packages into the repository until intake review and an explicit integration patch are prepared.
- Preserve the frozen original PPO and SciCF histories. New results use new directories, configuration identities, and manifests.

## 8. Current state at plan activation

| Component | State on 2026-09-04 | Claim boundary |
|---|---|---|
| Reconstructed AFP PPO compatibility baseline | complete | Engineering/runtime baseline; not paper-result reproduction |
| Historical SciCF gates | archived no-go | No SciCF performance, pairwise-refinement, or PPO-integration claim |
| Stage 0 v2.3 implementation candidate | mask-refinement development acceptance passed on n001 | Same-state mask and replay checks passed; clean formal coordinate remains open |
| MCC-PPO reference files | received and hash-identified | Toy/reference validation only; not a unified runnable project |
| Unified Stage 0 + PPO engine | not started | No implementation claim |
| PPO / Policy-CC / MCC-PPO pilot | closed | No performance claim |
| Formal comparison | closed | No method claim |
| Independent validation | not started | No physical-property or experimental claim |

## 9. Immediate work order

1. Preserve the completed reference-artifact intake and Stage-0 v2.2/v2.3 development evidence.
2. Freeze a clean immutable Git coordinate and rerun P1 manifests and acceptance on n001.
3. Profile accepted-mask throughput under the planned multi-worker P2 engine before freezing training budgets.
4. Design the single PPOEngine/CreditEstimator seam and its test matrix.
5. Integrate PPO first, then Policy-CC (`eta = 0`), then MCC-PPO.
6. Complete P3 estimator validation and P4 baseline evidence.
7. Draft and freeze the P5 pilot protocol before running it.

The next planned activity is **clean-coordinate Stage-0 P1 acceptance**.
No PPO, external API experiment, formal experiment, or scientific claim
promotion is opened by the current development evidence.
