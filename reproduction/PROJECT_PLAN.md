# Active project plan: terminal-evaluable counterfactual credit for DAPiGen

Status date: 2026-09-08

Plan status: **the schema-3 integration-v2 run remains archived as a no-go after
zero strict pairs; the K=5 optional-LLM v3 single-iteration run completed in Job
4736; the separately authorized six-iteration short-horizon engineering run
completed in n001 Slurm Job 4748. It is not a matched comparison and does not
authorize a rerun, longer or formal training, sealed-test access, an
algorithm-effectiveness claim, or a scientific claim**

## 2026-09-06 architecture-first amendment

The user selected a bounded online-engineering route: first make the complete
LLM-SciCF architecture executable, then harden and evaluate each module. This
changes the immediate engineering order without converting any earlier no-go
into a go decision.

The first implementation coordinate is
`dapigen-scicf-online-architecture-smoke-v1`. It runs one standard PPO update,
one blinded DeepSeek acquisition request over a 24-candidate cross-timestep
pool, selected-only matched factual/counterfactual Oracle verification, and at
most one KL-protected component-local pairwise update. It is an engineering
smoke only: no baseline is overwritten, no sealed test is opened, and no
algorithm-effectiveness or scientific claim may be made from it.

After this smoke, the module order is: acquisition context/schema, candidate
coverage, matched verifier and budget accounting, pairwise stability,
multi-iteration PPO integration, matched Random/heuristic controls, and only
then a frozen multi-seed experiment. The earlier Gate 1B.3 result remains a
failed, non-blocking descriptor diagnostic.

The first coordinate passed in n001 Slurm Job 4684 at clean commit
`d05a05c556958a798c3f4eab217eed587ed0309b`. It exercised the complete path
with one `K=1` request/verification/refinement cycle. This closes only the
architecture-reachability smoke; it does not admit multi-iteration training.

The second coordinate,
`dapigen-scicf-acquisition-verifier-dev-v1`, passed in n001 Slurm Job 4685 at
clean commit `d58b3f9d2d5eaa6060e39c93d3149f84e559120e`. Four outcome-blind
trajectory pools compared DeepSeek, Random, frozen-policy probability, and a
Morgan-distance chemistry heuristic on identical 24-candidate pools at maximum
`B=4`. All 96 candidates were evaluated with matched `K=2` continuations for
development-only scoring; the policy was not updated. The verifier yielded 25
sign-consistent pairs (13 positive, 12 negative). DeepSeek exceeded Random on
the pre-declared mean NDCG and effective-BestGain checks and had three of four
paired NDCG non-losses, so pairwise-stability development is admitted. This is
not a claim that DeepSeek is best: its mean NDCG (0.2777) remained below the
chemistry heuristic (0.3423), and four pools cannot establish effectiveness or
generality. Evidence is under
`reproduction/results/scicf-acquisition-verifier-dev-20260906/`.

The third coordinate, `dapigen-scicf-pairwise-stability-dev-v1`, passed in n001
Slurm Job 4686 at clean commit
`06ec14f3a6c136cdb9895ff31bcc1b475acc3b61`. It hash-reconstructed the same
rollout, 96 candidate IDs, initial policy, and 25 accepted `K=2` pairs from Job
4685, then ran six one-step update scenarios twice each from the same restored
initial state. All six replica groups were exact, every training subset improved
its signed margin, and all five held-out folds had positive mean-margin change.
Maximum joint KL across all 96 candidate states was `4.94e-7`; the value-head
parameters were unchanged, while shared-trunk critic predictions moved by at
most `0.00412`, below the frozen `0.01` bound. No LLM or fresh counterfactual
label call occurred and no PPO optimizer step was taken. This admits only a
separately frozen single-iteration integration smoke. Full-corpus pairwise
accuracy remained below `0.5` after one step (`0.48`), so no learning-quality
or effectiveness claim follows. Evidence is under
`reproduction/results/scicf-pairwise-stability-dev-20260906/`.

The fourth coordinate,
`dapigen-scicf-single-iteration-integration-smoke-v1`, did not pass. Slurm Job
4687 at commit `64b086142a6a971b80c8de3675b1d200e6b7413d` completed the full PPO,
two-request acquisition, `K=2` verification, pairwise update, and checkpoint
path, but an incorrect exact-evaluator-call assertion misclassified valid
pre-QSPR structural terminations as missing ledger calls. The assertion was
corrected at clean commit `b61d9e5c81dbef5108a2d796e0acd7ecf3a7640b`
without changing the experiment seed, candidate construction, model, or gate
thresholds. The corrected Slurm Job 4688 then failed closed before Oracle
verification because DeepSeek invented the out-of-pool ID `cf-281`. The result
is a no-go; neither an automatic rerun nor the short-horizon multi-iteration
smoke is admitted. Any further work requires a separately frozen API response-
schema robustness gate with invalid-response retention. Evidence is under
`reproduction/results/scicf-single-iteration-integration-20260906/`.

The fifth coordinate, `dapigen-scicf-llm-schema-robustness-dev-v1`, passed in
n001 Slurm Job 4689 at clean commit
`5387611e1ba088b8dafd0c50a80751059e8fdfed`. Nineteen server tests and five
mock-only scenarios verified pre-call request receipts, raw-response persistence
before parsing, one-repair/two-attempt bounds, exact candidate allowlist and
budget preservation, valid partial selection and abstention, and empty
fail-closed exhaustion without fallback or cache substitution. The run used
nine scripted in-memory completions and zero external API calls, credential
loads, PPO steps, Oracle calls, local models, or sealed-test accesses. Passing
admits only a separately frozen integration-v2 protocol design; it does not
authorize a DeepSeek call, integration rerun, or multi-iteration training.
Evidence is under
`reproduction/results/scicf-schema-robustness-dev-20260907/`.

The sixth coordinate began as the unexecuted protocol
`dapigen-scicf-single-iteration-integration-smoke-v2`. It freezes v1's seed,
PPO, candidate construction, two 24-candidate pools, `B=4`, selected-only `K=2`
verification, pairwise update, and thresholds. The only permitted implementation
delta is to route both DeepSeek decisions through the accepted bounded response
guard and to emit a terminal fail-closed report if either guard is exhausted.
The protocol distinguishes two pool decisions, at most four schema attempts,
and at most eight HTTP transmissions. This protocol freeze authorizes no
real execution, credential load, DeepSeek request, PPO, Oracle verification,
rerun, or multi-iteration training.

The authorization-gated v2 runner was subsequently implemented at clean commit
`05c92e587b9e50c1f23b83f271c95243724da33c`. CPU-only n001 Slurm Job 4692
passed 25 tests and all 19 mock-preflight checks after a transport-accounting
correction. The accepted preflight used six in-memory completions and zero
external API requests, credential loads, PPO steps, Oracle calls, local models,
or sealed-test accesses. Its decision admits only requesting a separate,
single-run real-execution authorization; it does not provide that authorization
or open multi-iteration training. See
`reproduction/scicf/online/SINGLE_ITERATION_INTEGRATION_V2_PROTOCOL.md` and
`reproduction/results/scicf-single-iteration-integration-v2-preflight-20260907/`.

The separately authorized real-run attempt was submitted exactly once as n001
Slurm Job 4693. Its 52-test step passed, but the runner failed before PPO while
loading a submitted `QSPR/polyBERT` directory that contains source code rather
than the required Hugging Face model files. The credential configuration was
parsed, but no external API transmission, PPO iteration, candidate-pool build,
Oracle evaluation, or pairwise update occurred. The one-run authorization is
consumed and no automatic rerun was made. A future attempt requires a new
model-asset completeness and hash-binding implementation, a new n001 Slurm
mock/preflight, and a fresh exact one-run authorization. Evidence is under
`reproduction/results/scicf-single-iteration-integration-v2-real-20260907/`.

The model-asset preflight repair was implemented at clean commit
`46aef57c693a4644a52bdfcd337cbc90184ac31e`. The runner now verifies the exact
resolved path, 14 required-file hashes, and the accepted full checkpoint
fingerprint `6bdd24f951dd90d3031e749ef0130752811bfefe6c850af82b805cf015ea195f`
before credential loading. Future authorization schema version 2 binds that
path, asset-binding hash, and fingerprint in addition to the existing protocol,
implementation, output, run-count, and operation scope. CPU-only n001 Slurm Job
4694 passed 29 tests and all 23 mock/preflight checks with zero credential,
external API, PPO, Oracle, local-model, or sealed-test access. Its decision is
only `go_request_separate_real_single_iteration_execution_authorization`; no
new authorization or real run was created. Evidence is under
`reproduction/results/scicf-single-iteration-integration-v2-model-asset-preflight-20260907/`.

The resulting schema-v2 real attempt was submitted exactly once as n001 Slurm
Job 4695 at implementation commit `46aef57`. Fifty-four tests and the full
polyBERT path, 14-file, fingerprint, accepted-encoder, and authorization checks
passed. Runtime construction then failed before PPO because the clean
implementation worktree did not contain the ignored reconstructed AFP
compatibility weights, scalers, and settings expected under
`RL_PPO/GNN/model`. The credentials configuration and local polyBERT were
loaded, but no DeepSeek transmission, PPO iteration, Oracle evaluation, or
pairwise update occurred. The authorization is consumed and no automatic rerun
was submitted. A further attempt requires a separately authorized
evaluator-asset routing/binding repair, a no-credential n001 Slurm preflight
that constructs the complete Stage0 runtime, and only then a fresh one-run
authorization. Evidence is under
`reproduction/results/scicf-single-iteration-integration-v2-real-modelbound-20260907/`.

The evaluator-asset repair and exact inference-accounting report were frozen at
clean commit `39cd7fb33c1c88397e78808860d766c9876c12df`. Authorization schema
version 3 now
binds the exact AFP evaluator directory, binding hash, and canonical fingerprint
over four weights, four scalers, four settings files, and `fpscores.pkl.gz`.
Only `RL_PPO/envs/evaluator.py` differs from the accepted Stage0 source: it adds
an optional explicit model-directory route while preserving the legacy default.
n001 Slurm Job 4697 passed 31 server tests and all 33 preflight checks, loaded
polyBERT plus the four AFP models, and constructed the complete Stage0 and PPO
runtime with an empty evaluator ledger. The preflight accepted no credential
file and executed zero API requests, PPO iterations, Oracle evaluations, or AFP
property predictions. Stage0 construction performed exactly two local polyBERT
initial-embedding cache misses. This admits only requesting a fresh exact
one-run execution authorization; no authorization or real run was created.
Evidence is under
`reproduction/results/scicf-integration-v2-evaluator-asset-preflight-20260907/`.

The fresh schema-3 execution authorization bound implementation `39cd7fb`, the
accepted polyBERT and AFP fingerprints, and one previously absent output
directory. It was consumed by exactly one n001 Slurm submission, Job 4713.
Fifty-six server tests passed and the job completed one 128-transition PPO
iteration, two valid DeepSeek decisions over separate 24-candidate pools,
selected-only `K=2` verification of eight candidates, and a terminal
checkpoint. Both API responses validated on their first attempt, using two HTTP
transmissions total. Verification completed all 32 branches and used 12 AFP
terminal evaluations, but every selected candidate's two deltas contained a
zero or changed sign. The accepted pair count was therefore zero and the
pairwise optimizer correctly took zero steps. The authoritative decision is
`no_go_multi_iteration_protocol_freeze`, even though Slurm and the bounded
runner completed normally with no integrity failures. No automatic rerun was
submitted and no next scope is authorized. Evidence is under
`reproduction/results/scicf-single-iteration-integration-v2-real-schema3-job4713-20260907/`.

The follow-up component coordinate,
`dapigen-scicf-soft-pair-resilience-dev-v1`, was implemented at clean commit
`7d005b07df0cb3022ead92efedd13d3aec04fc3c`. It replaces the future strict
K=2 auxiliary-label rule with exactly five matched continuations and a
continuous weight based on Jeffreys-shrunk sign support, non-tie fraction, and
median effect magnitude. Standard PPO is defined as the primary transaction
and must be checkpointed before the optional LLM stage. Bounded provider/schema
failures, valid abstention, circuit-open cooldown, and insufficient effective
pair mass yield an explicit `ppo_only_degraded` iteration and cannot roll back
PPO. Integrity errors remain fatal, and silent fallback under the SciCF label is
forbidden. CPU-only n001 Slurm Job 4714 passed 18 server tests and all 20
mock checks with zero credentials, external API requests, PPO iterations,
Oracle calls, local models, or sealed-test accesses. This admits only separately
gated v3 runner development and a later no-credential full-runtime preflight;
it does not authorize a real run or multi-iteration training. Evidence is under
`reproduction/results/scicf-soft-pair-resilience-preflight-20260907/`.

The separately frozen v3 integration runner was implemented at clean commit
`93501f33cf4594791b31a805b153bf1accc94538`. It validates schema-4 execution
authorization and all frozen assets before PPO, commits and hashes the primary
PPO checkpoint before reading private API settings, and then treats the LLM and
K=5 soft-pair refinement as an optional auxiliary transaction. Expected
provider/transport/schema failures, valid abstention, insufficient pair mass,
and pairwise KL rollback retain the primary PPO result as an explicit
`ppo_only_degraded` iteration. Authorization, asset, budget, leakage,
allowlist, and unexpected programming errors remain fatal. n001 Slurm Job 4715
passed 25 server tests and all 31 no-credential full-runtime checks. It loaded
the scientific runtime and executed the expected polyBERT initialization, but
used zero credentials, API requests, PPO iterations, AFP Oracle calls, local
LLMs, or sealed-test accesses. The decision permits only requesting a new exact
single-run schema-4 authorization; it does not authorize that run or any
multi-iteration training. Evidence is under
`reproduction/results/scicf-single-iteration-v3-preflight-20260907/`.

After explicit authorization of the blinded external payload and one real
PPO/AFP execution, schema-4 authorization
`scicf-single-iteration-v3-schema4-one-run-93501f3-20260907` was consumed by
exactly one n001 Slurm submission, Job 4736. Sixty-nine server tests passed and
the job completed one 128-transition PPO iteration, durably wrote the primary
checkpoint before credentials/LLM, validated both DeepSeek pool decisions on
their first attempt, verified eight selected candidates with `K=5`, and applied
one weighted pairwise step from four non-zero empirical weights. The soft gate
had total weight `3.3433`, effective mass `1.6717`, and no failures. The value
head was unchanged, full-support maximum joint KL was `2.70e-7`, and there were
no integrity failures. The authoritative module decision is
`primary_ppo_committed_auxiliary_applied`. This proves only the bounded
single-iteration engineering path. The consumed authorization cannot be reused,
and no automatic rerun or multi-iteration training is authorized. Evidence is
under
`reproduction/results/scicf-single-iteration-v3-real-schema4-job4736-20260907/`.

The bounded short-horizon controller was then implemented at clean commit
`5f2a6a100edcfd0b15fe82228a0eef8e1c2c2904`. It caps the coordinate at six
iterations, persists circuit-breaker state and cumulative LLM wall time in an
exactly hashed control checkpoint after every iteration, caps LLM time at 60
seconds per iteration and 180 seconds total, and clamps only KL diagnostics in
`[-1e-6, 0)` while rejecting materially negative values. Standard PPO remains
the primary transaction and is checkpointed before any credential read; bounded
LLM availability failures yield an explicit `ppo_only_degraded` iteration.
n001 Slurm Job 4737 passed 29 server tests and all 30 no-credential full-runtime
checks. Its synthetic sequence proved two failures, iterations 3-5 skipped by a
persisted open circuit, and retry at iteration 6 while cumulative LLM time
remained 25 seconds across checkpoint reloads. The run used zero credentials,
API requests, PPO iterations, AFP Oracle calls, AFP property inferences, local
LLMs, or sealed-test accesses. Its decision permits only requesting a separate
exact schema-5 authorization; no real short-horizon run, automatic resume/rerun,
or formal training is authorized. Evidence is under
`reproduction/results/scicf-short-horizon-preflight-20260907/`.

The new schema-5 authorization was then bound to implementation `5f2a6a1`, the
short-horizon protocol, both accepted asset fingerprints, iterations 1-6, no
resume point, and one unique output directory. It was consumed by exactly one
n001 Slurm submission, Job 4748. Sixty-seven server tests passed and all six
iterations completed without integrity failures. The run accumulated 768 PPO
transitions, 158 successful terminals, 12 DeepSeek pool decisions, 47 selected
and `K=5` verified candidates, 20 non-zero empirical weights, and six one-step
SciCF auxiliary updates. One schema-invalid response in iteration 3 was retained
before validation and repaired within the frozen bound. Total LLM time was
22.689 seconds, evaluator accounting was 299 requested / 255 unique / 255
backend / 44 cache-hit calls, and the largest corrected auxiliary KL was
`3.72e-7`. Six independently rehashed control checkpoints formed the exact
previous-hash chain. The circuit remained closed because no provider failure
occurred; the cooldown failure path therefore remains preflight evidence rather
than a live observation. The authoritative decision is
`short_horizon_engineering_run_complete_no_further_scope_authorized`. This is
not a PPO comparison or effectiveness result, and the consumed authorization
cannot be reused. Evidence is under
`reproduction/results/scicf-short-horizon-real-schema5-20260907/`.

## 1. Authority and scope

On 2026-09-08, the user authorized freezing a matched PPO-only control and
analysis protocol before requesting execution. The frozen coordinate is
`reproduction/scicf/PPO_ONLY_MATCHED_SHORT_HORIZON_V1.md`, with configuration
`reproduction/scicf/online/configs/ppo_only_matched_short_horizon_v1.json`.
It fixes seed 20260907, six iterations and 768 transitions against Job 4748,
with the same PPO hyperparameters, asset bindings and evaluator ceilings.
Actual auxiliary cost is reported separately. This is a one-seed exploratory
comparison specified after observing SciCF results. Runner implementation,
server preflight and the single real PPO-only submission are the proposed next
execution scope; none has been performed by the protocol-freeze task.

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
- The `legacy_effective` configuration is frozen for the post-P1 baseline
  regression; no PPO regression is part of the environment-acceptance gate.
- Environment, budget, action-catalog, model-asset, and source hashes are emitted in a contract manifest.

Stop condition: any parity, custom-chemistry, replay, ledger, or contract mismatch keeps all algorithm-training gates closed.

Current status: **passed under the reconstructed AFP compatibility contract** at
clean Git commit `373b3291ac04dbf654c134bee2ca61a0c86d1a68`. Slurm jobs 4659,
4660 and 4661 passed 44 tests, custom BRICS, polyBERT parity, 100-PI evaluator
parity, 1,000-transition replay/restore, Ray checkpoint accounting and formal
manifest generation on `yanlih100n1`. On the same 100-state sequence used for
v2.2, `closure_exact_cached` removed all 3,742/11,878 label-mask false positives
with zero false negatives and zero false positives, and was 128.834x faster than
the independent full-exact equivalent. The formal manifest records a clean Git
coordinate. No PPO was launched. Author-original model identity and scientific
validation remain outside P1; the separate `ssh cpu` alias is not n001 evidence.

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

Current status: **in progress**. P2-A accepted-mask throughput profiling passed
at clean profile commit `d06365e4485baddcdad65386c69cc47056bf94be`, bound to
the P1 coordinate without Stage-0 source changes. Slurm Job 4663 passed five
profiler tests and all functional, determinism, specification and throughput
checks. Median throughput scaled from 7.329 masks/s at one worker to 33.807
masks/s at eight workers (4.613x speedup; 0.577 efficiency); the minimum
adjacent ratio was 1.159. This is a mask-only admission using a stub encoder
outside the timed region. The v1 single-engine / credit-estimator interface and
13-item test matrix are now frozen; their contract-validator layer passed and
initially left all six integrated I-series tests pending. The real-observation /
evaluator-ledger characterization also passed at clean commit
`7e3ec9d4436f94af40f6204788ac8dec9dd9466e` in Slurm Job 4665: model/stack
construction took 4.535 s, the first 100-transition pass achieved 6.751
transitions/s, and exact cache replay achieved 216.812 transitions/s. Each pass
produced 31 successful terminals with matching trajectory, observation and
evaluation digests. The first pass recorded 62 requested / 31 unique / 31
backend / 31 cache-hit calls; replay recorded 62 / 0 / 0 / 62. This is a
single-worker pre-training characterization, not a throughput admission. The
common native PPO engine and environment-return GAE provider subsequently passed
at clean commit `6a4bd2c21b657c9f1b9c734028322cf66af6367f` in Slurm Job 4671.
The job passed 26 tests and executed two distinct 64-transition PPO iterations
on the accepted real polyBERT/QSPR stack. Its checkpoint/resume replay matched
all 13 declared identities, while the GAE provider made no evaluator call.
P2-I01, I05 and I06 are accepted for their engine/PPO scope. Policy-CC,
MCC-PPO and I02-I04 remain pending; no production training budget is frozen.

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
- Matched original-compatibility, Stage-0 `legacy_effective` six-step and
  Stage-0 standard five-step baselines to locate task-semantic changes without
  attributing them to the learning algorithm.
- Multiple declared seeds and matched evaluator checkpoints.
- Learning curves versus requested evaluator calls and environment transitions.
- Final validity, objective, uniqueness, novelty, diversity, Frag, and SNN metrics under a common evaluation protocol.
- Failure-mode and molecule-duplication audits.

This baseline is distinct from both the original reconstructed compatibility run and paper-result reproduction.

Current status: **P4-A protocol frozen before results; single-seed preflight
passed in Slurm Job 4675 and the five-seed formal array was launched as Job
4676**. P4-A covers the standard five-step native-PPO arm only. The original
RLlib compatibility evidence and `legacy_effective` six-step semantic control
remain required before the complete P4 gate can close.

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
| Stage 0 v2.3 environment | P1 engineering acceptance passed at clean commit `373b3291` | Reconstructed-asset compatibility scope; not author-original model or scientific validation |
| MCC-PPO reference files | received and hash-identified | Toy/reference validation only; not a unified runnable project |
| Unified Stage 0 + PPO engine | P2-A/P2-C profiles and native PPO/GAE smoke passed; counterfactual estimators pending | PPO engineering smoke only; no production-training or performance claim |
| P4-A native standard PPO baseline | preflight passed; formal Job 4676 running | Preflight engineering evidence only; no completed baseline or performance claim yet |
| PPO / Policy-CC / MCC-PPO pilot | closed | No performance claim |
| Formal comparison | closed | No method claim |
| Independent validation | not started | No physical-property or experimental claim |

## 9. Immediate work order

1. **Complete:** preserve the reference-artifact intake and Stage-0 v2.2/v2.3 development evidence.
2. **Complete:** preserve the clean P1 run coordinate and its synchronized, hash-verified evidence.
3. **Complete for mask scope:** profile accepted-mask throughput with 1, 2, 4 and 8 workers; retain both the failed development diagnostic and the corrected formal run.
4. **Complete for contract scope:** freeze the single `PPOEngine` / `CreditEstimator` seam and its 13-item matrix; validator tests pass, while I02-I04 remain pending implementation.
5. **Complete for single-worker characterization:** profile real polyBERT observation encoding and evaluator-ledger overhead; parallel scaling and budget selection remain open.
6. **Complete for PPO/GAE scope:** implement the common engine and GAE provider; engine-path, ledger-exhaustion and exact checkpoint/resume checks passed in Job 4671.
7. **Frozen before execution:** P4-A standard-five-step native-PPO protocol,
   seed `20260910` preflight, and formal seeds `20260911-20260915`; formal jobs
   may launch only after every preflight gate passes.
8. **In progress:** P4-A preflight passed all frozen gates in Job 4675; formal
   seeds `20260911-20260915` are running as Slurm array Job 4676. Completion,
   per-seed audit, aggregation and local evidence synchronization remain open.
9. Integrate Policy-CC (`eta = 0`) and close I02-I04 without changing the common PPO optimizer path; integrate MCC-PPO last.
10. Complete P3 estimator validation and draft/freeze P5 before running it.

The current authorized activity is **completion and audit of the already
launched five-seed formal P4-A array**. Neither the smoke nor P4-A alone
authorizes external API work, counterfactual training, a matched algorithm
comparison or scientific claim promotion.
