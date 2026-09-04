# Stage-0 v2.3 mask-refinement acceptance record

Status date: 2026-09-04

Status: **mask-refinement development acceptance passed; subsequently promoted
at the clean coordinate recorded in `VALIDATION_V2.3_P1.md`**

## Frozen task decisions

- Reward objective: `dapigen-paper-equation-1-weighted-average-v1`.
- Exact Markov state: immutable `DAPiGenState` schema 3, bound to `environment_id`.
- Policy observation: `augmented_v3`; this is a lossy learned representation, not an injective Markov encoding claim.
- Product-selection estimand: common-quantile selection over sorted unique enumerated products.
- Formal horizon: five actions.
- Complete blocks: `pristine_only`.
- Primary budget: requested complete-PI evaluations; unique molecules, cache-miss backend submissions and cache reuse are secondary quantities.

## v2.3 mask design

The formal mask mode is `closure_exact_cached`:

1. apply the cheap BRICS attachment-label compatibility rule;
2. allow compatible actions directly while a connection point remains;
3. when the proposed connection consumes the final attachment on both reactants,
   run only the matching released reverse-BRICS reaction and require a valid
   complete side-specific monomer;
4. cache this closure result by side, exact partial structure, growth count and
   action ID.

The optimized closure path is not the audit oracle. `exact_cached` independently
uses the original full custom `BRICSBuild` over every catalog action. Selected
environment transitions also retain full candidate enumeration and
reward-independent candidate selection.

## Required evidence

| Check | Required result | v2.3 development result |
|---|---|---|
| unit tests | all pass on target environment | passed: 44 tests, job 4658 |
| targeted closure equivalence | known one-attachment products equal full legacy BRICS | passed |
| custom DAPiGen BRICS | backend contains `dapigen_custom` | passed: `dapigen_custom-sha256-e9f66fcccf1fe399` |
| same 100-state mask audit | zero false negatives; false-positive reduction reported | passed: zero false negatives and zero false positives on both sides |
| mask performance | materially faster than full exact reference | passed: 5.371 s versus 695.502 s, 129.5x |
| state/action/seed replay | 1,000 real transitions, field-identical | passed: 1,000/1,000, job 4657 |
| snapshot restore | all replayed snapshots field-identical | passed: 1,000/1,000 |
| invalid selected mask actions | no `no_reaction_product` terminations | passed: 0 |
| evaluator/polyBERT parity | frozen panels pass | passed: 100/100 evaluator and 5/5 polyBERT, job 4656 |
| Ray checkpoint/accounting | shared actor and restored ledger agree | passed |
| immutable run coordinate | clean recognized Git commit | **not met in development staging** |

## Same-state mask audit

Job 4658 used seed `20260904` and the same 100-state sequence as v2.2: 58
distinct states with step-index frequencies 43/28/18/9/2 for indices 0..4.
The independent exact action totals were also unchanged: 23,770 dianhydride and
44,437 diamine actions.

The historical label-only mask allowed 27,512 dianhydride actions and 56,315
diamine actions. It therefore retained 3,742 false positives (13.60%) and
11,878 false positives (21.09%), respectively. The v2.3 mask removed all
15,620 of those false positives and had zero false negatives and zero false
positives on the audited states.

Mask-build timing on the same run was:

```text
label compatibility                 0.125662 s
closure_exact_cached                5.370791 s
independent full exact equivalent 695.501842 s
refined/full-exact speed ratio        129.5x
```

These are bounded development measurements on this state panel, not a universal
complexity or chemistry-validity proof.

## 1,000-step replay and throughput

The final optimized replay (job 4657) reproduced 1,000 transitions and restored
1,000 transition snapshots. It covered 552 distinct input states in 449
episodes, reached step 5, preserved global Python and NumPy RNG state, and
reported 231 successful terminals, 214 atom-limit terminals, four horizon
terminals and zero `no_reaction_product` terminals.

An initial semantically correct implementation used full `BRICSBuild` inside
every closure-mask query. Its chemistry replay took 29 min 59 s and issued
80,144 general assembly requests. The targeted implementation preserved the
same replay statistics while taking 2 min 10 s, a 13.8x improvement, and reduced
general assembly requests to 1,009. It still executed 79,402 targeted closure
checks. This is acceptable for the requested Stage-0 development audit, but PPO
throughput must be profiled under the eventual multi-worker P2 engine before a
training budget is frozen.

## Model and contract evidence

The final development manifest records:

```text
environment_id       dapigen:8004c1fa2055a186b4a4f3ed
task_contract_id     fe3750d9dbb26d97a9b9ea02040424f2c4c05cdeca3a64a9500d8f8892b88e4e
budget_contract_id   159ccbaa934ffd69b72f4d125be01a6cf72691589a1b46f973512413b8bb3842
runtime_contract_id  ea5a805dd837627305ee51cdddad8b37c55cb24adea73f97829126368348d25f
stage0 source tree   e3a7024d68de81d0f7c30d9368d271f49f1ab8f058d03557896321b5cd58a10a
```

The v2.3 task and budget IDs supersede the preliminary v2.3 values produced
before targeted closure optimization. They do not supersede the preserved v2.2
record.

## Evidence location and boundary

The remote development staging was
`/home/wch/workspaces/DAPiGen-reproduction/stage0-v23-mask-dev-20260904` on
`yanlih100n1`. Final jobs 4656, 4657 and 4658 completed with exit code 0. The
synced evidence is under
`reproduction/results/stage0-v23-mask-dev-20260904/`.

This is development evidence only. The staging directory intentionally has no
Git metadata, so `dapigen_git_commit` and `dapigen_git_dirty` are null. The
candidate was subsequently rerun and accepted at clean commit `373b3291`;
`VALIDATION_V2.3_P1.md` is the formal P1 record. No PPO, Policy-CC or MCC-PPO
run is authorized by this development record.

The reconstructed AFP/QSPR assets and installed polyBERT checkpoint passed
engineering parity and do not require retraining for the next environment gate.
They are not verified author-original assets. A new policy must eventually be
trained because the observation and task semantics differ from the historical
PPO baseline.
