# Stage-0 v2.2 acceptance record

Status date: 2026-09-04

Status: **development validation mostly passed; Gate P1 remains closed**

## Frozen task decisions

- Reward objective: `dapigen-paper-equation-1-weighted-average-v1`.
- Exact Markov state: immutable `DAPiGenState` schema 3, bound to `environment_id`.
- Policy observation: `augmented_v3`; this is an explicitly lossy learned representation, not a claim of injective Markov encoding.
- Product-selection estimand: common-quantile selection over sorted unique enumerated products. It is not a reaction-yield model.
- Formal horizon: five actions.
- Complete blocks: `pristine_only`.
- Primary budget: requested complete-PI evaluations; unique molecules,
  cache-miss backend submissions and cache reuse are secondary reported quantities.

## Required evidence

| Check | Required result | Current status |
|---|---|---|
| v2.2 unit tests | all pass in target environment | passed: 42 tests, job 4646 |
| custom DAPiGen BRICS | backend identity contains `dapigen_custom` | passed: `dapigen_custom-sha256-e9f66fcccf1fe399` |
| reachable mask audit | zero false negatives; false-positive rate reported and reviewed | **blocking review**: zero false negatives, but 13.60%/21.09% false positives |
| paper-objective evaluator parity | 100 fixed PI, `mismatch_count = 0` | passed: 100/100, job 4649 |
| polyBERT parity | fixed panel within frozen tolerance and actual content hash verified | passed: 5/5, maximum absolute difference 0 |
| state/action/seed replay | at least 1,000 real transitions, field-identical | passed: 1,000/1,000, job 4647 |
| checkpoint/restore | next transition plus evaluator cache/ledger identical | passed for 1,000 chemistry transitions and Ray evaluator state |
| Ray concurrency | one actor, duplicate requests, exact requested/unique/backend/cache totals | passed: 9/1/1/8 after restore, job 4649 |
| runtime contract | identical across method manifests | development manifest generated; formal check pending clean coordinate |
| task/budget contracts | identical for PPO, Policy-CC and MCC-PPO | blocked on P2 |

## Development evidence on n001

The candidate was staged outside the historical checkout at
`/home/wch/workspaces/DAPiGen-reproduction/stage0-v22-dev-20260904` and run on
`yanlih100n1`. Jobs 4646, 4647 and 4649 completed with exit code 0. Job 4648 is
superseded by 4649 because the latter contains the corrected comparable
`cache_scope=per_run` budget contract.

- [core, custom chemistry, assets and 100-state mask audit](../results/stage0-v22-dev-20260904/final-mask-100/)
- [1,000-transition replay and evaluator checkpoint](../results/stage0-v22-dev-20260904/final-replay/)
- [final model, evaluator, Ray and development-contract checks](../results/stage0-v22-dev-20260904/final-models-contractfix/)

The mask audit visited 100 nonterminal states (58 distinct states across 43
episodes). It found zero compatibility false negatives. It found 3,742 false
positives among 27,512 compatibility-allowed dianhydride actions (13.60%) and
11,878 among 56,315 diamine actions (21.09%). Exact enumeration required 692.8
seconds for the audit, so changing the formal rollout mask to `exact_cached`
without a performance design is not accepted.

The replay check reproduced and restored all 1,000 real chemistry transitions
without changing global Python or NumPy RNG state. It also observed 271
`no_reaction_product` terminations, consistent with the material
compatibility-mask false-positive risk rather than a replay failure.

The development manifest records:

```text
environment_id       dapigen:823784dc7e495332b89d305a
task_contract_id     0eeeeda87fe8a5b56d4a245c9d5713b94ffd8b2d92aa1f1370beab429eae86bb
budget_contract_id   4ae7ef868f971ebf57292346fe2af2fb42ab03cd928d998451ade6d01b04c587
runtime_contract_id  ea5a805dd837627305ee51cdddad8b37c55cb24adea73f97829126368348d25f
stage0 source tree   9b123fecb5534dbe39dc236b701ada4012a6908fbd9dccafcef54bda04bc2d19
```

This is development evidence only: the isolated staging directory intentionally
has no Git metadata, so `dapigen_git_commit` and `dapigen_git_dirty` are null.
Formal P1 evidence must be regenerated from a clean, immutable Git coordinate.

## Asset and retraining boundary

The reconstructed AFP/QSPR assets and the installed polyBERT checkpoint passed
their engineering hash and parity checks. They do not need to be retrained to
continue Stage-0 engineering. They are not author-original assets: the asset
report keeps `original_author_weights=false` and polyBERT source identity is not
verified. Existing PPO policy checkpoints are not reusable as the new standard
policy because `augmented_v3` has 1,246 dimensions and the task semantics have
changed; a new policy training run is required only after P1 and P2 open it.

## Stop condition

Any reward, chemistry, mask false-negative, encoder, replay, checkpoint, ledger,
runtime or contract mismatch keeps PPO/Policy-CC/MCC-PPO training closed. A
successful engineering gate does not establish QSPR validity or experimental
polyimide performance.

For this candidate, the compatibility-mask false-positive rates and the missing
clean Git coordinate are the two active P1 blockers. No PPO run is authorized.

## Current infrastructure observation

On 2026-09-04, the explicit `n001` SSH coordinate resolved to `yanlih100n1`.
The historical checkout, Python 3.8 environment, polyBERT mirror and reconstructed
AFP compatibility assets were all present, and the recorded reconstructed-asset
hash gate passed. The separate `ssh cpu` alias resolves to `yanli8488` and must
not be used as a substitute for n001 evidence.
