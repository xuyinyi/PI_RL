# Stage-0 v2.2 development validation on n001

Date: 2026-09-04

Status: **development evidence; formal Gate P1 remains closed**

This directory records an isolated validation of the uncommitted Stage-0 v2.2
candidate on `yanlih100n1`. The remote staging coordinate was
`/home/wch/workspaces/DAPiGen-reproduction/stage0-v22-dev-20260904`; it did not
modify the historical DAPiGen worktree or model assets.

## Jobs and results

| Job | Scope | Result |
|---:|---|---|
| 4646 | 42 tests, custom chemistry, reconstructed assets, 100-state mask audit | completed 0; mask needs review |
| 4647 | 1,000 chemistry transitions and Ray evaluator checkpoint | completed 0; passed |
| 4649 | polyBERT, 100 PI evaluator parity, Ray concurrency, corrected development manifest | completed 0; passed |

Job 4648 is retained in `final-models/` and `logs/` but superseded by 4649
because its manifest was created before `run_name` was separated from the
comparable `cache_scope`.

`SHA256SUMS.txt` covers every copied JSON, CSV and Slurm log in this directory
and was verified locally after transfer.

## Evidence summary

- Unit tests: 42 passed.
- Chemistry backend: `dapigen_custom-sha256-e9f66fcccf1fe399`, RDKit 2022.09.5.
- Catalogs: 576 dianhydride and 1,004 diamine actions; no invalid or canonical-duplicate rows.
- Mask audit: 100 states, 58 distinct, zero false negatives.
- Compatibility false positives: 3,742/27,512 dianhydride (13.60%) and 11,878/56,315 diamine (21.09%).
- polyBERT: 5/5 structures passed, dimension 600, maximum absolute difference 0.
- Evaluator: 100/100 PI passed with `mismatch_count=0` under `dapigen-paper-equation-1-weighted-average-v1`.
- Replay/restore: 1,000/1,000 transitions reproduced and restored; global Python and NumPy RNG unchanged.
- Ray after checkpoint restore: 9 requested, 1 unique, 1 backend and 8 cache hits.

The replay observed 271 `no_reaction_product` terminations. This does not break
deterministic replay, but it is consistent with the material compatibility-mask
false-positive rate and is an active task-quality concern.

## Contract identifiers

```text
environment_id       dapigen:823784dc7e495332b89d305a
task_contract_id     0eeeeda87fe8a5b56d4a245c9d5713b94ffd8b2d92aa1f1370beab429eae86bb
budget_contract_id   4ae7ef868f971ebf57292346fe2af2fb42ab03cd928d998451ade6d01b04c587
runtime_contract_id  ea5a805dd837627305ee51cdddad8b37c55cb24adea73f97829126368348d25f
stage0 source tree   9b123fecb5534dbe39dc236b701ada4012a6908fbd9dccafcef54bda04bc2d19
```

The development manifest has null Git commit/dirty fields because the isolated
staging directory has no `.git` metadata. It cannot be promoted to formal P1
evidence. The reconstructed assets are engineering-usable but remain explicitly
non-author-original, and polyBERT source identity is not verified.

## Decision

Do not launch PPO from this coordinate. First resolve and reaudit the mask, then
create a clean immutable Git coordinate and regenerate the formal manifest.
Existing polyBERT and QSPR assets do not require retraining for that engineering
work. Existing PPO policies must not be continued as the v2.2 standard policy.
