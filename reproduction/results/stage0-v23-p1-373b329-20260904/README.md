# Stage-0 v2.3 clean-coordinate P1 evidence

Status date: 2026-09-04

Status: **Gate P1 engineering acceptance passed under the reconstructed AFP
compatibility contract**

## Immutable coordinate

- DAPiGen Git commit: `373b3291ac04dbf654c134bee2ca61a0c86d1a68`
- branch used for checkout: `codex/stage0-v23-p1`
- remote checkout:
  `/home/wch/workspaces/DAPiGen-reproduction/stage0-v23-p1-373b329-clean`
- execution host: `yanlih100n1`, submitted through Slurm on `n001`
- source bundle SHA-256:
  `94cacd2befe8123f9106dce7d5987b4e35ba03bf811fb0b9bc820a802c62892d`
- Git status before and after formal model/manifest validation: clean

Run outputs were written outside the checkout under
`/home/wch/workspaces/DAPiGen-reproduction/runs/stage0-v23-p1-373b329`.
The complete remote and local file-hash lists matched after synchronization.

## Slurm jobs

| Job | Scope | Elapsed | Exit | Result |
|---:|---|---:|---:|---|
| 4659 | 44 tests, core/asset gates, same 100-state mask audit | 12:01 | 0:0 | passed |
| 4660 | 1,000 transition replay/restore and Ray checkpoint | 2:29 | 0:0 | passed |
| 4661 | polyBERT, 100-PI evaluator, Ray and formal manifest | 0:41 | 0:0 | passed |

## Acceptance results

- All 44 tests passed.
- The custom DAPiGen chemistry backend was loaded as
  `dapigen_custom-sha256-e9f66fcccf1fe399`.
- The same 100-state sequence contained 58 distinct states with step-index
  frequencies 43/28/18/9/2.
- Independent exact totals were 23,770 dianhydride and 44,437 diamine actions.
- `closure_exact_cached` had zero false negatives and zero false positives on
  both sides.
- It removed all label-only false positives: 3,742 dianhydride and 11,878
  diamine actions.
- Refined-mask time was 5.448 s versus 701.898 s for the independent full-exact
  equivalent, a 128.834x ratio on this bounded panel.
- Replay passed for 1,000/1,000 transitions and 1,000/1,000 restored snapshots,
  covering 552 distinct input states in 449 episodes. Terminations were 231
  success, 214 atom limit and 4 horizon, with zero `no_reaction_product`.
- polyBERT parity passed on 5/5 structures with maximum absolute difference
  zero. Persistent-versus-legacy evaluator parity passed on 100/100 PI
  molecules with tolerance zero and no mismatch.
- Shared Ray evaluator ledger/cache checkpoint and restoration passed.

The normalized 100-state audit, exact replay JSON and Ray JSON matched the
final v2.3 development evidence. Timing fields and absolute checkout paths were
excluded from the normalized mask comparison.

## Formal contract

```text
environment_id       dapigen:8004c1fa2055a186b4a4f3ed
task_contract_id     237bfedb2bfa364f3251b47c4e6f83f1ba99ad698c7e1cdde65c62c9f1d0768e
budget_contract_id   ae3040512c12039bf2a63fdcdaf29ac766ff3754753c77290077051583b9e54e
runtime_contract_id  ea5a805dd837627305ee51cdddad8b37c55cb24adea73f97829126368348d25f
stage0 source tree   e3a7024d68de81d0f7c30d9368d271f49f1ab8f058d03557896321b5cd58a10a
```

The task and budget IDs differ from the development-staging IDs because the
formal contract now binds the clean Git commit and `dapigen_git_dirty=false`.
The environment, runtime and Stage-0 implementation identities are unchanged.

## Evidence boundary

This acceptance uses the declared **reconstructed AFP compatibility** assets.
`reconstructed_asset_gate.json` records `original_author_weights=false`,
`polybert_identity_verified=false` and `ppo_assets_ready=false`; it also records
that all required runtime files are present and the reconstructed compatibility
gate passed. P1 therefore establishes a reproducible engineering environment,
not paper-original model provenance or a scientific property claim.

The `legacy_effective` six-step PPO regression belongs to the post-P1 baseline
matrix and remains unrun. No PPO, Policy-CC, MCC-PPO or external API experiment
was launched by this acceptance.
