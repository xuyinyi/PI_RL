# Stage-0 v2.3 clean-coordinate P1 acceptance record

Status date: 2026-09-04

Status: **passed under the reconstructed AFP compatibility contract**

The v2.3 implementation was accepted from clean Git commit
`373b3291ac04dbf654c134bee2ca61a0c86d1a68` on `yanlih100n1`. Slurm jobs
4659, 4660 and 4661 all completed with exit code `0:0`; the checkout was clean
before and after formal manifest creation.

The formal run reproduced the development evidence: 44 tests, custom DAPiGen
BRICS, 5/5 polyBERT parity, 100/100 evaluator parity, shared Ray ledger/cache
restoration, 1,000 transition replays and 1,000 snapshot restores all passed.
On the same 100 states, `closure_exact_cached` had zero false negatives and zero
false positives for both action sides, removed all 3,742/11,878 label-only false
positives, and took 5.448 s versus 701.898 s for the independent full-exact
equivalent (128.834x on this bounded panel).

The formal contract records:

```text
dapigen_git_commit   373b3291ac04dbf654c134bee2ca61a0c86d1a68
dapigen_git_dirty    false
environment_id       dapigen:8004c1fa2055a186b4a4f3ed
task_contract_id     237bfedb2bfa364f3251b47c4e6f83f1ba99ad698c7e1cdde65c62c9f1d0768e
budget_contract_id   ae3040512c12039bf2a63fdcdaf29ac766ff3754753c77290077051583b9e54e
runtime_contract_id  ea5a805dd837627305ee51cdddad8b37c55cb24adea73f97829126368348d25f
stage0 source tree   e3a7024d68de81d0f7c30d9368d271f49f1ab8f058d03557896321b5cd58a10a
```

Evidence and file hashes are under
`reproduction/results/stage0-v23-p1-373b329-20260904/`.

## Boundary

This is an engineering acceptance against reconstructed compatibility assets,
not proof of author-original checkpoint identity, paper-result reproduction,
physical-property improvement or experimental validity. The asset report keeps
`original_author_weights=false`, `polybert_identity_verified=false` and
`ppo_assets_ready=false` explicit.

The frozen `legacy_effective` six-step configuration is a post-P1 baseline
regression used to separate task-semantic changes from algorithm changes. It is
not a P1 environment check and remains unrun until the unified P2 engine exists.
No PPO, Policy-CC, MCC-PPO or external API experiment was launched.
