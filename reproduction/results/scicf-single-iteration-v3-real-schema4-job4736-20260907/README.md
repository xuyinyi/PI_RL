# SciCF v3 real single-iteration run: Job 4736

## Decision

The exact schema-4 authorization was consumed by one n001 Slurm submission.
Job 4736 completed successfully at clean implementation commit
`93501f33cf4594791b31a805b153bf1accc94538`:

```text
primary_ppo_committed_auxiliary_applied
```

The observed method was `ppo_plus_scicf_soft_pair`. There were no integrity
failures. This is a successful single-iteration engineering integration result,
not evidence of algorithmic effectiveness. It does not authorize an automatic
rerun, multi-iteration training, formal training, sealed-test access, or a
scientific claim.

## Primary PPO transaction

- 128 on-policy transitions
- 27 successful terminal episodes
- 27 requested/unique/backend on-policy evaluator calls
- 8 PPO optimizer steps
- approximate KL: `0.008422786369919777`
- primary checkpoint was written and hashed before credentials or LLM access
- primary checkpoint SHA-256:
  `b627609ba6d1c65a1c0865e0cc3720cba35a8ed5dc74878ad3613aa680a9b041`

The initial policy hash differs from the post-PPO policy hash, and the primary
checkpoint remained unchanged after the optional auxiliary stage.

## DeepSeek acquisition

- provider: `deepseek-official`
- model ID: `deepseek-v4-flash`
- revision: `DeepSeek-V4-Flash-0731-api-snapshot-2026-09-03`
- two 24-candidate pools
- eight selected candidates total
- two semantic attempts and two HTTP transmissions
- prompt tokens: 15,481
- completion tokens: 268
- token accounting complete: yes
- API key logged: no
- credential path logged: no

Both first responses passed the bounded response guard. No repair attempt,
heuristic fallback, or cached-response substitution was used.

## K=5 soft verification

All eight selected candidates received five matched factual/counterfactual
deltas, for a ceiling of 40 matched pairs / 80 branches. Structural termination
and cache reuse meant the AFP evaluator ledger recorded 37 requested calls, 27
unique/backend calls, and 10 cache hits during verification.

Four candidates received non-zero empirical weights:

| Candidate | Sign pattern | Preferred | Weight |
|---|---:|---|---:|
| `cf-6ff223aff62da0b8` | 1 positive, 4 negative | factual | 1.0000 |
| `cf-9eaa24c0e992c61f` | 3 positive, 2 ties | counterfactual | 0.9000 |
| `cf-710f4dd423428638` | 3 positive, 2 ties | counterfactual | 0.9000 |
| `cf-58eda5102428641e` | 5 positive | counterfactual | 0.5433 |

The other four candidates abstained because of a direction tie, weight below
the minimum, or five practical ties. The soft gate passed with:

- weighted candidates: 4
- total training weight: `3.343333333333333`
- normalized effective mass: `1.6716666666666664`
- largest candidate mass fraction: `0.29910269192422734`
- gate failures: none

Weights came only from matched Oracle evidence. LLM confidence did not enter
the loss, and counterfactual actions did not enter PPO clipping.

## Auxiliary update

Exactly one weighted pairwise optimizer step was applied:

- weighted pairwise loss: `0.75445296` before to `0.75410109` after
- weighted mean signed margin: `-0.11768591` to `-0.11702014`
- weighted preference accuracy: `0.0` before and after
- full-support maximum joint KL: `2.69834345090203e-7`
- maximum absolute value drift: `0.0012129470705986023`
- value-head SHA-256: unchanged
- final checkpoint SHA-256:
  `787188787b078b7db8892bf4b7cfebb6f0f7c9007106f1360811e1c47599642b`

The update moved all four margins in the intended aggregate direction but did
not cross zero for any weighted pair, hence accuracy remained zero. This is a
small, KL-bounded mechanics check only.

The raw auxiliary receipt reports a tiny negative KL
(`-2.893630153266713e-7`) from floating-point estimation. The independently
reported full-support KL is small and positive. Before multi-iteration use, the
raw KL diagnostic should be made explicitly non-negative within a declared
numerical tolerance; this does not invalidate the completed single update.

## Slurm and provenance

- Job: 4736
- State / exit: `COMPLETED` / `0:0`
- Elapsed: 61 seconds
- Allocation: 8 CPU, 96 GiB, one H100 GPU
- Tests: `69 passed in 4.51s`
- authorization SHA-256:
  `44de7d380ed438baceca184ed10fd31a23c26261f5703a4537d21faa5727d29a`
- report SHA-256:
  `c3c83dcdeb1dafe1175d6013749ced030e9d82459f4f54bdd2caa59c156035dc`
- protocol SHA-256:
  `57d7a06c8e519370b2cbb70330db4132bc08e7ff24e1fefa393ef13467b2d925`
- soft-pair config SHA-256:
  `97ed1bcf468293faa5c9e0eafb2dfde819061c201996ccbbcaf6b66bacce7173`

The two checkpoint binaries remain on n001 at the paths recorded in
`checkpoint-sha256.txt`; the repository archives their hashes rather than
duplicating the binaries.

## Next boundary

No rerun may use this authorization. Before requesting a bounded
multi-iteration engineering smoke, freeze and preflight a separate runner that
persists circuit-breaker state across iteration checkpoints, enforces a total
per-iteration LLM wall-time budget, retains the primary PPO checkpoint on every
recoverable path, and fixes the tiny-negative-KL diagnostic convention. Only
after a no-credential n001 Slurm preflight should a separate short-horizon
multi-iteration authorization be considered.

