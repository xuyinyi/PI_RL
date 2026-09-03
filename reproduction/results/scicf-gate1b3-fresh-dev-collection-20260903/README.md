# SciCF Gate 1B.3 fresh-development collection

The authorized Gate 1B.3 fresh-development collection completed and passed
metadata-only integrity validation. At collection completion, the labels had
not been evaluated.

## Scope

- Split: `dev` only.
- Stages: `early` and `middle` only.
- Seeds: `4241` through `4256`.
- Candidate pool: 24 per trajectory, budget four, unchanged source quotas.
- Structure exclusion: all 200 structure keys previously observed in Gate
  1B.1/B.2 development data.
- Late: not recollected; remains a non-gating saturation diagnostic.
- Test: not collected, accessed, or evaluated.

The collector required the versioned authorization receipt with SHA-256
`609120c8c7cb9c8008f5b63eadaeaeafad823b40152c3e5353fc784bb5d67f36`.
That receipt authorizes collection only; fresh-dev evaluation, test access,
Gate 1C, pairwise refinement, and PPO integration remain unauthorized.

## Collection result

| Stage | Trajectories | Unique structure keys | Atomic Oracle calls | Report SHA-256 |
|---|---:|---:|---:|---|
| early | 16 | 63 | 85 | `9799e54e74810a36b53401adc1ae0865b90a039557c2dbbfaa5b4bd0ff997316` |
| middle | 16 | 81 | 1,406 | `457142ffbbc68e29c16a1e467a890e0a88d1dde61923dddf637131992bce4815` |
| combined | 32 | 85 | 1,491 | — |

The combined 1,491 calls comprise 784 factual, 707 counterfactual, and zero
evaluation calls. The fresh candidate structures overlap neither the original
160 train keys nor the 200 prior-development keys. Every 24-candidate pool
covered all timesteps that existed in its factual trajectory.

## Provenance and boundary

- Collection authorization/enforcement commit:
  `bbac223d123588619675539e989e3010c079ddea`.
- Metadata validator commit:
  `be50578d63f90272340955c79cba8f30a5589451`.
- Slurm 4634: 50/50 tests passed after receipt enforcement.
- Slurm array 4635: early and middle collection completed with exit code 0.
- Slurm 4637: metadata-only collection validation completed.
- Collection validation SHA-256:
  `6f260f26700f4fc5c8da65ab173f9920989f37fd372b1a8b5a44e25f9e1ba13a`.

Full server data:

`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1b3-stage-routed-20260903-v1/fresh-dev-v1`

The run tree contains `dev/early`, `dev/middle`, and
`collection-validation.json`. It contains no model fit, development decision,
or test directory. No LLM call or PPO update occurred.

This artifact establishes collection completeness and structure isolation
only. It is not evidence that the Gate 1B.3 algorithm passes its fresh-dev
metrics.

## Evaluation follow-up

The user separately authorized fresh-dev evaluation on 2026-09-03. Slurm job
4640 subsequently evaluated these checksum-bound reports and returned a
development no-go. The sealed test remained untouched. See
`reproduction/results/scicf-gate1b3-fresh-dev-evaluation-20260903` for the
evaluation decision and exact evidence hashes.
