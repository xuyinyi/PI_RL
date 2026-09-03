# SciCF-PPO DeepSeek Flash offline Gate 1 result

The independent DeepSeek Flash Gate 1 decision is **failed / no-go**. Zero of
three PPO stages satisfied the pre-declared requirement that the lower bound of
the paired-bootstrap 95% confidence interval for LLM-minus-comparator NDCG@4
be strictly positive against both Random and the chemistry heuristic. Pairwise
refinement remains unauthorized.

This result does not overwrite or reinterpret the frozen local-Qwen Gate 1
result in `reproduction/results/scicf-gate1-20260830`. It is a separate model
run over the same blinded candidate pools and existing oracle evaluations.

`decision-summary.json` is a compact, repository-tracked record extracted from
the full n001 artifact. The full decision remains at:

`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1-formal-ad3889d-v1/gate1-decision-api-deepseek-v4-flash-20260903-v1.json`

DeepSeek API compatibility and security tests passed in Slurm job 4601
(23/23 tests), and shell validation passed in job 4602. Credential provisioning,
the one-request smoke, formal 24-request ranking, and aggregation completed in
jobs 4603-4606. All formal responses passed exact-pool and exact-budget schema
validation on the first attempt, with no transport retry and no formal cache
hit.

The public `deepseek-v4-flash` name is a rolling API alias. This run records the
dated release/snapshot marker `DeepSeek-V4-Flash-0731-api-snapshot-2026-09-03`,
the provider response fingerprint, prompt hashes, and response hashes; it does
not claim that the public alias is an immutable model endpoint.
