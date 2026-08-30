# SciCF-PPO offline Gate 1 result

The valid blinded Gate 1 decision is **failed / no-go**. Zero of three PPO
stages satisfied the pre-declared requirement that the lower bound of the
paired-bootstrap 95% confidence interval for LLM-minus-comparator NDCG@4 be
strictly positive against both Random and the chemistry heuristic. Pairwise
refinement is therefore not authorized.

`decision-summary.json` is a compact, repository-tracked record extracted from
the full n001 artifact. The full decision, including all 24 per-trajectory rows,
remains at:

`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1-formal-ad3889d-v1/gate1-decision-blinded-v2.json`

The formal data collection used Slurm array job 4573. Candidate presentation
blinding, LLM ranking, and aggregation used jobs 4584, 4585, and 4586 after
server tests passed in job 4583. Jobs 4581-4582 are excluded order-confounded
diagnostics, not scientific evidence.
