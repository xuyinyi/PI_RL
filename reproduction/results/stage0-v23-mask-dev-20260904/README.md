# Stage-0 v2.3 mask-refinement development evidence

This directory contains the n001 development evidence for the final targeted
closure implementation. It is not formal Gate P1 evidence because the remote
staging directory had no recognized Git metadata.

- `final-mask-100/`: job 4658, 44 tests plus the same 100-state independent
  mask audit; zero false positives and zero false negatives for the refined
  mask; 5.371 s refined versus 695.502 s full-exact equivalent.
- `final-replay/`: job 4657, 1,000 deterministic transitions and 1,000 snapshot
  restores; zero `no_reaction_product`; Ray evaluator checkpoint passed.
- `final-models/`: job 4656, 5/5 polyBERT parity, 100/100 evaluator parity,
  shared Ray ledger and final development contract.
- `superseded-full-builder-closure-replay/`: the slower but semantically
  equivalent first implementation, retained only to document the targeted-path
  performance repair.
- `logs/`: selected Slurm stdout/stderr for the smoke, superseded replay and
  final jobs.

See `../../stage0/VALIDATION_V2.3.md` for the interpretation and gate boundary.
