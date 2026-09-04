# Changelog

## v2.3 P1 acceptance

- froze clean implementation coordinate
  `373b3291ac04dbf654c134bee2ca61a0c86d1a68`;
- reran the full P1 acceptance on `yanlih100n1` through Slurm jobs 4659, 4660
  and 4661, all completing with exit code `0:0`;
- regenerated the formal manifest with `dapigen_git_dirty=false` and preserved
  environment, runtime and Stage-0 implementation identities;
- reproduced the same 100-state mask decisions and exact 1,000-step replay;
- retained reconstructed-asset provenance limits and kept all PPO and
  counterfactual-credit experiments closed.

## v2.3 candidate

- replaced the formal label-only mask with `closure_exact_cached`;
- retained cheap label compatibility while attachments remain and required an
  exact side-specific monomer at final attachment closure;
- indexed the released reverse-BRICS reactions by attachment-label pair so the
  closure check does not traverse the complete reaction table;
- kept selected transitions and the independent `exact_cached` audit reference
  on the original full custom `BRICSBuild` path;
- added targeted-versus-full legacy closure-product equivalence coverage;
- passed 44 tests and the same 100-state audit with zero false positives and
  zero false negatives on both sides;
- reduced audited mask construction from a 695.502-second full-exact equivalent
  to 5.371 seconds (129.5x);
- passed 1,000 transition replays and 1,000 snapshot restores with zero
  `no_reaction_product`, while targeted closure reduced chemistry replay time
  from 29:59 to 2:10 for the same observed trajectory statistics;
- identified the clean immutable Git coordinate as the final development-stage
  P1 blocker; it was later closed by the v2.3 P1 acceptance above.

## v2.2 candidate

- froze the paper/archived-bytecode weighted-average objective as a named task contract;
- bound state and transition snapshots to `environment_id` and rejected malformed state types;
- added evaluator cache, budget, source-ledger and audit-event checkpoint restoration;
- separated all-run unique molecules from cache-miss backend submissions after
  cache eviction or retry;
- verified declared local polyBERT fingerprints against checkpoint contents;
- added attachment-label histograms to the formal `augmented_v3` observation and stopped claiming the embedding is strictly Markov;
- included the complete Stage-0 source tree, QSPR Python tree and runtime versions in the task contract;
- isolated persistent-evaluator batch failures to individual molecules;
- added reachable-state compatibility-versus-exact mask auditing;
- separated instance-specific `run_name` from the comparable `per_run` cache
  lifetime contract;
- completed n001 development validation: 42 tests, model/evaluator parity,
  1,000-transition replay/restore and Ray checkpoint accounting passed, while
  the 100-state audit exposed blocking 13.60%/21.09% mask false-positive rates;
- retained v2.1 as an immutable, hash-identified intake artifact rather than accepted evidence.

## v2.1

- changed the scalable primary mask mode to BRICS-label compatibility with exact selected-action validation;
- made formal factory construction fail rather than silently fall back to stock RDKit BRICS;
- included the actual chemistry backend and RDKit version in environment identity;
- added strict integer action parsing;
- rejected non-restorable legacy state snapshots;
- preserved terminal molecule/candidate data in snapshots;
- added per-side growth counts to state and Markov observation;
- added explicit NOOP actions;
- separated pure chemistry transitions from terminal evaluation;
- removed reward-dependent terminal-product selection;
- added persistent polyBERT checkpoint content fingerprinting;
- added persistent QSPR evaluator with model/scaler/settings hashes;
- added requested/unique/cache-hit/source budget accounting;
- added evaluator source allow-list and finite-output validation;
- added shared Ray evaluator-service interface;
- added task and budget contract IDs plus implementation hashes;
- added standard, long-horizon, exact-mask audit and legacy-effective configs;
- expanded automated checks to 25 tests;
- added content hashes for the loaded custom BRICS implementation and task-critical DAPiGen sources;
- added direct RLlib EnvContext construction, explicit iteration-boundary ledger reads, polyBERT parity, contract comparison and Ray concurrency scripts.
