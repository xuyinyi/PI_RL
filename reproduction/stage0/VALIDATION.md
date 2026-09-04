# Historical validation report — DAPiGen Stage-0 Environment v2.1

> Evidence boundary: this file was supplied with the v2.1 intake package. It is
> retained as historical evidence only. Stage-0 v2.2 and v2.3 change
> state/snapshot, observation, evaluator, budget, mask and contract semantics;
> none of the results below count as current acceptance. Each current result
> must be written to a new version-bound output directory by an n001 Slurm job.

## 1. Checks executed in this container

```text
PYTHONPATH=overlay pytest -q
25 passed

python -m compileall
passed

Python 3.7 grammar parsing
32 / 32 Python files passed

python scripts/validate_core_without_models.py
passed
```

Execution image used for these checks:

```text
Python 3.13
NumPy 2.3.5
RDKit 2025.09.4
PyTorch 2.10 CPU
```

The smoke test used stock RDKit BRICS only because the complete DAPiGen checkout is not mounted in this container. The resulting backend identity explicitly contains `rdkit_fallback`; formal experiments are configured to reject this fallback.

## 2. Behaviors covered by automated tests

1. exact `(state, action, seed)` replay;
2. no consumption of Python or NumPy global RNG state;
3. stable state serialization and `state_id`;
4. strict rejection of legacy/incomplete snapshots;
5. rejection of fractional and Boolean action IDs;
6. Markov-v2 observation schema;
7. exact and compatibility action-mask behavior;
8. prevention of complete-block prefix erasure;
9. controlled minimum-growth long-horizon variants;
10. explicit NOOP after one side completes;
11. exact horizon without the released one-step offset;
12. separation of terminal failure and time-limit truncation;
13. no evaluator call for intermediate, failed, or truncated incomplete states;
14. reward-independent terminal-product selection;
15. successful terminal snapshot restoration including selected PI;
16. requested/unique/cache-hit budget accounting;
17. per-source evaluator ledger and source allow-list;
18. hard requested and unique oracle budgets;
19. environment fingerprint stability and change detection;
20. task-contract identity separated from budget-contract identity;
21. action-catalog export including explicit NOOP rows;
22. controller branch operations do not mutate the main episode;
23. real RDKit BRICS + DAPiGen two-stage PI reaction smoke path;
24. critical DAPiGen source changes alter the task contract;
25. concurrent duplicate evaluator requests share one transactional cache entry.

## 3. What was directly demonstrated

The no-model integration script generated a complete PI through the two-stage reaction chain, replayed it exactly under the same seed, restored its successful terminal snapshot, and evaluated it twice through a cached terminal evaluator. The resulting ledger was:

```text
requested_calls = 2
unique_calls    = 1
cache_hits      = 1
```

No intermediate structure was sent to the evaluator.

## 4. Checks that require the user's complete DAPiGen runtime

These are supplied as executable scripts but could not be completed in this container:

### 4.1 Custom BRICS parity

Formal runs require `RL_PPO.moldr.utils.BRICSBuild`. Run:

```bash
python scripts/audit_stage0_catalogs.py --dapigen-root /path/to/DAPiGen
```

The reported `chemistry_backend` must contain `dapigen_custom`, not `rdkit_fallback`.

### 4.2 Persistent polyBERT parity

The local polyBERT checkpoint is not present here. The persistent encoder must be checked against the released helper on a fixed partial-SMILES panel before training.

### 4.3 QSPR evaluator parity

The GitHub checkout exposes evaluator code, but the trained `.pt`, scaler and settings artifacts required by `RL_PPO/GNN/benchmarks.py` are not available in this container. Run:

```bash
python scripts/compare_evaluators.py \
  --dapigen-root /path/to/DAPiGen \
  --sample-size 100 \
  --device cpu
```

Primary experiments require `mismatch_count = 0` after the released rounding rules.

### 4.4 Gym 0.19 / Ray 1.13

Gym 0.19 and Ray 1.13 are absent here, so the legacy RLlib wrapper, `DAPiGenRLlibEnv`, masked Tuple-action model and shared Ray evaluator actor were syntax-checked but not runtime-executed. They must be validated inside the released DAPiGen environment.

### 4.5 Distributed budget concurrency

A formal test should submit concurrent duplicate and unique terminal molecules from multiple rollout workers and verify that the shared actor produces one global, transactional ledger.

## 5. Current acceptance status

| Component | Status |
|---|---|
| Pure branchable core | Passed locally |
| State/action schema | Passed locally |
| Named RNG and replay | Passed locally |
| Terminal-only reward adapter | Passed locally |
| Budget/cache ledger | Passed locally |
| RDKit fallback chemistry smoke | Passed locally |
| DAPiGen custom BRICS parity | Pending in full checkout |
| polyBERT numerical parity | Pending local checkpoint |
| persistent QSPR parity | Pending trained artifacts |
| Gym/RLlib runtime | Pending released environment |
| shared Ray actor concurrency | Pending Ray runtime |

The environment implementation is ready for repository-level acceptance, but Stage 0 is not declared complete until all pending rows pass.
