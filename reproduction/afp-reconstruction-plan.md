# DAPiGen AFP reward assets: reconstruction decision and execution contract

Frozen public code coordinate: `xuyinyi/DAPiGen@5f692946cbe0d15eede882dfe4cff7fb26eb7d8c`

## Decision

Author-supplied assets remain the only route to an original-weight reproduction.
If they cannot be obtained, the public raw data and model code are sufficient for
a clearly labeled compatibility reconstruction, but not for recovering the
authors' exact serialized weights.

Do not run the committed `QSPR/GNN/Ensembling.py` unchanged. It fails before
training and contains scientific/output mismatches:

- package imports require inconsistent working directories and `PYTHONPATH`;
- `data/dataloading.py` resolves raw data under nonexistent `QSPR/raw_data`;
- scalers are saved as `<property>.pkl`, while PPO requires
  `<property>_scaler.pkl`;
- the ensembling script enables sigmoid output for all four properties, while
  optimization and PPO inference use sigmoid only for transmittance;
- generated weights/settings remain under the QSPR library tree and are not
  staged into `RL_PPO/GNN/model`;
- the compatibility environment uses PyTorch 2.1/DGL 2.1 rather than the
  repository's Windows/PyTorch 1.12/DGL 0.9 environment.

## Required PPO files

| Property | Split seed | Model id | Initialization seed | Sigmoid |
|---|---:|---:|---:|---|
| transmittance(400) | 825 | 43 | 922 | true |
| cte | 854 | 56 | 125 | false |
| strength | 331 | 84 | 734 | false |
| tg | 525 | 64 | 109 | false |

For each property, PPO requires one scaler, one selected weight, and one settings
table. The expected names are enumerated by `check_assets.py`.

## Minimal reconstruction route

1. Preserve the clean upstream and implement a separate reconstruction driver in
   the patch worktree.
2. Read the four public CSVs from repository-root `raw_data/` and reproduce the
   committed heavy-atom filtering, canonicalization, and fixed random split.
3. Create the four deterministic scalers with exact PPO filenames.
4. Reuse the hard-coded optimized architecture/training parameters in
   `Ensembling.py`; do not rerun Bayesian optimization.
5. Train only the four members actually loaded by PPO, using the ids and
   initialization seeds above. A 100-row ensemble is not required for PPO
   inference.
6. Write each settings CSV so its index equals the selected model id; the PPO
   loader indexes the table by 43, 56, 84, or 64.
7. Stage outputs first under a reconstruction-specific directory. Promote them
   to `RL_PPO/GNN/model` only after all gates pass.

## Gates before PPO

1. Two-epoch pilot: data load, graph cache, forward/backward, early stopping, and
   serialization all succeed for every property configuration.
2. Full selected-member training completes with saved split, seed, environment,
   metrics, and SHA-256 manifests.
3. Each staged model loads through `RL_PPO/GNN/benchmarks.py::load_model` and
   reproduces its recorded validation/test metrics.
4. Predictions on the public datasets have finite values and physically
   plausible ranges after inverse scaling.
5. `Benchmark` produces all four property values and a finite terminal reward on
   fixed smoke molecules.
6. The PPO environment passes deterministic reset/step and terminal-reward
   smokes before any training run.

Even after all gates pass, label the result `reconstructed AFP compatibility
baseline`; do not call it the authors' original reward model unless the weight
identity is independently verified.
