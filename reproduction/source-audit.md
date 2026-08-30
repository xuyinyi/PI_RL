# DAPiGen original PPO: source and execution audit

Audit date: 2026-08-30  
Paper DOI: `10.1002/adma.202511099`  
Official repository: `https://github.com/xuyinyi/DAPiGen.git`  
Frozen branch: `main`  
Frozen commit: `5f692946cbe0d15eede882dfe4cff7fb26eb7d8c`

The public source at the frozen commit is the implementation coordinate. Bundled
`CPython 3.7` bytecode is retained only as historical evidence because it differs
from the committed source in imports, reward-model class names, and reward
aggregation.

| Paper/repository locator | Concept or claim | Code path and symbol | Config/default | Data/execution path | Status | Evidence | Uncertainty / next check |
|---|---|---|---|---|---|---|---|
| README, RL_PPO section | PPO agent trains from decomposed dianhydride and diamine blocks | `RL_PPO/moldr/train.py::main`; `RL_PPO/moldr/env.py::PIEnvValueMax` | 100 training iterations; evaluation every 10; 10,000 generated samples per evaluation | `RL_PPO/outputs/building_blocks/*.csv` | partial | Entry, environment, blocks, and loop are public | Published PPO checkpoint and run manifest are absent |
| README, prerequisites | Recreate the published software environment | `env.yml` | Python 3.7.13, Torch 1.12.1+cu116, DGL 0.9.1/cu113, Ray 1.13.0, Gym 0.19.0 | Windows build strings and Windows prefix | partial | Versions are recorded | Export is Windows-specific and Torch/CUDA pair is unsuitable for H100; compatibility environment is not bitwise-identical |
| `train.py:21-31` | Construct PPO/environment configuration | `RL_PPO/moldr/config.py::get_default_config` | caller passes `num_gpus=4` | Direct `python train.py` | missing_in_code | Function signature comments out `num_gpus`; unmodified run raises `TypeError` | Minimal compatibility repair must be documented |
| `train.py:35-39`; `config.py:33-79` | Apply policy network and PPO hyperparameters | `PPOTrainer(...)`; `ppo.DEFAULT_CONFIG` | FC 256/128/128; ReLU; batch 500; workers 15 | Trainer receives `config={"env_config": config}` | ambiguous | RLlib settings are nested inside `env_config`, so they are not necessarily applied to the trainer | Resolve against paper/SI or author run manifest before changing semantics |
| `env.py:18-221` | Two-part tuple action, 1,200-D state, terminal property reward | `PIEnvValueMax.reset`, `step`, `compute_score`, `is_done` | length 60; step length 5 | polyBERT embeddings + building blocks + `Benchmark` | matched | Spaces and terminal-reward path are explicit | `env_step > step_length` permits one more step than a `>=` interpretation; global NumPy RNG is not controlled by `seed()` |
| Main text p. 5, Eq. (1); Experimental Section p. 11 | High-transparency PPO uses the weighted-average reward retained in the paper | `RL_PPO/GNN/benchmarks.py::Benchmark.score` | `R_T400 * (1 + R_CLTE + R_strength + R_Tg + R_SA) / 5` | terminal reward only | partial | Paper formula and bundled 2024 bytecode agree | Public `.py` activates the geometric-average variant; reproduction branch restores Eq. (1) and records the upstream variant |
| `polyBERT.py:5-22`; README | Encode each monomer as a 600-D polyBERT fingerprint | `RL_PPO/utils/polyBERT.py::Embedding_smiles` | primary `kuelumbus/polyBERT` not pinned in README; mirror frozen at `xushijie/polyBERT@e7dce434fb3eff37905dc114008660e5479ca9a8` | installed at `RL_PPO/models` | partial | All 14 mirror files are downloaded and hashed; source-path smoke returns a finite `(600,)` `float32` vector | Primary repository remains inaccessible, so mirror-to-primary weight identity is unverified |
| `benchmarks.py:61-211` | Reward uses transmittance, CTE, strength, Tg, and SA | `RL_PPO/GNN/benchmarks.py::Benchmark` | ensemble IDs 43/56/84/64 | expected under `RL_PPO/GNN/model` | partial | Four selected members, scalers, and settings are reconstructed, hash-locked, and runtime-tested | Author-supplied binaries remain absent; these assets only authorize the explicitly labeled compatibility mode |
| `QSPR/GNN/Ensembling.py` | Train 100 AFP initializations for four properties | `main`; `save_model` | split seeds 825/854/331/525; seed 2023; 100 members; up to 1,000 epochs | raw data exists under repository `raw_data` | partial | Training code, raw property data, seeds, and architecture settings are present | Published entry has broken import/path assumptions; retrained weights on Torch/DGL compatibility stack are reconstructions, not original weights |
| `QSPR/GNN/data/dataloading.py:21-41`; PPO scaler paths | Construct and serialize four target scalers | `import_dataset`; `Standardization`; `zero_oneNormalization` | transparency range 100; other properties use population mean/deviation | committed code reads nonexistent `QSPR/raw_data` and writes `QSPR/GNN/model/<property>.pkl` | missing_in_code | Public CSVs are present and the four scaler parameters are deterministic | Repair repository-root resolution and stage as `<property>_scaler.pkl` |
| `QSPR/GNN/Ensembling.py:91-181`; PPO model ids | Produce the four AFP members actually loaded by PPO | `AttentiveFPNet`; `save_model`; `Benchmark.load_model` | ids 43/56/84/64 map to initialization seeds 922/125/734/109 | QSPR library output must be staged into `RL_PPO/GNN/model` | partial | Model ids, split seeds, initialization sequence, and fixed architectures are recoverable | Exact author weights are unrecoverable; selected-member retraining is a compatibility reconstruction |
| `Optimization.py:144-152`; `Ensembling.py:107-130`; PPO prediction methods | Select output activation by property | `AttentiveFPNet.linear_predict` | sigmoid only for transparency; linear for CTE/strength/Tg | training and PPO inference | ambiguous | Optimization and PPO agree on `[true,false,false,false]` | Ensembling hard-codes sigmoid true for all four; reconstruction must choose the cross-file-consistent interpretation and disclose it |
| Public git history | Recover missing PPO reward assets from another public revision | repository object history | all public commits checked | git objects and paths | missing_in_code | No `.pt`, scaler, settings, or polyBERT model files occur in public history | Requires author-supplied assets or a clearly labeled re-training route |

## Reproduction gates

1. **Source frozen**: complete.
2. **n001 compatibility environment**: complete; core CUDA and imports pass.
3. **Original command failure captured**: complete.
4. **Original scientific assets present and verified**: blocked; default gate remains fail-closed.
5. **Explicit reconstructed AFP compatibility asset gate**: complete; 12 reward and 14 polyBERT hashes verified.
6. **Environment reset/step smoke with installed polyBERT and reconstructed reward models**: complete through Slurm job `4558`.
7. **PPO training topology smoke**: complete through Slurm jobs `4560` and `4561`.
8. **Full compatibility PPO training run**: complete through Slurm job `4562`; exit `0:0`, 100 iterations, 99,000 environment steps, 11 checkpoints, and 11 matched 10,000-sample evaluations.
9. **Paper-level metric reproduction**: not established.

## Main-text comparison targets

The paper evaluates 10,000 generated samples before training and every 10 epochs.
The principal reported checkpoints are:

| Stage | Validity | Novelty | Diversity | Uniqueness | Frag | SNN |
|---|---:|---:|---:|---:|---:|---:|
| Untrained | 0.4626 | 89.5% | 0.848 | 0.992 | 0.486 | 0.306 |
| Epoch 10 | 0.7577 | 88.2% | 0.845 | 0.977 | 0.480 | 0.310 |
| Epoch 30 | 0.9822 | 87.3% | 0.836 | 0.612 | 0.397 | 0.307 |
| Epoch 50 | 0.9973 | 94.8% | 0.691 | 0.087 | 0.147 | 0.277 |
| Epoch 70 | 0.9989 | 82.1% | 0.328 | 0.024 | 0.014 | 0.342 |

These are comparison targets, not acceptance evidence until the same metric
implementations, sample counts, checkpoint identities, and seeds are verified.
