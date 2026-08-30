# DAPiGen original PPO reproduction on n001

This directory records a fail-closed reproduction of the public DAPiGen PPO
implementation. It does not treat an import smoke test, a reconstructed reward
model, or a started PPO job as reproduction of the paper's results.

Project gate: new-algorithm engineering may begin from the frozen reconstructed
compatibility baseline. Comparative scientific claims remain blocked until the
common paper-metric evaluator and matched multi-seed experiments are complete.

## Frozen coordinate

- DOI: `10.1002/adma.202511099`
- Repository: `https://github.com/xuyinyi/DAPiGen.git`
- Branch: `main`
- Commit: `5f692946cbe0d15eede882dfe4cff7fb26eb7d8c`
- Clean upstream on n001: `/home/wch/workspaces/DAPiGen-reproduction/upstream`
- Patch worktree on n001: `/home/wch/workspaces/DAPiGen-reproduction/worktree`
- Conda prefix on n001: `/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38`

## Current status

- Linux/H100 compatibility environment: **ready**.
- PyTorch CUDA matrix multiplication on H100: **passed**.
- DGL CUDA graph smoke: **passed**.
- Gym 0.19, Ray/RLlib 1.13, Transformers 4.20 imports: **passed**.
- Repaired RLlib/environment configuration construction: **passed**.
- Paper Equation (1) weighted-average reward restored in patch worktree: **asset-backed test passed**.
- Original README command: **fails before training**.
- polyBERT files from user-specified mirror: **downloaded; 600-D embedding smoke passed**.
- Mirror identity against the inaccessible primary repository: **not verified**.
- Four selected AFP members, scalers, and settings: **reconstructed and promoted as a compatibility baseline**.
- AFP two-epoch pilot and full training through Slurm: **passed**.
- `Benchmark` load/prediction and environment terminal reset/step through Slurm: **passed**.
- PPO runtime-file gate: **ready**.
- Original-asset scientific gate: **not ready**.
- Reconstructed-asset PPO topology smokes through Slurm: **passed** (jobs `4560` and `4561`).
- Full reconstructed-asset PPO baseline: **completed through Slurm** (job `4562`; exit `0:0`, 100 iterations, 99,000 environment steps).
- Baseline checkpoints/evaluations: **complete** (epochs `0,10,...,100`; 10,000 generated samples per checkpoint).
- Frozen compatibility-baseline record: **complete**; see `baseline-freeze.md`.
- Paper result reproduction: **not established**.

## Network helper on n001

The remote host uses a user-level Mihomo installation only for authorized model
downloads. It is not installed as root and does not expose a LAN listener.

- Version: `Mihomo Meta v1.19.30 linux amd64`
- Binary: `/home/wch/workspaces/DAPiGen-reproduction/tools/mihomo/mihomo`
- Configuration root: `/home/wch/workspaces/DAPiGen-reproduction/tools/mihomo/config`
- HTTP proxy: `127.0.0.1:17890`
- SOCKS proxy: `127.0.0.1:17891`
- Controller: `127.0.0.1:19090`
- Subscription and runtime configuration permissions: `0600`

`prepare_mihomo_config.py` preserves the resolved subscription while replacing
only the runtime listener settings. `select_mihomo_proxy.py` tests candidates
against Hugging Face and selects a working candidate without printing node names.
At the current repository state, public Hugging Face downloads work through the
proxy, while the pinned `kuelumbus/polyBERT` revision returns an authentication
challenge and therefore still requires an authorized Hugging Face login.

## Verified environment smoke

```bash
cd /home/wch/workspaces/DAPiGen-reproduction/worktree
DGLBACKEND=pytorch \
  /home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python \
  reproduction/smoke_core.py
```

## Asset gate

```bash
cd /home/wch/workspaces/DAPiGen-reproduction/worktree
/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python \
  reproduction/check_assets.py
```

The report now distinguishes two gates. The default `original` mode remains
fail-closed because neither the mirror identity nor the reconstructed AFP
weights are verified as the authors' original binary assets. The explicitly
selected `reconstructed-afp-compatibility` mode verifies the archived AFP
validation-report hash, all 12 reward-asset hashes, and all 14 polyBERT mirror
hashes before returning success. Random weights, empty files, or an implicit
fallback never satisfy either gate.

```bash
cd /home/wch/workspaces/DAPiGen-reproduction/worktree
/home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python \
  reproduction/check_assets.py \
  --asset-mode reconstructed-afp-compatibility
```

The reconstruction driver and Slurm entry points are:

```text
reconstruct_afp_assets.py
slurm/reconstruct_afp_array.sbatch
validate_afp_runtime.py
slurm/validate_afp_runtime.sbatch
```

The complete run record, model metrics, hashes, Slurm job ids, and final runtime
report are archived under `results/afp-full-20260830/`.

## PPO compatibility execution

The full run is labeled `reconstructed AFP compatibility baseline`; its manifest
also records `original_asset_reproduction=false`. It uses the public PPO topology
and schedule: 15 rollout workers, four GPUs, batch size 500, FC layers
256/128/128 with ReLU, 100 iterations, checkpoints every 10 iterations, and
10,000 generated samples before training and at every checkpoint. The public
source does not specify a PPO seed, so the full run intentionally leaves it
unset. RLlib 1.13 automatically reconciles the requested rollout fragment length
of 200 with 15 workers and batch size 500 to an effective fragment length of 33.

Slurm jobs `4560` (one-GPU functional smoke), `4561` (15-worker/four-GPU
topology smoke), and `4562` (full compatibility baseline) completed
successfully. The immutable full-run artifacts remain at:

```text
/home/wch/workspaces/DAPiGen-reproduction/runs/ppo-compat/full-20260830-v1
```

The full run contains 11 checkpoints and 110,000 generated evaluation rows.
This is execution evidence for the reconstructed compatibility baseline only.
Paper-result reproduction still requires matched paper-metric and multi-seed
comparison audits.

## Common algorithm framework

New algorithms must use the versioned adapter/evaluator/runner contract under
`framework/`; see `framework/README.md`. Formal runs use
`slurm/run_algorithm.sbatch` and a JSON configuration under `configs/`.

The framework freezes task and reward semantics, requires an explicit seed,
compares algorithms at exact environment-step checkpoints, evaluates with
`explore=false`, and writes a common artifact schema. The original PPO runner is
retained as frozen baseline evidence and is not silently rewritten.

## Configuration-only smoke

This test constructs the repaired RLlib configuration and the environment
object without exercising model inference. The separate Slurm runtime validator
now covers model loading, `reset()`, a generated-PI terminal `step()`, and reward
calculation.

```bash
cd /home/wch/workspaces/DAPiGen-reproduction/worktree
PYTHONPATH=. DGLBACKEND=pytorch \
  /home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python \
  reproduction/smoke_config.py
```

The primary polyBERT coordinate originally frozen for this reproduction is:

```text
kuelumbus/polyBERT@3675d021d0938179b6b57dfcca4753ca34048182
```

That repository is currently inaccessible to the authenticated account. At the
user's direction, the following Hugging Face mirror snapshot is installed under
`RL_PPO/models/`:

```text
xushijie/polyBERT@e7dce434fb3eff37905dc114008660e5479ca9a8
```

The DAPiGen `Embedding_smiles` path loads this snapshot and returns a finite
600-dimensional `float32` vector. Its file hashes are recorded in
`polybert-e7dce434.sha256`. The mirror must not be described as the original
asset until its model hashes can be compared with the primary repository.

The four selected reward-model members and associated scalers/settings are
installed under `RL_PPO/GNN/model/`; see `check_assets.py` for exact filenames.
They are reconstructed compatibility assets, not author-supplied weights.

## Captured unmodified failures

Running `python train.py` from `RL_PPO/moldr`, as shown in the README, first
fails with `ModuleNotFoundError: No module named 'RL_PPO'`. Adding only the
repository root to `PYTHONPATH` then fails with `No module named 'model'`.
Adding the two source roots reaches the deterministic source error:

```text
TypeError: get_default_config() got an unexpected keyword argument 'num_gpus'
```

These failures are upstream evidence, not environment-install failures. See
`source-audit.md` before applying compatibility changes.

## Reward coordinate

The paper's high-transparency baseline retains the weighted-average reward from
main-text Equation (1). The public Python source instead activates its geometric
variant, while an older bundled Python 3.7 bytecode artifact uses Equation (1).
This branch restores the paper equation and records the public-source variant as
a separate implementation state; the two must not be pooled in one baseline.

```bash
cd /home/wch/workspaces/DAPiGen-reproduction/worktree
PYTHONPATH=. DGLBACKEND=pytorch \
  /home/wch/workspaces/DAPiGen-reproduction/envs/dapigen-py38/bin/python \
  reproduction/smoke_reward_formula.py
```
