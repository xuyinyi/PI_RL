# DAPiGen common algorithm framework

This framework compares optimization algorithms without changing the frozen
chemistry task. The baseline coordinate is the annotated Git tag
`dapigen-ppo-compat-baseline-v1`.

## Frozen and replaceable layers

The following task semantics are frozen: building-block tables, 1,200-D
polyBERT observation, tuple action space, BRICS transition, PI reaction,
episode limits, AFP compatibility assets, paper Equation (1), and common metric
definitions. An algorithm adapter may replace the policy model, optimizer,
exploration rule, replay/on-policy storage, or update rule.

Observation augmentation, action masking, reward shaping, alternative AFP
models, or reaction changes define a task-interface variant. Such variants must
use a separate configuration and cannot be pooled with pure algorithm results.

## Adapter contract

Implement `AlgorithmAdapter` from `contracts.py` and expose the class through a
`module:Class` locator in the experiment JSON. The required methods are:

```text
initialize(context, parameters)
train_step(target_environment_steps) -> cumulative normalized metrics
act(observation, explore=False)
save(checkpoint_dir) -> checkpoint locator
restore(checkpoint_locator)
effective_config()
close()
```

`train_step()` must report cumulative `environment_steps`. The common runner
rejects missing, non-increasing, or overshooting values at evaluation budgets.
This prevents comparisons by ambiguous algorithm-specific "epochs".

## Common evaluator

`CommonEvaluator` always uses `explore=false`, an explicit evaluation seed, and
the same CSV schema. Its `dapigen-common-v1` summary contains validity,
canonical uniqueness, mean/max reward, novelty, diversity, Frag, and SNN.

The protocol uses Morgan fingerprints with radius 3 and 2,048 bits. Novelty is
the fraction of valid generated molecules whose maximum training-set Tanimoto
similarity is below 0.4. Diversity is one minus mean pairwise generated-set
Tanimoto similarity. Frag and SNN reuse the repository metric implementations.

Existing generation CSVs can be evaluated with:

```bash
python reproduction/evaluate_generated_csv.py \
  --input-csv /path/to/generate.csv \
  --reference-csv raw_data/PI.csv \
  --output-json /path/to/evaluation-summary.json
```

Formal offline evaluations can use
`slurm/evaluate_generated_csv.sbatch`; its output also records the evaluator
Git identity and Slurm job id. The Slurm entry point enables `--formal`, which
fails if either the allocation or clean-worktree requirement is absent.

## Slurm execution

Formal runner configurations require a clean Git worktree descended from the
frozen baseline tag and a Slurm allocation. Direct interactive training fails
closed. On n001 the generic runner uses the separate clean worktree
`/home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree`; the frozen
baseline worktree remains untouched.

```bash
mkdir -p /home/wch/workspaces/DAPiGen-reproduction/logs
sbatch reproduction/slurm/run_algorithm.sbatch \
  reproduction/configs/ppo-compat-common-v1.json \
  /home/wch/workspaces/DAPiGen-reproduction/runs/algorithms/ppo/seed-2023 \
  ppo-compat-common-seed-2023
```

Use `--seed N` after the three positional arguments to override the declared
seed while retaining it in `resolved-config.json` and `run-manifest.json`.

Each run writes:

```text
run-manifest.json
resolved-config.json
algorithm-config.pkl
algorithm-config-summary.json
training-metrics.jsonl
evaluation-metrics.jsonl
checkpoints/step_*/
evaluations/step_*/generate.csv
native-logs/
```
