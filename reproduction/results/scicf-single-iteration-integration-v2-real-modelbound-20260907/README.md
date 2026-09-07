# LLM-SciCF integration-v2 model-bound real-run attempt

## Decision

- Frozen protocol:
  `dapigen-scicf-single-iteration-integration-smoke-v2`
- Protocol SHA-256:
  `f05df22b15c069f43b6e9cc5013550a97b6c46f79db059e3674f5395de8c3955`
- Implementation coordinate:
  `46aef57c693a4644a52bdfcd337cbc90184ac31e`
- Authorization schema: 2
- Authorization SHA-256:
  `22d7550b22132f9f33f4bfc4730aabf279a509f1e8ade6725dad9ac42ea26ad4`
- Execution: n001 (`yanlih100n1`) Slurm Job 4695, `FAILED 1:0`
- Server tests: 54 passed in 4.75 seconds
- Decision: `no_go_environment_preflight_failure_before_ppo`

The newly authorized real attempt was submitted exactly once. Its authorization
is consumed and no automatic rerun was submitted. Multi-iteration and formal
training remain closed.

## What passed

The new polyBERT boundary worked as intended. Before credentials were loaded,
the runner validated:

- exact resolved model path;
- all 14 required-file hashes;
- asset-binding SHA-256
  `7e2f1cf5caa38d3d42e7ebe2599435ec306aa74e2cffeb9c2cac23408876a7ed`;
- full checkpoint fingerprint
  `6bdd24f951dd90d3031e749ef0130752811bfefe6c850af82b805cf015ea195f`;
- accepted Stage0 encoder version
  `polybert-sha256:6bdd24f951dd90d3031e749e`; and
- schema-v2 authorization bindings for the exact path and fingerprint.

The run intent retains these values and shows a clean source coordinate. The
private credential configuration was subsequently parsed, and only the
non-sensitive DeepSeek public identity was written. No API key or credential
path was logged.

## Failure

The Stage0 runtime then attempted to construct
`PersistentDAPiGenBenchmarkEvaluator`. Unlike polyBERT, its model directory is
currently derived from the source checkout root. The implementation worktree
contained only `__init__.py` and `fpscores.pkl.gz` under `RL_PPO/GNN/model` and
did not contain the ignored reconstructed AFP compatibility assets.

Construction stopped on the first missing file:

`RL_PPO/GNN/model/Ensemble_transmittance(400)_AFP_43.pt`

The accepted copies of all 12 required AFP weights, scalers, and settings files
remain present under the accepted Stage0 worktree. Their hashes are recorded in
`evaluator-asset-audit.json`. This is an evaluator-asset routing and preflight
gap, not a polyBERT, LLM-schema, PPO-numerics, Oracle, or effectiveness result.

## Exact execution boundary

- polyBERT validation: passed
- credentials configuration loaded: yes
- polyBERT weights loaded: yes
- initial embedding path entered: yes
- evaluator construction completed: no
- DeepSeek API transmissions: 0
- PPO iterations started: 0
- candidate pools built: 0
- Oracle evaluations: 0
- pairwise updates: 0
- sealed-test accesses: 0
- baseline mutations: 0

The API client is created only after complete runtime construction, one PPO
iteration, and both candidate pools. The exception occurred during runtime
construction, so the provider was not contacted and no API cost was incurred.

## Required next gate

A future attempt must not reuse this authorization, implementation coordinate,
or output directory. Before another real authorization request, development
must:

1. separate the evaluator-asset root from the source checkout or install an
   exact read-only binding to the accepted compatibility assets;
2. validate all 12 AFP weights/scalers/settings and `fpscores.pkl.gz` before
   credential loading, including exact hashes and compatibility identity;
3. bind the resolved evaluator-asset path and manifest in the next authorization
   schema and run intent;
4. make a no-credential n001 Slurm preflight construct the complete Stage0
   runtime, not only hash polyBERT; and
5. request another fresh, exact one-run authorization only after that preflight
   passes.

This archive authorizes none of those changes or executions.

## Evidence identities

- Job 4695 stdout:
  `d3a4d21f62750b554de66f58c16e7383b5f04684003197c442009a9590174c73`
- Job 4695 stderr:
  `916cc332ab82f550cc193585b82bda38878a3b99a7b6e5d1ab2d8f459d94fadb`
- Run intent:
  `f259c74f838b0d4cfdb033f2c7b9c9863be6e17f95983a9b33684649125333d0`
- Provider public identity:
  `88e53b88fb3d2ee289d552b1f717c8974088ffc04d89cfbbd8c4a0c3066d844d`

The failed worktree remained clean.
