# LLM-SciCF integration-v2 authorized real-run attempt

## Decision

- Protocol: `dapigen-scicf-single-iteration-integration-smoke-v2`
- Frozen protocol coordinate:
  `76eef775c213c3bfefb99d627068567a915a5a6b`
- Implementation coordinate:
  `05c92e587b9e50c1f23b83f271c95243724da33c`
- Authorization SHA-256:
  `9366af4f16457c4aac6d0832a72bfd0a4d9d79c229cf35c683764d7e1c568c95`
- Execution: n001 (`yanlih100n1`) Slurm Job 4693, `FAILED 1:0`
- Decision: `no_go_environment_preflight_failure_before_ppo`

The one-run authorization was exercised exactly once and is consumed. No
automatic rerun was submitted. Multi-iteration and formal training remain
closed.

## What ran

The Slurm allocation exposed one H100 GPU, eight CPUs, and 96 GiB RAM. Its test
step completed with 52 passing tests in 4.74 seconds. The real-runner step then
failed after two seconds while constructing the Stage0 runtime.

The submitted `--polybert-path` was:

`/home/wch/workspaces/DAPiGen-reproduction/scicf-single-iteration-integration-v2-runner-dev/QSPR/polyBERT`

That directory exists, so the submission guard passed, but it is the DAPiGen
polyBERT Python source package. It contains no Hugging Face `config.json`,
tokenizer assets, or model weights. `AutoTokenizer.from_pretrained()` therefore
failed before the PPO engine could be constructed.

## Exact execution boundary

The authorization and repository binding passed and `run-intent.json` was
written. The private credential configuration was then successfully parsed,
which is evidenced by the subsequent non-sensitive
`provider-public-identity.json`. Neither the API key nor the credential path was
logged.

The source order and retained outputs establish the following terminal state:

- DeepSeek provider: `deepseek-official`
- Model: `deepseek-v4-flash`
- Public revision: `DeepSeek-V4-Flash-0731-api-snapshot-2026-09-03`
- external API transmissions: 0
- PPO iterations started: 0
- acquisition pools built: 0
- Oracle evaluations: 0
- pairwise updates: 0
- sealed-test accesses: 0
- baseline mutations: 0

The API client is created only after `build_runtime`, the PPO iteration, and
both candidate pools. The exception occurred inside `build_runtime`; no guarded
acquisition directory or PPO result exists. Consequently this is an engineering
environment-preflight failure, not an LLM-schema, PPO-numerics, Oracle, or
algorithm-effectiveness result.

## Model-path diagnosis

The existing accepted Stage0 compatibility asset is at:

`/home/wch/workspaces/DAPiGen-reproduction/stage0-v23-mask-dev-20260904/RL_PPO/models`

All required configuration, tokenizer, SentencePiece, and weight files are
present there. Their hashes match the existing reconstructed-asset gate; the
14 non-cache files give a sorted-manifest SHA-256 of
`30108fb589d941ab445f3560e27d41290e00fa13a213f0c071414bbe936571d6`.
The selected hashes and the original-path failure are recorded in
`polybert-path-audit.json`. The prior gate labels the asset ready for the
reconstructed AFP compatibility baseline while retaining
`polybert_identity_verified=false` for upstream-source identity.

## Required next gate

A future real attempt must not reuse this authorization or output directory.
Before another request for authorization, development should:

1. make the runner and Slurm guard fail before credential loading unless the
   complete polyBERT required-file set is present and bound to accepted hashes;
2. bind the selected model directory fingerprint in the run intent;
3. run a new mock/preflight through n001 Slurm at a new clean implementation
   coordinate; and
4. request a fresh exact one-run authorization and unique output directory.

This archive does not authorize those changes, a new real attempt, automatic
rerun, multi-iteration training, sealed-test access, or any performance claim.

## Evidence hashes

- Job 4693 stdout:
  `2062fc7928c9ea77253184d18529c6b08c17949295e568e44ef63e27ba0b07c4`
- Job 4693 stderr:
  `ccbff4e3867f497d9c4648fdbdabbf51bae5c4610c9346597681d0e0ba835f6a`
- Run intent:
  `abf75fac7a27bf1d6e8599c849137fa0ed3e627c9af9a23530e3cc91de56cc0e`
- Provider public identity:
  `88e53b88fb3d2ee289d552b1f717c8974088ffc04d89cfbbd8c4a0c3066d844d`

The submitted worktree remained clean after the failure.
