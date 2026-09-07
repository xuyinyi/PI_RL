# SciCF integration-v2 AFP evaluator asset and full-runtime preflight

> Superseded reporting evidence. Job 4696 passed the functional boundary, but
> its report used the over-broad field `model_inference_executed=false` even
> though Stage0 construction performs two initial polyBERT embeddings. Corrected
> Job 4697 at commit `39cd7fb` is the accepted preflight evidence. AFP property
> inference, evaluator calls, credentials, API requests, PPO, and Oracle were
> zero in both jobs.

## Decision

- Frozen protocol:
  `dapigen-scicf-single-iteration-integration-smoke-v2`
- Protocol SHA-256:
  `f05df22b15c069f43b6e9cc5013550a97b6c46f79db059e3674f5395de8c3955`
- Implementation coordinate:
  `845b1e2897d108a48d51c1688078327c990aeff8`
- Execution: n001 (`yanlih100n1`) Slurm Job 4696, `COMPLETED 0:0`
- Server tests: 31 passed in 2.33 seconds
- Preflight checks: 31 passed, 0 failed
- Decision: `go_request_separate_real_single_iteration_execution_authorization`

This decision permits only asking for a new exact one-run authorization. It
does not create that authorization and does not authorize credential loading,
an external DeepSeek request, PPO, Oracle evaluation, an automatic rerun, or
multi-iteration training.

## Implemented boundary

The real integration-v2 runner now requires authorization schema version 3.
In addition to the existing protocol, implementation, output-directory, and
polyBERT bindings, it binds:

- the exact resolved AFP evaluator asset directory;
- the evaluator-asset binding SHA-256;
- a canonical fingerprint over all required evaluator assets; and
- exact hashes for four AFP weights, four scalers, four settings files, and
  `fpscores.pkl.gz`.

The asset validator also rejects missing, empty, changed, and unmanifested
top-level evaluator assets. All validation and authorization checks occur before
the credentials loader in the real runner.

The only Stage0 source delta relative to accepted commit `373b329` is
`RL_PPO/envs/evaluator.py`. It adds an optional explicit `model_dir` route to
`PersistentDAPiGenBenchmarkEvaluator`; the legacy default remains the checkout's
`RL_PPO/GNN/model` directory. The frozen baseline branch was not modified.

## Full-runtime evidence

Job 4696 used one Slurm-allocated H100 and constructed:

- the accepted polyBERT encoder from the exact 14-file checkpoint fingerprint;
- all four AFP property models and scalers from the explicit external directory;
- the SA fragment-score table;
- `Stage0Components` with environment ID
  `dapigen:8004c1fa2055a186b4a4f3ed`; and
- the SciCF `PPOEngine` and its method run contract.

The runtime-reported evaluator version remained
`dapigen-persistent-qspr-v2:4e76ba44f9e1e7a0`. Its loaded 13-file hashes exactly
matched the prevalidated binding, and its initial requested, unique, and backend
evaluator-call counters were all zero.

## Exact no-execution boundary

- credential-file argument accepted by preflight: no
- credentials loaded: no
- external API requests: 0
- PPO iterations executed: 0
- Oracle evaluations: 0
- evaluator requested/unique/backend calls: 0/0/0
- local models loaded: yes
- model inference executed: no
- sealed-test accesses: 0
- baseline mutations: 0
- real-run authorization created: no

The six completion objects used for schema-guard checks were scripted in-memory
mocks. They were not network transmissions.

## Bound evaluator asset identity

- Binding ID: `dapigen-reconstructed-afp-compatibility-evaluator-v1`
- Binding SHA-256:
  `67cd8ed2c2cf29c823985a889033bc7ee3f0c898b82e32d992c2c34917d80eb9`
- Asset fingerprint:
  `0bdcea6155f5a53acd94afd322d408fe3e31c02cf13778a0c65f5f0938096ab4`
- Asset classification: `reconstructed AFP compatibility baseline`
- Original author weights: false

## Evidence identities

- Preflight report:
  `933067f729af9133fc21994f82987ea4959a6e13f921f374d11bfa7dbf696d35`
- Run intent:
  `2f480fde028e7604f56c5b315f3c9dfec9c198da6ec4b5caac3bce0703d92759`
- Slurm stdout:
  `a093dfadd5b4f37fa9b0e45afac0515593aaed4716664667988686674b9d0b80`
- Slurm stderr:
  `ebdac9c5d016af0150d1675d442bf3d3d6b351d760552679702af37641a37e01`
- Routed evaluator source:
  `90b27f5947dc9fbe9c6b4220f2fbbbb43ad82ff4ca552d7e7ac7ce82c6e420e1`

The n001 implementation worktree remained clean.
