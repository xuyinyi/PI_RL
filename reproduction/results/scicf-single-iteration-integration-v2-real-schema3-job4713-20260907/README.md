# SciCF integration-v2 schema-3 real single-iteration evidence (Job 4713)

## Decision

The one authorized real n001 Slurm execution completed normally, but the
frozen integration gate returned:

```text
no_go_multi_iteration_protocol_freeze
```

This is an engineering-gate no-go, not a Slurm or integrity failure. The full
runtime, standard PPO update, two guarded DeepSeek acquisitions, selected-only
Oracle verification, terminal report, and checkpoint all completed. However,
none of the eight selected interventions produced a non-zero, sign-consistent
paired delta across the two frozen matched seeds. The accepted pair count was
therefore zero, so pairwise refinement was correctly skipped.

No automatic rerun was submitted. Multi-iteration training, formal training,
sealed-test access, baseline mutation, effectiveness claims, and scientific
claims remain unauthorized.

## Exact execution coordinate

- Host: `yanlih100n1` (SSH alias `n001`)
- Slurm Job: `4713`
- Slurm state / exit: `COMPLETED`, `0:0`
- Slurm elapsed / allocation: `00:00:49`, 8 CPU, 96 GiB, 1 GPU
- Implementation: `39cd7fb33c1c88397e78808860d766c9876c12df`
- Source dirty: `false`
- Protocol ID: `dapigen-scicf-single-iteration-integration-smoke-v2`
- Protocol SHA-256:
  `f05df22b15c069f43b6e9cc5013550a97b6c46f79db059e3674f5395de8c3955`
- Authorization ID:
  `scicf-integration-v2-schema3-39cd7fb-one-real-slurm-run-20260907`
- Authorization SHA-256:
  `d467b4efff6f576a09928abed2789e9bebc60ee914667c0f71cf3f60ee0706d8`
- Authorized unique output:
  `/home/wch/workspaces/DAPiGen-reproduction/runs/scicf-single-iteration-integration-v2-real-schema3-39cd7fb-20260907`

The submitted server tests passed: `56 passed in 4.73s`. The runner-reported
elapsed time was `41.06083119288087` seconds and peak RSS was
`1600.46484375` MiB.

## Bound model assets

The runner validated both model families before credentials:

- polyBERT binding SHA-256:
  `7e2f1cf5caa38d3d42e7ebe2599435ec306aa74e2cffeb9c2cac23408876a7ed`
- polyBERT checkpoint fingerprint:
  `6bdd24f951dd90d3031e749ef0130752811bfefe6c850af82b805cf015ea195f`
- AFP evaluator binding SHA-256:
  `67cd8ed2c2cf29c823985a889033bc7ee3f0c898b82e32d992c2c34917d80eb9`
- AFP evaluator asset fingerprint:
  `0bdcea6155f5a53acd94afd322d408fe3e31c02cf13778a0c65f5f0938096ab4`
- AFP boundary: 13 files, `reconstructed-afp-compatibility`, original author
  weights `false`

The exact evaluator source delta remained limited to
`RL_PPO/envs/evaluator.py` and the report recorded no integrity failures.

## Executed work and accounting

### Standard PPO

- Transitions: 128
- Successful terminal episodes: 27
- On-policy terminal evaluator calls: 27 requested / 27 unique / 27 backend
- PPO optimizer steps: 8
- Policy version: `0 -> 1`
- Approximate KL: `0.008422786369919777`
- Clip fraction: `0.05078125`
- Actor loss: `-0.028126218356192112`
- Value loss: `0.1083287859801203`

The post-PPO policy hash differed from the initial policy hash, so the standard
PPO update itself was real and completed.

### Guarded DeepSeek acquisition

- Provider/model: `deepseek-official` / `deepseek-v4-flash`
- Revision: `DeepSeek-V4-Flash-0731-api-snapshot-2026-09-03`
- Pool decisions: 2 of 2, each over 24 blinded candidates
- Selected: 4 per pool, 8 total
- HTTP transmissions: 2 total; frozen maximum was 8
- Semantic attempts: 2 total
- Schema repairs: 0
- Tokens reported: 15,481 prompt and 263 completion
- Both outcomes validated before Oracle execution

No local LLM was invoked. The API key and credential path were not written to
the run artifacts.

### Selected-only matched Oracle verification

- Selected candidates verified: 8
- Replicates per candidate: 2
- Completed factual/counterfactual branches: 32 of 32
- Terminal AFP evaluations: 12
- Verification evaluator calls: 12 requested / 12 unique / 12 backend
  (`7` counterfactual and `5` factual)
- Accepted sign-consistent pairs: 0 (`0` positive, `0` negative)
- Unselected candidates verified: `false`

The two replicate deltas for the eight selected candidates were:

```text
cf-50cc6aeebb4bb7b0   [ 0.0000, -0.0220]
cf-f5bd7c8579d79372   [-0.1618,  0.0934]
cf-170ca968d1effe63   [ 0.0000,  0.0000]
cf-eb2c755c0867c3a8   [ 0.0017, -0.1802]
cf-9eaa24c0e992c61f   [ 0.4239,  0.0000]
cf-924bf3bdc10273e4   [-0.1011,  0.0000]
cf-710f4dd423428638   [ 0.0000,  0.3349]
cf-db42dfb825ada4e5   [ 0.0000,  0.0000]
```

Each record was rejected as `zero_or_sign_inconsistent_delta`. This directly
caused the three declared integration failures:

1. `insufficient_sign_consistent_pairs`;
2. `pairwise_update_did_not_change_post_ppo_policy`;
3. `pairwise_update_not_applied_once`.

### Pairwise and total budget

- Pairwise optimizer steps: 0
- Pairwise status: `skipped_no_verified_pairs`
- Policy version after pairwise: 1 (unchanged from post-PPO)
- Value-head parameters changed: `false`
- Maximum measured policy/value drift from pairwise: 0
- Final evaluator ledger: 39 requested / 39 unique / 39 backend, under the
  frozen 512-call ceiling

## Integrity and scope interpretation

`execution_status: passed` means the bounded engineering runner completed and
produced all required receipts. It does not override the authoritative
`module_decision: no_go_multi_iteration_protocol_freeze`.

The report records:

- integrity failures: none;
- sealed test accessed: `false`;
- multi-iteration training authorized: `false`;
- formal training authorized: `false`;
- next scope authorized: `false`;
- algorithm effectiveness established: `false`;
- scientific claim authorized: `false`.

The evidence supports only a diagnosis: under the frozen `K=2` strict
sign-consistency rule, this single pair of pools yielded no usable refinement
pair. It does not establish that SciCF is ineffective, and it does not justify
silently relaxing the rule or rerunning until a favorable pool appears. Any
change to replication, acceptance, pool selection, or pair construction needs
a separately frozen development protocol and separate authorization.

## Artifacts

- `execution-authorization.json`: exact consumed schema-3 authorization
- `slurm-accounting.txt`: Slurm job and step accounting
- `logs/`: immutable stdout/stderr copies
- `run/run-intent.json`: pre-credential source/protocol/asset declaration
- `run/provider-public-identity.json`: non-secret provider identity
- `run/guarded-acquisition/`: request, raw response, validation, and outcome
  receipts for both decisions
- `run/pools/` and `run/requests/`: frozen blinded candidate-pool artifacts
- `run/oracle-verification.json`: all selected-only paired branch receipts
- `run/integration-v2-report.json`: authoritative terminal report; SHA-256
  `1e650475e261e1100ef2270d1efaf1090aba68bd62e1bc5524c42bd21b15922b`

The 4.6 MiB checkpoint is retained only in the immutable server output and was
not copied into Git. Its recorded SHA-256 is
`67dd8e9a8d753d31602fd7a8a6fa42965db3ef1502346d3c35688e7234ce9e35`.
The credential file was neither copied nor archived. A filename-only scan of
the server output found no API-key assignment, bearer token, key-shaped value,
or private credential path.
