# LLM-SciCF integration-v2 polyBERT asset preflight evidence

## Decision

- Frozen protocol:
  `dapigen-scicf-single-iteration-integration-smoke-v2`
- Protocol SHA-256:
  `f05df22b15c069f43b6e9cc5013550a97b6c46f79db059e3674f5395de8c3955`
- Model-asset implementation coordinate:
  `46aef57c693a4644a52bdfcd337cbc90184ac31e`
- Execution: n001 (`yanlih100n1`) Slurm Job 4694, `COMPLETED 0:0`
- Server tests: 29 passed in 4.40 seconds
- Preflight checks: 23 passed, 0 failed
- Module decision:
  `go_request_separate_real_single_iteration_execution_authorization`

The model-integrity repair and mock-only preflight pass. This decision permits
only a request for a new exact one-run real-execution authorization. No such
authorization was created in this stage, and no real run was submitted.

## Repair implemented

The runner now validates polyBERT before creating the run output, loading
credentials, constructing the PPO runtime, or creating the API client. The
validation binds all three of the following:

1. the exact resolved server model path;
2. the repository model-asset binding file, SHA-256
   `7e2f1cf5caa38d3d42e7ebe2599435ec306aa74e2cffeb9c2cac23408876a7ed`;
3. the accepted full-checkpoint fingerprint
   `6bdd24f951dd90d3031e749ef0130752811bfefe6c850af82b805cf015ea195f`.

The full-checkpoint fingerprint uses the existing Stage0 algorithm and covers
every file except `.git` and `__pycache__`. It therefore covers the Hugging Face
download metadata and lock files whose omission previously changed the accepted
environment identity. In addition, the validator independently checks the 14
manifested core files for presence, non-zero size, and exact SHA-256.

The verified server asset is:

`/home/wch/workspaces/DAPiGen-reproduction/stage0-v23-mask-dev-20260904/RL_PPO/models`

It resolves to encoder version `polybert-sha256:6bdd24f951dd90d3031e749e`, matching
the accepted Stage0 manifest. The binding continues to declare
`upstream_polybert_identity_verified=false`; this compatibility asset is not
relabelled as the original upstream checkpoint.

## Authorization hardening

Any future real-run authorization must use schema version 2 and bind:

- the frozen protocol hash;
- the exact implementation commit;
- one unique output directory;
- the exact resolved polyBERT path;
- the asset-binding-file hash;
- the full checkpoint fingerprint;
- exactly one Slurm run and the unchanged bounded operation map.

Changing the path, binding hash, fingerprint, implementation coordinate,
output directory, or operation scope causes fail-closed rejection. The consumed
Job 4693 authorization is schema version 1 and cannot authorize this runner.

## Mock/preflight coverage

The server tests include negative cases for a missing required model file, an
unmanifested full-tree change, an authorization-path mismatch, and an
authorization-fingerprint mismatch. The live preflight then hashed the complete
accepted server checkpoint and verified its binding to the accepted Stage0
encoder before running the existing bounded response-guard scenarios.

All 23 report checks are true, including model validation before credential
loading, complete fingerprint and required-file binding, closed-authorization
rejection, successful repair plus abstention, schema exhaustion, and transport
exhaustion without token imputation.

## Zero-execution boundary

- private credential file supplied: no
- credentials loaded: false
- external API requests: 0
- local model invocation: false
- PPO execution: false
- Oracle execution: false
- sealed-test access: false
- automatic rerun authorization: false
- multi-iteration authorization: false
- formal-training authorization: false
- algorithm effectiveness established: false

The sensitive-pattern scan over the complete output and Slurm logs returned no
match. Job 4694 stderr is empty and the submitted worktree remained clean.

## Evidence identities

- Model-asset validator:
  `acaff54e319d0f21870bf0eb551a9ed6a2305382e57fc7775c666cd33ae0564a`
- Real runner:
  `24a6309cecca7cac4e42dc1fbdfd76f455efa2ec50b81ad4d138acf59c307c34`
- Preflight runner:
  `4d4d90096e1cfa82e4e9bd042a21bcc8db6b267950ccd0b63bfc618722f0776f`
- Preflight report:
  `29276f4a201744c4b938a474cc392592bce649c2debc5e9e9554262f7f184d60`
- Run intent:
  `196c1fb213289727da4a338964d1155c226614998323270bcd9d2a317c752b42`
- Job 4694 stdout:
  `3a84e05fd979e2361e5856990463de3427d170572a44adcffcbee923e0d3c11b`
- Job 4694 stderr:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

## Stop boundary

This preflight authorizes only requesting a new one-run authorization. It does
not authorize creating that manifest, loading credentials, making a DeepSeek
request, executing PPO or Oracle verification, rerunning automatically, opening
multi-iteration training, or making an effectiveness or scientific claim.
