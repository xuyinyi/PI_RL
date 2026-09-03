# Private LLM API boundary

Future SciCF Gate 1 runs use a private OpenAI-compatible
`POST /chat/completions` API instead of loading an LLM on n001. The historical
local-Qwen Gate 1 result remains frozen and is not overwritten.

## Security contract

- Put API settings in a private file on n001 with mode `0600` or stricter.
- Never pass the API key as a command-line argument or commit the populated
  file. The tracked `reproduction/configs/scicf-api.env.example` contains only
  placeholders.
- The client rejects credentials embedded in the URL, group/world-readable
  credential files, missing model revision/deployment identity, and plain HTTP
  unless it is explicitly enabled for a trusted private network.
- Manifests contain only a provider ID, model ID, immutable model/deployment
  revision, endpoint SHA-256, decoding settings, request IDs, token accounting,
  and response fingerprints. They never contain the key or credential path.

## DeepSeek Flash profile

For the official DeepSeek low-cost path, provision the private file with
`reproduction/slurm/provision_scicf_api_credentials.sbatch`. The fixed profile
uses `https://api.deepseek.com/chat/completions`, model alias
`deepseek-v4-flash`, JSON-object response mode, no unsupported `seed` field,
and `thinking.type=disabled`. Record a dated provider release/snapshot marker as
the model revision because the public alias can move; the response fingerprint
is retained per call.

Credential provisioning and API acquisition both fail unless they run inside
Slurm. A separate one-request smoke job is available at
`reproduction/slurm/smoke_scicf_gate1_api.sbatch`; it uses a cache and output
namespace separate from the formal 24-request acquisition.

## Scientific and reproducibility contract

- Candidate pools, gain blinding, exact acquisition budget, response schema,
  retries, and fail-closed membership validation are unchanged.
- The API model ranks candidates only. It is not a reward, value, advantage,
  or scientific truth source.
- `run-intent.json` is written before the first API request. Successful
  responses are cached by prompt, provider identity, model revision, and
  decoding identity.
- API outputs and decisions use a user-supplied `RUN_ID` and never overwrite
  the frozen local-Qwen output.
- A formal API result remains a new Gate 1 experiment. It does not retroactively
  change the archived local-Qwen no-go decision.

## Server invocation

After creating the private credentials file and choosing a unique run ID:

```bash
cd /home/wch/workspaces/DAPiGen-reproduction/algorithm-worktree
sbatch reproduction/slurm/rank_scicf_gate1_api.sbatch \
  /home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1-formal-ad3889d-v1 \
  /absolute/private/path/scicf-api.env \
  RUN_ID
```

Only after that ranking job completes successfully:

```bash
sbatch reproduction/slurm/aggregate_scicf_gate1_api.sbatch \
  /home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1-formal-ad3889d-v1 \
  RUN_ID
```

Both commands must be submitted on n001. The API ranking job requests no GPU.

## Validation evidence

- Slurm job 4589: `COMPLETED 0:0`, 22/22 tests passed. The HTTP integration
  test forced one invalid invented ID, verified exact-pool repair, then proved
  identical prompt/model/decoding identity reused the successful cache without
  another API call. It also verified token accounting and that neither the API
  key nor credential path appeared in the manifest.
- Slurm job 4591: `COMPLETED 0:0`, both API Slurm scripts passed `bash -n` and
  the API Gate 1 configuration passed JSON parsing.
- Job 4590 is an excluded diagnostic: its ad hoc validation command
  mistakenly passed the non-JSON `.env.example` file to `jq`. No implementation
  failure or scientific output came from that job.
- Slurm job 4601: `COMPLETED 0:0`, 23/23 tests passed after adding the official
  DeepSeek Flash profile, Slurm-only secret provisioning, disabled thinking,
  omitted seed, and JSON-object mode. Job 4602 passed shell validation.
- Jobs 4603-4606: credential provisioning, one-request smoke, 24-request formal
  ranking, and aggregation all completed successfully. All 24 formal responses
  validated on the first attempt with no transport retry. The resulting
  independent Gate 1 decision is failed/no-go (0/3 successful stages), and
  pairwise refinement remains unauthorized.
