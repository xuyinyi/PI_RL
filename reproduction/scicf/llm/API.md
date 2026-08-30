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
