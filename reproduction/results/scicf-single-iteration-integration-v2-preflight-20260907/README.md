# LLM-SciCF single-iteration integration-v2 runner preflight evidence

## Decision

- Protocol: `dapigen-scicf-single-iteration-integration-smoke-v2`
- Frozen protocol coordinate:
  `76eef775c213c3bfefb99d627068567a915a5a6b`
- Final implementation coordinate:
  `05c92e587b9e50c1f23b83f271c95243724da33c`
- Final execution: n001 (`yanlih100n1`) Slurm Job 4692, `COMPLETED 0:0`
- Server tests: 25 passed
- Preflight decision:
  `go_request_separate_real_single_iteration_execution_authorization`
- Real single-iteration execution, automatic rerun, multi-iteration training,
  formal training, sealed-test access, effectiveness claims, and scientific
  claims: not authorized

The implementation and mock-only preflight pass. This decision permits only a
request for a separate real single-iteration execution authorization. It is not
that authorization and no execution manifest has been created.

## Implemented runner boundary

- A separate exact-scope authorization manifest is mandatory and is validated
  before the private credentials file can be loaded.
- The authorization binds the frozen protocol SHA-256, exact implementation
  commit, one output directory, and exactly one Slurm run.
- Both 24-candidate pools are built and persisted before the first provider
  call. Both guarded decisions must validate before selected-only Oracle
  verification can start.
- Each pool has at most two schema attempts, one semantic repair, and four HTTP
  transmissions; the complete run has an eight-transmission upper bound.
- Schema or transport exhaustion stops before Oracle verification and pairwise
  refinement. Partial selection from another pool is discarded.
- A valid abstention remains valid but does not receive fallback padding.
- Attempt artifacts, invalid raw model content, validation receipts, token
  accounting, and terminal failure reports are retained without logging the API
  key or credentials path.
- Unknown transport-failure token usage is recorded as incomplete with null
  token counts; it is not imputed as zero.

## Final mock-only verification

Job 4692 used six scripted in-memory completions across three orchestration
scenarios:

1. invented ID, successful repair, then a valid abstention in the second pool;
2. two invalid schema responses, stopping before the second pool and Oracle;
3. transport exhaustion, stopping before semantic repair, the second pool, and
   Oracle, with unknown token usage retained as incomplete.

All 19 declared report checks are true. The closed authorization fixture was
rejected, and static ordering verified that authorization is checked before
credential loading. External API requests, credential loads, PPO execution,
Oracle execution, local-model invocation, and sealed-test access were all zero
or false.

## Superseded diagnostics

- Job 4690 failed before pytest collection because the submitted base Python
  environment did not expose pytest. It performed no test, API, PPO, or Oracle
  work.
- Job 4691 passed 25 tests and the main orchestration checks, but its transport-
  exhaustion fixture exposed an accounting defect: absent provider token usage
  was represented as complete zero usage. That report is retained under
  `superseded-run-4691/` and is not the accepted preflight.
- Commit `05c92e5` corrected the accounting without changing the protocol,
  response guard, PPO, candidate, Oracle, pairwise, seed, or thresholds. Job
  4692 is the accepted preflight.

## Evidence identities

- Frozen protocol JSON:
  `f05df22b15c069f43b6e9cc5013550a97b6c46f79db059e3674f5395de8c3955`
- Final real-runner source:
  `7eb9f5a2e84e7b445e9c4c49aed2a1da92b522220df2844dc872cda399dbef1d`
- Final preflight report:
  `e3a9d79b0965653967a92bcf72b57a45f8b99fa82abbee7034f9970b7f25d891`
- Final run intent:
  `a7f7ca1288c5dfd5ea974ebdbaed488d8535d029b4af3359718d8e966fa72bc3`
- Job 4692 stdout:
  `871894e44a9572c5bdb2757378d959378df214ac568e6b86113b2a1c3b12742b`
- Job 4692 stderr is empty, SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Credential-pattern scanning across the complete archived preflight evidence
returned no match.

## Claim and execution boundary

This is mock-only runner engineering evidence. It does not show that a real
DeepSeek response will validate or repair, that the PPO/Oracle numerical path
will complete, or that SciCF improves PPO. A real run requires a new, explicit,
one-run authorization and a unique manifest bound to the protocol and final
implementation commit. Multi-iteration training remains closed even if that
future single run passes.
