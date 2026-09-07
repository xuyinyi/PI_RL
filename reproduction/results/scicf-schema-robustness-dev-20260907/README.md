# LLM-SciCF response-schema robustness development evidence

## Decision

- Protocol: `dapigen-scicf-llm-schema-robustness-dev-v1`
- Code coordinate: clean commit
  `5387611e1ba088b8dafd0c50a80751059e8fdfed`
- Execution: n001 (`yanlih100n1`) Slurm Job 4689, `COMPLETED 0:0`
- Server tests: 19 passed
- Five frozen mock scenarios: all passed
- Module decision: `go_separate_single_iteration_integration_v2_protocol_freeze`
- DeepSeek/API calls, credentials, PPO, Oracle, local model, sealed test,
  integration rerun, and multi-iteration training: not used or authorized

This development gate passes only the response-boundary implementation. It
admits writing a separate integration-v2 protocol; it does not authorize that
protocol's execution.

## Frozen behavior verified

- One initial schema attempt plus at most one semantic repair; a third semantic
  attempt is structurally unavailable.
- At most one transport retry per logical attempt, giving an absolute bound of
  four HTTP transmissions if this guard is later connected to an authorized
  external client.
- Each attempt request receipt is written before the provider call.
- Every returned raw model-content string is atomically written before JSON
  extraction or ranked-response validation.
- The repair prompt reuses the original blinded messages, exact presented
  candidate-ID allowlist, and original maximum budget.
- The invalid raw response is retained locally but not replayed to the model;
  only its SHA-256 and a stable validation-error class enter the repair prompt.
- Partial selection and explicit abstention retain the online schema semantics.
- Two invalid responses yield an empty fail-closed outcome with Oracle selection
  disabled; no heuristic fallback or cached-response substitution occurs.

## Scenarios

The runner executed nine in-memory mock completions across five scenarios:

1. valid first attempt;
2. invented candidate ID followed by valid repair;
3. malformed JSON followed by valid repair;
4. over-budget ranking followed by valid abstention; and
5. two invalid attempts followed by exact exhaustion and fail-closed output.

All per-attempt request, raw-response, validation, and outcome receipts are
retained under `run/scenarios/`. Every declared check in the report is true.

## Evidence identities

- `schema-robustness-report.json`:
  `2cc45c0c79d63d71abb6d20d7fe82936ce02a109c332d8395ea10682c40017c7`
- `run-intent.json`:
  `8fe29bde02c6ef13809db261d63479a8b7491c322388567a301ce45b5444f994`
- Slurm stdout:
  `742e6ea750c104f6d7500a8b8d5fd6d4f0b5774be1d2e8fa94fcac6b63c0b9ca`
- Slurm stderr is empty, SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

Credential-pattern scanning across the complete run and both logs returned no
match. The incident archive and Job 4688 stderr were verified against their
frozen SHA-256 values before the scenarios ran.

## Claim boundary

This is mock-only engineering evidence. It does not show that DeepSeek will
repair a real response, improve acquisition quality, or improve PPO. It does
not reopen the failed single-iteration smoke, access sealed test data, or admit
multi-iteration/formal training. Any real API or PPO execution requires a new,
separately frozen and authorized protocol.
