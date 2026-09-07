# LLM-SciCF response-schema robustness development v1

Status: **mock-only development evaluation passed; external API, PPO, Oracle,
integration rerun, and multi-iteration training remain disabled**.

## Question

Can the online acquisition boundary retain every returned model-content string
before attempting to parse it, permit at most one schema-only repair request,
and fail closed after exhaustion without changing the blinded candidate pool,
selection budget, reward boundary, or policy-score boundary?

## Frozen mechanism

For each initial online acquisition request:

1. Write an attempt request receipt before calling the provider.
2. On a returned completion, atomically write the complete raw model-content
   string and public provider metadata before JSON extraction or validation.
3. If valid, return the validated partial selection or abstention.
4. If invalid, permit exactly one schema-repair attempt. The repair reuses the
   original blinded messages and exact presented candidate-ID allowlist. It
   includes the invalid response hash and an error class, but does not replay
   the invalid text, reward truth, policy scores, credentials, or API key.
5. Capture and validate the repair response under the same rules.
6. If the repair also fails, return an empty fail-closed outcome. Do not pad,
   use a heuristic fallback, substitute a cached response, or authorize Oracle
   selection.

The semantic-attempt limit is two: one initial response plus at most one repair.
Each logical attempt permits at most one transport retry, so the absolute HTTP
transmission bound is four. Transport retrying and semantic repair remain
separately counted.

## Development scenarios

The single mock-only Slurm evaluation must cover:

- a valid first response with no repair;
- an invented ID followed by a valid repair;
- malformed JSON followed by a valid repair;
- an over-budget response followed by valid abstention; and
- two invalid responses producing an empty fail-closed outcome after exactly
  two attempts.

Every scenario must prove attempt-count bounds, raw-before-validation receipts,
exact allowlist and budget preservation, absence of fallback selection, and no
external API, PPO, Oracle, model, or sealed-test access.

## Stop and claim boundary

Any missing raw capture, overwritten attempt, third semantic attempt, candidate
allowlist change, budget expansion, invalid-text replay, reward/policy leakage,
fallback selection, or non-empty exhausted outcome fails this development gate.

Passing admits only a separately frozen design for a future single-iteration
integration-v2 rerun. It does not authorize that rerun itself, DeepSeek calls,
PPO, Oracle verification, multi-iteration training, sealed-test access,
algorithm-effectiveness claims, or scientific claims.

## Execution outcome

The frozen coordinate `5387611e1ba088b8dafd0c50a80751059e8fdfed`
completed on n001 in Slurm Job 4689 with 19 tests passed and all five scenarios
passing. Nine scripted in-memory completions were used; external API request
count, credential loads, PPO execution, Oracle execution, local-model use, and
sealed-test access were all zero or false. The next admitted action is only a
separate integration-v2 protocol freeze, not its execution. Evidence is under
`reproduction/results/scicf-schema-robustness-dev-20260907/`.
