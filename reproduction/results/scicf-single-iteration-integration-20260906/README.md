# LLM-SciCF single-iteration integration smoke evidence

## Decision

- Protocol: `dapigen-scicf-single-iteration-integration-smoke-v1`
- Final execution coordinate: clean commit
  `b61d9e5c81dbef5108a2d796e0acd7ecf3a7640b`
- Final execution: n001 (`yanlih100n1`) Slurm Job 4688, `FAILED 1:0`
- Server tests: 36 passed in both Job 4687 and Job 4688
- Gate decision: **no-go** for a bounded short-horizon multi-iteration smoke
- Formal training, multi-iteration training, sealed-test access, algorithm-
  effectiveness claims, and scientific claims: not authorized

Job 4688 failed closed before counterfactual verification because the first
DeepSeek response invented the out-of-pool identifier `cf-281`. The response
therefore failed the frozen candidate-ID schema. No Oracle verification,
pairwise update, checkpoint, or integration report was produced by Job 4688.
Because the configured API endpoint does not accept a deterministic seed, an
earlier valid response cannot be substituted for this failed response.

## Superseded diagnostic Job 4687

Job 4687 ran clean commit
`64b086142a6a971b80c8de3675b1d200e6b7413d` and completed the full numerical
path, but its report failed two ledger assertions. The assertions incorrectly
required every matched environment branch to invoke the QSPR evaluator. The
Stage-0 contract instead terminates structurally invalid branches before QSPR;
all 32 branch records were present, while only 12 branches produced terminal
molecular evaluations. Commit `b61d9e5` corrected the assertion without
changing the seed, candidate construction, model, or thresholds. Job 4687 is
retained as superseded code-diagnostic evidence and is not a passing gate.

Its non-admissive diagnostics were:

- one 128-transition PPO rollout and update;
- actor-credit SHA-256 exactly equal to the rollout GAE SHA-256, and critic-
  return SHA-256 exactly preserved by the PPO receipt;
- two outcome-blind, cross-timestep pools of 24 candidates, with two blinded
  DeepSeek requests completed before verification;
- eight selected candidates and 32 completed matched branches at `K=2`;
- 12 terminal-evaluator requests: four factual and eight counterfactual, with
  ten backend calls, two cache hits, and zero invalid evaluator results;
- two sign-consistent pairs, one positive and one negative;
- one pairwise update, with mean signed margin change `+5.55515e-5` and
  preference accuracy unchanged at `0.5`;
- maximum full-support joint KL `2.24566e-7`, maximum non-target-factor KL
  `0.0`, maximum absolute critic drift `0.00137779`, and unchanged value-head
  parameters;
- checkpoint SHA-256
  `bf27a826ff59b12289bd9adc132268803fed874142099d66e89aa0e375408223`.

These values show that the downstream PPO, verifier, and pairwise mechanics
were reachable in that attempt. They do not override either Job 4687's failed
frozen report or Job 4688's genuine schema failure.

## Audit boundary

The Job 4688 runner validated the API body before persisting a response file,
so the invalid raw body is not retained; the exact rejected identifier and
traceback are retained in `logs/scicf-one-iter-ledgerfix-4688.err`. This is an
evidence-retention limitation to fix before another independently authorized
API robustness run. Credential scans over both archived run directories and
all four Slurm logs found no API key or bearer-token pattern.

The next admissible work, if separately frozen and authorized, is API response-
schema robustness development with raw-invalid-response retention. The present
result does not admit a rerun, a short-horizon multi-iteration smoke, formal
training, sealed-test access, or any scientific conclusion.
