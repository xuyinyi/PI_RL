# P4-A native PPO baseline protocol v1

Status: **frozen before preflight execution**

## Scope

This protocol freezes the native-PyTorch PPO baseline on the accepted Stage-0
standard five-step task. It is the baseline arm that later Policy-CC and
MCC-PPO runs must share. It does not complete the whole P4 gate: the frozen
original RLlib compatibility result and the six-step `legacy_effective`
semantic-control arm remain separate evidence obligations.

The machine-readable authority is
`configs/native_ppo_baseline_protocol_v1.json`. A failed preflight cannot be
used to tune this file in place. Any behavior-changing repair or hyperparameter
change requires a new protocol version and a new preflight seed.

## Frozen task and optimizer

- Accepted environment: `dapigen:8004c1fa2055a186b4a4f3ed` at the P1 binding
  recorded by commit `373b3291`.
- Standard task: five steps, `closure_exact_cached`, `pristine_only`,
  `seeded_uniform`, 1,246-D `augmented_v3` observation and terminal-only paper
  Equation (1) reward.
- Native PPO: 128-transition rollouts, `gamma=1`, `lambda=0.95`, four update
  epochs, minibatch 64, clip/value clip 0.2, entropy coefficient 0.01,
  learning rate 3e-4 and a 256/256 Tanh trunk.
- Training calls are tagged `ppo/on_policy`; checkpoint evaluation uses a
  fresh, independent `evaluation` ledger and deterministic masked actions.
- The critic always uses environment-return targets and PPO actor credit is
  byte-identical to GAE.

## Preflight and automatic launch gate

Preflight uses seed `20260910`, a 512 requested-call cap, a 512 unique-call cap
and deterministic evaluations of 128 episodes at 0, 256 and the terminal
budget boundary. Because a rollout pre-reserves 128 worst-case calls, training
stops before another rollout when fewer than 128 calls remain; at least 385
requested calls must therefore be consumed.

The formal array may be submitted only if the preflight records all of the
following as true:

- clean Slurm execution on `yanlih100n1` and exact accepted-environment binding;
- requested/unique/backend/cache/source ledger within the frozen caps;
- only `ppo/on_policy` training calls and only `evaluation` checkpoint calls;
- finite PPO metrics, exact GAE actor credit and a changed policy;
- all checkpoint evaluations contain the frozen metrics and at least two valid
  molecules;
- final checkpoint round-trip preserves policy, version and evaluator ledger;
- no invalid evaluator result, sealed-test access or external API call.

Infrastructure or contract failure closes the launch gate. The preflight is
not a performance screen: low reward cannot be used to cancel, tune or repeat
the frozen formal seeds.

## Formal P4-A arm

Formal seeds are `20260911` through `20260915`. Each seed has a 10,000
requested-call and 10,000 unique-call training cap, with deterministic 1,000
episode evaluations at targets 0, 2,500, 5,000, 7,500 and the terminal budget
boundary. Minimum accepted budget use is 9,873 requested calls per seed.

The primary curve axis is actual requested training evaluator calls. Unique and
backend calls, cache hits, transitions, wall time and GPU memory are reported
separately. Checkpoint metrics are validity, objective mean/maximum, uniqueness,
novelty, diversity, Frag and SNN under the frozen `raw_data/PI.csv` reference.

Five completed seeds constitute the P4-A native-standard baseline evidence.
They do not by themselves close complete P4 or support a comparison,
optimization-superiority claim or scientific claim.
