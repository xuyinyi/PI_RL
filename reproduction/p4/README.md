# P4-A native PPO baseline

Current status: **preflight Job 4675 passed; formal array Job 4676 running**.

This directory contains the frozen standard-five-step native-PPO baseline arm.
The machine protocol fixes the task, optimizer, training evaluator budget,
evaluation checkpoints, seeds, metrics, data boundary and fail-closed launch
gate before any P4-A result exists.

Files:

- `P4_NATIVE_PPO_BASELINE_PROTOCOL_V1.md`: human-readable protocol and claim boundary;
- `configs/native_ppo_baseline_protocol_v1.json`: machine-readable authority;
- `protocol.py`: strict validation and acceptance checks;
- `scripts/run_native_ppo_baseline.py`: common preflight/formal runner;
- `scripts/audit_native_ppo_baseline.py`: independent per-run gate audit;
- `scripts/aggregate_native_ppo_baseline.py`: five-seed aggregation without claim promotion;
- `tests/test_protocol.py`: protocol immutability and fail-closed tests.

The preflight and formal array must execute through the corresponding Slurm
entry points in `../slurm/`. A preflight pass authorizes submission of the
already frozen formal seeds; it does not authorize retuning. A preflight failure
closes the formal launch gate until a separately versioned protocol is reviewed.

P4-A is not the complete P4 gate. The original RLlib compatibility evidence and
the `legacy_effective` six-step semantic-control arm remain separate required
evidence.
