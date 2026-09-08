# Read-only n001 handoff status snapshot

Checked 2026-09-08, approximately 09:11–09:14 Asia/Shanghai, using SSH shell
metadata reads only. No test, training, evaluation or API request was launched.
Remote hostname: `yanlih100n1`; `squeue -u wch` returned its header and no jobs.
This is a dated snapshot, not a continuing monitor.

## Latest engineering jobs

| Job | State | Exit | Elapsed |
|---|---|---|---|
| 4748 SciCF six iterations | COMPLETED | 0:0 | 00:04:54 |
| 4750 PPO-only preflight | COMPLETED | 0:0 | 00:00:12 |
| 4751 PPO-only six iterations and analysis | COMPLETED | 0:0 | 00:02:39 |

The isolated runtime checkout
`/home/wch/workspaces/DAPiGen-reproduction/ppo-only-control-a1059ca`
was clean at `a1059ca9d3dcd2d9b111de2472f90966ebadb815`.
Remote `upstream/main` remained at `5f69294`; it is not the latest development
checkout. The latest local evidence commit before documentation was
`4d35d3e3bfb3b615928e63fe3c624bf92af192d6`.

## P4-A formal array 4676

Common report root:
`/home/wch/workspaces/DAPiGen-reproduction/runs/p4a-native-ppo-0c33bce/formal`.
Each report is `seed-<seed>/run_report.json` beneath this root.

| Task | Seed | Slurm state / exit | Elapsed | Report status | evaluation_metrics_complete |
|---|---:|---|---|---|---|
| 4676_0 | 20260911 | FAILED / 1:0 | 00:10:36 | failed_gate | false |
| 4676_1 | 20260912 | COMPLETED / 0:0 | 00:11:41 | passed | true |
| 4676_2 | 20260913 | COMPLETED / 0:0 | 00:11:00 | passed | true |
| 4676_3 | 20260914 | FAILED / 1:0 | 00:10:23 | failed_gate | false |
| 4676_4 | 20260915 | COMPLETED / 0:0 | 00:12:10 | passed | true |

For both failed reports, the displayed acceptance map marks
`evaluation_metrics_complete=false`; the other listed checks, including source,
task, training budget, GAE updates and checkpoint roundtrip, are true.
Both record stop reason `insufficient_remaining_budget_for_reserved_rollout`.
That budget stop is not itself the reported failed acceptance check. This
snapshot does not determine every failed evaluation checkpoint or replace the
pending full per-seed audit. Do not drop those seeds from the aggregate.

Raw report SHA-256 values, recomputed remotely:

```text
b8a8faec93c5164bf1ebd582f6e73229bb65dce266de6a0b0a7eaeb6542cdbcb  seed-20260911/run_report.json
98a9e0fcb9db32f20a0ef925fa4299606a35dcc8076d461ac7c45eed6f61d8f5  seed-20260912/run_report.json
22b1eb126f4fc360813d16cf4c5bbd8797f1dfd373db024534b01bffd71528e9  seed-20260913/run_report.json
1b9874c2ce13d55bc5e7defc94c687716384b8ba8455d88cd778faa7bb04ddfd  seed-20260914/run_report.json
cfd42142dd794ddae7da8cbc096c2fe3db9e227ea1ae6d8ab6af40687b6a2152  seed-20260915/run_report.json
```

The local repository currently archives P4-A preflight evidence but not the
complete formal five-seed dataset. No aggregate file was located in the limited
`formal/` JSON inventory at depth two; a full-tree aggregation audit remains open.

## Data-access scope

The historical Gate 1B.1 root
`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1b1-structure-split-20260903-v1`
listed `dev`, `frozen-model-v1`, and `train`, with no `test` entry. This is a
directory inventory, not a comprehensive access-log audit. Earlier Gate 1B.3
reports explicitly record no sealed-test collection/access/evaluation.

Proxy liveness, API account/balance, primary Hugging Face access and independent
model validity were not checked in this documentation task.
