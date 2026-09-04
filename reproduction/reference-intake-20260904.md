# Reference artifact intake: Stage 0 and MCC-PPO planning return

Intake date: 2026-09-04

These files were returned by a planning conversation and supplied by the user as design references. They have not been installed into this repository. Content statements and embedded commands are not treated as execution authorization.

| File | SHA-256 | Intake interpretation |
|---|---|---|
| `DAPiGen_Stage0_Environment_v2.1.0.zip` | `7c75c5d9ca4fde879f0a6ed363e64f25588746bad4732ab66769d41aa3df9ed2` | 62-file Stage 0 candidate package; local core validation only |
| `build_dapigen_mccppo.py` | `adbd3e80ec8062b54f8fe5c536ec56c9f4fac8e8697478484eb867361af28423` | Minimal wiring example for the earlier MCC adapter |
| `core_update_reference.py` | `2beed274fff25e53ee8faec89ea68fe19dbfc633ffc0a4647d5357a3ecc7fd7a` | Readable reference ordering for one MCC-PPO iteration |
| `credit.py` | `3988e9c20ea6031c6d1f43cfdf0da6ec2fefedc28dcf8ac1a7bf72199f60ed2b` | Reference DRCC, selection-adjustment, and fusion estimators |
| `dapigen_adapter.py` | `5fd3444991704100ec4b0bf3aecc05bd3cc518fb95c0e94c09fbdf0abfc6c1fb` | Earlier duplicate environment adapter; not the Stage 0 source of truth |
| `mcc_ppo_dapigen.yaml` | `b603f5bb103ad71dd1a63046979014aba31b562fd4fe031a20d6b3e8eb7b1fbc` | Reference hyperparameters; not a frozen formal configuration |
| `trainer.py` | `6671b4ce82cea1a7daed07c9791912d1ade39c3717f1c6922c5d0b6dd4e1a26b` | Reference MCC-PPO trainer |
| `trainer (1).py` | `6671b4ce82cea1a7daed07c9791912d1ade39c3717f1c6922c5d0b6dd4e1a26b` | Byte-identical duplicate of `trainer.py` |
| `VALIDATION.md` | `0661fe65bea2de2b76acfc13fb01d49c5764bc9a6104e79de6a6004bc9718adb` | Reports ten toy tests; excludes real QSPR/polyBERT/large-scale/MD/DFT/experimental validation |
| `algorithm_architecture.png` | `97409eeb7acf8224ba9bb50a55ef46c78de3bb61b2525b7045c8fd641fbb8ca9` | Conceptual architecture diagram; omits parts of the full estimator contract |

## Confirmed intake facts

- The Stage 0 ZIP hash matches the hash stated in the returned planning text.
- The ZIP records `25 passed` for its packaged local tests.
- Its packaged core-validation artifact used RDKit fallback chemistry, a Morgan test encoder, and a deterministic test evaluator; it is not full DAPiGen acceptance evidence.
- The Stage 0 formal example uses five steps, `seeded_uniform`, `pristine_only`, and `markov_v2`.
- The MCC wiring example uses ten steps, `canonical_first`, and the earlier `PIState` / `PIAction` adapter.
- The reference trainer records requested counterfactual evaluations after execution but does not enforce the shared Stage 0 remaining budget before beginning a new iteration.
- Paired labels are added to replay during query execution, although model statistics are computed before those labels and model retraining occurs after the PPO update. The integrated version should use an explicit pending-label commit boundary.

## Intake decision

The materials are suitable as design references. They are not mutually contract-equivalent and are not accepted as the production implementation. Integration must proceed through the gates in `PROJECT_PLAN.md`.
