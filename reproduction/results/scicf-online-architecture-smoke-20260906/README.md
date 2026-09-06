# LLM-SciCF online architecture smoke

Final status: **engineering architecture smoke passed; no effectiveness or
scientific claim is authorized**.

## Governed coordinate

- Git branch: `codex/scicf-online-architecture-v1`
- Git commit: `d05a05c556958a798c3f4eab217eed587ed0309b`
- Host: `yanlih100n1`
- Final Slurm job: `4684`, `COMPLETED 0:0`, 26 seconds
- Protocol: `dapigen-scicf-online-architecture-smoke-v1`
- LLM: official DeepSeek API, `deepseek-v4-flash`, no local LLM
- Prompt: `scicf-dapigen-online-acquisition-blinded-v1`
- Stage-0 environment: `dapigen:8004c1fa2055a186b4a4f3ed`

## End-to-end result

1. All 33 P2 and online contract tests passed under Slurm.
2. One 64-transition on-policy rollout produced 12 successful terminals and
   12 requested/unique evaluator calls, all tagged `scicf_ppo/on_policy`.
3. The standard PPO update used byte-identical environment-return GAE for the
   actor and retained environment-return critic targets.
4. One reward-free, policy-score-free blinded API request presented 24 opaque
   candidates spanning timesteps 0, 1 and 2. DeepSeek selected four and did not
   abstain. The provider reported 7,839 prompt, 140 completion and 7,979 total
   tokens.
5. Only the four selected candidates were verified with one matched
   factual/counterfactual replicate each. Five terminal scientific objects
   reached the Oracle: one factual and four counterfactual. Three factual
   branches terminated for `atom_limit_exceeded` before Oracle scoring.
6. All four comparisons produced non-zero pairs. Deltas were `+0.4493`,
   `+0.4180`, `+0.4457`, and `-0.4176`; the first three preferred the
   counterfactual and the fourth preferred the factual action.
7. Exactly one component-local pairwise optimizer step was applied after PPO.
   Pair weights used only verified deltas. Mean joint KL was
   `5.792708179797046e-07`, below the frozen `0.01` rollback threshold.
8. Policy hashes changed independently across standard PPO and pairwise
   refinement. The remote checkpoint SHA-256 is
   `fe9a9f268571f7073d404dfc31f0b74d065432c2671cf2ea961ac9b8e5b07313`.

The checkpoint remains at
`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf-online-architecture-d05a05c/scicf-online-smoke-checkpoint.pt`
on n001 and is not stored in Git.

## Retained failed engineering attempts

- Job 4681: tests passed; stopped before API because the fresh worktree lacked
  a local polyBERT asset binding.
- Job 4682: tests passed; stopped before API because the fresh worktree lacked
  the ignored AFP/QSPR model files.
- Job 4683: PPO ran; stopped before API because the first successful episode
  had only timestep 0 and failed the cross-timestep pool requirement.
- Job 4684: passed after binding the accepted assets and selecting the first
  complete multi-step episode, preferring a successful one for architecture
  reachability only.

These failures are infrastructure/control-flow diagnostics. No API request was
made in Jobs 4681-4683.

## Evidence hashes

```text
9ee4969d83f6c08cb4e8a6100af24a11dbab81d10e1f45a87ed7c9dcdcca24f9  candidate-pool-audit.json
1a0b2fdeec1391fd191a7359867abc2517bf5224a6054ccc72c9546bcf27abb6  llm-request.json
03f7b0625f8553210449856ce982a20d43e290fe606dc40e5bf3b85d5483aaec  llm-response.json
85024719a302327056f1201b25178ca315d58565ebb5a7d2f36260d37946de75  oracle-verification.json
0e08448945b9cae31e95ff3bc6c991d4632ca770bb7e266f59062750c22f02f9  run-intent.json
f959cb5b974620b559e85a6f2a880a6b3aded404478321caa8894fcd902470c8  smoke-report.json
```

## Claim boundary and next gate

This run proves only that the architecture can execute with blinded LLM
acquisition, selected-only matched verification, source-level budget
accounting, separate PPO/pairwise updates, KL protection and checkpointing.
`K=1`, one reachability-selected episode and one API request cannot test
ranking reliability, learning benefit, Oracle efficiency or structural
generality. Gate 1B.3 remains a failed non-blocking descriptor diagnostic.

The next admissible work is module hardening, beginning with a small
development-only acquisition/verification smoke that adds matched Random and
policy/chemistry controls and `K>=2`. Formal multi-iteration training remains
closed until that protocol is frozen and separately authorized.
