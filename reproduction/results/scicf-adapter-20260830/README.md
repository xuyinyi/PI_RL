# SciCF DAPiGen adapter validation

- Execution host: n001 SSH target / Slurm node `yanlih100n1`
- Passed job: 4567 (`COMPLETED`, exit `0:0`, elapsed `00:00:05`)
- Server test job: 4568 (`COMPLETED`, exit `0:0`, 13/13 tests passed)
- Source commit: `1d2f0e17c29b2ab1d81616e5bd9c49c495391302`
- Remote report: `/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/adapter-validation-20260830-v2.json`

The direct environment and SciCF-wrapped environment produced the same
terminal polymer, reward, and five property values for the archived valid
action. Matched identity replay produced zero counterfactual effect, equal
terminal objects, and equal factual/counterfactual oracle counts. Restoring the
pre-action state exposed 1,578 unique, action-component-local interventions.

Job 4566 failed before the candidate-enumeration assertion because the
validation script attempted to enumerate from the terminal state left by the
identity replay. The script was corrected to restore the captured pre-action
snapshot before enumeration; no scientific result from job 4566 is used.

This is an engineering/runtime validation. It is not offline Gate 1 evidence
and does not establish SciCF performance or paper-result reproduction.
