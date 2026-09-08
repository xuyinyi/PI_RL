# Figure 2 Internal Publication-Success Criteria

**Date:** 2026-09-08

**Target:** ICASSP 2027 regular paper

**Status:** Prospective planning thresholds; not observed results and not official ICASSP acceptance thresholds.

## 1. Core principle

ICASSP does not define a universal percentage improvement required for acceptance. The internal gate must jointly require:

1. a practically non-trivial improvement;
2. paired seed-level uncertainty excluding no improvement;
3. consistency across seeds;
4. superiority to matched non-LLM controls if claiming an LLM contribution;
5. final-checkpoint held-out support;
6. no material validity, uniqueness, or diversity collapse; and
7. complete evaluator/API/compute cost accounting.

The current 158-versus-146 evaluated-terminal count is not usable for this gate. It is a one-seed transition-yield diagnostic, lacks matched rewards and held-out evaluation, and used 299 versus 146 requested evaluator calls.

## 2. Primary estimand

For each paired seed `i`, calculate:

```text
A_i(method) = normalized AUC of best-so-far valid, unique, on-policy
              terminal objective versus all requested training evaluator calls

D_i = A_i(LLM-SciCF-Soft) - A_i(PPO)
```

Use the paired absolute difference `D_i` for inference. Report relative improvement only as an interpretation:

```text
relative improvement = 100% * [mean(A_method) - mean(A_PPO)] / mean(A_PPO)
```

Both absolute and relative differences must be reported because relative percentages can become unstable when baseline AUC is small.

## 3. Recommended internal thresholds

| Evidence item | Minimum defensible result | ICASSP target | Strong result |
|---|---:|---:|---:|
| Primary AUC versus PPO | >=5% relative and paired 95% CI above 0 | **>=8% relative**, paired 95% CI above 0 | >=12% relative |
| Positive paired seeds | >=7/10 with magnitude-aware paired test passing | **>=8/10** | >=9/10 |
| Calls to reach a frozen PPO-quality target | >=10% fewer | **>=15--20% fewer** | >=25% fewer |
| Final held-out mean objective | no degradation; preferably >=3% higher | **>=5% higher** | >=8% higher |
| Frozen high-objective success rate | >=3 percentage-point increase | **>=5 percentage points** | >=10 percentage points |
| AUC versus Random-SciCF | positive paired estimate | **>=3% relative and CI above 0** | >=5% relative |
| AUC versus Diversity-SciCF | non-inferior if not claiming LLM superiority | **>=3% relative and CI above 0** for an LLM-selection claim | >=5% relative |
| Soft versus Hard LLM-SciCF | positive effect or clearly higher effective-pair rate | **>=2--3% AUC improvement** | >=5% improvement |
| Validity | no more than 3 percentage-point decrease | no decrease outside uncertainty | improved |
| Uniqueness | no more than 5 percentage-point decrease | no decrease outside uncertainty | improved |
| Diversity | no more than 10% relative decrease | no material decrease | improved |

These percentages are proposed internal decision thresholds. They are not promises of acceptance and must be frozen before formal outcomes are examined.

## 4. Four publication-strength tiers

### Tier 0 -- No performance claim

Any of the following is sufficient:

- primary AUC is not higher than PPO;
- paired confidence interval is centered near zero with no useful precision;
- improvement appears only in the last iteration or one/two seeds;
- final frozen-policy quality is materially worse;
- gains disappear against Random-SciCF;
- validity, uniqueness, or diversity collapses;
- cost or failure accounting is incomplete.

Use the run as a diagnostic/negative result; do not write a superiority claim.

### Tier 1 -- Weak or borderline

- AUC improvement is roughly 3--5%;
- uncertainty overlaps zero or seed consistency is weak;
- no clear advantage over Random/Diversity-SciCF; or
- held-out quality is essentially unchanged.

This may support an engineering demonstration if novelty and analysis are unusually strong, but it is a high-risk ICASSP submission.

### Tier 2 -- Defensible ICASSP result

- at least 5% and preferably about **8% AUC improvement over PPO**;
- paired 95% confidence interval excludes zero and the prespecified paired test passes;
- at least 8 of 10 paired seeds favor the method;
- at least 15% fewer evaluator calls to reach a frozen quality target, when that target is reached by both methods;
- final held-out mean improves by about 5% or the frozen success rate improves by at least 5 percentage points;
- LLM-SciCF exceeds Random-SciCF by about 3% or more;
- no material quality/diversity collapse.

This supports the bounded claim that SciCF improves evaluator-call efficiency on the frozen surrogate task.

### Tier 3 -- Strong result

- at least 12% AUC improvement over PPO;
- 9/10 or 10/10 paired seeds favor the method;
- at least 25% evaluator-call saving at matched target quality;
- at least 8% held-out objective improvement;
- at least 5% AUC improvement over both Random- and Diversity-SciCF;
- stable behavior under an alternative evaluator or held-out structural region.

This supports both the SciCF-framework contribution and a credible incremental LLM-selection contribution.

## 5. Claim-promotion rules

| Observed contrast | Claim allowed |
|---|---|
| LLM-SciCF > PPO, but LLM-SciCF ~= Random-SciCF | Counterfactual refinement helps; do not claim LLM selection is responsible |
| LLM-SciCF > PPO and > Random-SciCF, but ~= Diversity-SciCF | LLM is competitive with a strong outcome-blind heuristic; avoid claiming unique LLM superiority |
| LLM-SciCF > PPO, Random, and Diversity | LLM-guided acquisition provides incremental value |
| Soft > Hard, with more effective verified pairs | Empirical soft weighting improves use of sparse counterfactual evidence |
| Training AUC improves but final held-out quality does not | Training-time efficiency signal only; no final-policy superiority claim |
| Surrogate score improves without independent validation | Optimization relative to the frozen surrogate only |

## 6. Ten-seed power interpretation

With ten paired seeds and a two-sided 5% test, approximately 80% power generally requires a standardized paired effect close to `d_z = 1.0`. Therefore, the detectable percentage gain is roughly the paired seed-to-seed standard deviation of the percentage difference.

| Paired SD of relative AUC difference | Approximate improvement needed with 10 paired seeds |
|---:|---:|
| 5% | about 5% |
| 8% | about 8% |
| 10% | about 10% |
| 15% | about 15% |

Consequences:

- If the paired SD is about 8%, an 8% mean gain is a sensible target.
- If the paired SD is 12--15%, ten seeds will likely be underpowered for a 5--8% gain.
- Do not add seeds after inspecting significance unless an adaptive sample-size rule was frozen in advance.
- If resources permit, 15--20 paired seeds reduce the required standardized effect, but cannot compensate for protocol drift or biased comparisons.

## 7. Figure 2 visual success pattern

### Figure 2a

A convincing visual should show:

- separation beginning before the final budget checkpoint, not only at the last point;
- LLM-SciCF generally above PPO across the useful budget region;
- an uncertainty band that narrows enough to distinguish the paired AUC effect;
- Random/Diversity curves that reveal which part of the mechanism contributes;
- no smoothing that creates artificial separation.

### Figure 2b

A convincing held-out CCDF should show:

- rightward/upward displacement across a meaningful objective range;
- no advantage caused only by a few extreme molecules;
- seed-level uncertainty or inset summaries;
- failed/invalid samples included according to the frozen rule;
- consistency with the AUC conclusion.

Crossing CCDF curves are not automatically a failure, but they require threshold-specific interpretation and make a broad stochastic-dominance claim inappropriate.

## 8. Recommended formal success rule

Freeze the following as the primary internal gate:

> LLM-SciCF-Soft passes the main performance gate if its seed-level normalized AUC exceeds PPO by at least 5% relative, the paired 95% confidence interval for the absolute AUC difference is above zero, the prespecified paired test gives `p < 0.05`, and at least 8 of 10 paired seeds have a positive difference. For the desired ICASSP-strength result, target an improvement of at least 8%, at least 15% fewer evaluator calls to a frozen target quality, and a held-out objective improvement of approximately 5% without material validity, uniqueness, or diversity degradation.

The LLM-specific claim additionally requires superiority to Random-SciCF and preferably Diversity-SciCF. If those contrasts fail, retain only the counterfactual-refinement claim.
