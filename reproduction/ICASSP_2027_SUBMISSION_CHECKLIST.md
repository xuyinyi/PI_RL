# ICASSP 2027 Submission Checklist: DAPiGen / LLM-SciCF

**Audit date:** 2026-09-08

**Target:** ICASSP 2027 regular paper, Toronto, 16--21 May 2027

**Official regular-paper deadline:** 16 September 2026; the submission system lists the equivalent Beijing deadline as **20:00, 17 September 2026**. Use **20:00, 16 September 2026** as the internal upload target.

**Paper limit:** 4 pages of technical content, with an optional fifth page containing only references, funding acknowledgments, and a Compliance with Ethical Standards statement.

**Current readiness verdict:** **Not ready because of named blockers.** The engineering path and a one-seed exploratory comparison exist, but the common evaluator, frozen formal protocol, matched multi-seed comparison, formal figures, and manuscript are incomplete.

Official sources:

- [ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php)
- [ICASSP 2027 Author Guidelines](https://2027.ieeeicassp.org/author-guidelines/)
- [Submission deadline by time zone](https://cmsworkshops.com/ICASSP2027/papers.php)
- [ICASSP 2027 paper topics](https://cmsworkshops.com/ICASSP2027/Papers/paper_topics.php)

## 1. Package boundary and current evidence

### Confirmed engineering evidence

- [x] Stage 0 v2.3 compatibility environment accepted for the current compatibility assets.
- [x] Shared native PPO/GAE path implemented and smoke-tested.
- [x] Online K=5 SciCF selection/verifier/auxiliary-update path implemented and engineering-tested.
- [x] Six-iteration SciCF Job 4748 archived.
- [x] Matched PPO-only Job 4751 and its exploratory comparison archived.
- [x] First-iteration equivalence evidence retained.
- [x] Remote status snapshot records no active jobs on 2026-09-08.

### Evidence that must not be promoted to a scientific result

- [x] Treat Jobs 4748/4751 as a one-seed engineering/exploratory pair only.
- [x] Do not interpret 158 versus 146 evaluable terminals as reward or quality improvement.
- [x] Do not treat the last rollout as evaluation of the last updated checkpoint.
- [x] Do not treat small auxiliary KL as evidence of policy improvement.
- [x] Do not treat AFP/QSPR optimization as experimentally verified material-property improvement.
- [x] Do not drop failed P4-A seeds or equate Slurm `COMPLETED` with gate acceptance.

Primary evidence locators:

- `DEVELOPMENT_HANDOFF_20260908.md`, Sections 9, 12, and 14.
- `PROJECT_PLAN.md`, current status table and Gates P6--P7.
- `results/handoff-status-20260908/REMOTE_STATUS.md`.

## 2. Submission-blocking engineering tasks

These items must close before a formal comparison is launched.

- [ ] **B01 -- Freeze the paper claim.** Use the bounded claim: LLM-guided, verifier-weighted counterfactual refinement improves evaluator-call efficiency for a frozen surrogate molecular-design task. Do not claim physical material improvement without independent validation.
- [ ] **B02 -- Freeze the method identity.** Record Git commit, environment manifest, Stage 0 configuration, policy checkpoint, AFP/polyBERT fingerprints, LLM provider/model/version, prompt template, response schema, and all algorithm configurations.
- [ ] **B03 -- Implement unified on-policy event logging (H01).** Every arm must record reward, terminal reason, molecule identity, validity, duplicate status, requested evaluator call, unique evaluator call, cache hit, transition index, checkpoint identity, and explicit denominator.
- [ ] **B04 -- Implement the common 1,246-D final-checkpoint evaluator (H02).** Training and evaluation ledgers must be separate; evaluation seeds and budgets must be frozen; no silent reuse of the legacy 1,200-D evaluator is allowed.
- [ ] **B05 -- Freeze the formal comparison protocol (H03).** Specify endpoints, budgets, checkpoints, stopping rules, invalid/duplicate/missing-result handling, seed mapping, statistical tests, confidence intervals, multiplicity policy, success criteria, and claim-promotion rules. Bind the frozen protocol to the prospective thresholds in `FIG2_PUBLICATION_SUCCESS_CRITERIA.md`; do not revise those thresholds after viewing formal outcomes.
- [ ] **B06 -- Isolate RNG consumption (H05).** Demonstrate that enabling candidate selection or auxiliary processing does not unintentionally perturb the next PPO sampling RNG stream beyond the intended policy update.
- [ ] **B07 -- Implement and test the comparison arms.** All arms must use the same environment, reward, action mask, initial checkpoint, candidate construction, and accounting schema.
- [ ] **B08 -- Run a no-cost/local or governed Slurm preflight.** Check schema completeness, deterministic identities, budget stops, checkpoint loading, failure recording, and report aggregation before any formal API/evaluator spend.
- [ ] **B09 -- Freeze a new, explicit formal-run authorization.** Earlier Job 4748/4751 authorization is already consumed and does not authorize new seeds, training, evaluation, API use, or sealed-test access.

### Required algorithm arms

- [ ] `PPO`
- [ ] `PPO + Random-SciCF-K5-Soft`
- [ ] `PPO + Diversity-SciCF-K5-Soft`
- [ ] `PPO + LLM-SciCF-K5-Soft` (**ours**)
- [ ] `PPO + LLM-SciCF-K5-Hard` (ablation; may be reported in a compact table rather than Figure 2)

For all K=5 arms, freeze the same legal candidate pool, verifier, auxiliary optimizer, rollback/KL guard, and requested-call accounting. The diversity selector must be outcome-blind. Prefer a frozen existing polyBERT representation and farthest-point selection; otherwise freeze one alternative before seeing formal results.

## 3. Formal experiment checklist

### Protocol and scale

- [ ] Use paired seeds across all arms.
- [ ] Target **10 paired seeds**. Five seeds may be reported as exploratory, but a two-sided exact paired sign/permutation test cannot attain conventional `p < 0.05` with only five pairs under the most discrete sign-test setting.
- [ ] Freeze a common requested evaluator-call budget as the primary resource constraint.
- [ ] Freeze a maximum transition cap as a safety limit and report actual transitions; do not force equal transitions if the scientific claim is evaluator-call efficiency.
- [ ] Freeze reporting checkpoints on the requested-call axis.
- [ ] Use the same starting policy checkpoint for paired runs.
- [ ] Record wall time, CPU/GPU hours, and LLM/API calls and cost.
- [ ] Count failures, timeouts, malformed LLM responses, invalid structures, duplicates, and missing evaluator results; never remove them silently.
- [ ] Preserve every seed, including failed or prematurely stopped runs, in the aggregate report.

### Primary endpoint

- [ ] For each seed, calculate normalized AUC of **best-so-far valid, unique, on-policy terminal objective versus all requested training evaluator calls**.
- [ ] Define the x-axis integration range and normalization constant before viewing formal outcomes.
- [ ] Define how runs that terminate before the common budget contribute after their last valid point.
- [ ] Compare methods at the seed level using paired uncertainty; molecules are not independent experimental replicates.

### Final-policy endpoint

- [ ] Evaluate every frozen final checkpoint using the same independent held-out seed set and budget.
- [ ] Plot the objective CCDF, `P(R >= r)`, with failed/invalid evaluations handled according to the frozen rule (recommended: objective 0 when the evaluator contract defines failure as 0).
- [ ] Report seed-level means/medians and confidence intervals, not only pooled molecules.
- [ ] Keep final evaluation calls outside the training-call ledger.

### Secondary metrics

- [ ] Terminal objective distribution and threshold success rate.
- [ ] Validity.
- [ ] Uniqueness among valid outputs.
- [ ] Novelty against the frozen reference/training set.
- [ ] Diversity, Frag, and SNN under one frozen implementation.
- [ ] Requested evaluator calls, unique calls, cache hits, transitions, wall time, GPU/CPU use, and API cost.
- [ ] SciCF opportunity rate, selection rate, verifier completion rate, accepted-pair/effective-weight rate, auxiliary update count, rollback count, and auxiliary KL.
- [ ] LLM invalid-ID, malformed-response, timeout, retry, and abstention rates.

### Required statistical analysis

- [ ] Predeclare the primary contrast: `LLM-SciCF-Soft` versus `PPO` on seed-level normalized AUC.
- [ ] Predeclare the selector contrast: `LLM-SciCF-Soft` versus `Random-SciCF-Soft`.
- [ ] Predeclare the strong heuristic contrast: `LLM-SciCF-Soft` versus `Diversity-SciCF-Soft`.
- [ ] Predeclare the weighting ablation: `LLM-SciCF-Soft` versus `LLM-SciCF-Hard`.
- [ ] Report paired effect size, paired confidence interval, and exact/permutation p-value where meaningful.
- [ ] State the multiplicity treatment for secondary contrasts.
- [ ] Report all prespecified endpoints, including null or negative results.

### Predeclared Figure 2 publication-strength thresholds

These are internal prospective decision thresholds, not official ICASSP acceptance thresholds. Use the paired absolute normalized-AUC difference for inference and report relative improvement only as an interpretation. The full rationale and power notes are in `FIG2_PUBLICATION_SUCCESS_CRITERIA.md`.

| Evidence item | Minimum defensible result | ICASSP target | Strong result |
|---|---:|---:|---:|
| Primary normalized AUC versus PPO | >=5% relative and paired 95% CI above 0 | **>=8% relative**, paired 95% CI above 0 | >=12% relative |
| Positive paired seeds | majority plus a passing magnitude-aware paired test | **>=8/10** | >=9/10 |
| Evaluator calls to a frozen PPO-quality target | >=10% fewer | **>=15--20% fewer** | >=25% fewer |
| Final held-out mean objective | no degradation; preferably >=3% higher | **>=5% higher** | >=8% higher |
| Frozen high-objective success rate | >=3 percentage points | **>=5 percentage points** | >=10 percentage points |
| AUC versus Random-SciCF | positive paired estimate | **>=3% relative and CI above 0** | >=5% relative |
| AUC versus Diversity-SciCF | non-inferior if no LLM-superiority claim is made | **>=3% relative and CI above 0** for an LLM-selection claim | >=5% relative |
| Soft versus Hard LLM-SciCF | positive effect or higher effective-pair rate | **>=2--3% AUC improvement** | >=5% improvement |
| Validity | decrease <=3 percentage points | no decrease outside uncertainty | improved |
| Uniqueness | decrease <=5 percentage points | no decrease outside uncertainty | improved |
| Diversity | relative decrease <=10% | no material decrease | improved |

#### Main internal performance gate

- [ ] `LLM-SciCF-Soft` improves seed-level normalized AUC over PPO by at least 5% relative.
- [ ] The paired 95% CI for the **absolute** AUC difference is above zero.
- [ ] The prespecified paired test gives `p < 0.05`.
- [ ] At least 8 of 10 paired seeds have a positive AUC difference.
- [ ] The desired ICASSP-strength target is reached: approximately 8% AUC improvement, 15--20% fewer evaluator calls to a frozen target quality, and approximately 5% higher final held-out objective without material validity, uniqueness, or diversity degradation.

#### LLM-specific claim gate

- [ ] To claim that LLM selection adds value, `LLM-SciCF-Soft` exceeds `Random-SciCF-Soft`; target >=3% relative AUC improvement with paired CI above zero.
- [ ] To claim superiority over a strong outcome-blind heuristic, `LLM-SciCF-Soft` exceeds `Diversity-SciCF-Soft`; target >=3% relative AUC improvement with paired CI above zero.
- [ ] If LLM-SciCF exceeds PPO but not Random-SciCF, restrict the claim to the counterfactual-refinement framework.
- [ ] If LLM-SciCF exceeds PPO and Random-SciCF but not Diversity-SciCF, state that LLM selection is competitive with the heuristic; do not claim unique LLM superiority.
- [ ] If training AUC improves but final held-out quality does not, restrict the claim to a training-time efficiency signal.

#### Ten-seed power and variance check

- [ ] Treat ten paired seeds as capable of reliably detecting only a relatively large and consistent effect; approximately 80% power generally requires a paired standardized effect close to `d_z = 1.0`.
- [ ] Estimate and report the paired SD of relative AUC differences. As a planning approximation, paired SDs of 5%, 8%, 10%, and 15% require mean improvements of roughly 5%, 8%, 10%, and 15%, respectively, with ten pairs.
- [ ] If paired SD exceeds roughly 10%, either use a larger seed count under a predeclared rule or label a 5--8% result underpowered/exploratory.
- [ ] Do not add seeds after inspecting significance unless an adaptive sample-size rule was frozen before formal outcomes.
- [ ] Do not use the existing 158-versus-146 evaluated-terminal count to satisfy this gate: it is one seed, has no matched reward or held-out endpoint, and used 299 versus 146 requested evaluator calls.

### Formal-run evidence package

- [ ] One immutable run manifest per arm and seed.
- [ ] Code/config/environment/model/prompt fingerprints.
- [ ] Slurm submission receipt, job ID, stdout/stderr, exit state, and acceptance decision.
- [ ] Raw event ledger and final-evaluation ledger.
- [ ] Per-seed report and all-seed aggregate.
- [ ] SHA-256 manifest for raw data, processed tables, plots, and manuscript source data.
- [ ] One verifier that reconstructs every Figure 2 point from the archived ledgers.

## 4. Figure and table checklist

### Figure 1 -- Method architecture

- [ ] Show PPO rollout, legal counterfactual pool, LLM/random/diversity selector, K=5 matched evaluation, empirical soft weight, at-most-one auxiliary update, KL guard/rollback, and return to PPO.
- [ ] Visually separate policy sampling, LLM selection, AFP/QSPR evaluation, and PPO/auxiliary optimization.
- [ ] State explicitly that LLM scores, confidence, and prose do not enter reward or PPO targets.
- [ ] Avoid calling AFP a real scientific oracle; use `surrogate evaluator` or another accurately bounded term.

### Figure 2a -- Budget-indexed learning efficiency

- [ ] Four curves: PPO, Random-SciCF, Diversity-SciCF, and LLM-SciCF-Soft.
- [ ] X-axis: all requested training evaluator calls.
- [ ] Y-axis: best-so-far valid, unique, on-policy terminal objective.
- [ ] Show paired-seed mean/median plus 95% uncertainty band.
- [ ] Caption states seed count, aggregation unit, budget, failure rule, and whether smoothing is used.
- [ ] Include seed-level normalized AUC in the source-data table.

### Figure 2b -- Frozen final-policy quality

- [ ] CCDF of held-out terminal objective for the same four main methods.
- [ ] Use common evaluation seeds/budget and frozen failure handling.
- [ ] Include a compact seed-level summary inset only if it remains readable at final size.
- [ ] Do not pool molecules to create artificially narrow uncertainty intervals.

### Table 1 -- Compact result and ablation table

- [ ] Primary AUC with uncertainty and paired test.
- [ ] Final held-out objective/success summary.
- [ ] Validity, uniqueness, novelty, and diversity.
- [ ] Requested/unique calls, transitions, wall time, and LLM/API cost.
- [ ] Soft versus hard weighting ablation.
- [ ] Mark best and second-best results only after the full frozen analysis is complete.

### Figure production QA

- [ ] Every plotted number is traceable to a source-data CSV/TSV and generation script.
- [ ] Vector PDF/SVG preferred for line art; embedded raster elements have sufficient resolution.
- [ ] Curves remain distinguishable in grayscale and under common color-vision deficiencies.
- [ ] All text, legends, axes, and captions remain at least 9 pt in the final paper.
- [ ] Captions define metrics, denominators, uncertainty, seeds, and failure handling without requiring hidden supplementary material.

## 5. Scientific-claim and reproducibility checklist

- [ ] Write the main claim at the level supported by the experiment: evaluator-call-efficient optimization under a frozen surrogate task.
- [ ] If claiming real polymer-property improvement, complete alternative/held-out QSPR evaluation, domain-of-applicability analysis, and appropriate higher-fidelity or experimental validation first.
- [ ] If independent validation cannot be completed, include an explicit surrogate-validity limitation and do not use physical-property causal wording.
- [ ] Report the exact AFP/QSPR training boundary and whether generated structures are in-domain.
- [ ] Report LLM provider, exact model/version/date, temperature, prompt, legal-ID schema, retry/failure behavior, and whether any response was manually edited.
- [ ] State that the LLM does not supply reward, critic targets, or confidence weights if that remains true in the frozen implementation.
- [ ] Archive prompts and raw responses subject to provider/privacy constraints.
- [ ] Provide an environment file, exact runner commands, configuration files, and a minimal result verifier.
- [ ] Provide a code/data availability statement with actual public or archival locations; no placeholder URLs.
- [ ] Separate reproducibility of the surrogate optimization from validation of real material properties.

## 6. Four-page manuscript checklist

### Recommended structure and page budget

- [ ] **Page 1:** title, 100--150-word abstract, introduction, precise gap, and three contribution bullets/sentences.
- [ ] **Pages 1--2:** method and Figure 1; define candidate pool, K=5 acquisition, empirical soft weight, auxiliary loss, and KL guard.
- [ ] **Pages 2--3:** experimental contract, baselines, budgets, seed-level statistics, and Figure 2.
- [ ] **Pages 3--4:** main results, selector/weighting ablation, cost analysis, limitations, and conclusion.
- [ ] **Optional Page 5:** references, funding acknowledgments, and Compliance with Ethical Standards statement only.

### Content closure

- [ ] Title emphasizes the learning contribution, not an unsupported physical discovery.
- [ ] Abstract contains problem, method, experimental setting, principal quantitative result, and bounded conclusion.
- [ ] Introduction explains why sparse terminal feedback and expensive evaluation create the need for counterfactual refinement.
- [ ] Related work covers PPO, molecular RL/generation, LLM-guided scientific search, and verifier/oracle-guided learning.
- [ ] Method notation is dimensionally and procedurally consistent with the implementation.
- [ ] Experiments describe all five arms, paired seeds, budgets, endpoints, invalid/duplicate rules, and final evaluation.
- [ ] Results report uncertainty and cost, not just best examples.
- [ ] Limitations disclose surrogate bias, one task/environment if applicable, LLM reproducibility, and lack of wet-lab evidence.
- [ ] Conclusion does not broaden beyond the frozen evidence.
- [ ] Every factual or prior-work claim has a verified primary citation.
- [ ] Every numeric manuscript claim maps to a frozen table cell or figure source-data row.

## 7. ICASSP 2027 format and portal checklist

### Format

- [ ] Use the **ICASSP 2027** template; do not reuse the ICASSP 2025 template.
- [ ] Maximum four technical pages; optional fifth page restricted to references/funding/ethics.
- [ ] Two-column layout and font size no smaller than 9 pt, including captions.
- [ ] Title is bold, in ALL CAPITALS, contains no LaTeX math, and is representable in Unicode.
- [ ] Author names and affiliations are included because review is not blind.
- [ ] No page numbers.
- [ ] Figures fit the margins, have captions, and remain legible in black and white.
- [ ] PDF is A4 or US Letter, first-page-first, unprotected, and has all fonts embedded/subset.
- [ ] PDF is no larger than 5 MB.
- [ ] File is named with the first author's surname, e.g. `surname.pdf`.

### Metadata and authors

- [ ] Freeze title, author names, author order, affiliations, and corresponding author before upload.
- [ ] Every author has a valid ORCID; ICASSP 2027 requires ORCID for every listed author.
- [ ] PDF author list exactly matches the portal author list and order.
- [ ] Portal title exactly matches the PDF title.
- [ ] PDF abstract is approximately 100--150 words; portal abstract is identical and no more than 200 words.
- [ ] Provide up to five keywords, identical between PDF and portal.
- [ ] Confirm no author appears on more than nine ICASSP/OJSP-stream submissions.
- [ ] Recommended primary topic: **Machine Learning for Signal Processing -- Reinforcement learning [ML-REI]**.
- [ ] Recommended secondary topic: **Machine Learning for sciences [ML-APP-SCI]**.

### Policy and responsibility

- [ ] All authors approve the final manuscript, claims, author order, and submission.
- [ ] Verify whether funding acknowledgments or a Compliance with Ethical Standards statement is required.
- [ ] Review IEEE/ICASSP policy on AI-generated content. Disclose substantive AI-generated manuscript/code/figure content as required, and retain author verification records. Editing/grammar assistance should still be reviewed by the authors.
- [ ] Separately describe LLM use as part of the scientific method; this is not a substitute for any authorship-tool disclosure.
- [ ] Confirm any preprint/arXiv decision against the current Paper Kit before posting.

### Upload and receipt

- [ ] Run final PDF text extraction and page-by-page visual inspection.
- [ ] Check equations, symbols, line breaks, references, fonts, figure clipping, color/grayscale, and URL placeholders.
- [ ] Upload by the internal target: **20:00 Beijing time, 16 September 2026**.
- [ ] Verify the portal-rendered submission and metadata.
- [ ] Save paper number, access code, confirmation page, confirmation email, uploaded PDF checksum, and submission timestamp.
- [ ] Recheck submission status after automated inspection.

### If accepted

- [ ] Acceptance notification: 13 January 2027.
- [ ] Final paper deadline: 27 January 2027.
- [ ] Complete IEEE electronic copyright transfer for the accepted paper.
- [ ] At least one author completes the required non-student registration by 10 February 2027.
- [ ] One author presents in person; otherwise the paper may be removed from IEEE Xplore proceedings.

## 8. Nine-day critical path

This schedule is intentionally a go/no-go plan, not evidence that the work can be compressed safely.

### 8 September

- [ ] Freeze the bounded claim, five algorithm identities, primary endpoint, and exact formal protocol draft.
- [ ] Assign owners for engineering, formal runs, analysis/figures, manuscript, citations, and submission metadata.

### 9--10 September

- [ ] Implement B03--B07.
- [ ] Complete governed tests and a no-cost preflight.
- [ ] Prepare the analysis and plotting pipeline against synthetic fixtures before formal results exist.

### 11 September -- formal go/no-go gate

- [ ] Common event schema and final evaluator pass.
- [ ] All five arms pass identical accounting and checkpoint tests.
- [ ] Protocol, budgets, seeds, stopping rules, and hashes are frozen.
- [ ] Formal run scope and spend are explicitly authorized.
- [ ] **Stop the ICASSP 2027 regular-paper attempt if these conditions are not met.** Do not trade evidence integrity for the deadline.

### 11--13 September

- [ ] Run the preregistered paired-seed formal comparison through governed Slurm jobs.
- [ ] Audit every seed and resolve only implementation failures allowed by the frozen protocol; do not tune on formal outcomes.
- [ ] Run common final-checkpoint evaluation.

### 14 September

- [ ] Freeze aggregate tables, statistical report, Figure 2 source data, and Figure 2 exports.
- [ ] Decide the final claim from the prespecified success/claim-promotion rules.

### 15 September

- [ ] Write the full four-page paper from frozen results.
- [ ] Complete citations, Figure 1, Table 1, limitations, availability, and LLM disclosure.

### 16 September

- [ ] Complete author review and line-by-line numeric audit.
- [ ] Render and inspect the final PDF.
- [ ] Upload by the internal 20:00 Beijing target and retain at least one day for submission-system corrections.

### 17 September

- [ ] Final portal inspection and emergency correction only; do not introduce new scientific results or change the frozen analysis.
- [ ] Absolute listed Beijing deadline: 20:00.

## 9. Decision gates

### Gate A -- permit formal experiments

Pass only if B01--B09 are complete.

### Gate B -- permit manuscript-result claims

Pass only if all prespecified seeds are accounted for, final-checkpoint evaluation is complete, Figure 2 is reproducible from archived ledgers, and the primary statistical result has been evaluated under the frozen rule. Record which prospective threshold tier was reached and apply the corresponding claim restriction; a positive curve alone does not pass Gate B.

### Gate C -- permit submission

Pass only if the four-page manuscript, format checks, author/ORCID metadata, availability statements, AI/LLM disclosures, PDF inspection, and author approvals are complete.

### Scope-dependent claim gate

- **Surrogate-task claim:** may proceed without MD/DFT/wet-lab only if the paper explicitly limits conclusions to the frozen surrogate evaluator and reports applicability/bias limitations.
- **Real material-property claim:** blocked until independent model/applicability and appropriate higher-fidelity or experimental validation close.
