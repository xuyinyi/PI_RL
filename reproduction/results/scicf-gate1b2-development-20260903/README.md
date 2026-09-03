# SciCF Gate 1B.2 nonlinear-validity development gate

Gate 1B.2 stopped with a **development-entry no-go**. The sealed test set was
not collected, read, or evaluated.

The protocol changed one component from Gate 1B.1: the linear validity filter
was replaced by one pre-declared `RandomForestRegressor`. The model family and
all parameters were fixed before the additional development labels were
collected, and no hyperparameter or model-family search was performed. The
policy-conditioned Ridge gain ranker was reused byte-for-byte from the frozen
Gate 1B.1 model manifest and was not refit.

The original structure-isolated train, early-dev, middle-dev, and late-dev data
were reused. Twenty-four new late-dev trajectory seeds (`4210` through `4233`)
were added solely to test whether abstention calibration had both classes. The
collector was code-restricted to `split=dev` and `stage=late`.

## Development result

| Check | Result | Pre-declared requirement | Decision |
|---|---:|---:|---|
| train/combined-dev structure overlap | 0 keys | 0 keys | pass |
| original-dev cross-timestep trajectories | 8/24 = 0.3333 | >= 0.25 | pass |
| early rescue headroom captured | 0.6875 | positive and >= 0.25 | pass |
| middle BestGain@4 headroom captured | 0.8533 | positive and >= 0.25 | pass |
| middle NDCG@4 headroom captured | 0.2350 | positive and >= 0.25 | fail |
| late abstention calibration | 0 positive / 32 no-positive trajectories | both classes required | fail |

The nonlinear validity filter improved mean early RescueRate@4 over exact
expected Random by `0.515625`, capturing `68.75%` of available
Oracle-minus-Random headroom. This closes the specific early-rescue deficiency
seen in Gate 1B.1 on the unchanged early development set.

Middle BestGain@4 remained positive by `0.0159642975` and captured `85.33%` of
available headroom. Middle NDCG@4 remained positive by `0.1926581964`, but
captured only `23.50%`, below the frozen `25%` entry threshold. Although the
gain ranker itself was unchanged, the replacement validity filter participates
in the validity-first lexicographic selection rule and therefore can change the
four selected candidates and their NDCG.

All eight original and all twenty-four expanded late-dev pools lacked a
positive-gain candidate. An always-abstain rule would be untestable against
opportunity response on these data, so the threshold remained
`not-calibratable`. The implemented late selector still supports zero through
four selections; this dataset cannot calibrate when it should leave zero.

## Frozen model and data contract

The validity filter used all 576 original Gate 1B.1 train rows and 87
request-time descriptors without policy probability. Its fixed parameters are
512 trees, squared-error criterion, maximum depth 8, minimum split size 4,
minimum leaf size 2, square-root feature subsampling, bootstrap sampling,
random state 71021, and one model thread. Server versions were scikit-learn
1.2.2 and joblib 1.4.2.

The combined development set contained 200 unique structure keys versus 160
train keys, with zero overlap. The expanded late data used 2,002 new atomic
Oracle calls: 1,176 factual, 826 counterfactual, and zero evaluation calls. No
LLM call or PPO update occurred.

## Provenance and sealed-test boundary

- Protocol and implementation commit:
  `c899a1ac22ec3e627ce04e2b1000123cd8e624bd`.
- Slurm 4625: dependency/version probe completed.
- Slurm 4626: 43/43 SciCF tests passed.
- Slurm 4627: 24-trajectory late-dev-only collection completed.
- Slurm 4628: frozen development fit and no-go decision completed.
- Late expansion report SHA-256:
  `82c650cb69f11af18cb7aae54f069a96d9826f0c0b2580630193d5ed979128e8`.
- Development report SHA-256:
  `6d6127658d2855d77041dc08d041358479d5b88b1591ca7ac066efa57b8dfffe`.
- Frozen development model manifest SHA-256:
  `10b8c9b603bfaec1ff1600c22ca2fb1cb5cb96e5904d95267e42422a8202f8fd`.
- Serialized validity filter SHA-256:
  `9964539e604bde893f500beeedc39df1b0aa94bb42ebea692dc405414ab93047`.

Full server artifacts are under:

`/home/wch/workspaces/DAPiGen-reproduction/runs/scicf/gate1b2-development-20260903-v1`

The final artifact tree contains only `late-expansion-v1/dev/late` and the
development model directory. The parent Gate 1B.1 `test` directory remains
absent. The frozen manifest records `sealed_test_eligible=false`,
`test_collection_authorized=false`, `test_evaluation_authorized=false`, and
`test_evaluations_completed=0`.

This result supports only a development-set diagnosis: nonlinear deterministic
descriptors can rescue early-invalid candidates, while the current joint
validity-first/gain ordering misses the middle NDCG threshold and the frozen
late policy/pool is saturated for abstention calibration. It does not establish
sealed-test generalization, LLM acquisition benefit, PPO improvement,
independent-oracle robustness, or wet-lab validity.
