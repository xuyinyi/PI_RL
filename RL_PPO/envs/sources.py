"""Canonical evaluator-source labels for Stage-0 budget auditing."""

PPO_ON_POLICY = "ppo/on_policy"
POLICY_CC_ON_POLICY = "policy_cc/on_policy"
POLICY_CC_FACTUAL = "policy_cc/factual"
POLICY_CC_COUNTERFACTUAL = "policy_cc/counterfactual"
MCC_PPO_ON_POLICY = "mcc_ppo/on_policy"
MCC_PPO_FACTUAL = "mcc_ppo/factual"
MCC_PPO_COUNTERFACTUAL = "mcc_ppo/counterfactual"
EVALUATION = "evaluation"
ENVIRONMENT_REGRESSION = "environment_regression"

ALL_STAGE0_SOURCES = frozenset(
    (
        PPO_ON_POLICY,
        POLICY_CC_ON_POLICY,
        POLICY_CC_FACTUAL,
        POLICY_CC_COUNTERFACTUAL,
        MCC_PPO_ON_POLICY,
        MCC_PPO_FACTUAL,
        MCC_PPO_COUNTERFACTUAL,
        EVALUATION,
        ENVIRONMENT_REGRESSION,
    )
)
