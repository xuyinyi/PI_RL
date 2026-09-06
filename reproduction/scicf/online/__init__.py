"""Feature-gated online SciCF-PPO architecture smoke."""

from .contracts import (
    ONLINE_PROTOCOL_ID,
    OnlineCandidate,
    OnlineVerification,
    PairwiseRefinementConfig,
)

__all__ = (
    "ONLINE_PROTOCOL_ID",
    "OnlineCandidate",
    "OnlineVerification",
    "PairwiseRefinementConfig",
)
