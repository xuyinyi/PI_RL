"""Non-LLM chemistry heuristic used as a Gate 1 acquisition baseline."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

from reproduction.scicf.acquisition.base import AcquisitionCandidate
from reproduction.scicf.core.records import Intervention


def morgan_distance(factual_smiles: str, alternative_smiles: str) -> float:
    """Return Morgan-fingerprint Tanimoto distance between two building blocks."""

    from rdkit import Chem, DataStructs
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

    factual = Chem.MolFromSmiles(factual_smiles)
    alternative = Chem.MolFromSmiles(alternative_smiles)
    if factual is None or alternative is None:
        raise ValueError("chemistry heuristic requires valid factual and alternative SMILES")
    factual_fp = GetMorganFingerprintAsBitVect(factual, radius=2, nBits=2048)
    alternative_fp = GetMorganFingerprintAsBitVect(alternative, radius=2, nBits=2048)
    return 1.0 - float(DataStructs.TanimotoSimilarity(factual_fp, alternative_fp))


def annotate_candidates(
    interventions: Sequence[Intervention],
    policy_scores: Optional[Mapping[str, float]] = None,
) -> Tuple[AcquisitionCandidate, ...]:
    """Attach policy and chemistry scores without changing candidate legality."""

    result = []
    for intervention in interventions:
        factual_structure = intervention.metadata.get("factual_structure")
        if not factual_structure:
            raise ValueError("intervention is missing its factual chemical structure")
        distance = morgan_distance(
            str(factual_structure), intervention.alternative_structure
        )
        result.append(
            AcquisitionCandidate(
                intervention=intervention,
                policy_score=(
                    float(policy_scores[intervention.intervention_id])
                    if policy_scores is not None
                    else None
                ),
                structural_score=distance,
                heuristic_score=distance,
            )
        )
    return tuple(result)
