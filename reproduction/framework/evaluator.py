"""One generation and metric protocol shared by every algorithm adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

from reproduction.framework.contracts import AlgorithmAdapter, ContractError
from reproduction.framework.io import json_value


CSV_COLUMNS = (
    "PI",
    "A_1",
    "A_2",
    "transmittance",
    "cte",
    "strength",
    "tg",
    "SaScore",
    "reward",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CommonEvaluator:
    """Generate samples and compute the frozen DAPiGen comparison metrics."""

    def __init__(
        self,
        reference_csv: Optional[Path],
        reference_column: str = "smile",
        compute_paper_metrics: bool = True,
        protocol: str = "dapigen-common-v1",
    ) -> None:
        self.reference_csv = reference_csv
        self.reference_column = reference_column
        self.compute_paper_metrics = compute_paper_metrics
        self.protocol = protocol

    def generate(
        self,
        adapter: AlgorithmAdapter,
        env_class: Type[Any],
        env_config: Mapping[str, Any],
        sample_count: int,
        output_csv: Path,
        seed: int,
        explore: bool,
        progress_every: int,
        max_episode_steps: int,
    ) -> Dict[str, Any]:
        import numpy as np

        if explore:
            raise ContractError("common evaluation requires explore=false")
        random.seed(seed)
        np.random.seed(seed)
        environment = env_class(dict(env_config))
        if hasattr(environment, "seed"):
            environment.seed(seed)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        partial = output_csv.with_suffix(output_csv.suffix + ".partial")
        started = time.time()
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for sample_index in range(sample_count):
                action_1: List[int] = []
                action_2: List[int] = []
                observation = environment.reset()
                reward = 0.0
                for _ in range(max_episode_steps):
                    action = adapter.act(observation, explore=False)
                    observation, reward, done, info = environment.step(action)
                    previous_action = info["prev_action"]
                    action_1.append(int(previous_action[0]))
                    action_2.append(int(previous_action[1]))
                    if done:
                        writer.writerow(
                            {
                                "PI": environment.PI,
                                "A_1": ",".join(map(str, action_1)),
                                "A_2": ",".join(map(str, action_2)),
                                "transmittance": environment.transmittance,
                                "cte": environment.cte,
                                "strength": environment.strength,
                                "tg": environment.tg,
                                "SaScore": environment.SaScore,
                                "reward": float(reward),
                            }
                        )
                        break
                else:
                    raise ContractError(
                        "evaluation episode exceeded {} steps".format(max_episode_steps)
                    )

                completed = sample_index + 1
                if (
                    completed == 1
                    or completed % progress_every == 0
                    or completed == sample_count
                ):
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "event": "evaluation_progress",
                                "completed": completed,
                                "total": sample_count,
                                "elapsed_seconds": time.time() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        os.replace(str(partial), str(output_csv))
        summary = self.evaluate_csv(output_csv)
        summary.update(
            {
                "evaluation_seed": seed,
                "explore": False,
                "elapsed_seconds": time.time() - started,
                "output_csv": str(output_csv),
            }
        )
        return json_value(summary)

    def evaluate_csv(self, generated_csv: Path) -> Dict[str, Any]:
        import pandas as pd
        from rdkit import Chem, DataStructs
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan

        frame = pd.read_csv(generated_csv)
        missing_columns = sorted(set(CSV_COLUMNS) - set(frame.columns))
        if missing_columns:
            raise ContractError(
                "generation CSV is missing columns: {}".format(
                    ", ".join(missing_columns)
                )
            )

        valid_smiles: List[str] = []
        canonical_smiles: List[str] = []
        for value in frame["PI"].tolist():
            if not isinstance(value, str) or value == "None":
                continue
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                continue
            valid_smiles.append(value)
            canonical_smiles.append(Chem.MolToSmiles(molecule))

        rewards = pd.to_numeric(frame["reward"], errors="coerce").dropna()
        sample_count = int(len(frame))
        valid_count = len(valid_smiles)
        unique_count = len(set(canonical_smiles))
        summary: Dict[str, Any] = {
            "protocol": self.protocol,
            "samples": sample_count,
            "valid_samples": valid_count,
            "validity": valid_count / sample_count if sample_count else None,
            "canonical_unique_valid_PI": unique_count,
            "uniqueness": unique_count / valid_count if valid_count else None,
            "mean_reward": float(rewards.mean()) if len(rewards) else None,
            "max_reward": float(rewards.max()) if len(rewards) else None,
            "generation_csv_sha256": _sha256(generated_csv),
            "metric_definitions": {
                "validity_denominator": "all generated rows",
                "uniqueness_denominator": "RDKit-valid generated rows",
                "fingerprint": "Morgan radius=3 nBits=2048",
                "novelty": "fraction with max training-set Tanimoto < 0.4",
                "diversity": "1 - mean pairwise generated-set Tanimoto",
                "frag": "BRICS fragment-count cosine similarity to training set",
                "snn": "mean maximum Tanimoto similarity to training set",
            },
        }

        if not self.compute_paper_metrics:
            summary["paper_metrics_computed"] = False
            return json_value(summary)
        if self.reference_csv is None:
            raise ContractError("paper metrics require a reference CSV")
        reference_frame = pd.read_csv(self.reference_csv)
        if self.reference_column not in reference_frame.columns:
            raise ContractError(
                "reference column {} is missing from {}".format(
                    self.reference_column, self.reference_csv
                )
            )
        reference_smiles: List[str] = []
        reference_molecules = []
        for value in reference_frame[self.reference_column].tolist():
            if not isinstance(value, str):
                continue
            molecule = Chem.MolFromSmiles(value)
            if molecule is not None:
                reference_smiles.append(value)
                reference_molecules.append(molecule)
        if not reference_molecules:
            raise ContractError("reference set contains no valid molecules")

        summary["paper_metrics_computed"] = True
        summary["reference_csv"] = str(self.reference_csv)
        summary["reference_valid_samples"] = len(reference_smiles)
        if not valid_smiles:
            summary.update({"novelty": None, "diversity": None, "frag": None, "snn": None})
            return json_value(summary)

        generated_molecules = [Chem.MolFromSmiles(value) for value in canonical_smiles]
        reference_fps = [Morgan(molecule, 3, 2048) for molecule in reference_molecules]
        generated_fps = [Morgan(molecule, 3, 2048) for molecule in generated_molecules]

        novel_count = 0
        for fingerprint in generated_fps:
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprint, reference_fps
            )
            if max(similarities) < 0.4:
                novel_count += 1
        summary["novelty"] = novel_count / len(generated_fps)

        if len(generated_fps) < 2:
            summary["diversity"] = None
        else:
            similarity_sum = 0.0
            for index, fingerprint in enumerate(generated_fps):
                similarity_sum += sum(
                    DataStructs.BulkTanimotoSimilarity(
                        fingerprint, generated_fps[:index]
                    )
                )
            pair_count = len(generated_fps) * (len(generated_fps) - 1) / 2
            summary["diversity"] = 1.0 - similarity_sum / pair_count

        from RL_PPO.moldr.evaluation import FragMetric, SNNMetric

        summary["frag"] = float(
            FragMetric()(ref=reference_smiles, gen=valid_smiles)
        )
        summary["snn"] = float(
            SNNMetric()(ref=reference_smiles, gen=valid_smiles)
        )
        return json_value(summary)
