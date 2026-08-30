#!/usr/bin/env python3
"""Validate promoted reconstructed AFP assets through the real PPO interfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


EXPECTED = {
    "transmittance(400)": {"model_id": 43, "init_seed": 922, "sigmoid": True},
    "cte": {"model_id": 56, "init_seed": 125, "sigmoid": False},
    "strength": {"model_id": 84, "init_seed": 734, "sigmoid": False},
    "tg": {"model_id": 64, "init_seed": 109, "sigmoid": False},
}

SMOKE_SMILES = (
    "Cc1ccc(-c2c3ccccc3nc3cc(N4C(=O)CC(C)"
    "(c5ccc6c(c5)C(=O)N(C)C6=O)C4=O)ccc23)cc1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_mapping(values: Dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in values.values())


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    training_root = args.training_root.resolve()
    output_path = args.output.resolve()
    sys.path.insert(0, str(repo_root))

    import torch
    from rdkit.Chem import AllChem as Chem
    from reproduction.check_assets import build_report
    from RL_PPO.GNN.benchmarks import Benchmark
    from RL_PPO.moldr.config import get_default_config
    from RL_PPO.moldr.env import PIEnvValueMax
    from RL_PPO.utils.chemutils import get_mol
    from RL_PPO.utils.genPI import generate_PI

    if not torch.cuda.is_available():
        raise RuntimeError("runtime validation requires its allocated Slurm GPU")

    model_dir = repo_root / "RL_PPO" / "GNN" / "model"
    property_report = {}
    for property_name, expected in EXPECTED.items():
        model_stem = "Ensemble_{}_AFP".format(property_name)
        paths = {
            "scaler": model_dir / (property_name + "_scaler.pkl"),
            "weight": model_dir
            / (model_stem + "_{}.pt".format(expected["model_id"])),
            "settings": model_dir / (model_stem + "_settings.csv"),
        }
        manifest_path = training_root / "runs" / property_name / "run-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["property"]["model_id"] != expected["model_id"]:
            raise AssertionError("manifest model id mismatch for " + property_name)
        if manifest["property"]["init_seed"] != expected["init_seed"]:
            raise AssertionError("manifest initialization seed mismatch for " + property_name)

        hashes = {}
        for kind, path in paths.items():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(str(path))
            digest = sha256_file(path)
            hashes[kind] = digest
            recorded = manifest["assets"][path.name]["sha256"]
            if digest != recorded:
                raise AssertionError("promoted asset hash mismatch: " + path.name)

        settings = pd.read_csv(paths["settings"], index_col=-1)
        model_id = expected["model_id"]
        if settings.index.tolist() != [model_id]:
            raise AssertionError("settings index mismatch for " + property_name)
        if int(settings.loc[model_id, "init_seed"]) != expected["init_seed"]:
            raise AssertionError("settings initialization seed mismatch for " + property_name)
        if bool(settings.loc[model_id, "net_param:sigmoid"]) != expected["sigmoid"]:
            raise AssertionError("settings sigmoid mismatch for " + property_name)
        property_report[property_name] = {
            "model_id": model_id,
            "init_seed": expected["init_seed"],
            "sigmoid": expected["sigmoid"],
            "hashes": hashes,
            "metrics": manifest["metrics"],
        }

    benchmark = Benchmark([SMOKE_SMILES])
    benchmark_values = {
        "transmittance": float(benchmark.transmittance),
        "cte": float(benchmark.cte),
        "strength": float(benchmark.strength),
        "tg": float(benchmark.tg),
        "SA": float(benchmark.ScoreSA),
        "reward": float(benchmark.Score),
    }
    if not finite_mapping(benchmark_values):
        raise AssertionError("Benchmark produced non-finite values")
    if not 0.0 <= benchmark_values["transmittance"] <= 100.0:
        raise AssertionError("transmittance is outside its sigmoid-scaled range")
    if not -100.0 <= benchmark_values["cte"] <= 300.0:
        raise AssertionError("CTE smoke prediction is implausible")
    if not -100.0 <= benchmark_values["strength"] <= 800.0:
        raise AssertionError("strength smoke prediction is implausible")
    if not 0.0 <= benchmark_values["tg"] <= 800.0:
        raise AssertionError("Tg smoke prediction is implausible")
    if not 0.0 <= benchmark_values["reward"] <= 1.0:
        raise AssertionError("reward is outside [0, 1]")

    block_root = repo_root / "RL_PPO" / "outputs" / "building_blocks"
    dianhydrides = pd.read_csv(block_root / "blocks_dianhydride.csv")["block"].tolist()
    diamines = pd.read_csv(block_root / "blocks_diamine.csv")["block"].tolist()
    dianhydride_pattern = Chem.MolFromSmarts("[#8]=[#6]1[#6][#6][#6](=[#8])[#8]1")
    diamine_pattern = Chem.MolFromSmarts("[#7H2]")
    complete_dianhydrides = [
        (index, smiles)
        for index, smiles in enumerate(dianhydrides)
        if "*" not in smiles
        and len(get_mol(smiles).GetSubstructMatches(dianhydride_pattern)) == 2
    ]
    complete_diamines = [
        (index, smiles)
        for index, smiles in enumerate(diamines)
        if "*" not in smiles
        and len(get_mol(smiles).GetSubstructMatches(diamine_pattern)) == 2
    ]
    terminal_action = None
    for dianhydride_index, dianhydride in complete_dianhydrides:
        for diamine_index, diamine in complete_diamines:
            random.seed(2023)
            try:
                generated = generate_PI(dianhydride, diamine)
            except Exception:
                generated = None
            if generated:
                terminal_action = (dianhydride_index, diamine_index)
                break
        if terminal_action is not None:
            break
    if terminal_action is None:
        raise AssertionError("no complete-monomer action generated a valid PI")

    config = get_default_config(
        PIEnvValueMax,
        Benchmark,
        dianhydrides,
        diamines,
        model_path=repo_root / "RL_PPO" / "models",
        num_workers=0,
        num_gpus=1,
        length=60,
        step_length=5,
    )
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    environment = PIEnvValueMax(config["env_config"])
    observation = environment.reset()
    if observation.shape != (1200,) or not np.isfinite(observation).all():
        raise AssertionError("environment reset observation failed")
    random.seed(2023)
    preview_dianhydrides, preview_diamines = environment.reassemble(terminal_action)
    preview = {
        "action": list(terminal_action),
        "dianhydrides": [
            {
                "smiles": value,
                "contains_star": "*" in value,
                "matches": len(
                    get_mol(value).GetSubstructMatches(
                        environment.dianhydride_pattern
                    )
                ),
            }
            for value in preview_dianhydrides
        ],
        "diamines": [
            {
                "smiles": value,
                "contains_star": "*" in value,
                "matches": len(
                    get_mol(value).GetSubstructMatches(
                        environment.diamine_pattern
                    )
                ),
            }
            for value in preview_diamines
        ],
    }
    print("ENVIRONMENT_PREVIEW=" + json.dumps(preview, sort_keys=True), flush=True)
    next_observation, reward, done, info = environment.step(terminal_action)
    print(
        "ENVIRONMENT_RESULT="
        + json.dumps(
            {
                "done": bool(done),
                "flag_dianhydride": bool(environment.flag_dianhydride),
                "flag_diamine": bool(environment.flag_diamine),
                "PI": environment.PI,
                "reward": float(reward),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if next_observation.shape != (1200,) or not np.isfinite(next_observation).all():
        raise AssertionError("environment step observation failed")
    if not done or environment.PI in (None, "None"):
        raise AssertionError("the fixed complete-monomer action did not reach a PI terminal state")
    if not math.isfinite(float(reward)) or not 0.0 <= float(reward) <= 1.0:
        raise AssertionError("environment returned an invalid terminal reward")

    asset_report = build_report(repo_root)
    if not asset_report["ppo_runtime_files_ready"]:
        raise AssertionError("PPO runtime file gate is not ready")
    report = {
        "status": "passed",
        "label": "reconstructed AFP compatibility baseline",
        "original_author_weights": False,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "properties": property_report,
        "benchmark_smoke": {"smiles": SMOKE_SMILES, **benchmark_values},
        "environment_smoke": {
            "reset_shape": list(observation.shape),
            "action": list(terminal_action),
            "done": bool(done),
            "reward": float(reward),
            "generated_PI": environment.PI,
            "properties": {
                "transmittance": float(environment.transmittance),
                "cte": float(environment.cte),
                "strength": float(environment.strength),
                "tg": float(environment.tg),
                "SA": float(environment.SaScore),
            },
            "info_keys": sorted(info),
        },
        "asset_gate": asset_report,
        "scientific_identity_boundary": (
            "runtime files are complete, but the user-selected polyBERT mirror and "
            "reconstructed AFP weights are not verified as the authors' original assets"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
