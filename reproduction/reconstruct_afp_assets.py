#!/usr/bin/env python3
"""Reconstruct one AFP reward-model member as a compatibility baseline.

This driver intentionally does not modify the committed QSPR training script.
It fixes only the data-path, scaler-name, selected-member, and sigmoid
inconsistencies documented in ``afp-reconstruction-plan.md``.  Outputs are
written to a staging directory and must be validated before promotion into the
PPO model directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PropertyConfig:
    name: str
    split_seed: int
    model_id: int
    init_seed: int
    sigmoid: bool
    weight_decay: float
    depth: int
    dropout: float
    hidden_dim: int
    init_lr_exponent: float
    layers: int
    lr_reduce_factor: float


CONFIGS: Dict[str, PropertyConfig] = {
    "transmittance(400)": PropertyConfig(
        "transmittance(400)", 825, 43, 922, True, 0.0, 3, 0.25, 116, -3.0, 2, 0.90
    ),
    "cte": PropertyConfig(
        "cte", 854, 56, 125, False, 0.0, 2, 0.35, 87, -1.92401, 1, 0.60
    ),
    "strength": PropertyConfig(
        "strength", 331, 84, 734, False, 0.0, 3, 0.20, 97, -2.25647, 4, 0.80
    ),
    "tg": PropertyConfig(
        "tg", 525, 64, 109, False, 0.0, 1, 0.10, 111, -3.0, 4, 0.90
    ),
}

ATOM_REMOVE = ("Sn", "As", "Ti", "Ca", "Fe")
CODE_COMMIT = "5f692946cbe0d15eede882dfe4cff7fb26eb7d8c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", required=True, choices=tuple(CONFIGS))
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-epoch", type=int, default=1000)
    parser.add_argument("--earlystopping-patience", type=int, default=150)
    parser.add_argument("--lr-schedule-patience", type=int, default=30)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-label", default="manual")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    if args.max_epoch < 1:
        parser.error("--max-epoch must be positive")
    if args.earlystopping_patience < 1:
        parser.error("--earlystopping-patience must be positive")
    return args


def add_repo_to_path(repo_root: Path) -> None:
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def canonicalize_and_scale(
    repo_root: Path, config: PropertyConfig
) -> Tuple[pd.DataFrame, object, Dict[str, object]]:
    from rdkit import Chem
    from RL_PPO.GNN.model.utils.scaler import (
        Standardization,
        zero_oneNormalization,
    )

    source_path = repo_root / "raw_data" / (config.name + ".csv")
    frame = pd.read_csv(source_path)
    raw_rows = len(frame)
    for symbol in ATOM_REMOVE:
        frame = frame.loc[~frame["smile"].str.contains(symbol, na=False)].copy()
    frame.reset_index(drop=True, inplace=True)

    canonical_smiles: List[str] = []
    for row, value in enumerate(frame["smile"]):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError("invalid SMILES at row {}: {!r}".format(row, value))
        canonical_smiles.append(Chem.MolToSmiles(molecule))
    frame["smile"] = canonical_smiles

    if config.sigmoid:
        scaler = zero_oneNormalization()
    else:
        scaler = Standardization(frame["value_mean"])
    original_values = frame["value_mean"].astype(float).to_numpy(copy=True)
    frame["value_mean"] = scaler.Scaler(frame["value_mean"])

    audit = {
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "raw_rows": raw_rows,
        "filtered_rows": len(frame),
        "removed_rows": raw_rows - len(frame),
        "duplicate_canonical_smiles": len(canonical_smiles)
        - len(set(canonical_smiles)),
        "target_min": float(np.min(original_values)),
        "target_max": float(np.max(original_values)),
    }
    if config.sigmoid:
        audit["scaler"] = {"type": "zero_one", "range": float(scaler.range)}
    else:
        audit["scaler"] = {
            "type": "population_standardization",
            "mu": float(scaler.mu),
            "dev": float(scaler.dev),
        }
    return frame, scaler, audit


def split_indices(length: int, seed: int) -> Dict[str, List[int]]:
    indices = list(range(length))
    np.random.RandomState(int(seed)).shuffle(indices)
    train_end = int(0.8 * length)
    validation_end = train_end + int(0.1 * length)
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
        "all": list(range(length)),
    }


def metric_values(metrics: object) -> Dict[str, float]:
    return {
        "R2": float(metrics.R2),
        "MAE": float(metrics.MAE),
        "RMSE": float(metrics.RMSE),
        "SSE": float(metrics.SSE),
        "MAPE": float(metrics.MAPE),
        "MaxErr": float(metrics.MaxErr),
        "AIC": float(metrics.AIC),
        "BIC": float(metrics.BIC),
    }


def train_batch(model, optimizer, scaling, batch, device, parameter_count):
    from QSPR.GNN.utils.metrics import Metrics

    graph, targets, smiles = batch
    graph = graph.to(device=device)
    targets = targets.float().to(device=device)
    model.train()
    optimizer.zero_grad()
    score = model.forward(graph, graph.ndata["feat"], graph.edata["feat"])
    loss = model.loss(score, targets)
    loss.backward()
    optimizer.step()
    true = scaling.ReScaler(targets.detach().cpu())
    predict = scaling.ReScaler(score.detach().cpu())
    return float(loss.detach().cpu()), Metrics(true, predict, parameter_count), smiles


def evaluate_batch(model, scaling, batch, device, parameter_count):
    from QSPR.GNN.utils.metrics import Metrics

    graph, targets, smiles = batch
    graph = graph.to(device=device)
    targets = targets.float().to(device=device)
    model.eval()
    with __import__("torch").no_grad():
        score = model.forward(graph, graph.ndata["feat"], graph.edata["feat"])
        loss = model.loss(score, targets)
    true = scaling.ReScaler(targets.detach().cpu())
    predict = scaling.ReScaler(score.detach().cpu())
    metrics = Metrics(true, predict, parameter_count)
    return float(loss.detach().cpu()), metrics, predict, true, smiles


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    add_repo_to_path(repo_root)

    import dgl
    import rdkit
    import torch
    from torch.utils.data import DataLoader, Subset
    from QSPR.GNN.data.csv_dataset import MoleculeCSVDataset
    from QSPR.GNN.networks.AttentiveFP import AttentiveFPNet
    from QSPR.GNN.src.dgltools import collate_molgraphs
    from QSPR.GNN.src.feature.atom_featurizer import classic_atom_featurizer
    from QSPR.GNN.src.feature.bond_featurizer import classic_bond_featurizer
    from QSPR.GNN.src.feature.mol_featurizer import classic_mol_featurizer
    from QSPR.GNN.utils.Set_Seed_Reproducibility import set_seed
    from QSPR.GNN.utils.mol2graph import smiles_2_bigraph
    from RL_PPO.GNN.model.networks.AttentiveFP import AttentiveFPNet as PPOAttentiveFPNet

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config = CONFIGS[args.property]
    run_dir = output_root / "runs" / config.name
    asset_dir = output_root / "assets"
    cache_dir = output_root / "cache"
    error_dir = output_root / "error_log"
    for directory in (run_dir, asset_dir, cache_dir, error_dir):
        directory.mkdir(parents=True, exist_ok=True)

    set_seed(seed=2023)
    random.seed(2023)
    expected_init_seeds = [random.randint(0, 1000) for _ in range(100)]
    if expected_init_seeds[config.model_id] != config.init_seed:
        raise RuntimeError("target initialization seed no longer matches source sequence")

    frame, scaling, data_audit = canonicalize_and_scale(repo_root, config)
    scaler_path = asset_dir / (config.name + "_scaler.pkl")
    with scaler_path.open("wb") as handle:
        pickle.dump(scaling, handle)

    cache_path = cache_dir / config.name
    if args.rebuild_cache and cache_path.exists():
        raise RuntimeError(
            "refusing to delete an existing cache; use a new output root instead"
        )
    dataset = MoleculeCSVDataset(
        frame,
        smiles_2_bigraph,
        classic_atom_featurizer,
        classic_bond_featurizer,
        classic_mol_featurizer,
        str(cache_path),
        load=cache_path.exists(),
        error_log=str(error_dir / (config.name + ".csv")),
    )
    indices = split_indices(len(dataset), config.split_seed)
    if sorted(indices["train"] + indices["validation"] + indices["test"]) != list(
        range(len(dataset))
    ):
        raise RuntimeError("split coverage check failed")

    split_rows = []
    for split_name in ("train", "validation", "test"):
        for index in indices[split_name]:
            split_rows.append(
                {
                    "row_index": index,
                    "split": split_name,
                    "smile": dataset.smiles[index],
                }
            )
    pd.DataFrame(split_rows).sort_values("row_index").to_csv(
        run_dir / "split.csv", index=False
    )

    def make_batch(split_name: str):
        subset = Subset(dataset, indices[split_name])
        loader = DataLoader(
            subset,
            collate_fn=collate_molgraphs,
            batch_size=len(subset),
            shuffle=False,
            num_workers=0,
        )
        return next(iter(loader))

    batches = {name: make_batch(name) for name in ("train", "validation", "test", "all")}

    params = {
        "Dataset": config.name,
        "init_lr": 10 ** config.init_lr_exponent,
        "min_lr": args.min_lr,
        "sigmoid": config.sigmoid,
        "weight_decay": config.weight_decay,
        "lr_reduce_factor": config.lr_reduce_factor,
        "lr_schedule_patience": args.lr_schedule_patience,
        "earlystopping_patience": args.earlystopping_patience,
        "max_epoch": args.max_epoch,
    }
    net_params = {
        "num_atom_type": 36,
        "num_bond_type": 12,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "depth": config.depth,
        "layers": config.layers,
        "sigmoid": config.sigmoid,
        "residual": False,
        "batch_norm": False,
        "layer_norm": False,
        "device": args.device,
    }

    torch.manual_seed(config.init_seed)
    device = torch.device(args.device)
    model = AttentiveFPNet(net_params).to(device=device)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["init_lr"], weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params["lr_reduce_factor"],
        patience=params["lr_schedule_patience"],
    )

    checkpoint_path = run_dir / "best-checkpoint.pt"
    history_path = run_dir / "history.jsonl"
    best_validation_loss = float("inf")
    best_epoch = -1
    earlystop_counter = 0
    stop_reason = "max_epoch"
    start_time = time.time()

    with history_path.open("w", encoding="utf-8") as history_handle:
        for epoch in range(args.max_epoch):
            epoch_start = time.time()
            train_loss, train_metrics, _ = train_batch(
                model, optimizer, scaling, batches["train"], device, parameter_count
            )
            validation_loss, validation_metrics, _, _, _ = evaluate_batch(
                model, scaling, batches["validation"], device, parameter_count
            )
            test_loss, test_metrics, _, _, _ = evaluate_batch(
                model, scaling, batches["test"], device, parameter_count
            )
            scheduler.step(validation_loss)
            current_lr = float(optimizer.param_groups[0]["lr"])

            improved = validation_loss <= best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                earlystop_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                earlystop_counter += 1

            record = {
                "epoch": epoch,
                "seconds": time.time() - epoch_start,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "test_loss": test_loss,
                "train_R2": float(train_metrics.R2),
                "validation_R2": float(validation_metrics.R2),
                "test_R2": float(test_metrics.R2),
                "improved": improved,
                "earlystop_counter": earlystop_counter,
            }
            history_handle.write(json.dumps(record, sort_keys=True) + "\n")
            history_handle.flush()
            if epoch == 0 or improved or (epoch + 1) % 10 == 0:
                print(json.dumps(record, sort_keys=True), flush=True)

            if current_lr < params["min_lr"]:
                stop_reason = "minimum_learning_rate"
                break
            if earlystop_counter >= params["earlystopping_patience"]:
                stop_reason = "early_stopping"
                break

    if not checkpoint_path.exists():
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    metric_report: Dict[str, Dict[str, float]] = {}
    prediction_rows: List[Dict[str, object]] = []
    for split_name in ("train", "validation", "test", "all"):
        loss, metrics, predict, true, smiles = evaluate_batch(
            model, scaling, batches[split_name], device, parameter_count
        )
        metric_report[split_name] = {"scaled_mse_loss": loss, **metric_values(metrics)}
        for smile, true_value, prediction in zip(
            smiles, true.numpy().reshape(-1), predict.numpy().reshape(-1)
        ):
            prediction_rows.append(
                {
                    "split": split_name,
                    "smile": smile,
                    "true": float(true_value),
                    "prediction": float(prediction),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    if not np.isfinite(predictions[["true", "prediction"]].to_numpy()).all():
        raise RuntimeError("non-finite value detected in final predictions")
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    model_name = "Ensemble_{}_AFP".format(config.name)
    weight_path = asset_dir / (model_name + "_{}.pt".format(config.model_id))
    settings_path = asset_dir / (model_name + "_settings.csv")
    torch.save(model.state_dict(), weight_path)

    settings = {}
    settings.update({"param:" + key: value for key, value in params.items()})
    settings.update({"net_param:" + key: value for key, value in net_params.items()})
    for split_name, prefix in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
        ("all", "all"),
    ):
        for metric_name in ("R2", "MAE", "RMSE", "SSE", "MAPE"):
            settings[prefix + "_" + metric_name] = metric_report[split_name][metric_name]
    settings.update(
        {
            "init_seed": config.init_seed,
            "seed": config.split_seed,
            "best_epoch": best_epoch,
            "stop_reason": stop_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "comment": "reconstructed AFP compatibility baseline; not author-supplied weights",
            "index": config.model_id,
        }
    )
    pd.DataFrame([settings]).to_csv(settings_path, index=False)

    # Exercise the exact PPO network copy and strict state-dict loading before
    # declaring this property complete.
    reloaded_settings = pd.read_csv(settings_path, index_col=-1)
    loaded_net_params = {
        column.split(":", 1)[1]: reloaded_settings.loc[config.model_id, column]
        for column in reloaded_settings.columns
        if column.startswith("net_param:")
    }
    compatibility_model = PPOAttentiveFPNet(loaded_net_params).to(device=device)
    compatibility_model.load_state_dict(
        torch.load(weight_path, map_location=device), strict=True
    )

    write_json(run_dir / "metrics.json", metric_report)
    assets = {
        path.name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (scaler_path, weight_path, settings_path)
    }
    manifest = {
        "label": "reconstructed AFP compatibility baseline",
        "original_author_weights": False,
        "run_label": args.run_label,
        "property": asdict(config),
        "parameters": params,
        "network_parameters": net_params,
        "data_audit": data_audit,
        "split_rows": {name: len(values) for name, values in indices.items()},
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.time() - start_time,
        "parameter_count": parameter_count,
        "code": {
            "expected_upstream_commit": CODE_COMMIT,
            "worktree_commit": git_commit(repo_root),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "dgl": dgl.__version__,
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "assets": assets,
        "metrics": metric_report,
    }
    write_json(run_dir / "run-manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "property": config.name,
                "best_epoch": best_epoch,
                "test_R2": metric_report["test"]["R2"],
                "test_RMSE": metric_report["test"]["RMSE"],
                "assets": sorted(assets),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
