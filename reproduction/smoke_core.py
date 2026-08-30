#!/usr/bin/env python3
"""Verify the compatibility environment without loading scientific assets."""

import json
import sys

import dgl
import gym
import numpy
import pandas
import pyarrow
import ray
import scipy
import sklearn
import torch
import transformers
from ray.rllib.agents.ppo import PPOTrainer
from rdkit import Chem


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    left = torch.randn((2048, 2048), device="cuda")
    right = torch.randn((2048, 2048), device="cuda")
    product = left @ right
    torch.cuda.synchronize()

    graph = dgl.graph(([0, 1], [1, 2])).to("cuda")
    ethanol = Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))
    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_0": torch.cuda.get_device_name(0),
        "cuda_matmul_finite": bool(torch.isfinite(product).all().item()),
        "dgl": dgl.__version__,
        "dgl_graph_device": str(graph.device),
        "dgl_graph_nodes": graph.num_nodes(),
        "rdkit_smiles": ethanol,
        "gym": gym.__version__,
        "ray": ray.__version__,
        "rllib_trainer": PPOTrainer.__name__,
        "transformers": transformers.__version__,
        "pyarrow": pyarrow.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

