from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np


def _checkpoint_fingerprint(path_or_identifier: str) -> str:
    """Content fingerprint for a local Hugging Face checkpoint.

    Local filesystem paths are never embedded in the benchmark identity. If the
    argument is not a local directory, the identifier itself is hashed; callers
    should prefer a pinned revision or pass ``checkpoint_fingerprint`` explicitly.
    """

    root = Path(path_or_identifier)
    digest = hashlib.sha256()
    if not root.is_dir():
        digest.update(("identifier:" + str(path_or_identifier)).encode("utf-8"))
        return digest.hexdigest()
    files = [
        item
        for item in root.rglob("*")
        if item.is_file()
        and ".git" not in item.relative_to(root).parts
        and "__pycache__" not in item.relative_to(root).parts
    ]
    if not files:
        raise ValueError("No checkpoint files were found under %s" % root)
    for item in sorted(files, key=lambda value: str(value.relative_to(root))):
        relative = str(item.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        with item.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


class CachedMoleculeEncoder(object):
    """Bounded LRU cache around any single-SMILES encoder."""

    def __init__(
        self,
        encoder: Callable[[str], np.ndarray],
        maximum_entries: int = 100000,
        encoder_version: str = "callable-unknown",
    ) -> None:
        if int(maximum_entries) <= 0:
            raise ValueError("maximum_entries must be positive.")
        self.encoder = encoder
        self.maximum_entries = int(maximum_entries)
        self.encoder_version = str(encoder_version)
        self._cache = OrderedDict()
        self.output_dim = None  # type: Optional[int]
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, smiles: str) -> np.ndarray:
        if smiles in self._cache:
            self.cache_hits += 1
            value = self._cache.pop(smiles)
            self._cache[smiles] = value
            return value.copy()
        self.cache_misses += 1
        value = np.asarray(self.encoder(smiles), dtype=np.float32).reshape(-1)
        if self.output_dim is None:
            self.output_dim = int(value.size)
        elif int(value.size) != self.output_dim:
            raise ValueError("Molecular encoder returned an inconsistent dimension.")
        self._cache[smiles] = value.copy()
        while len(self._cache) > self.maximum_entries:
            self._cache.popitem(last=False)
        return value.copy()

    def encode_batch(self, smiles_batch: Sequence[str]) -> np.ndarray:
        items = [self(item) for item in smiles_batch]
        if not items:
            dim = 0 if self.output_dim is None else int(self.output_dim)
            return np.zeros((0, dim), dtype=np.float32)
        return np.stack(items, axis=0)

    def clear(self) -> None:
        self._cache.clear()

    def diagnostics(self):
        return {
            "encoder_version": self.encoder_version,
            "cache_entries": len(self._cache),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "output_dim": self.output_dim,
        }


class PersistentPolyBERTEncoder(object):
    """Load polyBERT once and provide cached, batched mean-pooled embeddings.

    The released helper reloads the tokenizer and model on every environment
    observation. This implementation loads them once per process and caches
    canonicalized partial structures.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 128,
        maximum_entries: int = 200000,
        maximum_length: Optional[int] = None,
        checkpoint_fingerprint: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(maximum_entries) <= 0:
            raise ValueError("maximum_entries must be positive.")
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.maximum_entries = int(maximum_entries)
        self.maximum_length = maximum_length
        self.model_path = str(model_path)
        local_checkpoint = Path(self.model_path).is_dir()
        if checkpoint_fingerprint is None and not local_checkpoint:
            raise ValueError(
                "A non-local Hugging Face identifier requires an explicit pinned "
                "checkpoint_fingerprint for a reproducible Stage-0 contract."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        self._cache = OrderedDict()
        self.output_dim = int(self.model.config.hidden_size)
        actual_fingerprint = (
            _checkpoint_fingerprint(self.model_path) if local_checkpoint else None
        )
        if checkpoint_fingerprint is not None:
            fingerprint = str(checkpoint_fingerprint)
            if local_checkpoint and fingerprint != actual_fingerprint:
                raise ValueError(
                    "Declared polyBERT fingerprint does not match local checkpoint: "
                    "%s != %s" % (fingerprint, actual_fingerprint)
                )
        else:
            fingerprint = str(actual_fingerprint)
        self.checkpoint_fingerprint = fingerprint
        self.encoder_version = "polybert-sha256:%s" % fingerprint[:24]
        self.cache_hits = 0
        self.cache_misses = 0

    def _encode_uncached(self, smiles_batch: Sequence[str]) -> np.ndarray:
        kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.maximum_length is not None:
            kwargs["max_length"] = int(self.maximum_length)
        encoded = self.tokenizer(list(smiles_batch), **kwargs)
        encoded = dict(
            (name, tensor.to(self.device)) for name, tensor in encoded.items()
        )
        with self.torch.no_grad():
            hidden = self.model(**encoded)[0]
            mask = encoded["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        return pooled.detach().cpu().numpy().astype(np.float32)

    def encode_batch(self, smiles_batch: Sequence[str]) -> np.ndarray:
        items = list(smiles_batch)
        missing = []  # type: List[str]
        seen = set()
        for smiles in items:
            if smiles in self._cache:
                self.cache_hits += 1
            elif smiles not in seen:
                self.cache_misses += 1
                missing.append(smiles)
                seen.add(smiles)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors = self._encode_uncached(batch)
            for smiles, vector in zip(batch, vectors):
                self._cache[smiles] = vector.copy()
                while len(self._cache) > self.maximum_entries:
                    self._cache.popitem(last=False)
        outputs = []
        for smiles in items:
            value = self._cache.pop(smiles)
            self._cache[smiles] = value
            outputs.append(value.copy())
        if not outputs:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.stack(outputs, axis=0)

    def __call__(self, smiles: str) -> np.ndarray:
        return self.encode_batch([smiles])[0]

    def diagnostics(self):
        return {
            "encoder_version": self.encoder_version,
            "cache_entries": len(self._cache),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "output_dim": int(self.output_dim),
            "device": str(self.device),
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
        }


class MorganFingerprintEncoder(object):
    """Deterministic lightweight encoder for environment tests and profiling.

    It is not the default scientific observation encoder. Main PPO comparisons
    should use the same frozen polyBERT checkpoint across all methods.
    """

    def __init__(self, radius: int = 2, number_of_bits: int = 2048) -> None:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        self.Chem = Chem
        self.radius = int(radius)
        self.output_dim = int(number_of_bits)
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.output_dim
        )
        self.encoder_version = "morgan-r%d-b%d" % (self.radius, self.output_dim)

    def __call__(self, smiles: str) -> np.ndarray:
        molecule = self.Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid SMILES supplied to Morgan encoder: %s" % smiles)
        array = np.zeros((self.output_dim,), dtype=np.uint8)
        fingerprint = self.generator.GetFingerprint(molecule)
        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array.astype(np.float32)

    def encode_batch(self, smiles_batch: Sequence[str]) -> np.ndarray:
        vectors = [self(item) for item in smiles_batch]
        if not vectors:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.stack(vectors, axis=0)

    def diagnostics(self):
        return {
            "encoder_version": self.encoder_version,
            "output_dim": int(self.output_dim),
        }
