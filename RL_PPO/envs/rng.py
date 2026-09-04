from __future__ import annotations

import hashlib
import struct
from typing import Any, Sequence, TypeVar

T = TypeVar("T")


def _encode(value: Any) -> bytes:
    type_name = "%s.%s" % (type(value).__module__, type(value).__qualname__)
    payload = (type_name + ":" + repr(value)).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def stable_uint64(seed: int, *keys: Any) -> int:
    """Process-independent stateless pseudorandom integer.

    This does not consume global Python/NumPy RNG state. Named semantic keys
    prevent an extra draw in one branch from shifting unrelated choices, which
    is required for common-random-number factual/counterfactual completions.
    """

    digest = hashlib.blake2b(digest_size=8, person=b"DAPIGEN0")
    digest.update(struct.pack(">Q", int(seed) & ((1 << 64) - 1)))
    for key in keys:
        digest.update(_encode(key))
    return int.from_bytes(digest.digest(), byteorder="big", signed=False)


def uniform01(seed: int, *keys: Any) -> float:
    value = stable_uint64(seed, *keys) >> 11
    return value / float(1 << 53)


def named_index(seed: int, stream: str, size: int) -> int:
    if int(size) <= 0:
        raise ValueError("size must be positive.")
    index = int(uniform01(seed, stream) * int(size))
    return min(index, int(size) - 1)


def named_choice(values: Sequence[T], seed: int, stream: str) -> T:
    if len(values) == 0:
        raise ValueError("Cannot choose from an empty sequence.")
    return values[named_index(seed, stream, len(values))]


def derive_seed(seed: int, *keys: Any) -> int:
    """Return a signed-int64-compatible child seed."""

    return stable_uint64(seed, *keys) & ((1 << 63) - 1)
