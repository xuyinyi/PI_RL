#!/usr/bin/env python3
"""Create a loopback-only Mihomo runtime config without exposing proxy data."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path


TOP_LEVEL = re.compile(
    r"^(allow-lan|bind-address|port|socks-port|external-controller):.*$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--http-port", type=int, default=17890)
    parser.add_argument("--socks-port", type=int, default=17891)
    parser.add_argument("--controller-port", type=int, default=19090)
    args = parser.parse_args()

    replacements = {
        "allow-lan": "allow-lan: false\n",
        "bind-address": "bind-address: 127.0.0.1\n",
        "port": f"port: {args.http_port}\n",
        "socks-port": f"socks-port: {args.socks_port}\n",
        "external-controller": (
            f"external-controller: 127.0.0.1:{args.controller_port}\n"
        ),
    }

    source_lines = args.source.read_text(encoding="utf-8").splitlines(keepends=True)
    seen: set[str] = set()
    output_lines: list[str] = []

    for line in source_lines:
        match = TOP_LEVEL.match(line)
        if match:
            key = match.group(1)
            output_lines.append(replacements[key])
            seen.add(key)
        else:
            output_lines.append(line)

    for key in replacements:
        if key not in seen:
            output_lines.append(replacements[key])

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.destination.parent,
        prefix=f".{args.destination.name}.",
        delete=False,
    ) as handle:
        handle.writelines(output_lines)
        temporary_path = Path(handle.name)

    os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, args.destination)


if __name__ == "__main__":
    main()
