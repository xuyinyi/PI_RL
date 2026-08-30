#!/usr/bin/env python3
"""Select a working Mihomo proxy without printing subscription node names."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import urllib.error
import urllib.parse
import urllib.request


GROUP_TYPES = {"Selector", "URLTest", "Fallback", "LoadBalance", "Relay"}
NON_PROXY_TYPES = GROUP_TYPES | {"Direct", "Reject", "Pass", "Compatible"}


def matches_region(name: str, region: str) -> bool:
    if region == "any":
        return True
    folded = name.casefold()
    if region == "us":
        markers = (
            "美国",
            "美國",
            "🇺🇸",
            "united states",
            "usa",
            "u.s.",
            "us-",
            "us_",
            "us ",
        )
        return any(marker in folded for marker in markers)
    raise ValueError(f"unsupported region: {region}")


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    timeout: float = 15.0,
) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return {} if not body else json.loads(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="http://127.0.0.1:19090")
    parser.add_argument("--test-url", default="https://huggingface.co/robots.txt")
    parser.add_argument("--timeout-ms", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--region", choices=("any", "us"), default="any")
    args = parser.parse_args()

    state = request_json(f"{args.controller}/proxies")
    proxies = state.get("proxies")
    if not isinstance(proxies, dict):
        raise SystemExit("Mihomo controller returned no proxy inventory")

    global_group = proxies.get("GLOBAL")
    if not isinstance(global_group, dict):
        raise SystemExit("Mihomo GLOBAL selector is unavailable")

    members = global_group.get("all")
    if not isinstance(members, list):
        raise SystemExit("Mihomo GLOBAL selector has no candidates")

    candidates: list[str] = []
    for name in members:
        entry = proxies.get(name)
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        if (
            entry.get("type") not in NON_PROXY_TYPES
            and matches_region(name, args.region)
        ):
            candidates.append(name)

    def measure(name: str) -> tuple[str, int] | None:
        encoded_name = urllib.parse.quote(name, safe="")
        query = urllib.parse.urlencode(
            {"url": args.test_url, "timeout": args.timeout_ms}
        )
        try:
            result = request_json(
                f"{args.controller}/proxies/{encoded_name}/delay?{query}",
                timeout=(args.timeout_ms / 1000) + 3,
            )
            delay = result.get("delay")
            if isinstance(delay, int) and delay > 0:
                return name, delay
        except (OSError, TimeoutError, urllib.error.URLError, ValueError):
            return None
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [result for result in pool.map(measure, candidates) if result]

    if not results:
        raise SystemExit(
            f"No working candidates for the test URL (tested={len(candidates)})"
        )

    selected_name, selected_delay = min(results, key=lambda item: item[1])
    request_json(f"{args.controller}/configs", method="PATCH", payload={"mode": "global"})
    encoded_global = urllib.parse.quote("GLOBAL", safe="")
    request_json(
        f"{args.controller}/proxies/{encoded_global}",
        method="PUT",
        payload={"name": selected_name},
    )

    print(f"candidates={len(candidates)}")
    print(f"working={len(results)}")
    print(f"selected_delay_ms={selected_delay}")
    print(f"region={args.region}")
    print("selected_name=REDACTED")


if __name__ == "__main__":
    main()
