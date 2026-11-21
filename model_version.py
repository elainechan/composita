"""Which model version an OpenRouter route serves right now, and whether that is the pinned one.

Rule: results are only attributed to a model if every call came from one verified version.
SF has no dated routes, but OpenRouter publishes a dated `canonical_slug` per alias
(e.g. openai/gpt-5.6-luna -> openai/gpt-5.6-luna-20260709) and the hosts that serve it
(provider, quantization). run_eval.py checks these against configs/model_versions.json before
a paid run (refuses on mismatch) and again after it (marks the run invalid on mismatch).

Free: public OpenRouter catalogue + SF catalogue, no model calls.

    python model_version.py            # check every pinned route now
    python model_version.py --pin      # write current versions to configs/model_versions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PINS = Path(__file__).parent / "configs" / "model_versions.json"
API = "https://openrouter.ai/api/v1"


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def fingerprint(route: str, catalog: dict | None = None) -> dict:
    """Version facts for one SF route `openrouter/<vendor>/<model>`."""
    slug = route.removeprefix("openrouter/")
    if catalog is None:
        catalog = {m["id"]: m for m in _get(f"{API}/models")["data"]}
    m = catalog.get(slug)
    if m is None:
        return {"route": route, "error": "not in OpenRouter catalogue"}
    hosts = _get(f"{API}/models/{m['canonical_slug']}/endpoints")["data"].get("endpoints", [])
    return {
        "route": route,
        "canonical_slug": m["canonical_slug"],
        "created": m.get("created"),
        "hosts": sorted(f"{h['provider_name']}:{h.get('quantization') or 'unknown'}"
                        for h in hosts),
    }


def compare(pinned: dict, now: dict) -> list[str]:
    """Differences that mean a different model version. Host changes are reported too."""
    problems = []
    if now.get("error"):
        return [now["error"]]
    for key in ("canonical_slug", "created"):
        if pinned.get(key) != now.get(key):
            problems.append(f"{key}: pinned {pinned.get(key)} now {now.get(key)}")
    if set(now.get("hosts", [])) - set(pinned.get("hosts", [])):
        problems.append(f"new hosts: {sorted(set(now['hosts']) - set(pinned['hosts']))}")
    return problems


def load_pins() -> dict:
    return json.loads(PINS.read_text())["routes"]


def check(route: str) -> tuple[dict, list[str]]:
    """Current fingerprint and problems vs the pin. Unpinned route is itself a problem."""
    now = fingerprint(route)
    pins = load_pins()
    if route not in pins:
        return now, [f"{route} not pinned in {PINS.name}"]
    return now, compare(pins[route], now)


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--pin", action="store_true", help="overwrite pins with current versions")
    o = a.parse_args()
    models = json.loads((PINS.parent / "models.json").read_text())
    routes = sorted({r for p in models["cascade_pairs"] for r in (p["cheap"], p["expensive"])}
                    | set(models["frontier_ensemble"]))
    catalog = {m["id"]: m for m in _get(f"{API}/models")["data"]}
    current = {r: fingerprint(r, catalog) for r in routes}
    if o.pin:
        PINS.write_text(json.dumps({
            "pinned_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rule": "results count for a model only if its canonical_slug matched before and "
                    "after every run; never combine runs across versions",
            "routes": current}, indent=1) + "\n")
        print(f"pinned {len(current)} routes in {PINS}")
        return 0
    pins = load_pins()
    bad = 0
    for r in routes:
        problems = compare(pins[r], current[r]) if r in pins else ["not pinned"]
        bad += bool(problems)
        print(f"{'CHANGED' if problems else 'ok':8} {current[r].get('canonical_slug', r)}"
              + (f"  <- {'; '.join(problems)}" if problems else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
