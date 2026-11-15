"""Snapshot OpenRouter prices for SF's OpenRouter routes, plus SF parameter details for candidates.

Writes configs/prices_<date>.json (all routes) and configs/model_params_<date>.json
(sf.models.get() for CANDIDATES).

Free: OpenRouter's public /api/v1/models (no key) + SF's catalog (sf.models.list(), no model
calls). Prices are USD per 1M tokens. Routers (negative price) are dropped.

    python snapshot_prices.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")

OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

# Day 3 shortlist candidates: cascade pairs (same provider) + frontier ensemble.
CANDIDATES = [
    "openrouter/openai/gpt-5.6-luna", "openrouter/openai/gpt-5.6-terra",
    "openrouter/openai/gpt-5.6-sol", "openrouter/openai/gpt-6-astra",
    "openrouter/qwen/qwen3.7-flash", "openrouter/qwen/qwen3.8-2.4t-a95b",
    "openrouter/google/gemini-3.5-flash-lite", "openrouter/google/gemini-3.6-flash",
    "openrouter/google/gemini-3.1-pro-preview", "openrouter/google/gemini-4-argon",
    "openrouter/anthropic/claude-haiku-4.5", "openrouter/anthropic/claude-sonnet-5",
    "openrouter/anthropic/claude-opus-5.5", "openrouter/anthropic/claude-fable-5.2",
    "openrouter/deepseek/deepseek-v4-flash", "openrouter/deepseek/deepseek-v4-pro",
    "openrouter/x-ai/grok-4.6",
]


def per_million(pricing: dict, key: str) -> float | None:
    value = pricing.get(key)
    return None if value in (None, "") else round(float(value) * 1e6, 6)


def main() -> None:
    import screamingface as sf

    from run_eval import preflight, setup_logging

    setup_logging()
    preflight()
    sf_routes = [m.id for m in sf.models.list() if m.provider == "openrouter"]

    with urllib.request.urlopen(OPENROUTER_MODELS, timeout=30) as resp:
        catalog = {m["id"]: m for m in json.load(resp)["data"]}

    now = datetime.now(timezone.utc)
    rows, missing = {}, []
    for route in sorted(sf_routes):
        m = catalog.get(route.removeprefix("openrouter/"))
        if m is None:
            missing.append(route)
            continue
        p = m.get("pricing", {})
        price_in, price_out = per_million(p, "prompt"), per_million(p, "completion")
        if price_in is None or price_in < 0:
            continue  # router, not a model
        rows[route] = {
            "in_per_m": price_in,
            "out_per_m": price_out,
            "reasoning_per_m": per_million(p, "internal_reasoning"),
            "cache_read_per_m": per_million(p, "input_cache_read"),
            "context": m.get("context_length"),
            "reasoning": "reasoning" in (m.get("supported_parameters") or []),
            "expiration_date": m.get("expiration_date"),
            "created": m.get("created"),
        }

    out = {
        "snapshot_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": OPENROUTER_MODELS,
        "units": "USD per 1M tokens",
        "reasoning_note": "reasoning_per_m null = reasoning tokens billed at out_per_m",
        "missing_from_openrouter": missing,
        "routes": rows,
    }
    path = Path("configs") / f"prices_{now:%Y-%m-%d}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}: {len(rows)} routes, {len(missing)} missing")

    details = {}
    for route in CANDIDATES:
        try:
            d = sf.models.get(route)
        except Exception as exc:  # record, don't abort the snapshot
            details[route] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        details[route] = {
            "upstream_id": d.upstream_id,
            "observed_at": d.observed_at.isoformat() if d.observed_at else None,
            "parameters": {
                name: {
                    "provider_support": prm.provider_support,
                    "gateway_status": prm.gateway_status,
                    "deprecated": prm.provider_deprecated,
                    "cache_behavior": prm.cache_behavior,
                    "schema": None if prm.schema is None else {
                        k: v for k, v in dataclasses.asdict(prm.schema).items() if v is not None
                    },
                }
                for name, prm in d.parameters.items()
            },
        }
    ppath = Path("configs") / f"model_params_{now:%Y-%m-%d}.json"
    ppath.write_text(json.dumps(details, indent=1, default=str) + "\n")
    print(f"wrote {ppath}: {len(details)} candidates")


if __name__ == "__main__":
    main()
