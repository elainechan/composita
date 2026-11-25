"""Summarise SF export files: accuracy, refusals, failures, cost and latency per model.

Free: reads local exports only, calls nothing.

Accuracy rules (decided 2026-11-22):
- sf_score: correct / all cases (SF's own score; refusals and failures count as wrong).
- acc_answered: correct / (cases - provider refusals). Only refusals are excluded;
  token-cap hits and provider errors still count as wrong.
- acc_common: correct / cases in the common set, i.e. cases that no sample of any listed
  model refused. Those cases leave both numerator and denominator.

    python analysis/pilot_summary.py EXPORT.json [...] --out results/x.csv --runs results/y.csv
    python analysis/pilot_summary.py --hashes results/z_sha256.txt exports/*   # tamper-evidence
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as st
import sys
from pathlib import Path


def load(path: Path) -> dict:
    """One export plus its version sidecar. Refuses runs whose version check failed."""
    version = json.loads(path.with_suffix(".version.json").read_text())
    if not version["valid"]:
        raise SystemExit(f"{path}: version check failed; not combinable")
    return {"path": path, "model": version["model"], "report": json.loads(path.read_text()),
            "slug": version["after"]["fingerprint"]["canonical_slug"]}


def codes(case: dict) -> set[str]:
    return {f["code"] for f in case.get("failures", [])}


def correct(case: dict) -> bool:
    return case["status"] == "scored" and (case["grade"].get("score") or 0) > 0


def common_set(runs: list[dict]) -> set[int]:
    ids = None
    for r in runs:
        for c in r["report"]["candidates"]:
            ok = {x["case_id"] for x in c["cases"] if "provider_refusal" not in codes(x)}
            ids = ok if ids is None else ids & ok
    return ids or set()


def summarise(run: dict, common: set[int]) -> dict:
    cands = run["report"]["candidates"]
    cases = [x for c in cands for x in c["cases"]]
    ops = [op.get("accounting") or {} for x in cases for op in x.get("operations") or []]
    n, refused = len(cases), sum("provider_refusal" in codes(x) for x in cases)
    right = sum(correct(x) for x in cases)
    in_common = [x for x in cases if x["case_id"] in common]
    lat = sorted(a["provider_latency_ms"] / 1000 for a in ops if a.get("provider_latency_ms"))
    out = [a["usage"]["output_tokens"] for a in ops if a.get("usage")]
    cost = float(run["report"]["usage"]["cost_usd"])
    return {
        "model": run["model"], "canonical_slug": run["slug"], "k": len(cands), "cases": n,
        "correct": right, "sf_score": round(right / n, 4),
        "acc_answered": round(right / (n - refused), 4),
        "acc_common": round(sum(correct(x) for x in in_common) / len(in_common), 4),
        "refused": refused,
        "token_cap": sum("model_token_cap" in codes(x) for x in cases),
        "provider_error": sum(bool(codes(x) & {"provider_error", "upstream_error"}) for x in cases),
        "cost_usd": round(cost, 6), "cost_per_call": round(cost / n, 6),
        "out_tokens_per_call": round(st.mean(out)),
        "latency_median_s": round(st.median(lat), 1),
        "latency_p90_s": round(lat[int(0.9 * (len(lat) - 1))], 1),
        "export": run["path"].name,
    }


def run_ids(run: dict) -> list[dict]:
    """SF identifiers per run: verifiable against SF's records (pinned cache replay)."""
    return [{"model": run["model"], "candidate": c["name"], "run_id": c["run_id"],
             "pinned_cache": c.get("frozen_copy_id"), "capture_status": c.get("capture_status"),
             "started_at": c["started_at"], "completed_at": c["completed_at"],
             "client_version": c.get("client_version"), "export": run["path"].name}
            for c in run["report"]["candidates"]]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("files", nargs="+", type=Path)
    a.add_argument("--out", type=Path, help="per-model summary CSV")
    a.add_argument("--runs", type=Path, help="per-run SF identifiers CSV")
    a.add_argument("--hashes", type=Path, help="write SHA-256 of every given file instead")
    o = a.parse_args()
    if o.hashes:
        lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.as_posix()}"
                 for p in sorted(o.files) if p.is_file()]
        o.hashes.write_text("\n".join(lines) + "\n")
        print(f"wrote {o.hashes} ({len(lines)} files)")
        return 0
    runs = [load(p) for p in o.files if not p.name.endswith(".version.json")]
    common = common_set(runs)
    print(f"common set: {len(common)} cases; excluded {sorted(set().union(*({x['case_id'] for c in r['report']['candidates'] for x in c['cases']} for r in runs)) - common)}")
    rows = [summarise(r, common) for r in runs]
    for r in rows:
        print(f"{r['model']:42} k={r['k']} sf={r['sf_score']:.3f} answered={r['acc_answered']:.3f} "
              f"common={r['acc_common']:.3f} refused={r['refused']} ${r['cost_usd']:.3f}")
    if o.out:
        write_csv(o.out, rows)
    if o.runs:
        write_csv(o.runs, [x for r in runs for x in run_ids(r)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
