"""Estimate credit spend, the stop rule, and run length from the price snapshot and configs.

Free: reads configs only, calls nothing. Token counts are assumptions until the Day 6 pilot
measures them; pass --out-tokens to try other values.

    python cost_model.py
    python cost_model.py --out-tokens 1500 3000 6000 --latency 20 40 60
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

CFG = Path(__file__).parent / "configs"
PRICES = CFG / "prices_2026-11-15.json"
MMLU_TOTAL = 12032
IN_TOKENS = {"seqqa": 600, "mmlu_pro": 900}  # per call; guesses until the pilot
CONCURRENCY = 4  # runnerIoConcurrency: calls to one provider at a time, per run


def plan() -> dict:
    return json.loads((CFG / "budget.json").read_text())["policy"]


def samples_per_model() -> dict[str, int]:
    """route -> k, from k_by_role (cheap models k, expensive/frontier 1)."""
    m = json.loads((CFG / "models.json").read_text())
    k = json.loads((CFG / "sampling.json").read_text())["k_by_role"]
    out = {r: k["frontier"] for r in m["frontier_ensemble"]}
    out.update({p["expensive"]: k["expensive"] for p in m["cascade_pairs"]})
    out.update({p["cheap"]: k["cheap"] for p in m["cascade_pairs"]})
    return out


def case_cost(ks: dict[str, int], tin: int, tout: int, prices: dict) -> float:
    """USD for one question across all models, each with its k samples."""
    return sum(n * (tin * prices[r]["in_per_m"] + tout * prices[r]["out_per_m"]) / 1e6
               for r, n in ks.items())


def max_case_cost(n_seqqa: int, n_mmlu: int, rho: float, p: dict, spent: float = 0) -> float:
    """Largest SeqQA cost per case at which the plan fits the credit.

    (spent + n_seqqa*c*(1+retry) + n_mmlu*c*rho*(1+retry) + reserves) * (1+contingency) <= credit
    rho = MMLU-Pro cost per case / SeqQA cost per case. Stop rule: the pilot's upper 95%
    bound on c must not exceed this (n_mmlu = the plan's minimum).
    """
    room = p["credit_usd"] / (1 + p["contingency"]) - spent - sum(p["reserves_usd"].values())
    return room / ((n_seqqa + n_mmlu * rho) * (1 + p["retry"]))


def max_mmlu_cases(c_seq: float, n_seqqa: int, rho: float, p: dict, spent: float = 0) -> int:
    room = (p["credit_usd"] / (1 + p["contingency"]) - spent - sum(p["reserves_usd"].values())
            - n_seqqa * c_seq * (1 + p["retry"]))
    return max(0, min(MMLU_TOTAL, math.floor(room / (c_seq * rho * (1 + p["retry"])))))


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--out-tokens", type=int, nargs="+", default=[1500, 3000, 6000])
    a.add_argument("--latency", type=float, nargs="+", default=[20, 40, 60],
                   help="seconds per call, for run length")
    a.add_argument("--run-budget", type=int, default=3000,
                   help="seconds a chunk may run (under --run-timeout 3600)")
    o = a.parse_args()

    p = plan()
    prices = json.loads(PRICES.read_text())["routes"]
    seqqa = json.loads((CFG / "benchmarks.json").read_text())["seqqa"]["case_count"]
    ks = samples_per_model()
    calls = sum(ks.values())
    print(f"{len(ks)} models, {calls} calls per question, SeqQA {seqqa}, prices {PRICES.name}\n")

    print("out tokens | SeqQA $/case | SeqQA pass | MMLU-Pro $/1,000 | MMLU-Pro cases that fit")
    for t in o.out_tokens:
        cs = case_cost(ks, IN_TOKENS["seqqa"], t, prices)
        cm = case_cost(ks, IN_TOKENS["mmlu_pro"], t, prices)
        print(f"{t:>10} | {cs:12.3f} | {cs * seqqa:10,.0f} | {cm * 1000:16,.0f} | "
              f"{max_mmlu_cases(cs, seqqa, cm / cs, p):,}")

    t = o.out_tokens[len(o.out_tokens) // 2]
    rho = (case_cost(ks, IN_TOKENS["mmlu_pro"], t, prices)
           / case_cost(ks, IN_TOKENS["seqqa"], t, prices))
    c = max_case_cost(seqqa, p["mmlu_min_cases"], rho, p)
    print(f"\nStop rule: pilot's upper 95% bound on SeqQA $/case must be <= ${c:.3f} "
          f"(SeqQA pass <= ${c * seqqa:,.0f}; MMLU-Pro minimum {p['mmlu_min_cases']:,}, "
          f"rho {rho:.3f})")

    print(f"\nPer model, one SeqQA pass ({t} out tokens):")
    for r, n in sorted(ks.items()):
        cost = case_cost({r: n}, IN_TOKENS["seqqa"], t, prices) * seqqa
        print(f"  k={n} {r.removeprefix('openrouter/'):38} ${cost:,.0f}")

    print(f"\nRun length per model (SeqQA, {CONCURRENCY} calls at a time; chunk <= "
          f"{o.run_budget}s):")
    for lat in o.latency:
        for k in sorted(set(ks.values())):
            hours = seqqa * k * lat / CONCURRENCY / 3600
            chunk = max(1, math.floor(o.run_budget * CONCURRENCY / (k * lat)))
            print(f"  {lat:>3.0f}s/call k={k}: {hours:4.1f} h unsplit; chunk {min(chunk, seqqa)} "
                  f"cases -> {math.ceil(seqqa / chunk)} runs")
        total_h = seqqa * calls * lat / CONCURRENCY / 3600
        print(f"  {lat:>3.0f}s/call all models: {total_h:.0f} h of run time "
              f"({total_h / 3:.0f} h wall-clock at 3 chunks at once)")


if __name__ == "__main__":
    main()
