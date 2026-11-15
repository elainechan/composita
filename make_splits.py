"""Write fixed random held-out splits of SeqQA case ids to configs/splits_seqqa.json.

Case ids are SF positions 1..N (what `sf.evaluate(..., cases=[...])` takes). Their order is
pinned by the benchmark revision and serving shuffle seed in configs/benchmarks.json, so
these ids stay valid while that revision is served.

Each split picks a "select" half (choose the cascade threshold there) and the rest is the
"report" half. The selected ids are stored, not just the seed, so results never depend on
the RNG staying stable across Python versions.

    python make_splits.py            # 1000 splits, 50/50, master seed 20261114
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BENCHMARKS = Path("configs/benchmarks.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=int, default=1000)
    p.add_argument("--select-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=20261114)
    a = p.parse_args()

    bench = json.loads(BENCHMARKS.read_text())["seqqa"]
    n = bench["case_count"]
    k = round(n * a.select_frac)
    rng = random.Random(a.seed)
    ids = list(range(1, n + 1))
    splits = [sorted(rng.sample(ids, k)) for _ in range(a.splits)]

    out = {
        "benchmark": bench["sf_id"],
        "revision": bench["revision"],
        "case_ids": [1, n],
        "master_seed": a.seed,
        "select_frac": a.select_frac,
        "n_splits": a.splits,
        "select": splits,
        "_note": "report half of split i = all ids in case_ids range not in select[i]",
    }
    path = Path("configs/splits_seqqa.json")
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path}: {a.splits} splits of {k}/{n - k} over ids 1..{n}")


if __name__ == "__main__":
    main()
