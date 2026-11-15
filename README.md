# composita

*Does Composition Pay?*

OpenMined Ensembles & Routing grant project.

- configs/: model lists, price snapshots, case-ID splits
- exports/: report JSON exported from ScreamingFace
- analysis/: hull, cascade and ensemble analysis

## Usage

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv "screamingface[notebook]"
.venv/bin/python run_eval.py --check          # free: Cloudflare login + engine reachable
.venv/bin/python run_eval.py MODEL BENCHMARK --limit 1 --queue-timeout 600 --run-timeout 600
.venv/bin/python make_splits.py               # regenerate configs/splits_seqqa.json
```

- `run_eval.py`: one evaluation with a queue timeout, a run timeout and a timestamped log; a timeout makes the SDK send the engine a stop.
- `configs/benchmarks.json`: pinned SF benchmark revisions, case digests and seeds.
- `configs/splits_seqqa.json`: 1000 fixed 50/50 held-out splits of SeqQA case ids.
