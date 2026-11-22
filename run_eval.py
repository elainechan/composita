"""Run one ScreamingFace evaluation with a queue timeout, a run timeout and a timestamped log.

Why this exists (Day 1 smoke run stalled 17 min in the runner queue):
- sf.evaluate() has no timeout; the hosted engine keeps an unclaimed queued Run ~16h.
- Only an exception raised inside sf.evaluate() makes the SDK send the Engine a stop.
  SIGTERM/SIGKILL (e.g. `timeout`) skip that, so timers and SIGTERM raise an exception instead.
- Queue/capacity notices arrive as SDK log warnings, not Events, so logging is captured.

Two timers, one SIGALRM:
- queue timeout: armed at submit. Nothing is billed while queued.
- run timeout: replaces it at the first `Started` event. That event comes from the URL4
  executor, which only runs inside a runner, so it marks "a runner picked the run up".

CLI:
    python run_eval.py --check                      # free: confirm login + engine reachable
    python run_eval.py MODEL BENCHMARK --limit 1    # spends credit
Notebook (one kernel = one Cloudflare login):
    from run_eval import preflight, run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")
# The SDK prints the Cloudflare login URL to stdout; unbuffered so it shows when redirected.
sys.stdout.reconfigure(line_buffering=True)
import screamingface as sf  # noqa: E402

import model_version  # noqa: E402

log = logging.getLogger("run_eval")


class Deadline(BaseException):
    """Raised by the alarm/SIGTERM handler. BaseException so no `except Exception` eats it."""


class _Phase:
    """What the alarm means right now: queue wait or execution."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.started_at: float | None = None


def _handler(phase: _Phase):
    def handle(signum, _frame):
        why = "SIGTERM" if signum == signal.SIGTERM else f"{phase.name} timeout"
        raise Deadline(why)
    return handle


def _with_timers(phase: _Phase, first_s: int, fn, *args, **kwargs):
    """Call fn with SIGALRM armed for first_s; SIGALRM/SIGTERM raise Deadline inside fn."""
    old_alrm = signal.signal(signal.SIGALRM, _handler(phase))
    old_term = signal.signal(signal.SIGTERM, _handler(phase))
    signal.alarm(first_s)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alrm)
        signal.signal(signal.SIGTERM, old_term)


def setup_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _log_event(event) -> None:
    extra = ""
    if event.kind == "span" and event.request_model:
        extra = (f" model={event.request_model} in={event.input_tokens}"
                 f" out={event.output_tokens} cache={event.cache_status}")
    elif event.kind == "log":
        extra = f" {event.severity_text}: {event.body}"
        if event.attributes:
            extra += f" {dict(event.attributes)}"
    log.info("event %s seq=%s run=%s%s", event.kind, event.sequence, event.run_id, extra)


def preflight(deadline_s: int = 300) -> int:
    """Free catalog call. Fails fast if the Cloudflare login would hang."""
    n = len(_with_timers(_Phase("login"), deadline_s, sf.benchmarks.list))
    log.info("preflight ok: engine reachable, %d benchmarks", n)
    return n


SAMPLING = Path(__file__).parent / "configs" / "sampling.json"


def role(model: str) -> str:
    """cheap / expensive / frontier, from configs/models.json (cheap wins if listed twice)."""
    m = json.loads((SAMPLING.parent / "models.json").read_text())
    if model in {p["cheap"] for p in m["cascade_pairs"]}:
        return "cheap"
    if model in {p["expensive"] for p in m["cascade_pairs"]}:
        return "expensive"
    if model in m["frontier_ensemble"]:
        return "frontier"
    raise ValueError(f"{model} is not in configs/models.json")


def candidates(model: str, k: int | None = None) -> list:
    """k seeded copies of one model, with the shared params from configs/sampling.json.

    Each copy gets its own seed (part of SF's cache key, so k real calls) and its own
    name (sf.evaluate rejects duplicate candidate names). k=None uses k_by_role.
    """
    cfg = json.loads(SAMPLING.read_text())
    n = cfg["k_by_role"][role(model)] if k is None else k
    seeds = cfg["seeds"][:n]
    if len(seeds) < n:
        raise ValueError(f"configs/sampling.json has {len(seeds)} seeds, need {n}")
    omit = (set(cfg["omit_params_by_vendor"].get(model.split("/")[1], []))
            | set(cfg.get("omit_params_by_model", {}).get(model, [])))
    short = model.rsplit("/", 1)[-1]
    out = []
    for s in seeds:
        params = {k: v for k, v in {**cfg["params"], "seed": s}.items() if k not in omit}
        out.append(sf.Model(model, name=f"{short}-s{s}", params=params))
    return out


class VersionMismatch(Exception):
    """The route does not serve the pinned model version (configs/model_versions.json)."""


def _version_check(model: str, when: str) -> dict:
    """Fingerprint now vs the pin. Logs and returns {fingerprint, problems}."""
    now, problems = model_version.check(model)
    if problems:
        log.error("version check %s: %s does NOT match pin: %s", when, model, "; ".join(problems))
    else:
        log.info("version check %s: %s = %s", when, model, now["canonical_slug"])
    return {"when": when, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": now, "problems": problems}


def run(model: str, benchmark: str, *, k: int | None = None, limit: int | None = None,
        cases: list[int] | None = None, queue_timeout_s: int = 600,
        run_timeout_s: int = 3600, export: Path | None = None,
        capture: bool = False, use_cache: bool = False):
    """One model, k seeded samples, one sf.evaluate. Spends credit once a runner starts it.

    Refuses to submit unless the route serves the pinned version; re-checks afterwards and
    writes <export>.version.json. A run whose version changed is INVALID (VersionMismatch).
    Cache is off by default: a cached answer can't be tied to a model version.
    """
    before = _version_check(model, "before")
    if before["problems"]:
        raise VersionMismatch(f"{model}: refusing to submit; nothing billed")
    t0 = time.time()
    phase = _Phase("queue")

    def on_event(event) -> None:
        if event.kind == "started" and phase.started_at is None:
            phase.name, phase.started_at = "run", time.time()
            signal.alarm(run_timeout_s)
            log.info("runner picked up the run after %.0fs in queue; run timeout %ss",
                     phase.started_at - t0, run_timeout_s)
        _log_event(event)

    cands = candidates(model, k)
    bad = model_version.unsupported_params(model, dict(cands[0].params))
    if bad:
        raise ValueError(f"{model}: OpenRouter lists no support for {bad}; refusing to submit")
    log.info("submit model=%s k=%d params=%s benchmark=%s limit=%s cases=%s "
             "queue_timeout=%ss run_timeout=%ss", model, len(cands), dict(cands[0].params),
             benchmark, limit, cases, queue_timeout_s, run_timeout_s)
    report = None
    try:
        report = _with_timers(
            phase, queue_timeout_s, sf.evaluate, cands, benchmark=benchmark,
            limit=limit, cases=cases, on_event=on_event,
            capture=capture, use_cache=use_cache,
        )
    except Deadline as exc:
        if phase.started_at is None:
            log.error("%s: never left the queue in %.0fs; stop sent, nothing billed",
                      exc, time.time() - t0)
        else:
            log.error("%s: ran %.0fs after start; stop sent, partial calls may be billed",
                      exc, time.time() - phase.started_at)
        raise
    finally:
        after = _version_check(model, "after")
        if export is not None:
            sidecar = export.with_suffix(".version.json")
            sidecar.write_text(json.dumps({
                "model": model, "valid": not after["problems"] and report is not None,
                "use_cache": use_cache, "before": before, "after": after}, indent=1) + "\n")
    log.info("done in %.1fs scores=%s usage=%s", time.time() - t0,
             {c.name: c.score for c in report.candidates}, report.usage)
    if export is not None:
        log.info("exported to %s", report.export(export))
    if after["problems"]:
        raise VersionMismatch(f"{model}: version changed during the run; results INVALID")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", nargs="?")
    p.add_argument("benchmark", nargs="?")
    p.add_argument("--check", action="store_true", help="preflight only (free)")
    p.add_argument("--k", type=int, help="samples per case (default: k_by_role in configs/sampling.json)")
    p.add_argument("--limit", type=int)
    p.add_argument("--cases", type=int, nargs="+")
    p.add_argument("--queue-timeout", type=int, default=600,
                   help="seconds to wait for a runner before cancelling (default 600)")
    p.add_argument("--run-timeout", type=int, default=3600,
                   help="seconds the run may execute once started (default 3600)")
    p.add_argument("--login-timeout", type=int, default=300,
                   help="seconds to finish the Cloudflare login in preflight (default 300)")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--use-cache", action="store_true",
                   help="allow cached answers (off by default: they can't be tied to a version)")
    a = p.parse_args()

    if a.check:
        setup_logging()
        try:
            preflight(a.login_timeout)
        except Deadline as exc:
            log.error("preflight %s: login not completed", exc)
            return 3
        return 0
    if not (a.model and a.benchmark):
        p.error("MODEL and BENCHMARK required unless --check")

    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{a.benchmark}_{a.model.replace('/', '_')}"
    setup_logging(Path("exports") / f"{stem}.log")
    try:
        preflight(a.login_timeout)
    except Deadline as exc:
        log.error("preflight %s: login not completed; nothing submitted", exc)
        return 3
    try:
        run(a.model, a.benchmark, k=a.k, limit=a.limit, cases=a.cases,
            queue_timeout_s=a.queue_timeout, run_timeout_s=a.run_timeout,
            export=Path("exports") / f"{stem}.json",
            capture=a.capture, use_cache=a.use_cache)
    except Deadline:
        return 2
    except VersionMismatch as exc:
        log.error("%s", exc)
        return 4
    except ValueError as exc:
        log.error("%s", exc)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
