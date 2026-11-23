"""Leaderboard baseline: one SF-native composition, run on the whole benchmark, then submitted.

Composition: plurality vote over the three cheap cascade models (configs/models.json), one
sample each, with the per-vendor params from configs/sampling.json and seed 0. The vote reads
each member's answer letter from its last 'ANSWER: X' line (SeqQA declares no vote rule); the
winning member's reply goes to the benchmark's grader unchanged; quorum 2, so one failed
or refused member doesn't sink the case.

The leaderboard ranks only runs covering the registered case count (SeqQA: 600), so this is ONE
run, never chunked. Cases run one at a time: expect ~5-6 h. Run under `caffeinate -i`.

    python baseline.py --show                     # free: print the composition, check versions
    python baseline.py --cases 1 2                # smoke test (spends credit)
    python baseline.py                            # full run (spends credit), saves the report
    python baseline.py --submit [REPORT_ID]       # submit a saved report (free); authors = you
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import run_eval
from run_eval import Deadline, _Phase, _version_check, _with_timers, log, sf

BENCHMARK = "inspect-lab_bench_seqqa"
NAME = "cheap_trio_vote"
CONFIGS = Path(__file__).parent / "configs"


def members() -> list:
    """flash, luna, haiku as sf.Model with the study's params (seed 0).

    optional=True: without it a member that fails (e.g. hits the token cap) fails the whole case
    and quorum is never consulted (seen on question 11 of the first full run, 2026-11-23).
    """
    pairs = json.loads((CONFIGS / "models.json").read_text())["cascade_pairs"]
    out = []
    for r in (p["cheap"] for p in pairs):
        m = run_eval.candidates(r, k=1)[0]
        out.append(sf.Model(m.model, name=m.name, params=dict(m.params), optional=True))
    return out


# SeqQA declares no answer rule for votes, so read the letter the same way the benchmark asks
# for it: the last 'ANSWER: X' line, X optionally wrapped as in '(B)', '$B' or '**B**'.
ANSWER = sf.Extract("regex", pattern=r"ANSWER:\s*[(*$]*([A-Z])", group=1, occurrence="last")


def composition():
    return sf.Fusion(members(), synthesizer=sf.Vote(over=ANSWER), quorum=2, name=NAME)


def run(cases: list[int] | None, run_timeout_s: int, export: Path):
    routes = [m.model for m in members()]
    before = [_version_check(r, "before") for r in routes]
    if any(b["problems"] for b in before):
        raise run_eval.VersionMismatch("a member's version changed; refusing to submit")
    for m in members():
        bad = run_eval.model_version.unsupported_params(m.model, dict(m.params))
        if bad:
            raise ValueError(f"{m}: unsupported params {bad}")
    phase, t0 = _Phase("queue"), time.time()

    def on_event(event):
        if event.kind == "started" and phase.started_at is None:
            phase.name, phase.started_at = "run", time.time()
            run_eval.signal.alarm(run_timeout_s)
            log.info("runner picked up the run after %.0fs; run timeout %ss",
                     phase.started_at - t0, run_timeout_s)
        run_eval._log_event(event)

    log.info("submit %s cases=%s", composition(), cases or "ALL")
    report = None
    try:
        report = _with_timers(phase, 600, sf.evaluate, [composition()], benchmark=BENCHMARK,
                              cases=cases, on_event=on_event, capture=True, use_cache=False)
    finally:
        after = [_version_check(r, "after") for r in routes]
        export.with_suffix(".version.json").write_text(json.dumps({
            "composition": NAME, "routes": routes,
            "valid": report is not None and not any(a["problems"] for a in after),
            "use_cache": False, "before": before, "after": after}, indent=1) + "\n")
    c = report.candidates[0]
    log.info("done in %.0fs score=%s cases=%d coverage=%s usage=%s pinned_cache=%s capture=%s",
             time.time() - t0, c.score, len(c.cases), c.coverage, report.usage,
             c.pinned_cache, c.capture_status)
    log.info("exported to %s", report.export(export))
    if any(a["problems"] for a in after):
        raise run_eval.VersionMismatch("a member's version changed during the run; INVALID")
    return report


def submit(report_id: str | None) -> int:
    """Submit a saved report (free). No id: the newest saved report of this composition."""
    if report_id is None:
        saved = [r for r in sf.reports.list() if r.candidates == (NAME,) and r.downloaded]
        if not saved:
            log.error("no saved report for %s", NAME)
            return 6
        report_id = saved[0].id  # sf.reports.list() is newest first
    log.info("submitting saved report %s", report_id)
    report = sf.reports.get(report_id)
    c = report.candidates[0]
    expected = json.loads((CONFIGS / "benchmarks.json").read_text())["seqqa"]["case_count"]
    if c.name != NAME or len(c.cases) < expected:
        log.error("refusing: %s has %d cases (leaderboard ranks only >= %d)",
                  c.name, len(c.cases), expected)
        return 6
    score = sf.leaderboards.submit(c)  # authors default to the authenticated submitter
    log.info("submitted: %s", score)
    return 0


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--show", action="store_true", help="print composition + version check (free)")
    a.add_argument("--cases", type=int, nargs="+", help="subset for a smoke test")
    a.add_argument("--run-timeout", type=int, default=10 * 3600)
    a.add_argument("--submit", nargs="?", const="", metavar="REPORT_ID",
                   help="submit a saved report (default: newest of this composition)")
    o = a.parse_args()
    if o.show:
        run_eval.setup_logging()
        print(composition())
        for m in members():
            print(" ", m.name, dict(m.params))
        return 0
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{BENCHMARK}_{NAME}"
    run_eval.setup_logging(Path("exports") / f"{stem}.log")
    if o.submit is not None:
        return submit(o.submit or None)
    try:
        run_eval.preflight()
        run(o.cases, o.run_timeout, Path("exports") / f"{stem}.json")
    except Deadline:
        return 2
    except run_eval.VersionMismatch as exc:
        log.error("%s", exc)
        return 4
    except ValueError as exc:
        log.error("%s", exc)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
