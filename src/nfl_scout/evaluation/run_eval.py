"""Generate eval runs: every frozen matchup, per config and repeat, saved to
runs/<config>/r<repeat>/<matchup>.json. Saved runs are skipped, so a batch resumes after an
interruption; failures go to runs/<config>/failures.jsonl and the batch carries on.

    uv run python -m nfl_scout.evaluation.run_eval --config full --repeats 1 2 3        # estimate
    uv run python -m nfl_scout.evaluation.run_eval --config full --repeats 1 2 3 --yes  # run
"""

from __future__ import annotations

import argparse
import json
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_scout.config import Settings, load_settings
from nfl_scout.evaluation.matchups import load_matchups
from nfl_scout.graph.build import CONFIGS, build_graph
from nfl_scout.graph.state import Request
from nfl_scout.runs import matchup_name, run_path, save_run

# Measured per-report cost on Haiku (one live run each), used until real eval runs exist.
FALLBACK_COST = {"full": 0.034, "no_checker": 0.025, "single": 0.011}


def _request(m: dict, season: int) -> Request:
    return Request(team=m["team"], opponent=m["opponent"], season=season, week=m["week"])


def pending(
    matchups: list[dict], config: str, repeats: list[int], runs_dir: Path, season: int
) -> list[dict]:
    """(matchup, repeat) pairs without a saved run, as matchup dicts with a "repeat" key."""
    return [
        {**m, "repeat": k}
        for k in repeats
        for m in matchups
        if not run_path(runs_dir, config, _request(m, season), k).exists()
    ]


def estimate_cost(config: str, n: int, runs_dir: Path) -> float:
    """n reports at the mean cost of this config's saved runs (or the measured fallback)."""
    costs = [
        json.loads(p.read_text())["cost_usd"]["total"]
        for p in (runs_dir / config).glob("r*/*.json")
    ]
    per_report = sum(costs) / len(costs) if costs else FALLBACK_COST[config]
    return n * per_report


def run_eval(
    matchups: list[dict],
    config: str,
    repeats: list[int],
    settings: Settings,
    build: Callable[[str], Any] | None = None,
    runs_dir: Path | None = None,
    workers: int = 4,
    limit: int | None = None,
) -> dict[str, Any]:
    runs_dir = runs_dir or settings.resolve(settings.paths.runs_dir)
    season = settings.data.eval_season
    todo = pending(matchups, config, repeats, runs_dir, season)
    skipped = len(matchups) * len(repeats) - len(todo)
    todo = todo[:limit] if limit is not None else todo
    graph = (build or (lambda c: build_graph(settings=settings, config=c)))(config)
    failures_path = runs_dir / config / "failures.jsonl"
    lock = threading.Lock()
    failed: list[dict] = []

    def one(m: dict) -> Path:
        request = _request(m, season)
        out = graph.invoke({"request": request})
        return save_run(out, config, settings, runs_dir=runs_dir, repeat=m["repeat"])

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, m): m for m in todo}
        for fut in as_completed(futures):
            m = futures[fut]
            name = matchup_name(_request(m, season))
            try:
                fut.result()
                done += 1
                print(f"  ok   r{m['repeat']} {name}", flush=True)
            except Exception as e:  # noqa: BLE001 - one bad matchup must not stop the batch
                entry = {
                    "matchup": name,
                    "repeat": m["repeat"],
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=5),
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                with lock:
                    failed.append(entry)
                    failures_path.parent.mkdir(parents=True, exist_ok=True)
                    with failures_path.open("a") as f:
                        f.write(json.dumps(entry) + "\n")
                print(f"  FAIL r{m['repeat']} {name}: {entry['error']}", flush=True)
    failed.sort(key=lambda f: (f["repeat"], f["matchup"]))
    return {"done": done, "skipped": skipped, "failed": failed}


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--repeats", type=int, nargs="+", default=[1])
    ap.add_argument("--matchups", default="v1", help="frozen matchup set version")
    ap.add_argument("--limit", type=int, help="run only the first N pending reports")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--yes", action="store_true", help="actually run (default: estimate only)")
    a = ap.parse_args()

    runs_dir = settings.resolve(settings.paths.runs_dir)
    matchups = load_matchups(a.matchups)
    todo = pending(matchups, a.config, a.repeats, runs_dir, settings.data.eval_season)
    n = min(len(todo), a.limit) if a.limit is not None else len(todo)
    cost = estimate_cost(a.config, n, runs_dir)
    print(f"{a.config}: {n} reports to run ({len(matchups) * len(a.repeats) - len(todo)} saved), "
          f"estimated ${cost:.2f}")  # fmt: skip
    if not a.yes:
        print("estimate only; pass --yes to run")
        return
    summary = run_eval(matchups, a.config, a.repeats, settings, workers=a.workers, limit=a.limit)
    print(f"done {summary['done']}, skipped {summary['skipped']}, failed {len(summary['failed'])}")


if __name__ == "__main__":
    main()
