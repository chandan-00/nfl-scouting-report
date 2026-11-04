"""Forward check: for each team-level claim a final report cites, compare the season-to-date
value (the cited call's exact value, through the week before the game) and a naive baseline
(the league average through the same cutoff) with the same metric in the game itself.

The 2026 games come after every model's training cutoff, so this is clean. Single-game samples
are noisy; claims whose game sample is under `min_n` plays (or drives) are skipped.

    uv run python -m nfl_scout.evaluation.forward --config full no_checker single
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import polars as pl

from nfl_scout.evaluation.independent import METRICS, _scrimmage, load_plays, week_plays
from nfl_scout.runs import load_run

MIN_N = 10


def forward_rows(run: dict[str, Any], plays: pl.DataFrame, min_n: int = MIN_N) -> list[dict]:
    r = run["request"]
    week = r["week"]
    matchup = f"w{week:02d}_{r['team']}_vs_{r['opponent']}"
    cited = {
        n["mention"]["claim_id"] for n in run.get("number_checks", []) if n["mention"]["claim_id"]
    }
    calls = {c["call_id"]: c for c in run["tool_calls"]}
    game = week_plays(plays, week)
    rows = []
    for section in run["sections"].values():
        for claim in section["claims"]:
            call = calls.get(claim.get("tool_call_id") or "")
            if (
                claim["claim_id"] not in cited
                or claim.get("measure", "value") != "value"
                or call is None
                or call.get("error")
                or call["tool"] not in METRICS
                or call["result"]["value"] is None
            ):
                continue
            args = {k: v for k, v in call["args"].items() if k not in ("season", "through_week")}
            fn = METRICS[call["tool"]]
            next_value, next_n = fn(game, **args)
            if next_value is None or next_n < min_n:
                continue
            league_args = {**args, "team": None}
            league, _ = fn(_scrimmage(plays, call["args"]["through_week"]), **league_args)
            report_value = float(call["result"]["value"])
            rows.append(
                {
                    "config": run["config"],
                    "matchup": matchup,
                    "claim_id": claim["claim_id"],
                    "tool": call["tool"],
                    "args": args,
                    "report_value": report_value,
                    "next_game": next_value,
                    "next_n": next_n,
                    "league": league,
                    "report_error": abs(report_value - next_value),
                    "naive_error": abs(league - next_value),
                }
            )
    return rows


def _closer(rows: list[dict]) -> float:
    return mean(1.0 if r["report_error"] < r["naive_error"] else 0.0 for r in rows)


def aggregate_forward(rows: list[dict]) -> dict[str, Any]:
    """Share of claims where the report beat the league average (unit-free, pooled), and mean
    absolute errors per metric only (EPA and percentages don't share units). Each distinct
    (game, metric, split) counts once."""
    # Repeats of a matchup cite the same calls: count each (game, metric, split) once.
    unique = {(r["matchup"], r["tool"], json.dumps(r["args"], sort_keys=True)): r for r in rows}
    rows = list(unique.values())
    if not rows:
        return {"claims": 0}
    by_tool = {}
    for tool in sorted({r["tool"] for r in rows}):
        rs = [r for r in rows if r["tool"] == tool]
        by_tool[tool] = {
            "claims": len(rs),
            "report_closer": _closer(rs),
            "report_mae": mean(r["report_error"] for r in rs),
            "naive_mae": mean(r["naive_error"] for r in rs),
        }
    return {"claims": len(rows), "report_closer": _closer(rows), "by_tool": by_tool}


def main() -> None:
    from nfl_scout.config import load_settings
    from nfl_scout.evaluation.snapshot import snapshot_connection

    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["full", "no_checker", "single"])
    ap.add_argument("--min-n", type=int, default=MIN_N)
    a = ap.parse_args()
    runs_dir: Path = settings.resolve(settings.paths.runs_dir)
    plays = load_plays(snapshot_connection(), settings.data.eval_season)
    out = {}
    for config in a.config:
        rows = [
            row
            for p in sorted((runs_dir / config).glob("r*/*.json"))
            for row in forward_rows(load_run(p), plays, a.min_n)
        ]
        out[config] = {"aggregate": aggregate_forward(rows), "rows": rows}
        agg = out[config]["aggregate"]
        if agg["claims"]:
            print(f"{config}: {agg['claims']} claims, report closer {agg['report_closer']:.1%}")
            for tool, t in agg["by_tool"].items():
                print(
                    f"  {tool:28} n={t['claims']:4} closer={t['report_closer']:.0%} "
                    f"MAE report={t['report_mae']:.3f} league={t['naive_mae']:.3f}"
                )
    path = runs_dir / "scores" / "forward.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
