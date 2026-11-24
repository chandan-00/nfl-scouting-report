"""Forward check: for each team-level claim a final report cites, compare the season-to-date
value (the cited call's exact value, through the week before the game) and a naive baseline
(the league average through the same cutoff) with the same metric in the game itself.

The 2026 games come after every model's training cutoff, so this is clean. Single-game samples
are noisy; claims whose game sample is under `min_n` plays (or drives) are skipped.

    uv run python -m nfl_scout.evaluation.forward --config full no_checker single
    uv run python -m nfl_scout.evaluation.forward --live 2026_w10 --config full

With --live, it scores saved live reports (runs/live/<dir>/) against the current data file, once
their games have been played, and writes runs/scores/forward_live_<dir>.json plus a document for
the week, docs/forward/<dir>.md, linked from docs/forward/README.md.
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


def _game_key(row: dict) -> str:
    """The two teams of a row's game, sorted: both reports of one game share it."""
    _, team, _, opponent = row["matchup"].split("_")
    return "_".join(sorted((team, opponent)))


def per_game(rows: list[dict]) -> list[dict]:
    """Each (game, metric, split) once. With one report per team, both reports of a game can cite
    the same stat; counting it twice would weight that game double."""
    unique = {(_game_key(r), r["tool"], json.dumps(r["args"], sort_keys=True)): r for r in rows}
    return list(unique.values())


def week_markdown(
    name: str,
    rows: list[dict],
    games: list[dict],
    reports: int,
    generated: str,
    data_loaded: str | None = None,
) -> str:
    """The week's forward check as a document: headline, by metric, by game."""
    season, week = name.split("_w")
    rows = per_game(rows)
    lines = [
        f"# Forward check: {season} week {int(week)}",
        "",
        f"Generated {generated}. {reports} reports, data through week {int(week) - 1}, scored "
        "against the games once they were final"
        + (f", on the data file loaded {data_loaded}." if data_loaded else "."),
        "",
        "nflverse revises play-by-play during the season (EPA and success were re-computed for "
        "every 2026 game between Nov 2 and Nov 24), so a re-run on a later data file can differ.",
        "",
        "For each team stat a report cites: is the report's season-to-date value closer to the "
        "same stat in the game itself than the league average through the same week? A stat "
        "cited in both reports of a game counts once. About 50% means no better than the league "
        "average.",
        "",
    ]
    if not rows:
        return "\n".join([*lines, "No claims could be scored.", ""])
    lines += [
        f"**Report closer than league average: {_closer(rows):.1%} of {len(rows)} claims.**",
        "",
        "## By metric",
        "",
        "| Metric | Claims | Report closer | MAE report | MAE league |",
        "|---|---|---|---|---|",
    ]
    for tool in sorted({r["tool"] for r in rows}):
        rs = [r for r in rows if r["tool"] == tool]
        mae_report = mean(r["report_error"] for r in rs)
        mae_league = mean(r["naive_error"] for r in rs)
        lines.append(
            f"| {tool} | {len(rs)} | {_closer(rs):.0%} | {mae_report:.3f} | {mae_league:.3f} |"
        )
    lines += ["", "## By game", "", "| Game | Claims | Report closer |", "|---|---|---|"]
    for g in sorted(games, key=lambda g: g["game_id"]):
        rs = [r for r in rows if _game_key(r) == "_".join(sorted((g["away"], g["home"])))]
        if rs:
            score = f"{g['away']} {g['away_score']} at {g['home']} {g['home_score']}"
            lines.append(f"| {score} | {len(rs)} | {_closer(rs):.0%} |")
    return "\n".join([*lines, ""])


def _write_week_doc(docs: Path, name: str, md: str) -> Path:
    """Write docs/forward/<name>.md and rebuild the index of weeks."""
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / f"{name}.md"
    path.write_text(md)
    weeks = sorted(p.stem for p in docs.glob("*_w*.md"))
    index = ["# Forward checks by week", "", "One document per week of live reports.", ""]
    index += [f"- [{w.replace('_w', ' week ')}]({w}.md)" for w in weeks]
    (docs / "README.md").write_text("\n".join([*index, ""]))
    return path


def main() -> None:
    from nfl_scout.config import load_settings
    from nfl_scout.evaluation.snapshot import snapshot_connection

    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["full", "no_checker", "single"])
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--live", help="score live reports in runs/live/<LIVE>/ instead of the eval")
    a = ap.parse_args()
    base: Path = settings.resolve(settings.paths.runs_dir)
    if a.live:
        from nfl_scout.tools.base import default_connection

        runs_dir, con = base / "live" / a.live, default_connection()
    else:
        runs_dir, con = base, snapshot_connection()
    plays = load_plays(con, settings.data.eval_season)
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
    name = f"forward_live_{a.live}.json" if a.live else "forward.json"
    path = base / "scores" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    if a.live:
        from datetime import date

        season, week = (int(x) for x in a.live.split("_w"))
        games = [
            dict(zip(("game_id", "away", "home", "away_score", "home_score"), g, strict=True))
            for g in con.execute(
                "SELECT game_id, away_team, home_team, away_score, home_score FROM schedules "
                "WHERE season = ? AND week = ? AND game_type = 'REG'",
                [season, week],
            ).fetchall()
        ]
        rows = [row for config in a.config for row in out[config]["rows"]]
        reports = sum(len(list((runs_dir / c).glob("r*/*.json"))) for c in a.config)
        loaded = con.execute(
            "SELECT strftime(loaded_at, '%Y-%m-%d %H:%M') FROM _ingest_log WHERE table_name = 'pbp'"
        ).fetchone()
        md = week_markdown(
            a.live, rows, games, reports, date.today().isoformat(), loaded[0] if loaded else None
        )
        print("wrote", _write_week_doc(settings.resolve(Path("docs/forward")), a.live, md))


if __name__ == "__main__":
    main()
