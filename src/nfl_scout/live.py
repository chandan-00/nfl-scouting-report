"""Live reports for an upcoming week: one per game, saved to runs/live/<season>_w<week>/.

    uv run python -m nfl_scout.live --week 9            # check data, estimate cost
    uv run python -m nfl_scout.live --week 9 --yes      # run

Each report is from the home team's view, except that games involving --focus (default BUF)
are always from the focus team's view. It refuses to run until the previous week is complete:
every scheduled game has play-by-play and FTN charting covers at least 95% of its plays.
"""

from __future__ import annotations

import argparse

import duckdb

from nfl_scout.config import load_settings
from nfl_scout.evaluation.run_eval import estimate_cost, run_eval
from nfl_scout.tools.base import default_connection

FTN_MIN_COVERAGE = 0.95


def week_matchups(
    con: duckdb.DuckDBPyConnection, season: int, week: int, focus: str | None = None
) -> list[dict]:
    rows = con.execute(
        """
        SELECT game_id, away_team, home_team FROM schedules
        WHERE season = ? AND game_type = 'REG' AND week = ? ORDER BY game_id
        """,
        [season, week],
    ).fetchall()
    out = []
    for game_id, away, home in rows:
        team, opponent = (away, home) if away == focus else (home, away)
        out.append(
            {"game_id": game_id, "week": week, "team": team, "opponent": opponent,
             "through_week": week - 1}
        )  # fmt: skip
    return sorted(out, key=lambda m: (m["team"] != focus, m["team"]))


def data_ready(con: duckdb.DuckDBPyConnection, season: int, week: int) -> tuple[bool, list[str]]:
    """Is `week` complete: every scheduled game in pbp, and FTN covering its plays?"""
    problems = []
    (scheduled,) = con.execute(
        "SELECT count(*) FROM schedules WHERE season = ? AND game_type = 'REG' AND week = ?",
        [season, week],
    ).fetchone()
    (loaded,) = con.execute(
        "SELECT count(DISTINCT game_id) FROM pbp WHERE season = ? AND week = ?", [season, week]
    ).fetchone()
    if loaded < scheduled:
        problems.append(f"play-by-play has {loaded} of {scheduled} week-{week} games")
    (plays, charted) = con.execute(
        """
        SELECT count(*), count(f.nflverse_play_id)
        FROM pbp p LEFT JOIN ftn_charting f
          ON f.nflverse_game_id = p.game_id AND f.nflverse_play_id = CAST(p.play_id AS INTEGER)
        WHERE p.season = ? AND p.week = ? AND p.play_type IN ('pass', 'run')
        """,
        [season, week],
    ).fetchone()
    coverage = charted / plays if plays else 0.0
    if coverage < FTN_MIN_COVERAGE:
        problems.append(f"FTN charting covers {coverage:.0%} of week-{week} plays")
    return not problems, problems


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True, help="the upcoming game week")
    ap.add_argument("--season", type=int, default=settings.data.eval_season)
    ap.add_argument("--focus", default="BUF", help="always report from this team's view")
    ap.add_argument("--yes", action="store_true", help="actually run (default: check only)")
    ap.add_argument("--force", action="store_true", help="run even if the data isn't complete")
    a = ap.parse_args()

    con = default_connection()
    ok, problems = data_ready(con, a.season, a.week - 1)
    for p in problems:
        print("not ready:", p)
    matchups = week_matchups(con, a.season, a.week, a.focus)
    runs_dir = settings.resolve(settings.paths.runs_dir)
    live_dir = runs_dir / "live" / f"{a.season}_w{a.week:02d}"
    cost = estimate_cost("full", len(matchups), runs_dir)
    print(f"week {a.week}: {len(matchups)} games, estimated ${cost:.2f}")
    if not a.yes or (not ok and not a.force):
        print("check only; pass --yes to run" + ("" if ok else " (and --force: data incomplete)"))
        return
    summary = run_eval(matchups, "full", [1], settings, runs_dir=live_dir)
    print(f"done {summary['done']}, skipped {summary['skipped']}, failed {len(summary['failed'])}")


if __name__ == "__main__":
    main()
