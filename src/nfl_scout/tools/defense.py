"""Defensive tendencies from FTN charting: blitz rate, pass rushers, box count.

Defensive efficiency (EPA allowed, sacks, pressure proxy, third downs, red zone) lives in
efficiency.py with side="defense".
"""

from __future__ import annotations

import duckdb

from nfl_scout.tools.base import PLAY_KIND_SQL, Measure, PlayKind, measure_plays, tool


def _params(team: str, season: int, through_week: int) -> dict:
    return {"team": team, "season": season, "through_week": through_week}


@tool(unit="percent")
def blitz_rate(
    con: duckdb.DuckDBPyConnection, team: str, season: int, through_week: int
) -> Measure:
    """Share of opponent dropbacks where the defense sent 5+ pass rushers, in percent (FTN)."""
    return measure_plays(
        con,
        "100.0 * avg(CASE WHEN f.n_pass_rushers >= 5 THEN 1.0 ELSE 0.0 END)",
        "p.defteam = $team AND p.qb_dropback = 1 AND f.n_pass_rushers IS NOT NULL",
        _params(team, season, through_week),
        "Percent of dropbacks faced with 5+ pass rushers (FTN charting).",
        ftn=True,
    )


@tool(unit="rushers")
def average_pass_rushers(
    con: duckdb.DuckDBPyConnection, team: str, season: int, through_week: int
) -> Measure:
    """Average number of pass rushers per opponent dropback (FTN)."""
    return measure_plays(
        con,
        "avg(f.n_pass_rushers)",
        "p.defteam = $team AND p.qb_dropback = 1 AND f.n_pass_rushers IS NOT NULL",
        _params(team, season, through_week),
        "Mean pass rushers per dropback faced (FTN charting).",
        ftn=True,
    )


@tool(unit="defenders")
def average_box_defenders(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    play_kind: PlayKind = "all",
) -> Measure:
    """Average defenders in the box at the snap (FTN), optionally on pass or run plays only."""
    return measure_plays(
        con,
        "avg(f.n_defense_box)",
        f"p.defteam = $team AND {PLAY_KIND_SQL[play_kind]} AND f.n_defense_box IS NOT NULL",
        _params(team, season, through_week),
        f"Mean defenders in the box on {play_kind} plays (FTN charting).",
        ftn=True,
    )
