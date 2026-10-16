"""Efficiency metrics for either side of the ball (side="defense" means allowed by `team`)."""

from __future__ import annotations

import duckdb

from nfl_scout.tools.base import (
    PLAY_KIND_SQL,
    PLAYS_CTE,
    Measure,
    PlayKind,
    Side,
    measure_plays,
    run_query,
    team_col,
    tool,
)

OFFENSE_HIGHER = {"offense": "higher", "defense": "lower"}
OFFENSE_LOWER = {"offense": "lower", "defense": "higher"}


def _params(team: str, season: int, through_week: int) -> dict:
    return {"team": team, "season": season, "through_week": through_week}


def _side_label(side: Side) -> str:
    return "by the offense" if side == "offense" else "allowed by the defense"


@tool(unit="epa_per_play", better=OFFENSE_HIGHER)
def epa_per_play(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    side: Side = "offense",
    play_kind: PlayKind = "all",
) -> Measure:
    """Expected points added per scrimmage play (pass = dropbacks incl. sacks and scrambles)."""
    return measure_plays(
        con,
        "avg(p.epa)",
        f"{team_col(side)} = $team AND {PLAY_KIND_SQL[play_kind]}",
        _params(team, season, through_week),
        f"Mean EPA per {play_kind} play {_side_label(side)}.",
    )


@tool(unit="percent", better=OFFENSE_HIGHER)
def success_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    side: Side = "offense",
    play_kind: PlayKind = "all",
) -> Measure:
    """Share of scrimmage plays with positive EPA (nflfastR `success`), in percent."""
    return measure_plays(
        con,
        "100.0 * avg(p.success)",
        f"{team_col(side)} = $team AND {PLAY_KIND_SQL[play_kind]}",
        _params(team, season, through_week),
        f"Percent of {play_kind} plays with EPA > 0, {_side_label(side)}.",
    )


@tool(unit="percent", better=OFFENSE_HIGHER)
def explosive_play_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    side: Side = "offense",
    play_kind: PlayKind = "all",
) -> Measure:
    """Share of plays gaining 20+ yards (passes) or 10+ yards (designed runs), in percent."""
    return measure_plays(
        con,
        "100.0 * avg(CASE WHEN (p.pass = 1 AND p.yards_gained >= 20)"
        " OR (p.rush = 1 AND p.yards_gained >= 10) THEN 1.0 ELSE 0.0 END)",
        f"{team_col(side)} = $team AND {PLAY_KIND_SQL[play_kind]}",
        _params(team, season, through_week),
        f"Percent of {play_kind} plays gaining 20+ yds (pass) or 10+ yds (run), "
        f"{_side_label(side)}.",
    )


@tool(unit="percent", better=OFFENSE_HIGHER)
def red_zone_td_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    side: Side = "offense",
) -> Measure:
    """Share of drives reaching the opponent's 20 that end in a touchdown, in percent.

    n_plays counts red-zone drives, not plays.
    """
    sql = f"""
        {PLAYS_CTE}
        , drives AS (
            SELECT p.game_id, p.fixed_drive, any_value(p.week) AS week,
                   bool_or(p.fixed_drive_result = 'Touchdown') AS td
            FROM plays p
            WHERE {team_col(side)} = $team
            GROUP BY p.game_id, p.fixed_drive
            HAVING min(p.yardline_100) <= 20
        )
        SELECT 100.0 * avg(td::DOUBLE), count(*), list_sort(list(DISTINCT week)),
               count(DISTINCT game_id)
        FROM drives
    """
    value, n, weeks, n_games = run_query(con, sql, _params(team, season, through_week))
    definition = f"Percent of red-zone drives ending in a TD, {_side_label(side)}."
    return Measure(value, n, {"definition": definition, "weeks": weeks or [], "n_games": n_games})


@tool(unit="percent", better=OFFENSE_HIGHER)
def third_down_conversion_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    side: Side = "offense",
) -> Measure:
    """Share of third-down plays converted, in percent. For a defense, 100 minus this is the
    third-down stop rate."""
    return measure_plays(
        con,
        "100.0 * avg(p.third_down_converted)",
        f"{team_col(side)} = $team AND p.down = 3",
        _params(team, season, through_week),
        f"Percent of third-down plays converted, {_side_label(side)}.",
    )


@tool(unit="percent", better=OFFENSE_LOWER)
def sack_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    side: Side = "offense",
) -> Measure:
    """Sacks per dropback, in percent (offense: sacks taken; defense: sacks made)."""
    return measure_plays(
        con,
        "100.0 * avg(p.sack)",
        f"{team_col(side)} = $team AND p.qb_dropback = 1",
        _params(team, season, through_week),
        f"Percent of dropbacks ending in a sack ({side}).",
    )


@tool(unit="percent", better=OFFENSE_LOWER)
def pressure_proxy_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    side: Side = "offense",
) -> Measure:
    """Dropbacks with a sack or QB hit, in percent. A proxy for pressure rate: true pressure
    data is not published for 2026, so always call this a pressure proxy in the report."""
    return measure_plays(
        con,
        "100.0 * avg(CASE WHEN p.sack = 1 OR p.qb_hit = 1 THEN 1.0 ELSE 0.0 END)",
        f"{team_col(side)} = $team AND p.qb_dropback = 1",
        _params(team, season, through_week),
        f"Pressure proxy: percent of dropbacks with a sack or QB hit ({side}). "
        "Not true pressure rate.",
    )
