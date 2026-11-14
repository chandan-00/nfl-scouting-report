"""Special-teams metrics: field goals, kickers, punting, kickoffs, returns and special-teams EPA.

nflverse records field goal, extra point and punt plays with the kicking team as `posteam`, and
kickoff plays with the receiving team as `posteam`. Every query reads only weeks <= through_week.
"""

from __future__ import annotations

from typing import Literal

import duckdb

from nfl_scout.tools.base import Measure, Side, run_query, tool

FgDistance = Literal["all", "under_40", "40_to_49", "50_plus"]
PuntStat = Literal["gross_average", "net_average", "inside_20_rate", "touchback_rate"]
ReturnKind = Literal["kickoff", "punt"]

KICKS_CTE = """
WITH kicks AS (
    SELECT * FROM pbp
    WHERE season = $season AND season_type = 'REG' AND week <= $through_week
      AND play_type IN ('field_goal', 'extra_point', 'punt', 'kickoff')
)
"""

FG_DISTANCE_SQL = {
    "all": "TRUE",
    "under_40": "k.kick_distance < 40",
    "40_to_49": "k.kick_distance >= 40 AND k.kick_distance < 50",
    "50_plus": "k.kick_distance >= 50",
}

# Net punt average: distance minus return yards, minus 20 yards for a touchback.
PUNT_STAT_SQL = {
    "gross_average": "avg(k.kick_distance)",
    "net_average": "avg(k.kick_distance - coalesce(k.return_yards, 0) - 20 * k.touchback)",
    "inside_20_rate": "100.0 * avg(k.punt_inside_twenty)",
    "touchback_rate": "100.0 * avg(k.touchback)",
}

# Which column names the returner, and which team returns, per kind of kick.
RETURNER = {"kickoff": "k.kickoff_returner_player_name", "punt": "k.punt_returner_player_name"}
RETURNING_TEAM = {"kickoff": "k.posteam", "punt": "k.defteam"}
KICKING_TEAM = {"kickoff": "k.defteam", "punt": "k.posteam"}


def _params(team: str, season: int, through_week: int) -> dict:
    return {"team": team, "season": season, "through_week": through_week}


def _measure(
    con: duckdb.DuckDBPyConnection, value_sql: str, where: str, params: dict, definition: str
) -> Measure:
    sql = f"""
        {KICKS_CTE}
        SELECT {value_sql}, count(*), list_sort(list(DISTINCT k.week)), count(DISTINCT k.game_id)
        FROM kicks k WHERE {where}
    """
    value, n, weeks, n_games = run_query(con, sql, params)
    meta = {"definition": definition, "weeks": weeks or [], "n_games": n_games}
    return Measure(value=value, n_plays=n, meta=meta)


@tool(unit="percent", better={"offense": "higher"})
def field_goal_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    distance: FgDistance = "all",
) -> Measure:
    """Share of the team's field goal attempts made, in percent, optionally by kick distance
    (blocked kicks count as misses). n_plays is the number of attempts."""
    return _measure(
        con,
        "100.0 * avg(CASE WHEN k.field_goal_result = 'made' THEN 1.0 ELSE 0.0 END)",
        f"k.play_type = 'field_goal' AND k.posteam = $team AND {FG_DISTANCE_SQL[distance]}",
        _params(team, season, through_week),
        f"Percent of field goal attempts made ({distance.replace('_', ' ')} yards).",
    )


@tool(unit="players", rankable=False)
def kicker_summary(
    con: duckdb.DuckDBPyConnection, team: str, season: int, through_week: int
) -> Measure:
    """Each kicker who tried a field goal or extra point for the team: attempts, makes, longest
    make and the first and last week they kicked, in order of first week. Shows kicker changes.
    n_plays is the team's field goal attempts."""
    sql = f"""
        {KICKS_CTE}
        SELECT k.kicker_player_name,
               count(*) FILTER (WHERE k.play_type = 'field_goal'),
               count(*) FILTER (WHERE k.play_type = 'field_goal'
                                AND k.field_goal_result = 'made'),
               max(k.kick_distance) FILTER (WHERE k.play_type = 'field_goal'
                                            AND k.field_goal_result = 'made'),
               count(*) FILTER (WHERE k.play_type = 'extra_point'),
               count(*) FILTER (WHERE k.play_type = 'extra_point'
                                AND k.extra_point_result = 'good'),
               min(k.week), max(k.week)
        FROM kicks k
        WHERE k.play_type IN ('field_goal', 'extra_point') AND k.posteam = $team
          AND k.kicker_player_name IS NOT NULL
        GROUP BY 1 ORDER BY 7, 1
    """
    rows = con.execute(sql, _params(team, season, through_week)).fetchall()
    kickers = [
        {"player": name, "fg_attempts": fga, "fg_made": fgm, "longest_made": long_,
         "xp_attempts": xpa, "xp_made": xpm, "first_week": first, "last_week": last}
        for name, fga, fgm, long_, xpa, xpm, first, last in rows
    ]  # fmt: skip
    weeks = sorted({w for *_, first, last in rows for w in range(first, last + 1)})
    meta = {"definition": "Field goals and extra points by kicker.", "weeks": weeks}
    return Measure(kickers, sum(k["fg_attempts"] for k in kickers), meta)


@tool(unit="yards_or_percent")
def punting(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    stat: PuntStat,
) -> Measure:
    """The team's punting: gross average (yards), net average (yards, after returns and
    touchbacks), inside-the-20 rate or touchback rate (percent of punts). n_plays is punts."""
    return _measure(
        con,
        PUNT_STAT_SQL[stat],
        "k.play_type = 'punt' AND k.posteam = $team",
        _params(team, season, through_week),
        f"Punting {stat.replace('_', ' ')}.",
    )


@tool(unit="percent")
def kickoff_touchback_rate(
    con: duckdb.DuckDBPyConnection, team: str, season: int, through_week: int
) -> Measure:
    """Share of the team's kickoffs that end in a touchback, in percent. n_plays is kickoffs."""
    return _measure(
        con,
        "100.0 * avg(k.touchback)",
        "k.play_type = 'kickoff' AND k.defteam = $team",
        _params(team, season, through_week),
        "Percent of kickoffs that are touchbacks.",
    )


@tool(unit="yards", better={"offense": "higher", "defense": "lower"})
def return_average(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    kind: ReturnKind,
    side: Side = "offense",
) -> Measure:
    """Average yards per kickoff or punt return: by the team's returners (side="offense") or
    allowed by its coverage (side="defense"). Touchbacks and fair catches aren't returns.
    n_plays is returns."""
    team_sql = RETURNING_TEAM[kind] if side == "offense" else KICKING_TEAM[kind]
    label = "by the return unit" if side == "offense" else "allowed by the coverage unit"
    return _measure(
        con,
        "avg(k.return_yards)",
        f"k.play_type = '{kind}' AND {team_sql} = $team AND {RETURNER[kind]} IS NOT NULL "
        f"AND coalesce(k.touchback, 0) = 0",
        _params(team, season, through_week),
        f"Yards per {kind} return, {label}.",
    )


@tool(unit="epa_per_play", better={"offense": "higher"})
def special_teams_epa(
    con: duckdb.DuckDBPyConnection, team: str, season: int, through_week: int
) -> Measure:
    """Expected points added per special-teams play (field goals, extra points, punts and
    kickoffs, kicking or receiving) from the team's point of view. n_plays is plays."""
    return _measure(
        con,
        "avg(CASE WHEN k.posteam = $team THEN k.epa ELSE -k.epa END)",
        "(k.posteam = $team OR k.defteam = $team) AND k.epa IS NOT NULL",
        _params(team, season, through_week),
        "Mean EPA per special-teams play, from the team's side.",
    )
