"""Offensive tendencies: play calling, run direction, target depth, player usage, FTN rates."""

from __future__ import annotations

from typing import Literal

import duckdb

from nfl_scout.tools.base import PLAYS_CTE, Measure, measure_plays, tool

Distance = Literal["short", "medium", "long"]
UsageKind = Literal["target", "rush"]
Direction = Literal["left", "middle", "right"]
FtnFeature = Literal["play_action", "screen", "motion", "rpo"]

DISTANCE_SQL = {
    "short": "p.ydstogo <= 3",
    "medium": "p.ydstogo BETWEEN 4 AND 7",
    "long": "p.ydstogo >= 8",
}

# Per kind: the plays that count, and the player id and name columns.
USAGE = {
    "target": (
        "p.pass_attempt = 1 AND p.sack = 0 AND p.receiver_player_id IS NOT NULL",
        "receiver_player_id",
        "receiver_player_name",
    ),
    "rush": (
        "p.rush = 1 AND p.rusher_player_id IS NOT NULL",
        "rusher_player_id",
        "rusher_player_name",
    ),
}

# Per FTN feature: the FTN column, and which plays form the denominator.
FTN_FEATURES = {
    "play_action": ("is_play_action", "p.qb_dropback = 1", "dropbacks"),
    "screen": ("is_screen_pass", "p.qb_dropback = 1", "dropbacks"),
    "motion": ("is_motion", "TRUE", "scrimmage plays"),
    "rpo": ("is_rpo", "TRUE", "scrimmage plays"),
}


def _params(team: str, season: int, through_week: int, **extra) -> dict:
    return {"team": team, "season": season, "through_week": through_week, **extra}


@tool(unit="percent")
def pass_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    down: Literal[1, 2, 3, 4] | None = None,
    distance: Distance | None = None,
    early_downs_only: bool = False,
) -> Measure:
    """Share of the offense's scrimmage plays that are dropbacks (passes, sacks, scrambles), in
    percent. Optional splits: down, distance to go (short 1-3, medium 4-7, long 8+ yards), or
    early downs (1st and 2nd)."""
    if early_downs_only and down is not None:
        raise ValueError("use either down or early_downs_only, not both")
    where = ["p.posteam = $team"]
    label = []
    if down is not None:
        where.append(f"p.down = {int(down)}")
        label.append(f"down {down}")
    if early_downs_only:
        where.append("p.down IN (1, 2)")
        label.append("early downs")
    if distance is not None:
        where.append(DISTANCE_SQL[distance])
        label.append(f"{distance} distance")
    return measure_plays(
        con,
        "100.0 * avg(p.pass)",
        " AND ".join(where),
        _params(team, season, through_week),
        f"Percent of plays that are dropbacks ({', '.join(label) or 'all downs'}).",
    )


@tool(unit="percentage_points")
def pass_rate_over_expected(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    early_downs_only: bool = False,
) -> Measure:
    """Dropback rate minus the nflfastR expected dropback rate (xpass), in percentage points.
    Positive means the offense passes more than the situation predicts."""
    where = "p.posteam = $team AND p.pass_oe IS NOT NULL"
    if early_downs_only:
        where += " AND p.down IN (1, 2)"
    return measure_plays(
        con,
        "avg(p.pass_oe)",
        where,
        _params(team, season, through_week),
        f"Mean pass_oe ({'early downs' if early_downs_only else 'all downs'}).",
    )


@tool(unit="yards")
def average_depth_of_target(
    con: duckdb.DuckDBPyConnection, team: str, season: int, through_week: int
) -> Measure:
    """Average air yards per pass attempt (aDOT); sacks and throws without air yards excluded."""
    return measure_plays(
        con,
        "avg(p.air_yards)",
        "p.posteam = $team AND p.pass_attempt = 1 AND p.sack = 0 AND p.air_yards IS NOT NULL",
        _params(team, season, through_week),
        "Mean air yards per pass attempt.",
    )


@tool(unit="percent")
def run_direction_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    direction: Direction,
) -> Measure:
    """Share of designed runs going left, middle or right, in percent."""
    return measure_plays(
        con,
        f"100.0 * avg(CASE WHEN p.run_location = '{direction}' THEN 1.0 ELSE 0.0 END)",
        "p.posteam = $team AND p.rush = 1 AND p.run_location IS NOT NULL",
        _params(team, season, through_week),
        f"Percent of designed runs to the {direction}.",
    )


def _resolve_player(
    con: duckdb.DuckDBPyConnection, plays_where: str, id_col: str, name_col: str, params: dict
) -> tuple[str, str, int]:
    """Match `player` (id or name, case-insensitive) among the team's plays: (id, name, count)."""
    rows = con.execute(
        f"""
        {PLAYS_CTE}
        SELECT p.{id_col}, any_value(p.{name_col}), count(*)
        FROM plays p
        WHERE p.posteam = $team AND {plays_where}
          AND (lower(p.{id_col}) = lower($player) OR lower(p.{name_col}) = lower($player))
        GROUP BY 1 ORDER BY 3 DESC
        """,
        params,
    ).fetchall()
    if not rows:
        raise ValueError(
            f"player {params['player']!r} not found for {params['team']}; "
            "use usage_leaders to list players"
        )
    if len(rows) > 1:
        options = ", ".join(f"{name} ({pid})" for pid, name, _ in rows)
        raise ValueError(f"player {params['player']!r} is ambiguous: {options}; pass the id")
    return rows[0]


@tool(unit="percent", rankable=False)
def player_share(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    player: str,
    kind: UsageKind,
) -> Measure:
    """A player's share of the team's targets or designed rushes, in percent. `player` is a
    gsis id or the play-by-play name (e.g. 'T.Kelce')."""
    plays_where, id_col, name_col = USAGE[kind]
    params = _params(team, season, through_week, player=player)
    pid, name, count = _resolve_player(con, plays_where, id_col, name_col, params)
    m = measure_plays(
        con,
        f"100.0 * avg(CASE WHEN p.{id_col} = $pid THEN 1.0 ELSE 0.0 END)",
        f"p.posteam = $team AND {plays_where}",
        _params(team, season, through_week, pid=pid),
        f"{name}'s percent of team {kind}s.",
    )
    m.meta["player"] = {"id": pid, "name": name, "count": count}
    return m


@tool(unit="players", rankable=False)
def usage_leaders(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    kind: UsageKind,
    top_n: int = 5,
) -> Measure:
    """The team's leaders in targets or designed rushes, with count and share (percent).
    n_plays is the team total."""
    plays_where, id_col, name_col = USAGE[kind]
    sql = f"""
        {PLAYS_CTE}
        SELECT p.{id_col}, any_value(p.{name_col}), count(*),
               sum(count(*)) OVER () AS total, list_sort(list(DISTINCT p.week)) AS weeks
        FROM plays p
        WHERE p.posteam = $team AND {plays_where}
        GROUP BY 1
        ORDER BY 3 DESC, 2
    """
    rows = con.execute(sql, _params(team, season, through_week)).fetchall()
    total = int(rows[0][3]) if rows else 0
    leaders = [
        {"player": name, "player_id": pid, "count": n, "share": 100.0 * n / total}
        for pid, name, n, _, _ in rows[:top_n]
    ]
    weeks = sorted({w for *_, ws in rows for w in ws})
    meta = {"definition": f"Top {top_n} players by team {kind}s.", "weeks": weeks}
    return Measure(leaders, total, meta)


@tool(unit="percent")
def ftn_play_rate(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    feature: FtnFeature,
) -> Measure:
    """How often the offense uses play action or screens (per dropback), or pre-snap motion or
    RPOs (per scrimmage play), in percent. From FTN charting; uncharted plays are excluded."""
    column, base, denominator = FTN_FEATURES[feature]
    return measure_plays(
        con,
        f"100.0 * avg(f.{column}::DOUBLE)",
        f"p.posteam = $team AND {base} AND f.{column} IS NOT NULL",
        _params(team, season, through_week),
        f"Percent of {denominator} with {feature.replace('_', ' ')} (FTN charting).",
        ftn=True,
    )
