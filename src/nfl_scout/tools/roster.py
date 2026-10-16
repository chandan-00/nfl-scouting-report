"""Injury report and snap-count tools. Both read only weeks <= through_week."""

from __future__ import annotations

import duckdb

from nfl_scout.tools.base import Measure, run_query, tool

STATUS_ORDER = "CASE report_status WHEN 'Out' THEN 0 WHEN 'Doubtful' THEN 1 ELSE 2 END"


@tool(unit="players", rankable=False)
def injury_report(
    con: duckdb.DuckDBPyConnection, team: str, season: int, through_week: int
) -> Measure:
    """Players with a game status (Out, Doubtful, Questionable) on the team's latest injury
    report up to through_week. n_plays is the number of players listed."""
    params = {"team": team, "season": season, "through_week": through_week}
    (report_week,) = run_query(
        con,
        """
        SELECT max(week) FROM injuries
        WHERE team = $team AND season = $season AND game_type = 'REG' AND week <= $through_week
        """,
        params,
    )
    rows = con.execute(
        f"""
        SELECT full_name, position, report_status, report_primary_injury, practice_status
        FROM injuries
        WHERE team = $team AND season = $season AND game_type = 'REG' AND week = $week
          AND report_status IS NOT NULL
        ORDER BY {STATUS_ORDER}, full_name
        """,
        {"team": team, "season": season, "week": report_week},
    ).fetchall()
    players = [
        {"player": n, "position": pos, "status": s, "injury": inj, "practice_status": prac}
        for n, pos, s, inj, prac in rows
    ]
    meta = {
        "definition": "Game statuses on the latest injury report up to through_week.",
        "report_week": report_week,
        "weeks": [report_week] if report_week else [],
    }
    return Measure(players, len(players), meta)


@tool(unit="percent", rankable=False)
def snap_share(
    con: duckdb.DuckDBPyConnection,
    team: str,
    season: int,
    through_week: int,
    player: str,
    last_n_weeks: int | None = None,
) -> Measure:
    """A player's average share of offensive or defensive snaps (whichever is larger) per game,
    in percent, optionally over only the last N weeks. n_plays is the number of games."""
    first_week = through_week - last_n_weeks + 1 if last_n_weeks else 1
    params = {
        "team": team,
        "season": season,
        "through_week": through_week,
        "first_week": first_week,
        "player": player,
    }
    name, value, n, weeks = run_query(
        con,
        """
        SELECT any_value(player), 100.0 * avg(greatest(offense_pct, defense_pct)), count(*),
               list_sort(list(DISTINCT week))
        FROM snap_counts
        WHERE team = $team AND season = $season AND game_type = 'REG'
          AND week BETWEEN $first_week AND $through_week
          AND (lower(player) = lower($player) OR pfr_player_id = $player)
        """,
        params,
    )
    if n == 0:
        raise ValueError(
            f"no {team} snap counts for player {player!r} in weeks {first_week}-"
            f"{through_week}; use the full name, e.g. 'Travis Kelce'"
        )
    meta = {
        "definition": f"{name}'s mean snap share per game, weeks {first_week}-{through_week}.",
        "weeks": weeks,
        "n_games": n,
    }
    return Measure(value, n, meta)
