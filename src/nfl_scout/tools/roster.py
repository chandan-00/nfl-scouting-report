"""Injury report and snap-count tools. Both read only weeks <= through_week."""

from __future__ import annotations

import re

import duckdb

from nfl_scout.tools.base import Measure, run_query, tool

STATUS_ORDER = "CASE report_status WHEN 'Out' THEN 0 WHEN 'Doubtful' THEN 1 ELSE 2 END"
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


@tool(unit="players", rankable=False)
def injury_report(
    con: duckdb.DuckDBPyConnection, team: str, season: int, through_week: int
) -> Measure:
    """Players with a game status (Out, Doubtful, Questionable) on the team's latest injury
    report up to through_week. n_plays is the number of players listed. player_id is the
    play-by-play id, which player_share accepts."""
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
        SELECT full_name, gsis_id, position, report_status, report_primary_injury, practice_status
        FROM injuries
        WHERE team = $team AND season = $season AND game_type = 'REG' AND week = $week
          AND report_status IS NOT NULL
        ORDER BY {STATUS_ORDER}, full_name
        """,
        {"team": team, "season": season, "week": report_week},
    ).fetchall()
    players = [
        {
            "player": n,
            "player_id": pid,
            "position": pos,
            "status": s,
            "injury": inj,
            "practice_status": prac,
        }
        for n, pid, pos, s, inj, prac in rows
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
        "first_week": first_week,
        "through_week": through_week,
    }
    pfr_id = _resolve_snap_player(con, player, params)
    name, value, n, weeks = run_query(
        con,
        """
        SELECT any_value(player), 100.0 * avg(greatest(offense_pct, defense_pct)), count(*),
               list_sort(list(DISTINCT week))
        FROM snap_counts
        WHERE team = $team AND season = $season AND game_type = 'REG'
          AND week BETWEEN $first_week AND $through_week AND pfr_player_id = $pfr_id
        """,
        {**params, "pfr_id": pfr_id},
    )
    meta = {
        "definition": f"{name}'s mean snap share per game, weeks {first_week}-{through_week}.",
        "weeks": weeks,
        "n_games": n,
    }
    return Measure(value, n, meta)


def _last_name(full: str) -> str:
    words = [w for w in re.split(r"\s+", full.lower().strip()) if w.strip(".") not in SUFFIXES]
    return words[-1] if words else ""


def _resolve_snap_player(con: duckdb.DuckDBPyConnection, player: str, params: dict) -> str:
    """The pfr id for `player` (pfr id or name) in the team's snap counts. Falls back to a unique
    last-name match, since sources spell first names differently ("Pat" / "Patrick")."""
    rows = con.execute(
        """
        SELECT DISTINCT player, pfr_player_id FROM snap_counts
        WHERE team = $team AND season = $season AND game_type = 'REG'
          AND week BETWEEN $first_week AND $through_week
        """,
        params,
    ).fetchall()
    exact = {pid for name, pid in rows if name.lower() == player.lower() or pid == player}
    if not exact:
        exact = {pid for name, pid in rows if _last_name(name) == _last_name(player)}
    if len(exact) == 1:
        return exact.pop()
    weeks = f"weeks {params['first_week']}-{params['through_week']}"
    if exact:
        options = ", ".join(sorted(f"{n} ({pid})" for n, pid in rows if pid in exact))
        raise ValueError(f"player {player!r} is ambiguous: {options}; pass the full name")
    raise ValueError(
        f"no snaps recorded for {player!r} with {params['team']} in {weeks} "
        "(the player may not have played)"
    )
