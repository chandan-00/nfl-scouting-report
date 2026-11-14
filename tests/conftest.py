"""Builders for small in-memory DuckDB databases shaped like the nflverse tables."""

import duckdb
import polars as pl
import pytest

PASS = {
    "season": 2026,
    "season_type": "REG",
    "week": 1,
    "game_id": "2026_01_DEN_KC",
    "posteam": "KC",
    "defteam": "DEN",
    "play_type": "pass",
    "pass": 1.0,
    "rush": 0.0,
    "qb_dropback": 1.0,
    "qb_scramble": 0.0,
    "pass_attempt": 1.0,
    "sack": 0.0,
    "qb_hit": 0.0,
    "two_point_attempt": 0.0,
    "epa": 0.0,
    "success": 0.0,
    "down": 1.0,
    "ydstogo": 10.0,
    "yardline_100": 50.0,
    "xpass": 0.5,
    "pass_oe": 0.0,
    "air_yards": 5.0,
    "yards_gained": 5.0,
    "receiver_player_id": "00-WR1",
    "receiver_player_name": "W.One",
    "rusher_player_id": None,
    "rusher_player_name": None,
    "run_location": None,
    "run_gap": None,
    "fixed_drive": 1.0,
    "fixed_drive_result": "Punt",
    "third_down_converted": 0.0,
    # Kicking-play columns, neutral on scrimmage plays.
    "kick_distance": None,
    "return_yards": 0.0,
    "touchback": 0.0,
    "punt_inside_twenty": 0.0,
    "punt_fair_catch": 0.0,
    "kickoff_fair_catch": 0.0,
    "field_goal_attempt": 0.0,
    "field_goal_result": None,
    "extra_point_attempt": 0.0,
    "extra_point_result": None,
    "kicker_player_name": None,
    "punter_player_name": None,
    "kickoff_returner_player_name": None,
    "punt_returner_player_name": None,
}

# Kicking plays: field goal and punt rows have the kicking team as posteam; kickoff rows have the
# receiving team as posteam (the nflverse convention).
KICK = {"pass": 0.0, "rush": 0.0, "qb_dropback": 0.0, "pass_attempt": 0.0, "air_yards": None,
        "receiver_player_id": None, "receiver_player_name": None, "down": None}  # fmt: skip

RUN = {
    **PASS,
    "play_type": "run",
    "pass": 0.0,
    "rush": 1.0,
    "qb_dropback": 0.0,
    "pass_attempt": 0.0,
    "air_yards": None,
    "receiver_player_id": None,
    "receiver_player_name": None,
    "rusher_player_id": "00-RB1",
    "rusher_player_name": "R.One",
    "run_location": "middle",
}

FTN = {
    "nflverse_game_id": "2026_01_DEN_KC",
    "nflverse_play_id": 1,
    "season": 2026,
    "week": 1,
    "n_blitzers": 0,
    "n_pass_rushers": 4,
    "n_defense_box": 6,
    "is_play_action": False,
    "is_motion": False,
    "is_screen_pass": False,
    "is_rpo": False,
}

INJURY = {
    "season": 2026,
    "game_type": "REG",
    "team": "KC",
    "week": 1,
    "gsis_id": "00-WR1",
    "position": "WR",
    "full_name": "Wide One",
    "report_primary_injury": "Hamstring",
    "report_status": "Questionable",
    "practice_status": "Limited Participation in Practice",
}

SNAP = {
    "game_id": "2026_01_DEN_KC",
    "season": 2026,
    "game_type": "REG",
    "week": 1,
    "player": "Wide One",
    "pfr_player_id": "OneWi00",
    "position": "WR",
    "team": "KC",
    "opponent": "DEN",
    "offense_snaps": 60.0,
    "offense_pct": 0.9,
    "defense_snaps": 0.0,
    "defense_pct": 0.0,
}

TEAMS = ["KC", "DEN", "LV", "LAC"]
TEAM_NAMES = {
    "KC": "Kansas City Chiefs",
    "DEN": "Denver Broncos",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
}

# A DEN offensive play against the KC defense.
KC_DEF_ROW = {"posteam": "DEN", "defteam": "KC"}


NUMERIC_NULLABLE = {"kick_distance", "air_yards", "down"}


def _create(con: duckdb.DuckDBPyConnection, name: str, rows: list[dict], template: dict) -> None:
    df = pl.DataFrame([{**template, **r} for r in rows] or [template], infer_schema_length=None)
    # All-null columns (e.g. rusher ids when every play is a pass) are strings in the real data,
    # except the numeric ones below.
    df = df.with_columns(
        pl.col(c).cast(pl.Float64 if c in NUMERIC_NULLABLE else pl.Utf8)
        for c, t in df.schema.items()
        if t == pl.Null
    )
    con.register(f"_{name}", df.to_arrow())
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _{name}")
    if not rows:
        con.execute(f"DELETE FROM {name}")
    con.unregister(f"_{name}")


def build_db(
    con: duckdb.DuckDBPyConnection,
    pbp: list[dict] = (),
    ftn: list[dict] = (),
    injuries: list[dict] = (),
    snaps: list[dict] = (),
) -> duckdb.DuckDBPyConnection:
    """(Re)create every table the tools read. Rows are partial dicts merged over the defaults;
    pbp rows get play_id 1, 2, 3, ... unless they set one."""
    _create(
        con,
        "pbp",
        [{**PASS, "play_id": float(i + 1), **r} for i, r in enumerate(pbp)],
        {**PASS, "play_id": 1.0},
    )
    _create(con, "ftn_charting", list(ftn), FTN)
    _create(con, "injuries", list(injuries), INJURY)
    _create(con, "snap_counts", list(snaps), SNAP)
    _create(
        con,
        "teams",
        [
            {"team_abbr": t, "team_name": n, "team_nick": n.split()[-1]}
            for t, n in TEAM_NAMES.items()
        ],
        {"team_abbr": "KC", "team_name": "Kansas City Chiefs", "team_nick": "Chiefs"},
    )
    return con


@pytest.fixture
def con():
    with duckdb.connect(":memory:") as c:
        yield c
