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
}

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

# A DEN offensive play against the KC defense.
KC_DEF_ROW = {"posteam": "DEN", "defteam": "KC"}


def _create(con: duckdb.DuckDBPyConnection, name: str, rows: list[dict], template: dict) -> None:
    df = pl.DataFrame([{**template, **r} for r in rows] or [template], infer_schema_length=None)
    # All-null columns (e.g. rusher ids when every play is a pass) are strings in the real data.
    df = df.with_columns(pl.col(c).cast(pl.Utf8) for c, t in df.schema.items() if t == pl.Null)
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
    _create(con, "teams", [{"team_abbr": t} for t in TEAMS], {"team_abbr": "KC"})
    return con


@pytest.fixture
def con():
    with duckdb.connect(":memory:") as c:
        yield c
