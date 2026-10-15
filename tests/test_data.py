import duckdb
import polars as pl
import pytest

from nfl_scout.data.coverage import ftn_coverage, pbp_field_coverage
from nfl_scout.data.ingest import latest_weeks, write_table


@pytest.fixture
def con():
    with duckdb.connect(":memory:") as c:
        yield c


def test_latest_weeks_per_season():
    df = pl.DataFrame({"season": [2025, 2025, 2026, 2026], "week": [21, 22, 4, 5]})
    assert latest_weeks(df) == "2025: wk 22, 2026: wk 5"


def test_latest_weeks_without_week_column():
    assert latest_weeks(pl.DataFrame({"team": ["KC"]})) is None


def test_write_table_replaces_and_logs(con):
    write_table(con, "t", pl.DataFrame({"season": [2026], "week": [1]}))
    write_table(con, "t", pl.DataFrame({"season": [2026, 2026], "week": [1, 2]}))
    assert con.execute("SELECT count(*) FROM t").fetchone()[0] == 2
    log = con.execute("SELECT n_rows, latest_week_by_season FROM _ingest_log").fetchall()
    assert log == [(2, "2026: wk 2")]


def _pbp(con, rows):
    """Create a minimal pbp table with every column the coverage queries touch."""
    base = {
        "season": 2026,
        "season_type": "REG",
        "game_id": "g1",
        "play_id": 1.0,
        "play_type": "pass",
        "qb_dropback": 1.0,
        "pass_attempt": 1.0,
        "sack": 0.0,
        "qb_scramble": 0.0,
        "run_location": None,
        "epa": 0.1,
        "yards_gained": 5.0,
        "success": 1.0,
        "down": 1.0,
        "ydstogo": 10.0,
        "yardline_100": 50.0,
        "shotgun": 1.0,
        "no_huddle": 0.0,
        "xpass": 0.5,
        "pass_oe": 1.0,
        "qb_hit": 0.0,
        "air_yards": 8.0,
        "pass_location": "left",
        "receiver_player_id": "r1",
        "cpoe": 2.0,
        "rusher_player_id": None,
        "run_gap": None,
    }
    df = pl.DataFrame([{**base, **r} for r in rows], infer_schema_length=None)
    con.register("_pbp", df.to_arrow())
    con.execute("CREATE TABLE pbp AS SELECT * FROM _pbp")


def test_pbp_coverage_uses_relevant_subset(con):
    # Two pass attempts (one missing air_yards) and one run, which must not count for air_yards.
    _pbp(
        con,
        [
            {"play_id": 1.0},
            {"play_id": 2.0, "air_yards": None},
            {
                "play_id": 3.0,
                "play_type": "run",
                "qb_dropback": 0.0,
                "pass_attempt": 0.0,
                "air_yards": None,
                "run_location": "middle",
                "rusher_player_id": "b1",
            },
        ],
    )
    cov = pbp_field_coverage(con, 2026)
    assert cov["air_yards"] == (0.5, 2)
    assert cov["epa"] == (1.0, 3)
    assert cov["rusher_player_id"] == (1.0, 1)
    # A run up the middle has no gap, so it is excluded from the run_gap denominator.
    assert cov["run_gap"][1] == 0


def test_ftn_coverage_counts_unmatched_plays(con):
    _pbp(con, [{"play_id": 1.0}, {"play_id": 2.0}])
    ftn = pl.DataFrame(
        {
            "nflverse_game_id": ["g1"],
            "nflverse_play_id": [1],
            "n_blitzers": [2],
            "n_pass_rushers": [5],
            "n_defense_box": [6],
            "n_offense_backfield": [1],
            "is_play_action": [False],
            "is_motion": [True],
            "is_screen_pass": [False],
            "is_rpo": [False],
            "is_no_huddle": [False],
        }
    )
    con.register("_ftn", ftn.to_arrow())
    con.execute("CREATE TABLE ftn_charting AS SELECT * FROM _ftn")
    matched, n, fields = ftn_coverage(con, 2026)
    assert (matched, n) == (0.5, 2)
    assert fields["n_blitzers"] == 1.0
