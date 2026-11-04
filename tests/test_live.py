"""Weekly live reports: one per game, focus team's view, and only on complete data."""

import pytest
from conftest import RUN, build_db

from nfl_scout.live import data_ready, week_matchups


@pytest.fixture
def db(con):
    # Week 1: two games, both with play-by-play; FTN charts KC's plays only.
    build_db(
        con,
        [{}, {**RUN}, {"game_id": "2026_01_LV_LAC", "posteam": "LV", "defteam": "LAC"}],
        ftn=[{"nflverse_play_id": 1}, {"nflverse_play_id": 2}],
    )
    con.execute(
        """
        CREATE TABLE schedules AS SELECT * FROM (VALUES
          ('2026_01_DEN_KC', 2026, 'REG', 1, 'DEN', 'KC'),
          ('2026_01_LV_LAC', 2026, 'REG', 1, 'LV', 'LAC'),
          ('2026_02_KC_LV', 2026, 'REG', 2, 'KC', 'LV'),
          ('2026_02_BUF_DEN', 2026, 'REG', 2, 'BUF', 'DEN')
        ) t(game_id, season, game_type, week, away_team, home_team)
        """
    )
    return con


def test_one_report_per_game_from_the_home_team_or_the_focus_team(db):
    ms = week_matchups(db, 2026, 2, focus="BUF")
    assert [(m["team"], m["opponent"]) for m in ms] == [("BUF", "DEN"), ("LV", "KC")]
    assert all(m["through_week"] == 1 for m in ms)


def test_data_ready_needs_every_game_and_ftn_coverage(db):
    ok, problems = data_ready(db, 2026, 1)
    assert not ok
    assert any("FTN" in p for p in problems)  # LV-LAC play has no FTN row
    db.execute(
        "INSERT INTO ftn_charting SELECT * REPLACE ('2026_01_LV_LAC' AS nflverse_game_id, "
        "3 AS nflverse_play_id) FROM ftn_charting LIMIT 1"
    )
    assert data_ready(db, 2026, 1) == (True, [])


def test_data_ready_flags_a_game_without_play_by_play(db):
    db.execute("INSERT INTO schedules VALUES ('2026_01_NYJ_MIA', 2026, 'REG', 1, 'NYJ', 'MIA')")
    ok, problems = data_ready(db, 2026, 1)
    assert not ok and any("2 of 3 week-1 games" in p for p in problems)
