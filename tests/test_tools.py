"""Metric tools against hand-computed values on small synthetic tables."""

import pytest
from conftest import FTN, RUN, build_db

from nfl_scout import tools as t

KC_DEF = {"posteam": "DEN", "defteam": "KC"}


def approx(x):
    return pytest.approx(x, abs=1e-9)


# --- Result shape and argument checks -----------------------------------------------------


def test_result_has_value_n_plays_and_meta(con):
    build_db(con, [{"epa": 0.4}, {**RUN, "epa": 0.0, "week": 2, "game_id": "2026_02_KC_LV"}])
    r = t.epa_per_play("KC", 2026, 2, con=con)
    assert r.value == approx(0.2)
    assert r.n_plays == 2
    assert r.query_meta["tool"] == "epa_per_play"
    assert r.query_meta["args"] == {
        "team": "KC",
        "season": 2026,
        "through_week": 2,
        "side": "offense",
        "play_kind": "all",
    }
    assert r.query_meta["unit"] == "epa_per_play"
    assert r.query_meta["weeks"] == [1, 2]
    assert r.query_meta["n_games"] == 2


def test_empty_sample_gives_none(con):
    build_db(con, [{"epa": 0.4}])
    r = t.epa_per_play("KC", 2026, 1, play_kind="run", con=con)
    assert r.value is None
    assert r.n_plays == 0


def test_unknown_team_raises(con):
    build_db(con, [{}])
    with pytest.raises(ValueError, match="team"):
        t.epa_per_play("XYZ", 2026, 1, con=con)


def test_through_week_must_be_positive(con):
    build_db(con, [{}])
    with pytest.raises(ValueError, match="through_week"):
        t.epa_per_play("KC", 2026, 0, con=con)


def test_bad_split_raises(con):
    build_db(con, [{}])
    with pytest.raises(ValueError, match="play_kind"):
        t.epa_per_play("KC", 2026, 1, play_kind="screen", con=con)


# --- Efficiency (offense and defense) ------------------------------------------------------


def test_epa_per_play_by_kind_and_side(con):
    build_db(
        con,
        [
            {"epa": 0.6},
            {"epa": -0.2},
            {**RUN, "epa": 0.5},
            {**KC_DEF, "epa": 1.0},
            # Not scrimmage plays: a punt and a two-point try.
            {"play_type": "punt", "pass": 0.0, "epa": 3.0},
            {"two_point_attempt": 1.0, "epa": 3.0},
        ],
    )
    assert t.epa_per_play("KC", 2026, 1, con=con).value == approx(0.3)
    assert t.epa_per_play("KC", 2026, 1, play_kind="pass", con=con).value == approx(0.2)
    assert t.epa_per_play("KC", 2026, 1, play_kind="run", con=con).value == approx(0.5)
    allowed = t.epa_per_play("KC", 2026, 1, side="defense", con=con)
    assert (allowed.value, allowed.n_plays) == (approx(1.0), 1)


def test_scramble_counts_as_pass(con):
    build_db(con, [{"epa": 0.0}, {**RUN, "pass": 1.0, "rush": 0.0, "qb_scramble": 1.0, "epa": 1.0}])
    assert t.epa_per_play("KC", 2026, 1, play_kind="pass", con=con).value == approx(0.5)
    assert t.epa_per_play("KC", 2026, 1, play_kind="run", con=con).n_plays == 0


def test_success_rate(con):
    build_db(con, [{"success": 1.0}, {"success": 0.0}, {"success": 0.0}, {**RUN, "success": 1.0}])
    assert t.success_rate("KC", 2026, 1, con=con).value == approx(50.0)
    assert t.success_rate("KC", 2026, 1, play_kind="pass", con=con).value == approx(100 / 3)


def test_explosive_play_rate(con):
    build_db(
        con,
        [
            {"yards_gained": 25.0},
            {"yards_gained": 19.0},
            {**RUN, "yards_gained": 12.0},
            {**RUN, "yards_gained": 9.0},
        ],
    )
    r = t.explosive_play_rate("KC", 2026, 1, con=con)
    assert (r.value, r.n_plays) == (approx(50.0), 4)
    assert t.explosive_play_rate("KC", 2026, 1, play_kind="run", con=con).value == approx(50.0)


def test_red_zone_td_rate_counts_drives(con):
    build_db(
        con,
        [
            # Drive 1 reaches the red zone and scores.
            {"fixed_drive": 1.0, "yardline_100": 30.0, "fixed_drive_result": "Touchdown"},
            {"fixed_drive": 1.0, "yardline_100": 15.0, "fixed_drive_result": "Touchdown"},
            # Drive 2 reaches the red zone and kicks a field goal.
            {"fixed_drive": 2.0, "yardline_100": 18.0, "fixed_drive_result": "Field goal"},
            # Drive 3 never gets inside the 20.
            {"fixed_drive": 3.0, "yardline_100": 40.0, "fixed_drive_result": "Touchdown"},
            # Opponent drive inside the KC 20 that scores: counts for KC's defense.
            {**KC_DEF, "fixed_drive": 4.0, "yardline_100": 5.0, "fixed_drive_result": "Touchdown"},
        ],
    )
    r = t.red_zone_td_rate("KC", 2026, 1, con=con)
    assert (r.value, r.n_plays) == (approx(50.0), 2)
    assert t.red_zone_td_rate("KC", 2026, 1, side="defense", con=con).value == approx(100.0)


def test_third_down_conversion_rate(con):
    build_db(
        con,
        [
            {"down": 3.0, "third_down_converted": 1.0},
            {"down": 3.0},
            {**RUN, "down": 3.0},
            {"down": 2.0, "third_down_converted": 0.0},
        ],
    )
    r = t.third_down_conversion_rate("KC", 2026, 1, con=con)
    assert (r.value, r.n_plays) == (approx(100 / 3), 3)


def test_sack_rate_and_pressure_proxy(con):
    build_db(
        con,
        [
            {"sack": 1.0, "qb_hit": 1.0},
            {"qb_hit": 1.0},
            {},
            {},
            {**RUN},
        ],
    )
    sack = t.sack_rate("KC", 2026, 1, con=con)
    assert (sack.value, sack.n_plays) == (approx(25.0), 4)
    pressure = t.pressure_proxy_rate("KC", 2026, 1, con=con)
    assert pressure.value == approx(50.0)
    assert "proxy" in pressure.query_meta["definition"]


# --- Offense tendencies ----------------------------------------------------------------------


def test_pass_rate_by_down_and_distance(con):
    build_db(
        con,
        [
            {"down": 1.0, "ydstogo": 10.0},
            {**RUN, "down": 1.0, "ydstogo": 10.0},
            {**RUN, "down": 3.0, "ydstogo": 2.0},
            {"down": 3.0, "ydstogo": 9.0},
        ],
    )
    assert t.pass_rate("KC", 2026, 1, con=con).value == approx(50.0)
    assert t.pass_rate("KC", 2026, 1, down=1, con=con).value == approx(50.0)
    r = t.pass_rate("KC", 2026, 1, down=3, distance="short", con=con)
    assert (r.value, r.n_plays) == (approx(0.0), 1)
    assert t.pass_rate("KC", 2026, 1, down=3, distance="long", con=con).value == approx(100.0)


def test_pass_rate_early_downs(con):
    build_db(con, [{"down": 1.0}, {**RUN, "down": 2.0}, {"down": 3.0}])
    r = t.pass_rate("KC", 2026, 1, early_downs_only=True, con=con)
    assert (r.value, r.n_plays) == (approx(50.0), 2)


def test_pass_rate_over_expected(con):
    build_db(con, [{"down": 1.0, "pass_oe": 10.0}, {**RUN, "down": 3.0, "pass_oe": -20.0}])
    assert t.pass_rate_over_expected("KC", 2026, 1, con=con).value == approx(-5.0)
    r = t.pass_rate_over_expected("KC", 2026, 1, early_downs_only=True, con=con)
    assert (r.value, r.n_plays) == (approx(10.0), 1)


def test_average_depth_of_target_skips_sacks(con):
    build_db(con, [{"air_yards": 4.0}, {"air_yards": 12.0}, {"sack": 1.0, "air_yards": None}])
    r = t.average_depth_of_target("KC", 2026, 1, con=con)
    assert (r.value, r.n_plays) == (approx(8.0), 2)


def test_run_direction_share(con):
    build_db(
        con,
        [
            {**RUN, "run_location": "left"},
            {**RUN, "run_location": "middle"},
            {**RUN, "run_location": "middle"},
            {**RUN, "run_location": "right"},
        ],
    )
    r = t.run_direction_rate("KC", 2026, 1, direction="middle", con=con)
    assert (r.value, r.n_plays) == (approx(50.0), 4)


def test_player_share_by_name_or_id(con):
    two = {"receiver_player_id": "00-TE1", "receiver_player_name": "T.Two"}
    build_db(con, [{}, {}, two, two, {"receiver_player_id": None}, {**RUN}])
    by_id = t.player_share("KC", 2026, 1, player="00-WR1", kind="target", con=con)
    assert (by_id.value, by_id.n_plays) == (approx(50.0), 4)
    by_name = t.player_share("KC", 2026, 1, player="w.one", kind="target", con=con)
    assert by_name.value == approx(50.0)
    assert by_name.query_meta["player"] == {"id": "00-WR1", "name": "W.One", "count": 2}
    rush = t.player_share("KC", 2026, 1, player="R.One", kind="rush", con=con)
    assert (rush.value, rush.n_plays) == (approx(100.0), 1)


def test_player_share_unknown_player_raises(con):
    build_db(con, [{}])
    with pytest.raises(ValueError, match="player"):
        t.player_share("KC", 2026, 1, player="Nobody", kind="target", con=con)


def test_usage_leaders(con):
    two = {"receiver_player_id": "00-TE1", "receiver_player_name": "T.Two"}
    build_db(con, [{}, two, two, {}, two])
    r = t.usage_leaders("KC", 2026, 1, kind="target", con=con)
    assert r.value == [
        {"player": "T.Two", "player_id": "00-TE1", "count": 3, "share": approx(60.0)},
        {"player": "W.One", "player_id": "00-WR1", "count": 2, "share": approx(40.0)},
    ]
    assert r.n_plays == 5


def test_ftn_play_rates(con):
    build_db(
        con,
        pbp=[{}, {}, {**RUN}, {}],
        ftn=[
            {**FTN, "nflverse_play_id": 1, "is_play_action": True, "is_motion": True},
            {**FTN, "nflverse_play_id": 2},
            {**FTN, "nflverse_play_id": 3, "is_motion": True, "is_rpo": True},
            # play 4 has no FTN row and is left out of the denominator.
        ],
    )
    pa = t.ftn_play_rate("KC", 2026, 1, feature="play_action", con=con)
    assert (pa.value, pa.n_plays) == (approx(50.0), 2)
    motion = t.ftn_play_rate("KC", 2026, 1, feature="motion", con=con)
    assert (motion.value, motion.n_plays) == (approx(200 / 3), 3)


# --- Defense (FTN) ---------------------------------------------------------------------------


def test_blitz_rate_and_rushers(con):
    build_db(
        con,
        pbp=[{**KC_DEF}, {**KC_DEF}, {**KC_DEF, **RUN, **KC_DEF}, {**KC_DEF}],
        ftn=[
            {**FTN, "nflverse_play_id": 1, "n_pass_rushers": 5},
            {**FTN, "nflverse_play_id": 2, "n_pass_rushers": 4},
            {**FTN, "nflverse_play_id": 3, "n_pass_rushers": 0},
            {**FTN, "nflverse_play_id": 4, "n_pass_rushers": 6},
        ],
    )
    blitz = t.blitz_rate("KC", 2026, 1, con=con)
    assert (blitz.value, blitz.n_plays) == (approx(200 / 3), 3)
    assert t.average_pass_rushers("KC", 2026, 1, con=con).value == approx(5.0)


def test_average_box_defenders(con):
    build_db(
        con,
        pbp=[{**KC_DEF}, {**KC_DEF, **RUN, **KC_DEF}],
        ftn=[
            {**FTN, "nflverse_play_id": 1, "n_defense_box": 6},
            {**FTN, "nflverse_play_id": 2, "n_defense_box": 8},
        ],
    )
    assert t.average_box_defenders("KC", 2026, 1, con=con).value == approx(7.0)
    assert t.average_box_defenders("KC", 2026, 1, play_kind="run", con=con).value == approx(8.0)


# --- League rank -----------------------------------------------------------------------------


def _three_offenses(con):
    rows = []
    for i, (team, epa) in enumerate([("KC", 0.1), ("DEN", 0.3), ("LV", -0.2)]):
        rows.append({"posteam": team, "defteam": "LAC", "epa": epa, "play_id": float(i)})
    # LAC's defense faced all three; give LAC an offense too.
    rows.append({"posteam": "LAC", "defteam": "KC", "epa": 0.3, "play_id": 9.0})
    build_db(con, rows)


def test_league_rank_higher_is_better(con):
    _three_offenses(con)
    r = t.league_rank("epa_per_play", "KC", 2026, 1, con=con)
    assert r.value == 3
    assert r.query_meta["team_value"] == approx(0.1)
    assert r.query_meta["n_teams"] == 4
    assert r.query_meta["rank_1_is"] == "best (highest)"


def test_league_rank_ties_share_the_best_rank(con):
    _three_offenses(con)
    r = t.league_rank("epa_per_play", "DEN", 2026, 1, con=con)
    assert (r.value, r.query_meta["tied"]) == (1, True)


def test_league_rank_lower_is_better_on_defense(con):
    _three_offenses(con)
    # KC's defense allowed 0.3 (only LAC's play); LAC's defense allowed mean(0.1, 0.3, -0.2).
    r = t.league_rank("epa_per_play", "LAC", 2026, 1, side="defense", con=con)
    assert r.value == 1
    assert r.query_meta["rank_1_is"] == "best (lowest)"


def test_league_rank_rejects_unrankable_tools(con):
    _three_offenses(con)
    with pytest.raises(ValueError, match="rank"):
        t.league_rank("usage_leaders", "KC", 2026, 1, kind="target", con=con)


# --- Injuries and snaps ----------------------------------------------------------------------


def test_injury_report_uses_latest_week_through_cutoff(con):
    build_db(
        con,
        injuries=[
            {"week": 1, "full_name": "Old News", "report_status": "Out"},
            {"week": 2, "full_name": "Wide One", "report_status": "Questionable"},
            {"week": 2, "full_name": "Practiced Fully", "report_status": None},
            {"week": 3, "full_name": "Future Report", "report_status": "Out"},
        ],
    )
    r = t.injury_report("KC", 2026, 2, con=con)
    assert r.value == [
        {
            "player": "Wide One",
            "position": "WR",
            "status": "Questionable",
            "injury": "Hamstring",
            "practice_status": "Limited Participation in Practice",
        }
    ]
    assert r.query_meta["report_week"] == 2


def test_snap_share_average_and_recent_weeks(con):
    build_db(
        con,
        snaps=[
            {"week": 1, "offense_pct": 0.5},
            {"week": 2, "offense_pct": 0.7},
            {"week": 3, "offense_pct": 0.9},
        ],
    )
    r = t.snap_share("KC", 2026, 3, player="Wide One", con=con)
    assert (r.value, r.n_plays) == (approx(70.0), 3)
    recent = t.snap_share("KC", 2026, 3, player="wide one", last_n_weeks=2, con=con)
    assert recent.value == approx(80.0)
    assert recent.query_meta["weeks"] == [2, 3]


def test_league_rank_names_the_valid_splits(con):
    _three_offenses(con)
    with pytest.raises(ValueError, match=r"epa_per_play accepts: play_kind, side"):
        t.league_rank("epa_per_play", "KC", 2026, 1, direction="left", con=con)
