"""Leakage guard: no tool may read data from weeks after through_week.

Every registered tool must appear in EXAMPLE_ARGS, so a new tool can't skip this test.
"""

import pytest
from conftest import FTN, INJURY, KICK, PASS, RUN, SNAP, build_db

from nfl_scout.tools import TOOLS

THROUGH_WEEK = 2

EXAMPLE_ARGS = {
    "epa_per_play": {"side": "defense", "play_kind": "pass"},
    "success_rate": {},
    "explosive_play_rate": {},
    "red_zone_td_rate": {},
    "third_down_conversion_rate": {},
    "sack_rate": {"side": "defense"},
    "pressure_proxy_rate": {},
    "pass_rate": {"down": 1},
    "pass_rate_over_expected": {},
    "average_depth_of_target": {},
    "run_direction_rate": {"direction": "left"},
    "player_share": {"player": "W.One", "kind": "target"},
    "usage_leaders": {"kind": "rush"},
    "ftn_play_rate": {"feature": "motion"},
    "blitz_rate": {},
    "average_pass_rushers": {},
    "average_box_defenders": {},
    "league_rank": {"metric": "epa_per_play"},
    "injury_report": {},
    "snap_share": {"player": "Wide One"},
    "field_goal_rate": {"distance": "50_plus"},
    "kicker_summary": {},
    "punting": {"stat": "net_average"},
    "kickoff_touchback_rate": {},
    "return_average": {"kind": "punt", "side": "defense"},
    "special_teams_epa": {},
}


def _week_rows(week: int, epa: float, flag: bool) -> dict:
    """One week of rows for every table, with values that would change any metric."""
    game = f"2026_{week:02d}_KC_DEN"
    plays = []
    for i, base in enumerate([PASS, RUN, {**PASS, "posteam": "DEN", "defteam": "KC"}]):
        plays.append(
            {
                **base,
                "week": week,
                "game_id": game,
                "play_id": float(i + 1),
                "epa": epa,
                "success": float(flag),
                "yards_gained": 30.0 if flag else 1.0,
                "down": 3.0 if flag else 1.0,
                "third_down_converted": float(flag),
                "sack": float(flag and base is not RUN),
                "qb_hit": float(flag),
                "pass_oe": 50.0 if flag else 0.0,
                "air_yards": (40.0 if flag else 2.0) if base is not RUN else None,
                "yardline_100": 5.0 if flag else 15.0,
                "fixed_drive_result": "Touchdown" if flag else "Field goal",
                "run_location": "left" if flag and base is RUN else base["run_location"],
                **({"rusher_player_id": "00-RB9", "rusher_player_name": "R.Nine"} if flag else {}),
            }
        )
    # Kicking plays (KC field goal, extra point, punt, kickoff, and a DEN kickoff KC returns),
    # with values that change every special-teams metric.
    kick = {**KICK, "week": week, "game_id": game, "epa": epa}
    plays += [
        {**kick, "play_type": "field_goal", "field_goal_attempt": 1.0,
         "kick_distance": 55.0, "field_goal_result": "missed" if flag else "made",
         "kicker_player_name": "K.Late" if flag else "K.One"},
        {**kick, "play_type": "extra_point", "extra_point_attempt": 1.0,
         "extra_point_result": "failed" if flag else "good", "kicker_player_name": "K.One"},
        {**kick, "play_type": "punt", "kick_distance": 70.0 if flag else 40.0,
         "return_yards": 50.0 if flag else 5.0, "punt_returner_player_name": "R.Den",
         "punt_inside_twenty": float(flag), "touchback": 0.0},
        {**kick, "play_type": "kickoff", "posteam": "DEN", "defteam": "KC",
         "touchback": float(flag), "kick_distance": 65.0},
        {**kick, "play_type": "kickoff", "posteam": "KC", "defteam": "DEN",
         "return_yards": 90.0 if flag else 20.0, "kickoff_returner_player_name": "R.Kc"},
    ]  # fmt: skip
    for i, p in enumerate(plays):
        p["play_id"] = float(i + 1)
    ftn = [
        {
            **FTN,
            "nflverse_game_id": game,
            "nflverse_play_id": i + 1,
            "week": week,
            "n_pass_rushers": 7 if flag else 4,
            "n_defense_box": 9 if flag else 6,
            "is_motion": flag,
        }
        for i in range(3)
    ]
    injuries = [{**INJURY, "week": week, "report_status": "Out" if flag else "Questionable"}]
    snaps = [{**SNAP, "week": week, "game_id": game, "offense_pct": 0.1 if flag else 0.8}]
    return {"pbp": plays, "ftn": ftn, "injuries": injuries, "snaps": snaps}


def _db(con, weeks):
    tables = {"pbp": [], "ftn": [], "injuries": [], "snaps": []}
    for week, epa, flag in weeks:
        for name, rows in _week_rows(week, epa, flag).items():
            tables[name] += rows
    return build_db(con, **tables)


def test_every_tool_has_example_args():
    assert set(EXAMPLE_ARGS) == set(TOOLS)


@pytest.mark.parametrize("name", sorted(EXAMPLE_ARGS))
def test_tool_ignores_weeks_after_through_week(con, name):
    fn = TOOLS[name].fn
    past = [(1, 0.1, False), (2, -0.3, False)]

    _db(con, past)
    before = fn(team="KC", season=2026, through_week=THROUGH_WEEK, con=con, **EXAMPLE_ARGS[name])

    _db(con, [*past, (3, 9.0, True), (4, -9.0, True)])
    after = fn(team="KC", season=2026, through_week=THROUGH_WEEK, con=con, **EXAMPLE_ARGS[name])

    assert after == before
    assert max(after.query_meta.get("weeks") or [0]) <= THROUGH_WEEK
