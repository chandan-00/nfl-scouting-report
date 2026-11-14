"""Special-teams tools against hand-computed values. Kickoff rows have the receiving team as
posteam; field goal and punt rows have the kicking team (nflverse)."""

import pytest
from conftest import KICK, build_db

from nfl_scout import tools as t


def approx(x):
    return pytest.approx(x, abs=1e-9)


def fg(dist, result, kicker, week=1, epa=0.0):
    return {**KICK, "play_type": "field_goal", "field_goal_attempt": 1.0, "kick_distance": dist,
            "field_goal_result": result, "kicker_player_name": kicker, "week": week,
            "game_id": f"2026_{week:02d}_DEN_KC", "epa": epa}  # fmt: skip


def xp(result, kicker, week=1):
    return {**KICK, "play_type": "extra_point", "extra_point_attempt": 1.0,
            "extra_point_result": result, "kicker_player_name": kicker, "week": week,
            "game_id": f"2026_{week:02d}_DEN_KC"}  # fmt: skip


def punt(dist, ret=0.0, returner=None, touchback=0.0, inside20=0.0, fair=0.0, team="KC", epa=0.0):
    opp = "DEN" if team == "KC" else "KC"
    return {**KICK, "play_type": "punt", "posteam": team, "defteam": opp, "kick_distance": dist,
            "return_yards": ret, "punt_returner_player_name": returner, "touchback": touchback,
            "punt_inside_twenty": inside20, "punt_fair_catch": fair, "punter_player_name": "P.One",
            "epa": epa}  # fmt: skip


def kickoff(kicking, ret=0.0, returner=None, touchback=0.0, epa=0.0):
    receiving = "DEN" if kicking == "KC" else "KC"
    return {**KICK, "play_type": "kickoff", "posteam": receiving, "defteam": kicking,
            "kick_distance": 65.0, "return_yards": ret, "kickoff_returner_player_name": returner,
            "touchback": touchback, "epa": epa}  # fmt: skip


@pytest.fixture
def st(con):
    rows = [
        fg(30.0, "made", "K.One", epa=0.5),
        fg(45.0, "missed", "K.One", epa=-1.5),
        fg(52.0, "made", "K.Two", week=2),
        fg(38.0, "blocked", "K.Two", week=2),
        xp("good", "K.Two", week=2),
        xp("failed", "K.One"),
        punt(50.0, ret=10.0, returner="R.Den", epa=-0.4),
        punt(40.0, touchback=1.0),
        punt(45.0, inside20=1.0, fair=1.0),
        punt(48.0, ret=12.0, returner="R.Kc", team="DEN"),  # a DEN punt that KC returns
        kickoff("KC", touchback=1.0, epa=0.2),
        kickoff("KC", ret=25.0, returner="R.Den", epa=0.6),
        kickoff("DEN", ret=30.0, returner="R.Kc", epa=0.3),
        {"epa": 9.0},  # a scrimmage play: never counted by these tools
    ]
    return build_db(con, rows)


def test_field_goal_rate_by_distance(st):
    assert t.field_goal_rate("KC", 2026, 2, con=st).value == approx(50.0)
    assert t.field_goal_rate("KC", 2026, 2, con=st).n_plays == 4
    under = t.field_goal_rate("KC", 2026, 2, distance="under_40", con=st)
    assert (under.value, under.n_plays) == (approx(50.0), 2)  # blocked counts as a miss
    assert t.field_goal_rate("KC", 2026, 2, distance="40_to_49", con=st).value == approx(0.0)
    assert t.field_goal_rate("KC", 2026, 2, distance="50_plus", con=st).value == approx(100.0)
    assert t.field_goal_rate("KC", 2026, 1, con=st).n_plays == 2


def test_kicker_summary_shows_each_kicker_and_when_they_kicked(st):
    r = t.kicker_summary("KC", 2026, 2, con=st)
    assert r.n_plays == 4
    assert r.value == [
        {"player": "K.One", "fg_attempts": 2, "fg_made": 1, "longest_made": 30.0,
         "xp_attempts": 1, "xp_made": 0, "first_week": 1, "last_week": 1},
        {"player": "K.Two", "fg_attempts": 2, "fg_made": 1, "longest_made": 52.0,
         "xp_attempts": 1, "xp_made": 1, "first_week": 2, "last_week": 2},
    ]  # fmt: skip


def test_punting_gross_net_inside_20_and_touchbacks(st):
    def stat(s):
        return t.punting("KC", 2026, 2, stat=s, con=st)

    assert stat("gross_average").value == approx(45.0)
    # Net: distance minus return yards, minus 20 for a touchback: (40 + 20 + 45) / 3.
    assert stat("net_average").value == approx(35.0)
    assert stat("inside_20_rate").value == approx(100 / 3)
    assert stat("touchback_rate").value == approx(100 / 3)
    assert stat("net_average").n_plays == 3


def test_kickoff_touchback_rate_counts_the_kicking_team(st):
    r = t.kickoff_touchback_rate("KC", 2026, 2, con=st)
    assert (r.value, r.n_plays) == (approx(50.0), 2)


def test_return_average_for_and_allowed(st):
    def ret(kind, side):
        return t.return_average("KC", 2026, 2, kind=kind, side=side, con=st)

    assert (ret("kickoff", "offense").value, ret("kickoff", "offense").n_plays) == (30.0, 1)
    assert (ret("kickoff", "defense").value, ret("kickoff", "defense").n_plays) == (25.0, 1)
    assert ret("punt", "offense").value == approx(12.0)
    # Touchbacks and fair catches aren't returns.
    assert (ret("punt", "defense").value, ret("punt", "defense").n_plays) == (10.0, 1)


def test_special_teams_epa_is_from_the_teams_point_of_view(st):
    r = t.special_teams_epa("KC", 2026, 2, con=st)
    # KC's own field goals and punts count as is; on KC kickoffs DEN has the ball, so the sign
    # flips (-0.2, -0.6); KC's kickoff return counts as is (+0.3); DEN's punt flips (0.0).
    plays = [0.5, -1.5, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0, -0.0, -0.2, -0.6, 0.3]
    assert r.n_plays == len(plays)
    assert r.value == approx(sum(plays) / len(plays))


def test_special_teams_tools_can_be_ranked(st):
    r = t.league_rank("field_goal_rate", "KC", 2026, 2, con=st)
    assert r.value == 1 and r.query_meta["rank_1_is"] == "best (highest)"
