"""Independent metric code: hand-computed values, and agreement with the tools."""

import pytest
from conftest import KC_DEF_ROW, RUN, build_db
from pytest import approx

from nfl_scout.evaluation.independent import SPLITS, cross_check, independent_value, load_plays


@pytest.fixture
def db(con):
    build_db(
        con,
        [
            {"epa": 0.6, "success": 1.0, "yards_gained": 25.0, "down": 3.0, "ydstogo": 9.0,
             "third_down_converted": 1.0, "yardline_100": 15.0, "fixed_drive_result": "Touchdown"},
            {"epa": -0.2, "sack": 1.0, "pass_attempt": 0.0, "air_yards": None, "down": 1.0},
            {**RUN, "epa": 0.1, "success": 1.0, "yards_gained": 12.0, "run_location": "left",
             "fixed_drive": 2.0},
            {**KC_DEF_ROW, "epa": 0.3, "qb_hit": 1.0},
            {"week": 2, "epa": 9.0},  # after the cutoff
            {"two_point_attempt": 1.0, "epa": 9.0},  # not a scrimmage play
        ],
    )  # fmt: skip
    return con


def value(db, tool, **args):
    return independent_value(
        load_plays(db, 2026), tool, {"season": 2026, "through_week": 1, **args}
    )


def test_hand_computed_values(db):
    assert value(db, "epa_per_play", team="KC") == (approx(0.5 / 3), 3)
    assert value(db, "epa_per_play", team="KC", play_kind="pass") == (approx(0.2), 2)
    assert value(db, "explosive_play_rate", team="KC") == (approx(200 / 3), 3)
    assert value(db, "sack_rate", team="KC") == (approx(50.0), 2)
    assert value(db, "pressure_proxy_rate", team="KC", side="defense") == (approx(100.0), 1)
    assert value(db, "red_zone_td_rate", team="KC") == (approx(100.0), 1)
    assert value(db, "pass_rate", team="KC", down=3, distance="long") == (approx(100.0), 1)
    assert value(db, "average_depth_of_target", team="KC") == (approx(5.0), 1)
    assert value(db, "run_direction_rate", team="KC", direction="left") == (approx(100.0), 1)


def test_uncovered_tools_return_none(db):
    assert value(db, "blitz_rate", team="KC") is None


def test_agrees_with_every_tool_on_every_split(db):
    n, bad = cross_check(db, 2026, 1)
    assert bad == []
    assert n == sum(len(s) for s in SPLITS.values()) * 2  # two teams have plays


def test_cross_check_catches_a_tool_bug(db, monkeypatch):
    # Sabotage one tool: the cross-check must notice.
    import dataclasses

    from nfl_scout.tools import TOOLS

    real = TOOLS["sack_rate"].fn

    def broken(*a, **k):
        r = real(*a, **k)
        return r.model_copy(update={"value": (r.value or 0) + 1})

    monkeypatch.setitem(TOOLS, "sack_rate", dataclasses.replace(TOOLS["sack_rate"], fn=broken))
    _, bad = cross_check(db, 2026, 1)
    assert bad and {b["tool"] for b in bad} == {"sack_rate"}
