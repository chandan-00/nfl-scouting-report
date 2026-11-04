"""Forward check: did the report's season-to-date tendencies predict the next game better than
the league average did?"""

import pytest
from conftest import KC_DEF_ROW, RUN, build_db
from pytest import approx

from nfl_scout.evaluation.forward import aggregate_forward, forward_rows
from nfl_scout.evaluation.independent import load_plays


@pytest.fixture
def plays(con):
    # Through week 1: KC passes on 2 of 3 plays, DEN on 1 of 1. League: 3 of 4 = 75%.
    # Week 2 (the game being scouted): KC passes on 6 of 10 plays.
    week2 = [{"week": 2}] * 6 + [{**RUN, "week": 2}] * 4
    build_db(con, [{}, {}, {**RUN}, {**KC_DEF_ROW}, *week2])
    return load_plays(con, 2026)


def run(claim_ids=("oc1",), value=66.67):
    call = {
        "call_id": "o1",
        "node": "offense_analyst",
        "tool": "pass_rate",
        "args": {"team": "KC", "season": 2026, "through_week": 1},
        "result": {"value": value, "n_plays": 3, "query_meta": {}},
        "error": None,
    }
    claims = [
        {"claim_id": "oc1", "text": "KC pass rate", "stated_value": 66.7, "tool_call_id": "o1",
         "derived_from": None, "op": None, "measure": "value"},
        {"claim_id": "oc2", "text": "KC sample", "stated_value": 3, "tool_call_id": "o1",
         "derived_from": None, "op": None, "measure": "n_plays"},
    ]  # fmt: skip
    return {
        "config": "full",
        "request": {"team": "DEN", "opponent": "KC", "season": 2026, "week": 2},
        "sections": {"offense_analyst": {"role": "offense_analyst", "prose": "", "claims": claims}},
        "number_checks": [{"mention": {"claim_id": c}, "status": "ok"} for c in claim_ids],
        "tool_calls": [call],
    }


def test_compares_report_value_and_league_average_with_the_next_game(plays):
    (row,) = forward_rows(run(), plays, min_n=5)
    assert row["tool"] == "pass_rate"
    assert row["report_value"] == approx(66.67)
    assert row["next_game"] == approx(60.0) and row["next_n"] == 10
    assert row["league"] == approx(75.0)
    assert row["report_error"] == approx(6.67) and row["naive_error"] == approx(15.0)


def test_only_value_claims_cited_in_the_report_with_enough_plays(plays):
    assert forward_rows(run(claim_ids=()), plays, min_n=5) == []  # not cited in the report
    assert forward_rows(run(), plays, min_n=11) == []  # next game too small


def test_aggregate_reports_closer_share_and_mean_errors(plays):
    other = [{**r, "matchup": "w02_X_vs_Y"} for r in forward_rows(run(value=90.0), plays, min_n=5)]
    rows = forward_rows(run(), plays, min_n=5) + other
    agg = aggregate_forward(rows)
    assert agg["claims"] == 2
    assert agg["report_closer"] == approx(0.5)
    # Errors are only averaged within a metric: EPA and percentages don't share units.
    assert "report_mae" not in agg
    assert agg["by_tool"]["pass_rate"]["report_mae"] == approx((6.67 + 30.0) / 2)
    assert agg["by_tool"]["pass_rate"]["naive_mae"] == approx(15.0)


def test_the_same_game_metric_and_split_counts_once_across_repeats(plays):
    # Three repeats citing the same call are one prediction, not three.
    rows = forward_rows(run(), plays, min_n=5) * 3
    assert aggregate_forward(rows)["claims"] == 1
