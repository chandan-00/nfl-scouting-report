"""Scorer: reads saved run files only (no API), per run and per config."""

import json

import pytest
from conftest import RUN, build_db
from test_graph import FakeClient, _offense, _run, _scripts, tool_response

from nfl_scout.config import load_settings
from nfl_scout.evaluation.score import aggregate, score_config, score_run
from nfl_scout.runs import load_run, save_run


@pytest.fixture
def runs(con, tmp_path):
    build_db(con, [{}, {**RUN}])
    clean = save_run(_run(con, FakeClient(_scripts())), "full", load_settings(), tmp_path, 1)
    scripts = _scripts()
    # Draft says 60.0 (the tool says 50.0); the revision fixes it.
    scripts["offense_analyst"] = [tool_response(("pass_rate", {"team": "KC"})), _offense(60.0)]
    scripts["offense_analyst"] += [_offense(50.0)]
    fixed = save_run(_run(con, FakeClient(scripts)), "full", load_settings(), tmp_path, 2)
    return load_run(clean), load_run(fixed), con, tmp_path


def test_clean_run_scores_perfect_draft_and_final_accuracy(runs):
    clean, _, con, _ = runs
    s = score_run(clean, con, seeds=2)
    assert s["draft"]["claims"] == 3 and s["draft"]["value_ok"] == 3
    assert s["final"]["claims"] == s["final"]["value_ok"] > 0
    assert s["corrected"] == 0 and s["dropped"] == 0
    # The editor's report has one number with no marker ("of 2 plays").
    assert s["final"]["report_numbers"] == 4 and s["final"]["unsupported"] == 1


def test_revised_run_counts_the_draft_error_and_the_correction(runs):
    _, fixed, con, _ = runs
    s = score_run(fixed, con, seeds=0)
    assert s["draft"]["claims"] == 2 and s["draft"]["value_ok"] == 1
    assert s["final"]["value_ok"] == s["final"]["claims"]
    assert s["corrected"] == 1 and s["dropped"] == 0 and s["checks_run"] == 2


def test_independent_sample_agrees_with_the_tools(runs):
    clean, _, con, _ = runs
    ind = score_run(clean, con, seeds=0)["independent"]
    assert ind["sampled"] >= 1
    assert ind["tool_agrees"] == ind["covered"]


def test_seeded_errors_are_scored_per_kind(runs):
    clean, _, con, _ = runs
    seeded = score_run(clean, con, seeds=3)["seeded"]
    assert set(seeded["by_kind"]) <= {"wrong_value", "wrong_team", "wrong_split", "unlinked_number"}
    assert all(v["caught"] <= v["seeded"] for v in seeded["by_kind"].values())


def test_scores_are_deterministic(runs):
    clean, _, con, _ = runs
    assert json.dumps(score_run(clean, con, seeds=2)) == json.dumps(score_run(clean, con, seeds=2))


def test_config_aggregate_pools_claims_and_reports_intervals(runs):
    _, _, con, tmp_path = runs
    rows = score_config("full", tmp_path, con, seeds=1)
    agg = aggregate(rows)
    assert agg["runs"] == 2
    assert agg["draft_accuracy"] == pytest.approx(4 / 5)
    lo, hi = agg["draft_accuracy_ci"]
    assert lo <= agg["draft_accuracy"] <= hi
    assert agg["final_accuracy"] == 1.0
    assert agg["cost_per_report"] >= 0


def test_unsourced_claims_are_counted(runs):
    clean, _, con, _ = runs
    clean["draft"]["offense_analyst"]["unsourced"] = [{"claim_id": "oc9", "text": "x"}]
    s = score_run(clean, con, seeds=0)
    assert s["draft"]["unsourced_claims"] == 1
    assert aggregate([s])["unsourced_claims"] == 1
