"""Eval runner: every frozen matchup per config and repeat, resumable, failures recorded."""

import json

import pytest

from nfl_scout.config import load_settings
from nfl_scout.evaluation.run_eval import estimate_cost, pending, run_eval

MATCHUPS = [
    {"game_id": "g1", "week": 4, "team": "DEN", "opponent": "KC", "through_week": 3},
    {"game_id": "g2", "week": 5, "team": "BUF", "opponent": "NE", "through_week": 4},
    {"game_id": "g3", "week": 6, "team": "SEA", "opponent": "LV", "through_week": 5},
]


class FakeGraph:
    def __init__(self, fail=()):
        self.fail = set(fail)
        self.seen = []

    def invoke(self, state):
        r = state["request"]
        self.seen.append(r.opponent)
        if r.opponent in self.fail:
            raise RuntimeError("boom")
        return {"request": r, "usage": []}


def run(tmp_path, graph, **kw):
    return run_eval(
        MATCHUPS, "full", [1], load_settings(), build=lambda config: graph,
        runs_dir=tmp_path, workers=1, **kw,
    )  # fmt: skip


def test_saves_one_file_per_matchup_and_repeat(tmp_path):
    summary = run(tmp_path, FakeGraph())
    files = sorted(p.name for p in (tmp_path / "full" / "r1").glob("*.json"))
    assert files == [
        "2026_w04_DEN_vs_KC.json",
        "2026_w05_BUF_vs_NE.json",
        "2026_w06_SEA_vs_LV.json",
    ]
    assert summary["done"] == 3 and summary["failed"] == []
    assert json.loads((tmp_path / "full" / "r1" / files[0]).read_text())["repeat"] == 1


def test_resumes_by_skipping_saved_runs(tmp_path):
    run(tmp_path, FakeGraph())
    graph = FakeGraph()
    summary = run(tmp_path, graph)
    assert graph.seen == [] and summary["skipped"] == 3


def test_a_failed_matchup_is_recorded_and_the_rest_continue(tmp_path):
    summary = run(tmp_path, FakeGraph(fail={"NE"}))
    assert summary["done"] == 2
    assert [f["matchup"] for f in summary["failed"]] == ["2026_w05_BUF_vs_NE"]
    log = (tmp_path / "full" / "failures.jsonl").read_text().splitlines()
    assert "boom" in json.loads(log[0])["error"]
    assert pending(MATCHUPS, "full", [1], tmp_path, 2026)[0]["opponent"] == "NE"


def test_limit_runs_only_the_first_pending(tmp_path):
    summary = run(tmp_path, FakeGraph(), limit=2)
    assert summary["done"] == 2


def test_cost_estimate_uses_measured_runs_when_there_are_some(tmp_path):
    assert estimate_cost("single", 10, tmp_path) == pytest.approx(10 * 0.011, rel=0.5)
    d = tmp_path / "single" / "r1"
    d.mkdir(parents=True)
    (d / "x.json").write_text(json.dumps({"cost_usd": {"total": 0.02}}))
    assert estimate_cost("single", 10, tmp_path) == pytest.approx(0.2)
