"""Results write-up rendered from score files."""

from nfl_scout.evaluation.results import render


def scores(acc, repeats=(1, 2, 3)):
    runs = [
        {"repeat": k, "draft": {"claims": 10, "value_ok": int(acc * 10)},
         "final": {"claims": 10, "value_ok": 10}}
        for k in repeats
    ]  # fmt: skip
    agg = {
        "runs": len(runs),
        "draft_claims": 30,
        "draft_accuracy": acc,
        "draft_accuracy_ci": [acc - 0.05, acc + 0.02],
        "final_accuracy": 1.0,
        "final_accuracy_ci": [1.0, 1.0],
        "report_unsupported_rate": 0.0,
        "seeded_catch_rate": 1.0,
        "seeded_by_kind": {"wrong_team": 1.0, "wrong_value": 0.9},
        "seeded_false_positives": 0,
        "independent_covered": 20,
        "independent_tool_agreement": 1.0,
        "cost_per_report": 0.031,
    }
    return {"aggregate": agg, "runs": runs}


def test_render_compares_configs_with_intervals_and_context():
    md = render(
        {"full": scores(0.9), "single": scores(1.0)},
        context={"model": "claude-haiku-5-5", "matchups": "v1", "snapshot": "75d2056509ab"},
    )
    assert "| Draft claim accuracy | 90.0% (85.0–92.0) | 100.0% (95.0–102.0) |" in md
    assert "| Cost per report | $0.031 | $0.031 |" in md
    assert "wrong_value" in md and "90.0%" in md
    assert "claude-haiku-5-5" in md and "75d2056509ab" in md


def test_render_shows_accuracy_per_repeat():
    md = render({"full": scores(0.9)}, context={})
    assert "| full | 90.0% | 90.0% | 90.0% |" in md


def test_missing_metrics_render_as_dash():
    s = scores(0.9)
    s["aggregate"]["final_accuracy"] = None
    s["aggregate"]["final_accuracy_ci"] = [None, None]
    assert "| Final claim accuracy | — |" in render({"full": s}, context={})


def test_render_adds_the_forward_check_when_given():
    forward = {
        "full": {"aggregate": {"claims": 100, "report_closer": 0.48,
                               "by_tool": {"pass_rate": {"claims": 40, "report_closer": 0.42,
                                                         "report_mae": 10.9, "naive_mae": 9.8}}}},
    }  # fmt: skip
    md = render({"full": scores(0.9)}, context={}, forward=forward)
    assert "## Forward check" in md
    assert "| Full graph | 100 | 48.0% |" in md
    assert "| pass_rate | 40 | 42.0% | 10.900 | 9.800 |" in md


def test_render_flags_reports_made_without_any_tool_call():
    s = scores(0.9)
    s["runs"][0]["tool_calls"] = 0
    s["runs"][1]["tool_calls"] = 30
    assert "| Reports with no tool calls | 1 |" in render({"full": s}, context={})
