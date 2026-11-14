"""The demo's report rendering: numbers become highlighted spans with their claim and check."""

from nfl_scout.app_render import annotate_report


def _run():
    report = {
        "title": "DEN vs KC, Week 2",
        "summary": "KC passes on 61.2% [oc1] of plays.",
        "sections": [
            {"heading": "Opponent offense", "body": "It ran 40 plays <b>[oc2]</b> and 3 more."}
        ],
    }
    text = report["summary"] + "\n\n" + report["sections"][0]["body"]
    checks = [
        {"mention": {"text": "61.2%", "start": text.index("61.2%"), "claim_id": "oc1"},
         "status": "ok"},
        {"mention": {"text": "40", "start": text.index("40"), "claim_id": "oc2"}, "status": "ok"},
        {"mention": {"text": "3", "start": text.index("3 more"), "claim_id": None},
         "status": "unlinked"},
    ]  # fmt: skip
    sections = {
        "offense_analyst": {
            "claims": [
                {
                    "claim_id": "oc1",
                    "text": "KC offense pass rate",
                    "stated_value": 61.2,
                    "tool_call_id": "o1",
                    "measure": "value",
                },
                {
                    "claim_id": "oc2",
                    "text": "KC plays",
                    "stated_value": 40,
                    "tool_call_id": "o1",
                    "measure": "n_plays",
                },
            ]  # fmt: skip
        }
    }
    calls = [{"call_id": "o1", "tool": "pass_rate",
              "args": {"team": "KC", "season": 2026, "through_week": 1},
              "result": {"value": 61.234, "n_plays": 40}, "error": None}]  # fmt: skip
    return {"report": report, "number_checks": checks, "sections": sections, "tool_calls": calls}


def test_numbers_become_spans_with_claim_tool_and_status():
    parts = annotate_report(_run())
    assert [h for h, _ in parts] == ["Summary", "Opponent offense"]
    summary = parts[0][1]
    assert 'class="num ok"' in summary and ">61.2%</span>" in summary
    assert "KC offense pass rate" in summary and "pass_rate" in summary and "61.234" in summary
    assert "[oc1]" not in summary  # markers are removed


def test_sample_sizes_show_the_n_plays_and_unlinked_numbers_are_flagged():
    body = annotate_report(_run())[1][1]
    assert "n_plays = 40" in body
    assert 'class="num unlinked"' in body and ">3</span>" in body
    assert "&lt;b&gt;" in body  # report text is escaped
