"""Demo: browse saved scouting reports and the evaluation results. No API calls.

    uv run streamlit run app/streamlit_app.py

Reports come from demo/ (committed copies of saved runs), or runs/live/ when it exists locally.
Hover a highlighted number to see its claim, the tool call it cites and the check result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# On Streamlit Community Cloud only app/requirements.txt is installed, not the package itself.
sys.path.insert(0, str(ROOT / "src"))

# Streamlit Cloud pulls new commits into the running process, where an older app_render may
# already be imported; Streamlit reruns only this script. Drop the cached copy so the helpers
# always match this script.
sys.modules.pop("nfl_scout.app_render", None)
from nfl_scout.app_render import annotate_report, cost_rows  # noqa: E402

SOURCES = {"Live reports": [ROOT / "demo" / "live", ROOT / "runs" / "live"]}
RESULTS = ROOT / "docs" / "results.md"

STYLE = """
<style>
.report p { line-height: 1.6; font-size: 1rem; }
.num { padding: 0 2px; border-radius: 3px; cursor: help; border-bottom: 1px dotted; }
.num.ok { background: rgba(46, 160, 67, 0.18); }
.num.unlinked, .num.unknown_claim, .num.value_mismatch { background: rgba(218, 54, 51, 0.25); }
</style>
"""


@st.cache_data
def load_runs() -> dict[str, dict[str, Path]]:
    """{week label: {"TEAM (vs OPP)": path}}, newest week first. Each report is written for TEAM
    about OPP. On duplicates demo/ wins over runs/, and a later folder wins over an earlier one."""
    weeks: dict[str, dict[str, Path]] = {}
    for base in reversed(SOURCES["Live reports"]):
        for path in sorted(base.glob("*/**/*.json")):
            if path.name == "failures.jsonl":
                continue
            run = json.loads(path.read_text())
            if "report" not in run:
                continue
            r = run["request"]
            week = f"{r['season']} week {r['week']}"
            weeks.setdefault(week, {})[f"{r['team']} (vs {r['opponent']})"] = path
    weeks = {w: dict(sorted(games.items())) for w, games in weeks.items()}
    return dict(sorted(weeks.items(), key=lambda kv: int(kv[0].split()[-1]), reverse=True))


def show_report(path: Path) -> None:
    run = json.loads(path.read_text())
    report = run["report"]
    statuses = [c["status"] for c in run.get("number_checks", [])]
    ok = sum(s == "ok" for s in statuses)
    cost = run.get("cost_usd", {}).get("total")

    st.subheader(report["title"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numbers checked", f"{ok}/{len(statuses)}")
    c2.metric("Fact-check rounds", run.get("checks_run", 0))
    c3.metric("Claims dropped", len(run.get("dropped", [])))
    c4.metric("Cost", f"${cost:.3f}" if cost is not None else "n/a")
    st.caption(
        "Green numbers are linked to a claim whose tool call was re-run and matched. Red numbers "
        "have no supporting claim. Hover a number for details."
    )

    parts = annotate_report(run)
    html = [STYLE, '<div class="report">']
    for heading, body in parts:
        if heading != "Summary":
            html.append(f"<h4>{heading}</h4>")
        html.append(f"<p>{body}</p>")
    html.append("</div>")
    st.html("".join(html))

    with st.expander("Fact-check log"):
        log = run.get("check_log", [])
        if not log:
            st.write("No failed checks.")
        for e in log:
            st.write(f"Round {e['round']} · {e.get('claim_id') or 'prose'} · {e['check']}: "
                     f"{e['detail']}")  # fmt: skip
    with st.expander(f"Tool calls ({len(run['tool_calls'])})"):
        rows = [
            {
                "id": c["call_id"],
                "analyst": c["node"],
                "tool": c["tool"],
                "args": {k: v for k, v in c["args"].items() if k not in ("season", "through_week")},
                "value": None if c.get("error") else str((c.get("result") or {}).get("value"))[:60],
                "error": c.get("error"),
            }
            for c in run["tool_calls"]
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("Cost by node"):
        st.dataframe(cost_rows(run.get("cost_usd", {})), width="stretch", hide_index=True)


def page_reports() -> None:
    st.title("NFL opponent scouting reports")
    runs = load_runs()
    if not runs:
        st.info("No saved reports found in demo/live or runs/live.")
        return
    week = st.sidebar.selectbox("Week", list(runs))
    games = runs[week]
    default = next((i for i, g in enumerate(games) if g.startswith("BUF")), 0)
    team = st.sidebar.selectbox("Your team", list(games), index=default)
    st.sidebar.caption("Each report scouts your team's opponent for that week.")
    show_report(games[team])


def page_results() -> None:
    st.markdown(RESULTS.read_text() if RESULTS.exists() else "No results yet.")


def page_about() -> None:
    st.title("How it works")
    st.markdown(
        """
A LangGraph pipeline of Claude Haiku 5.5 agents writes an opponent scouting report from 2026
NFL play-by-play data (nflverse), using only games played before the matchup.

1. **Planner** lists the questions for each analyst.
2. **Analysts** (offense, defense, injuries, and special teams in live reports) call
   deterministic SQL tools over DuckDB and write a section. Every number they write is a claim
   that cites one tool call.
3. **Matchup analyst** reads the sections and names the deciding matchups.
4. **Fact-checker** (no LLM) re-runs every cited tool call and checks each claim's value and
   wording. Failing sections are sent back for revision, up to two rounds; claims that still fail
   are dropped.
5. **Editor** writes the final report, and a number check links every number to its claim.

The model writes every number; tool outputs only appear in the trace. The evaluation compares
this graph with a no-checker version and a single agent on 40 matchups from weeks 4-7, after the
model's training cutoff. See the Results page.
"""
    )


st.set_page_config(page_title="NFL scouting reports", layout="wide")
nav = st.navigation(
    [
        st.Page(page_reports, title="Reports", default=True),
        st.Page(page_results, title="Evaluation results"),
        st.Page(page_about, title="How it works"),
    ]
)
nav.run()
