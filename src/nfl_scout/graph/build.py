"""Assemble the LangGraph:

planner -> (offense || defense || injury analysts) -> matchup analyst -> draft snapshot
  -> fact-checker -> (reviser per failing role -> fact-checker, at most max_revision_loops)
  -> editor -> number check

Baselines: "no_checker" skips the fact-checker; "single" is one agent with every tool.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import duckdb
from langgraph.graph import END, START, StateGraph

from nfl_scout.config import Settings, load_settings
from nfl_scout.graph import nodes
from nfl_scout.graph.llm import LLM, default_client
from nfl_scout.graph.llm_tools import tool_schemas
from nfl_scout.graph.state import ScoutState
from nfl_scout.tools.base import default_connection

# Tools per analyst. Each node gets only what its role needs: the API caps strict tools at 20 per
# request, and strict tools plus the structured-output schema must compile to a small enough
# grammar (all 20 tools + the section schema was rejected as too large).
EFFICIENCY = [
    "epa_per_play",
    "success_rate",
    "explosive_play_rate",
    "red_zone_td_rate",
    "third_down_conversion_rate",
    "sack_rate",
    "pressure_proxy_rate",
    "league_rank",
]
ROLE_TOOLS = {
    "offense_analyst": EFFICIENCY
    + [
        "pass_rate",
        "pass_rate_over_expected",
        "average_depth_of_target",
        "run_direction_rate",
        "player_share",
        "usage_leaders",
        "ftn_play_rate",
    ],
    "defense_analyst": EFFICIENCY + ["blitz_rate", "average_pass_rushers", "average_box_defenders"],
    "injury_analyst": ["injury_report", "snap_share", "player_share", "usage_leaders"],
    "matchup_analyst": EFFICIENCY
    + ["pass_rate", "pass_rate_over_expected", "blitz_rate", "average_box_defenders"],
}
# The single-agent baseline gets every tool the analysts have (exactly the 20-tool API limit).
SINGLE_TOOLS = sorted({name for names in ROLE_TOOLS.values() for name in names})
# Graph configurations compared in the eval.
CONFIGS = ("full", "no_checker", "single")
# Run in parallel after the planner; the matchup analyst reads all of their sections.
ANALYSTS = ("offense_analyst", "defense_analyst", "injury_analyst")


def season_teams(con: duckdb.DuckDBPyConnection, season: int) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT posteam FROM pbp WHERE season = ? AND posteam IS NOT NULL ORDER BY 1",
        [season],
    ).fetchall()
    return [t for (t,) in rows]


def build_graph(
    client: Any = None,
    con: duckdb.DuckDBPyConnection | None = None,
    teams: list[str] | None = None,
    settings: Settings | None = None,
    config: str = "full",
):
    """`config`: "full" (fact-checker + revise loop), "no_checker" (drafts go straight to the
    editor) or "single" (one agent with every tool, no checker)."""
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}, not {config!r}")
    settings = settings or load_settings()
    llm = LLM(client or default_client(), settings)
    teams = teams or season_teams(con or default_connection(), settings.data.eval_season)

    def analyst_node(role: str, **kwargs: Any):
        tools = tool_schemas(teams, ROLE_TOOLS[role])
        return partial(nodes.analyst, role=role, llm=llm, tools=tools, con=con, **kwargs)

    g = StateGraph(ScoutState)
    g.add_node("snapshot", nodes.snapshot)
    g.add_node("number_check", nodes.number_check)
    g.add_edge("number_check", END)

    if config == "single":
        tools = tool_schemas(teams, SINGLE_TOOLS)
        g.add_node("single_agent", partial(nodes.single_agent, llm=llm, tools=tools, con=con))
        g.add_edge(START, "single_agent")
        g.add_edge("single_agent", "snapshot")
        g.add_edge("snapshot", "number_check")
        return g.compile()

    g.add_node("planner", partial(nodes.planner, llm=llm))
    for role in ANALYSTS:
        g.add_node(role, analyst_node(role))
        g.add_edge("planner", role)
    g.add_node("matchup_analyst", analyst_node("matchup_analyst", reads_sections=True))
    g.add_node("editor", partial(nodes.editor, llm=llm))
    g.add_edge(START, "planner")
    g.add_edge(list(ANALYSTS), "matchup_analyst")  # waits for all three
    g.add_edge("matchup_analyst", "snapshot")
    g.add_edge("editor", "number_check")

    if config == "no_checker":
        g.add_edge("snapshot", "editor")
        return g.compile()

    g.add_node(
        "fact_checker",
        partial(nodes.fact_checker, max_loops=settings.models.max_revision_loops, con=con),
    )
    all_tools = {role: tool_schemas(teams, names) for role, names in ROLE_TOOLS.items()}
    g.add_node("reviser", partial(nodes.reviser, llm=llm, tools=all_tools, con=con))
    g.add_edge("snapshot", "fact_checker")
    g.add_conditional_edges("fact_checker", nodes.route_after_check, ["reviser", "editor"])
    g.add_edge("reviser", "fact_checker")
    return g.compile()
