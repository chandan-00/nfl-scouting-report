"""Assemble the LangGraph: planner -> (offense analyst || defense analyst).

Day 4 adds the injury/roster agent, matchup analyst and editor; day 5 the fact-checker.
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
}
ANALYSTS = tuple(ROLE_TOOLS)


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
):
    settings = settings or load_settings()
    llm = LLM(client or default_client(), settings)
    teams = teams or season_teams(con or default_connection(), settings.data.eval_season)

    g = StateGraph(ScoutState)
    g.add_node("planner", partial(nodes.planner, llm=llm))
    for role in ANALYSTS:
        g.add_node(
            role,
            partial(
                nodes.analyst,
                role=role,
                llm=llm,
                tools=tool_schemas(teams, ROLE_TOOLS[role]),
                con=con,
            ),
        )
        g.add_edge("planner", role)
        g.add_edge(role, END)
    g.add_edge(START, "planner")
    return g.compile()
