"""Graph nodes: planner and the offense/defense analysts (tool loop + LLM-written claims)."""

from __future__ import annotations

from typing import Any

import duckdb

from nfl_scout.graph import prompts
from nfl_scout.graph.llm import LLM, final_json
from nfl_scout.graph.llm_tools import ToolExecutor
from nfl_scout.graph.state import (
    PLAN_SCHEMA,
    ROLE_PREFIX,
    SECTION_SCHEMA,
    ScoutState,
    parse_section,
)

# Model turns per analyst before giving up (each turn may hold several parallel tool calls).
MAX_TURNS = 12


def planner(state: ScoutState, llm: LLM) -> dict[str, Any]:
    r = state["request"]
    msg = prompts.planner_message(r.team, r.opponent, r.season, r.week, r.through_week)
    response, usage = llm.call(
        "planner",
        system=prompts.PLANNER_SYSTEM,
        messages=[{"role": "user", "content": msg}],
        output_schema=PLAN_SCHEMA,
    )
    return {"plan": final_json(response), "usage": [usage]}


def analyst(
    state: ScoutState,
    role: str,
    llm: LLM,
    tools: list[dict],
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    r = state["request"]
    focus = state.get("plan", {}).get(prompts.FOCUS_KEY[role], [])
    executor = ToolExecutor(r.season, r.through_week, ROLE_PREFIX[role], node=role, con=con)
    messages: list[dict] = [
        {
            "role": "user",
            "content": prompts.analyst_message(
                role, r.team, r.opponent, r.season, r.week, r.through_week, focus
            ),
        }
    ]
    usages = []
    for _ in range(MAX_TURNS):
        response, usage = llm.call(
            role,
            system=prompts.SYSTEM,
            messages=messages,
            output_schema=SECTION_SCHEMA,
            tools=tools,
        )
        usages.append(usage)
        if response.stop_reason != "tool_use":
            section = parse_section(role, final_json(response))
            return {
                "sections": {role: section},
                "tool_calls": executor.calls,
                "usage": usages,
            }
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                content, is_error = executor.run(block.name, block.input)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                        "is_error": is_error,
                    }
                )
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"{role}: no final answer after {MAX_TURNS} turns")
