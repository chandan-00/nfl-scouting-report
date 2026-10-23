"""Graph nodes: planner, analysts (tool loop + LLM-written claims), editor and number check."""

from __future__ import annotations

from typing import Any

import duckdb

from nfl_scout.graph import prompts
from nfl_scout.graph.llm import LLM, final_json
from nfl_scout.graph.llm_tools import ToolExecutor
from nfl_scout.graph.state import (
    PLAN_SCHEMA,
    REPORT_SCHEMA,
    ROLE_PREFIX,
    SECTION_SCHEMA,
    Report,
    ScoutState,
    parse_section,
)
from nfl_scout.numbers import check_numbers

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
    reads_sections: bool = False,
) -> dict[str, Any]:
    """One analyst's tool loop. With `reads_sections` (the matchup analyst) the first message
    holds the other analysts' sections and claims instead of planner questions."""
    r = state["request"]
    focus = state.get("plan", {}).get(prompts.FOCUS_KEY.get(role, ""), [])
    sections = state.get("sections", {}) if reads_sections else None
    executor = ToolExecutor(r.season, r.through_week, ROLE_PREFIX[role], node=role, con=con)
    messages: list[dict] = [
        {
            "role": "user",
            "content": prompts.analyst_message(
                role, r.team, r.opponent, r.season, r.week, r.through_week, focus, sections
            ),
        }
    ]
    usages = []
    for turn in range(MAX_TURNS):
        # The last turn keeps the tools (cached prefix) but forbids calling them.
        response, usage = llm.call(
            role,
            system=prompts.SYSTEM,
            messages=messages,
            output_schema=SECTION_SCHEMA,
            tools=tools,
            tool_choice="none" if turn == MAX_TURNS - 1 else "auto",
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
        if turn == MAX_TURNS - 2:
            results.append({"type": "text", "text": prompts.LAST_TURN})
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"{role}: no final answer after {MAX_TURNS} turns")


def editor(state: ScoutState, llm: LLM) -> dict[str, Any]:
    r = state["request"]
    msg = prompts.editor_message(r.team, r.opponent, r.season, r.week, state["sections"])
    response, usage = llm.call(
        "editor",
        system=prompts.EDITOR_SYSTEM,
        messages=[{"role": "user", "content": msg}],
        output_schema=REPORT_SCHEMA,
    )
    return {"report": Report(**final_json(response)), "usage": [usage]}


def number_check(state: ScoutState) -> dict[str, Any]:
    """Link every number in the final report to a claim (no LLM)."""
    claims = [c for s in state["sections"].values() for c in s.claims]
    return {"number_checks": check_numbers(state["report"].text(), claims)}
