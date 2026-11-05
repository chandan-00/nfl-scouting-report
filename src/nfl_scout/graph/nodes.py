"""Graph nodes: planner, analysts (tool loop + LLM-written claims), editor and number check."""

from __future__ import annotations

import copy
from typing import Any

import duckdb
from langgraph.types import Send

from nfl_scout.checker import check_sections
from nfl_scout.graph import prompts
from nfl_scout.graph.llm import LLM, final_json
from nfl_scout.graph.llm_tools import ToolExecutor
from nfl_scout.graph.state import (
    PLAN_SCHEMA,
    REPORT_SCHEMA,
    ROLE_PREFIX,
    SECTION_SCHEMA,
    SINGLE_SCHEMA,
    Report,
    ScoutState,
    parse_section,
    parse_single,
)
from nfl_scout.numbers import check_numbers

# Model turns per analyst before giving up (each turn may hold several parallel tool calls).
MAX_TURNS = 12
# The single agent does every analyst's work, so it gets more tool turns, and its one final
# answer (the whole report and every claim) gets a larger, streamed output budget.
SINGLE_MAX_TURNS = 24
SINGLE_FINAL_MAX_TOKENS = 32000
SINGLE_MAX_NUDGES = 2


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
    revision: tuple[Any, list] | None = None,
) -> dict[str, Any]:
    """One analyst's tool loop. With `reads_sections` (the matchup analyst) the first message
    holds the other analysts' sections and claims instead of planner questions. With `revision`
    (previous section, failed checks) the analyst rewrites its section."""
    r = state["request"]
    focus = state.get("plan", {}).get(prompts.FOCUS_KEY.get(role, ""), [])
    sections = (
        {k: v for k, v in state.get("sections", {}).items() if k != role}
        if reads_sections
        else None
    )
    prefix = ROLE_PREFIX[role]
    done = sum(1 for c in state.get("tool_calls", []) if c.node == role)
    executor = ToolExecutor(r.season, r.through_week, prefix, node=role, con=con, start=done)
    content = prompts.analyst_message(
        role, r.team, r.opponent, r.season, r.week, r.through_week, focus, sections
    )
    if revision:
        content += prompts.revision_block(*revision)
    messages: list[dict] = [{"role": "user", "content": content}]
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
        results = _run_tools(response, executor)
        if turn == MAX_TURNS - 2:
            results.append({"type": "text", "text": prompts.LAST_TURN})
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"{role}: no final answer after {MAX_TURNS} turns")


def _run_tools(response: Any, executor: ToolExecutor) -> list[dict]:
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
    return results


def single_agent(
    state: ScoutState,
    llm: LLM,
    tools: list[dict],
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Baseline: one agent with every tool writes the whole report in the same claim format.

    Tool turns carry no output schema: all 20 strict tools plus a schema compile to a grammar
    the API rejects. When the agent stops calling tools (or runs out of turns), one more call
    with tool_choice "none" asks for the structured report; the tools stay in the request so
    the cached prefix holds.
    """
    r = state["request"]
    node = "single_agent"
    executor = ToolExecutor(r.season, r.through_week, ROLE_PREFIX[node], node=node, con=con)
    msg = prompts.single_message(r.team, r.opponent, r.season, r.week, r.through_week)
    messages: list[dict] = [{"role": "user", "content": msg}]
    usages = []
    nudges = 0
    for _ in range(SINGLE_MAX_TURNS):
        response, usage = llm.call(
            node, system=prompts.SINGLE_SYSTEM, messages=messages, output_schema=None, tools=tools
        )
        usages.append(usage)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            if not executor.calls:
                # About 10% of runs said "ready" on turn 1, with nothing to write a report from.
                nudges += 1
                if nudges > SINGLE_MAX_NUDGES:
                    raise RuntimeError("single_agent: no tool calls; nothing to report on")
                messages.append({"role": "user", "content": prompts.SINGLE_NO_TOOLS})
                continue
            messages.append({"role": "user", "content": prompts.SINGLE_FINAL})
            break
        results = _run_tools(response, executor)
        messages.append({"role": "user", "content": results})
    else:
        messages[-1]["content"].append({"type": "text", "text": prompts.SINGLE_FINAL})
    response, usage = llm.call(
        node,
        system=prompts.SINGLE_SYSTEM,
        messages=messages,
        output_schema=SINGLE_SCHEMA,
        tools=tools,
        tool_choice="none",
        max_tokens=SINGLE_FINAL_MAX_TOKENS,
    )
    usages.append(usage)
    section, report = parse_single(final_json(response))
    return {
        "sections": {node: section},
        "report": report,
        "tool_calls": executor.calls,
        "usage": usages,
    }


def snapshot(state: ScoutState) -> dict[str, Any]:
    """Freeze the analysts' sections before any fact-check, so drafts can be scored."""
    return {"draft": copy.deepcopy(state["sections"]), "checks_run": 0}


def fact_checker(
    state: ScoutState, max_loops: int, con: duckdb.DuckDBPyConnection | None = None
) -> dict[str, Any]:
    """Check every section. Roles with failures are sent back for revision until `max_loops`
    revisions have run; after that, failing claims are dropped (and logged)."""
    results = check_sections(state["sections"], state.get("tool_calls", []), con)
    failed = [f for f in results if not f.ok]
    n = state.get("checks_run", 0) + 1
    claims = {c.claim_id: c for s in state["sections"].values() for c in s.claims}
    log = [
        {
            "round": n,
            "role": f.role,
            "claim_id": f.claim_id,
            "check": f.check,
            "detail": f.detail,
            "text": claims[f.claim_id].text if f.claim_id in claims else None,
            "stated_value": claims[f.claim_id].stated_value if f.claim_id in claims else None,
        }
        for f in failed
    ]
    update: dict[str, Any] = {"checks": results, "check_log": log, "checks_run": n}
    if failed and n <= max_loops:
        update["revise_roles"] = sorted({f.role for f in failed})
        return update

    update["revise_roles"] = []
    bad = {f.claim_id for f in failed if f.claim_id in claims}
    update["dropped"] = [e for e in log if e["claim_id"] in bad or e["claim_id"] is None]
    if bad:
        update["sections"] = {
            role: s.model_copy(update={"claims": [c for c in s.claims if c.claim_id not in bad]})
            for role, s in state["sections"].items()
            if any(c.claim_id in bad for c in s.claims)
        }
    return update


def route_after_check(state: ScoutState) -> list[Send] | str:
    roles = state.get("revise_roles", [])
    if not roles:
        return "editor"
    return [Send("reviser", {**state, "revise_role": role}) for role in roles]


def reviser(
    state: ScoutState,
    llm: LLM,
    tools: dict[str, list[dict]],
    con: duckdb.DuckDBPyConnection | None = None,
) -> dict[str, Any]:
    """Re-run one analyst with its previous section and the checks it failed."""
    role = state["revise_role"]
    failures = [f for f in state["checks"] if f.role == role and not f.ok]
    return analyst(
        state,
        role,
        llm,
        tools[role],
        con,
        reads_sections=role == "matchup_analyst",
        revision=(state["sections"][role], failures),
    )


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
