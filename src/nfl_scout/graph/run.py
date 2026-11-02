"""Run the graph for one matchup, save it to runs/<config>/<matchup>.json and print the
sections, fact-check log, final report, number checks, usage and cost.

    uv run python -m nfl_scout.graph.run --team DEN --opponent KC --week 6
"""

from __future__ import annotations

import argparse
from collections import Counter

from nfl_scout.config import load_settings
from nfl_scout.costs import cost_by_node
from nfl_scout.graph.build import CONFIGS, build_graph
from nfl_scout.graph.state import Request
from nfl_scout.runs import save_run


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="the team the report is for")
    ap.add_argument("--opponent", required=True, help="the team being scouted")
    ap.add_argument("--week", type=int, required=True, help="game week (data through week - 1)")
    ap.add_argument("--season", type=int, default=settings.data.eval_season)
    ap.add_argument("--graph", choices=CONFIGS, default="full", help="graph configuration")
    ap.add_argument("--config", help="run name under runs/ (default: the graph name)")
    ap.add_argument("--no-save", action="store_true", help="don't write the run file")
    a = ap.parse_args()

    request = Request(team=a.team, opponent=a.opponent, season=a.season, week=a.week)
    out = build_graph(settings=settings, config=a.graph).invoke({"request": request})

    print("PLAN", out.get("plan"), sep="\n", end="\n\n")
    calls = {c.call_id: c for c in out["tool_calls"]}
    for role, section in out["sections"].items():
        print(f"== {role} ==\n{section.prose}\n")
        for c in section.claims:
            call = calls.get(c.tool_call_id or "")
            field = "n_plays" if c.measure == "n_plays" else "value"
            actual = call.result[field] if call and call.result else None
            source = c.tool_call_id or f"{c.op}({', '.join(c.derived_from or [])})"
            values = f"stated={c.stated_value!s:8} tool={actual!s:.8}"
            print(f"  {c.claim_id:5} {values} [{source}] {c.text}")
        print()
    print(f"fact-check rounds: {out.get('checks_run', 0)}")
    for e in out.get("check_log", []):
        print(f"  r{e['round']} {e['claim_id'] or '-':5} {e['check']:8} {e['detail']}")
    print(f"dropped: {[d['claim_id'] for d in out.get('dropped', [])]}\n")
    print("== REPORT ==", out["report"].markdown(), sep="\n", end="\n\n")
    checks = out["number_checks"]
    print("number checks:", dict(Counter(c.status for c in checks)))
    for c in checks:
        if c.status != "ok":
            print(f"  {c.status:15} {c.mention.text!r} -> {c.mention.claim_id}")
    print()
    errors = [c for c in out["tool_calls"] if c.error]
    print(f"tool calls: {len(calls)} ({len(errors)} errors)")
    for e in errors:
        print(f"  {e.call_id} {e.tool} {e.args}: {e.error}")
    tot = {k: 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")}
    for u in out["usage"]:
        print(
            f"  {u.node:16} in={u.input_tokens:6} out={u.output_tokens:5} "
            f"cache_read={u.cache_read_input_tokens:6} write={u.cache_creation_input_tokens:6} "
            f"{u.latency_s:.1f}s"
        )
        for k in tot:
            tot[k] += getattr(u, k)
    print("total", tot)
    costs = cost_by_node(out["usage"], settings)
    print(
        "cost USD", {k: round(v, 4) for k, v in costs.items()}, f"total={sum(costs.values()):.4f}"
    )
    if not a.no_save:
        print("saved", save_run(out, a.config or a.graph, settings))


if __name__ == "__main__":
    main()
