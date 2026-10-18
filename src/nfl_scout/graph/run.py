"""Run the graph for one matchup and print the sections, claims and usage.

    uv run python -m nfl_scout.graph.run --team DEN --opponent KC --week 6

Debug helper only; saving runs to runs/<config>/<matchup>.json comes on day 5.
"""

from __future__ import annotations

import argparse

from nfl_scout.config import load_settings
from nfl_scout.graph.build import build_graph
from nfl_scout.graph.state import Request


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="the team the report is for")
    ap.add_argument("--opponent", required=True, help="the team being scouted")
    ap.add_argument("--week", type=int, required=True, help="game week (data through week - 1)")
    ap.add_argument("--season", type=int, default=settings.data.eval_season)
    a = ap.parse_args()

    request = Request(team=a.team, opponent=a.opponent, season=a.season, week=a.week)
    out = build_graph(settings=settings).invoke({"request": request})

    print("PLAN", out["plan"], sep="\n", end="\n\n")
    calls = {c.call_id: c for c in out["tool_calls"]}
    for role, section in out["sections"].items():
        print(f"== {role} ==\n{section.prose}\n")
        for c in section.claims:
            call = calls.get(c.tool_call_id or "")
            actual = call.result["value"] if call and call.result else None
            source = c.tool_call_id or f"{c.op}({', '.join(c.derived_from or [])})"
            values = f"stated={c.stated_value!s:8} tool={actual!s:.8}"
            print(f"  {c.claim_id:5} {values} [{source}] {c.text}")
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


if __name__ == "__main__":
    main()
