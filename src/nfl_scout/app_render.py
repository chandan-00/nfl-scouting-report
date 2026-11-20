"""Render a saved run's report as HTML for the demo: each number becomes a highlighted span whose
tooltip shows its claim, the tool call it cites, the tool's value and the number check's status.
Reads saved run dicts only (no API calls)."""

from __future__ import annotations

import html
import re
from typing import Any

MARKER = re.compile(r"\s?\[[^\]]+\]")
INJECTED = ("season", "through_week")


def _tooltip(claim_id: str | None, status: str, claims: dict, calls: dict) -> str:
    if claim_id is None or claim_id not in claims:
        return f"No claim behind this number ({status})."
    claim = claims[claim_id]
    lines = [f"{claim_id}: {claim['text']}", f"stated: {claim['stated_value']}"]
    call = calls.get(claim.get("tool_call_id") or "")
    if call:
        args = ", ".join(f"{k}={v}" for k, v in call["args"].items() if k not in INJECTED)
        lines.append(f"tool: {call['tool']}({args})")
        result = call.get("result") or {}
        field = "n_plays" if claim.get("measure") == "n_plays" else "value"
        value = result.get(field)
        if not isinstance(value, list):
            lines.append(f"{field} = {value}")
    elif claim.get("derived_from"):
        lines.append(f"derived: {claim.get('op')} of {', '.join(claim['derived_from'])}")
    lines.append(f"check: {status}")
    return "\n".join(lines)


def _render(part: str, base: int, checks: list[dict], claims: dict, calls: dict) -> str:
    """HTML for one part of the report text, which starts at offset `base` in the full text."""
    out, pos = [], 0
    for c in checks:
        m = c["mention"]
        start = m["start"] - base
        if start < pos or start >= len(part):
            continue
        end = start + len(m["text"])
        tip = _tooltip(m.get("claim_id"), c["status"], claims, calls)
        out.append(html.escape(MARKER.sub("", part[pos:start])))
        out.append(
            f'<span class="num {html.escape(c["status"])}" title="{html.escape(tip)}">'
            f"{html.escape(m['text'])}</span>"
        )
        pos = end
    out.append(html.escape(MARKER.sub("", part[pos:])))
    return "".join(out)


def annotate_report(run: dict[str, Any]) -> list[tuple[str, str]]:
    """(heading, html) for the summary and each section. Number-check offsets index into the
    report text: the summary and section bodies joined by blank lines."""
    report = run["report"]
    claims = {c["claim_id"]: c for s in run["sections"].values() for c in s["claims"]}
    calls = {c["call_id"]: c for c in run["tool_calls"]}
    checks = sorted(run.get("number_checks", []), key=lambda c: c["mention"]["start"])
    parts = [("Summary", report["summary"])]
    parts += [(s["heading"], s["body"]) for s in report["sections"]]
    out, base = [], 0
    for heading, text in parts:
        out.append((heading, _render(text, base, checks, claims, calls)))
        base += len(text) + 2  # the "\n\n" between parts
    return out


def cost_rows(cost: dict[str, Any]) -> list[dict[str, Any]]:
    """Cost by node as table rows. Runs saved before 2026-11-20 stored a {node: cost} dict."""
    by_node = cost.get("by_node", [])
    if isinstance(by_node, dict):
        return [{"node": n, "cost_usd": c} for n, c in by_node.items()]
    return list(by_node)
