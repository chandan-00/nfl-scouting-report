"""Render docs/results.md from the score files (runs/scores/<config>.json).

uv run python -m nfl_scout.evaluation.results
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from nfl_scout.config import PROJECT_ROOT, load_settings
from nfl_scout.evaluation.matchups import EVAL_DIR

CONFIG_NAMES = {
    "full": "Full graph",
    "no_checker": "No fact-checker",
    "single": "Single agent",
}

# (label, aggregate key, format); "pct_ci" also reads <key>_ci.
ROWS = [
    ("Reports", "runs", "int"),
    ("Reports with no tool calls", "zero_tool_reports", "int"),
    ("Draft claims", "draft_claims", "int"),
    ("Draft claim accuracy", "draft_accuracy", "pct_ci"),
    ("Final claim accuracy", "final_accuracy", "pct_ci"),
    ("Draft wording correct", "draft_wording_ok", "pct"),
    ("Final wording correct", "final_wording_ok", "pct"),
    ("Claims written with no source", "unsourced_claims", "int"),
    ("Unsupported numbers per draft", "draft_unsupported_per_report", "num"),
    ("Report numbers per report", "report_numbers_per_report", "num"),
    ("Report numbers unsupported", "report_unsupported_rate", "pct"),
    ("Report numbers mismatched", "report_mismatch_rate", "pct"),
    ("Claims corrected by revision", "corrected", "int"),
    ("Claims dropped", "dropped", "int"),
    ("Independent sample: tool agrees", "independent_tool_agreement", "pct"),
    ("Independent sample: stated value agrees", "independent_stated_agreement", "pct"),
    ("Seeded errors caught", "seeded_catch_rate", "pct"),
    ("Checker false positives (seeded)", "seeded_false_positives", "int"),
    ("Tool error rate", "tool_error_rate", "pct"),
    ("Output tokens per report", "output_tokens_per_report", "int"),
    ("Cost per report", "cost_per_report", "usd"),
]


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _cell(agg: dict[str, Any], key: str, fmt: str) -> str:
    v = agg.get(key)
    if v is None:
        return "—"
    if fmt == "int":
        return f"{round(v):,}"
    if fmt == "num":
        return f"{v:.1f}"
    if fmt == "usd":
        return f"${v:.3f}"
    if fmt == "pct":
        return _pct(v)
    lo, hi = agg.get(f"{key}_ci", [None, None])
    ci = f" ({100 * lo:.1f}–{100 * hi:.1f})" if lo is not None and hi is not None else ""
    return _pct(v) + ci


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def render(
    scores: dict[str, dict[str, Any]],
    context: dict[str, str],
    forward: dict[str, dict[str, Any]] | None = None,
    grades: dict[str, Any] | None = None,
) -> str:
    configs = list(scores)
    names = [CONFIG_NAMES.get(c, c) for c in configs]
    for c in configs:  # an integrity check: a report with no tool calls has no data behind it
        zero = sum(r.get("tool_calls", 1) == 0 for r in scores[c]["runs"])
        scores[c]["aggregate"]["zero_tool_reports"] = zero
    lines = ["# Evaluation results", ""]
    if context:
        lines += [f"- **{k}:** {v}" for k, v in context.items()] + [""]
    lines += [
        "## Main comparison",
        "",
        "Percentages show a 95% bootstrap interval over reports.",
        "",
    ]
    rows = [[label] + [_cell(scores[c]["aggregate"], key, fmt) for c in configs]
            for label, key, fmt in ROWS]  # fmt: skip
    lines += _table(["Metric", *names], rows) + [""]

    kinds = sorted({k for c in configs for k in scores[c]["aggregate"].get("seeded_by_kind", {})})
    if kinds:
        lines += ["## Seeded errors caught, by kind", ""]
        rows = [[k] + [_pct(scores[c]["aggregate"]["seeded_by_kind"].get(k)) for c in configs]
                for k in kinds]  # fmt: skip
        lines += _table(["Kind", *names], rows) + [""]

    repeats = sorted({r["repeat"] for c in configs for r in scores[c]["runs"]} - {None})
    if repeats:
        lines += ["## Draft claim accuracy by repeat", ""]
        rows = []
        for c in configs:
            cells = []
            for k in repeats:
                runs = [r for r in scores[c]["runs"] if r["repeat"] == k]
                claims = sum(r["draft"]["claims"] for r in runs)
                ok = sum(r["draft"]["value_ok"] for r in runs)
                cells.append(_pct(ok / claims if claims else None))
            rows.append([c, *cells])
        lines += _table(["Config", *(f"r{k}" for k in repeats)], rows) + [""]
    if forward:
        lines += _forward_section(forward)
    if grades:
        lines += _grades_section(grades)
    return "\n".join(lines)


def _grades_section(grades: dict[str, Any]) -> list[str]:
    per = grades["per_config"]
    configs = list(per)
    n = max(g["n"] for g in per.values())
    lines = [
        "",
        "## Hand grading",
        "",
        f"Blind grades (1-5) of {n} matchups per config, one grader (the project owner). Each "
        "matchup's three reports were shown as A/B/C in a shuffled order, and ranked 1-3.",
        "",
        "| Criterion | " + " | ".join(CONFIG_NAMES.get(c, c) for c in configs) + " |",
        "|---|" + "---|" * len(configs),
    ]
    for crit in ("relevance", "reasoning", "actionability", "coverage"):
        row = [f"{per[c]['mean'][crit]:.1f}" for c in configs]
        lines.append(f"| {crit.capitalize()} | " + " | ".join(row) + " |")
    lines.append(
        "| Mean rank | " + " | ".join(f"{per[c]['mean_rank']:.1f}" for c in configs) + " |"
    )
    lines.append(
        "| Wins (ranked first) | " + " | ".join(str(per[c]["wins"]) for c in configs) + " |"
    )
    lines += [
        "",
        "Paired by matchup: difference in the average of the four grades, with a 95% bootstrap "
        "interval over matchups.",
        "",
        "| Comparison | Grade difference | Ranked higher |",
        "|---|---|---|",
    ]
    for p in grades.get("paired", []):
        lo, hi = p["ci"]
        name = f"{CONFIG_NAMES.get(p['a'], p['a'])} vs {CONFIG_NAMES.get(p['b'], p['b'])}"
        lines.append(
            f"| {name} | {p['mean_diff']:+.2f} ({lo:+.2f} to {hi:+.2f}) | "
            f"{p['a_ranked_higher']}/{p['n']} |"
        )
    return lines


def _forward_section(forward: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## Forward check",
        "",
        "For each team-level number a final report cites: is the season-to-date value closer to "
        "the same metric in the game itself than the league average through the same week? Each "
        "(game, metric, split) counts once.",
        "",
    ]
    rows = [
        [CONFIG_NAMES.get(c, c), f"{f['aggregate']['claims']:,}",
         _pct(f["aggregate"].get("report_closer"))]
        for c, f in forward.items()
        if f["aggregate"].get("claims")
    ]  # fmt: skip
    lines += _table(["Config", "Claims", "Report closer than league average"], rows) + [""]
    first = next((c for c, f in forward.items() if f["aggregate"].get("claims")), None)
    if first:
        lines += [f"By metric ({CONFIG_NAMES.get(first, first)}):", ""]
        rows = [
            [tool, str(t["claims"]), _pct(t["report_closer"]), f"{t['report_mae']:.3f}",
             f"{t['naive_mae']:.3f}"]
            for tool, t in forward[first]["aggregate"]["by_tool"].items()
        ]  # fmt: skip
        header = ["Metric", "Claims", "Report closer", "MAE report", "MAE league"]
        lines += _table(header, rows) + [""]
    return lines


def main() -> None:
    settings = load_settings()
    scores_dir = settings.resolve(settings.paths.runs_dir) / "scores"
    scores = {
        c: json.loads((scores_dir / f"{c}.json").read_text())
        for c in CONFIG_NAMES
        if (scores_dir / f"{c}.json").exists()
    }
    snapshot = json.loads((EVAL_DIR / "snapshot.json").read_text())
    context = {
        "Generated": datetime.now(UTC).strftime("%Y-%m-%d"),
        "Model": settings.node_model("offense_analyst").model + " on every node",
        "Matchups": "eval/matchups_v1.json (40 games, 2026 weeks 4–7, after every model's "
        "training cutoff)",
        "Data snapshot": f"{snapshot['path']} (sha256 {snapshot['sha256'][:12]})",
    }
    forward_path = scores_dir / "forward.json"
    forward = json.loads(forward_path.read_text()) if forward_path.exists() else None
    forward = {c: f for c, f in (forward or {}).items() if c in scores} or None
    grades_path = scores_dir / "grades.json"
    grades = json.loads(grades_path.read_text()) if grades_path.exists() else None
    notes_path = EVAL_DIR / "results_notes.md"  # hand-written context, kept across regenerations
    notes = "\n" + notes_path.read_text() if notes_path.exists() else ""
    out = PROJECT_ROOT / "docs" / "results.md"
    out.write_text(render(scores, context, forward, grades) + "\n" + notes)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
