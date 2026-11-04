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
    ("Draft claims", "draft_claims", "int"),
    ("Draft claim accuracy", "draft_accuracy", "pct_ci"),
    ("Final claim accuracy", "final_accuracy", "pct_ci"),
    ("Draft wording correct", "draft_wording_ok", "pct"),
    ("Final wording correct", "final_wording_ok", "pct"),
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


def render(scores: dict[str, dict[str, Any]], context: dict[str, str]) -> str:
    configs = list(scores)
    names = [CONFIG_NAMES.get(c, c) for c in configs]
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
    return "\n".join(lines)


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
    out = PROJECT_ROOT / "docs" / "results.md"
    out.write_text(render(scores, context) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
