"""Blind hand-grading of report usefulness (spec: 10 matchups x 3 configs, 4 criteria 1-5,
plus a ranking within each matchup).

    uv run python -m nfl_scout.evaluation.grading build   # writes runs/grading/
    uv run python -m nfl_scout.evaluation.grading score   # after filling grades.csv

The packet shows each matchup's reports as A/B/C in a seeded random order, with claim markers
stripped. Don't open runs/grading/key.json until grades.csv is filled in.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from statistics import mean
from typing import Any

from nfl_scout.config import load_settings
from nfl_scout.evaluation.matchups import load_matchups

CRITERIA = ("relevance", "reasoning", "actionability", "coverage")
CRITERIA_HELP = {
    "relevance": "insights matter for this game, not generic stat recitation",
    "reasoning": "football logic is sound; numbers are interpreted correctly",
    "actionability": "a coach could turn it into game-plan decisions",
    "coverage": "the key matchups of this game are identified",
}
MARKERS = re.compile(r"\s*\[[a-z]{1,3}\d+(?:\s*,\s*[a-z]{1,3}\d+)*\]", re.I)


def strip_markers(text: str) -> str:
    return MARKERS.sub("", text)


def _name(m: dict) -> str:
    return f"2026_w{m['week']:02d}_{m['team']}_vs_{m['opponent']}"


def _report_md(report: dict[str, Any]) -> str:
    parts = [f"### {strip_markers(report['title'])}", "", strip_markers(report["summary"])]
    for s in report["sections"]:
        parts += ["", f"**{s['heading']}**", "", strip_markers(s["body"])]
    return "\n".join(parts)


def build_packet(
    matchups: list[dict],
    configs: list[str],
    runs_dir: Path,
    out_dir: Path,
    repeat: int = 1,
    seed: int = 2026,
) -> Path:
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; grading packets are built once")
    graded = [m for m in matchups if m.get("hand_graded")]
    paths = {
        (_name(m), c): runs_dir / c / f"r{repeat}" / f"{_name(m)}.json"
        for m in graded
        for c in configs
    }
    missing = [f"{c}/{n}" for (n, c), p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing runs: {', '.join(missing)}")

    rng = random.Random(seed)
    key: dict[str, dict[str, str]] = {}
    sheet: list[dict[str, str]] = []
    doc = [
        "# Blind grading packet",
        "",
        "Score each report 1-5 on each criterion, then rank the reports within each matchup "
        "(1 = most useful). Fill in grades.csv. Don't open key.json until you're done.",
        "",
        *(f"- **{c}**: {h}" for c, h in CRITERIA_HELP.items()),
        "",
    ]
    for m in graded:
        name = _name(m)
        order = configs[:]
        rng.shuffle(order)
        key[name] = dict(zip("ABC", order, strict=False))
        doc += [f"## {m['team']} vs {m['opponent']}, week {m['week']} ({name})", ""]
        for label, config in key[name].items():
            report = json.loads(paths[(name, config)].read_text())["report"]
            doc += [f"## Report {label}", "", _report_md(report), "", "---", ""]
            sheet.append({"matchup": name, "label": label, **dict.fromkeys(CRITERIA, ""),
                          "rank": "", "notes": ""})  # fmt: skip

    out_dir.mkdir(parents=True)
    (out_dir / "packet.md").write_text("\n".join(doc))
    (out_dir / "key.json").write_text(json.dumps(key, indent=1))
    with (out_dir / "grades.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    return out_dir


def score_grades(out_dir: Path) -> dict[str, Any]:
    key = json.loads((out_dir / "key.json").read_text())
    with (out_dir / "grades.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["rank"].strip()]
    by_config: dict[str, list[dict]] = {}
    for r in rows:
        by_config.setdefault(key[r["matchup"]][r["label"]], []).append(r)
    return {
        config: {
            "n": len(rs),
            "mean": {c: mean(float(r[c]) for r in rs) for c in CRITERIA},
            "mean_rank": mean(float(r["rank"]) for r in rs),
            "wins": sum(r["rank"].strip() == "1" for r in rs),
        }
        for config, rs in sorted(by_config.items())
    }


def _by_matchup(out_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """{matchup: {config: {"score": mean of the criteria, "rank": rank}}}."""
    key = json.loads((out_dir / "key.json").read_text())
    with (out_dir / "grades.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["rank"].strip()]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for r in rows:
        config = key[r["matchup"]][r["label"]]
        score = mean(float(r[c]) for c in CRITERIA)
        out.setdefault(r["matchup"], {})[config] = {"score": score, "rank": int(r["rank"])}
    return out


def paired(
    by_matchup: dict[str, dict[str, dict[str, float]]],
    pairs: list[tuple[str, str]],
    n_boot: int = 5000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per pair of configs, over matchups graded for both: the mean difference in average grade
    (a - b) with a 95% bootstrap interval over matchups, and how often a was ranked above b."""
    out = []
    for a, b in pairs:
        both = [m for m in by_matchup.values() if a in m and b in m]
        diffs = [m[a]["score"] - m[b]["score"] for m in both]
        rng = random.Random(seed)
        boots = sorted(mean(rng.choice(diffs) for _ in diffs) for _ in range(n_boot))
        out.append(
            {
                "a": a,
                "b": b,
                "n": len(both),
                "mean_diff": mean(diffs),
                "ci": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]],
                "a_ranked_higher": sum(m[a]["rank"] < m[b]["rank"] for m in both),
            }
        )
    return out


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "score"])
    ap.add_argument("--configs", nargs="+", default=["full", "no_checker", "single"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026, help="label shuffle seed")
    a = ap.parse_args()
    runs_dir = settings.resolve(settings.paths.runs_dir)
    out_dir = runs_dir / "grading"
    if a.action == "build":
        build_packet(load_matchups(), a.configs, runs_dir, out_dir, a.repeat, a.seed)
        print(f"wrote {out_dir}/packet.md and grades.csv (don't open key.json)")
    else:
        configs = a.configs
        pairs = [(x, y) for i, x in enumerate(configs) for y in configs[i + 1 :]]
        result = {
            "per_config": score_grades(out_dir),
            "paired": paired(_by_matchup(out_dir), pairs),
        }
        scores = runs_dir / "scores" / "grades.json"
        scores.write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        print(f"wrote {scores}")


if __name__ == "__main__":
    main()
