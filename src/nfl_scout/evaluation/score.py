"""Score saved runs. Reads only runs/<config>/r*/<matchup>.json; never calls the API.

Per run:
- draft: claims in the analysts' drafts (before any fact-check) and how many pass the value and
  wording checks; numbers in the draft prose with no claim (unsupported)
- final: claims the final report cites and how many pass; report numbers and how many are
  unsupported or don't match their claim
- corrected (draft failures fixed by revision) and dropped claims
- independent: a 20% sample of draft claims re-derived with code independent of the tools
- seeded: errors injected into the draft, caught per kind, and false positives
- cost, tokens, tool calls and tool errors

    uv run python -m nfl_scout.evaluation.score --config full no_checker single
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from nfl_scout.checker import Checker, _close, check_sections
from nfl_scout.evaluation.independent import independent_value, load_plays
from nfl_scout.evaluation.seeding import score_seeded, seed_errors
from nfl_scout.graph.state import Section, ToolCall
from nfl_scout.runs import load_run

SAMPLE_FRACTION = 0.2


def _rng(run: dict[str, Any], salt: str) -> random.Random:
    """Seeded per run, so scores are reproducible and independent across runs."""
    r = run["request"]
    key = f"{run['config']}|{run.get('repeat')}|{r['week']}|{r['team']}|{r['opponent']}|{salt}"
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def _claim_checks(results, claim_ids: set[str] | None = None) -> dict[str, int]:
    by_claim: dict[str, dict[str, bool]] = {}
    for r in results:
        if r.claim_id and r.check in ("value", "wording"):
            by_claim.setdefault(r.claim_id, {})[r.check] = r.ok
    if claim_ids is not None:
        by_claim = {k: v for k, v in by_claim.items() if k in claim_ids}
    return {
        "claims": len(by_claim),
        "value_ok": sum(v.get("value", False) for v in by_claim.values()),
        "wording_ok": sum(v.get("wording", True) for v in by_claim.values()),
    }


def _independent(
    draft: dict[str, Section],
    calls: dict[str, ToolCall],
    plays: pl.DataFrame,
    rng: random.Random,
) -> dict[str, int]:
    cited = [c for s in draft.values() for c in s.claims if c.tool_call_id in calls]
    cited.sort(key=lambda c: c.claim_id)
    n = min(len(cited), max(1, round(SAMPLE_FRACTION * len(cited)))) if cited else 0
    sample = rng.sample(cited, n)
    out = {"sampled": n, "covered": 0, "tool_agrees": 0, "stated_agrees": 0}
    for claim in sample:
        call = calls[claim.tool_call_id]
        if call.result is None:
            continue
        ours = independent_value(plays, call.tool, call.args)
        if ours is None:
            continue
        out["covered"] += 1
        value, n_plays = ours
        tool_value, tool_n = call.result["value"], call.result["n_plays"]
        same = (value is None and tool_value is None) or (
            value is not None and tool_value is not None and abs(value - tool_value) < 1e-6
        )
        out["tool_agrees"] += int(same and n_plays == tool_n)
        truth = n_plays if claim.measure == "n_plays" else value
        try:
            stated = float(claim.stated_value)
        except (TypeError, ValueError):
            continue
        out["stated_agrees"] += int(truth is not None and _close(stated, float(truth)))
    return out


def score_run(
    run: dict[str, Any],
    con: duckdb.DuckDBPyConnection,
    seeds: int = 3,
    plays: pl.DataFrame | None = None,
) -> dict[str, Any]:
    calls = [ToolCall(**c) for c in run["tool_calls"]]
    by_id = {c.call_id: c for c in calls}
    checker = Checker(calls, con)
    draft = {k: Section(**v) for k, v in run["draft"].items()}
    final = {k: Section(**v) for k, v in run["sections"].items()}

    draft_results = check_sections(draft, calls, con, checker)
    draft_scores = _claim_checks(draft_results)
    draft_scores["unsourced_claims"] = sum(len(sec.unsourced) for sec in draft.values())
    draft_scores["unsupported"] = sum(
        r.check == "prose" and r.claim_id is None for r in draft_results
    )
    failing = {r.claim_id for r in draft_results if r.claim_id and r.check != "prose" and not r.ok}

    statuses = Counter(n["status"] for n in run.get("number_checks", []))
    cited_in_report = {
        n["mention"]["claim_id"] for n in run.get("number_checks", []) if n["mention"]["claim_id"]
    }
    final_scores = _claim_checks(check_sections(final, calls, con, checker), cited_in_report)
    final_scores.update(
        report_numbers=sum(statuses.values()),
        unsupported=statuses["unlinked"] + statuses["unknown_claim"],
        mismatched=statuses["value_mismatch"],
    )

    dropped = sum(1 for d in run.get("dropped", []) if d.get("claim_id"))
    plays = plays if plays is not None else load_plays(con, run["request"]["season"])
    seeded: dict[str, Any] = {"by_kind": {}, "false_positives": 0}
    for i in range(seeds):
        sections, manifest = seed_errors(draft, calls, _rng(run, f"seed{i}"), con=con)
        sc = score_seeded(draft, sections, manifest, calls, con, checker)
        for kind, v in sc["by_kind"].items():
            agg = seeded["by_kind"].setdefault(kind, {"seeded": 0, "caught": 0})
            agg["seeded"] += v["seeded"]
            agg["caught"] += v["caught"]
        seeded["false_positives"] += len(sc["false_positives"])

    usage = run.get("usage", [])
    r = run["request"]
    return {
        "config": run["config"],
        "repeat": run.get("repeat"),
        "matchup": f"{r['season']}_w{r['week']:02d}_{r['team']}_vs_{r['opponent']}",
        "draft": draft_scores,
        "final": final_scores,
        # Draft failures the run fixed: every failing draft claim not dropped. A no-checker run
        # can't correct anything, so its count is 0.
        "corrected": max(len(failing) - dropped, 0) if run.get("checks_run", 0) else 0,
        "dropped": dropped,
        "checks_run": run.get("checks_run", 0),
        "independent": _independent(draft, by_id, plays, _rng(run, "independent")),
        "seeded": seeded,
        "cost_usd": run["cost_usd"]["total"],
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "llm_calls": len(usage),
        "tool_calls": len(calls),
        "tool_errors": sum(1 for c in calls if c.error),
    }


def score_config(
    config: str, runs_dir: Path, con: duckdb.DuckDBPyConnection, seeds: int = 3
) -> list[dict[str, Any]]:
    paths = sorted((runs_dir / config).glob("r*/*.json"))
    runs = [load_run(p) for p in paths]
    plays = {s: load_plays(con, s) for s in {r["request"]["season"] for r in runs}}
    return [score_run(r, con, seeds, plays[r["request"]["season"]]) for r in runs]


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _bootstrap(rows: list[dict], num, den, n: int = 2000, seed: int = 0) -> list[float | None]:
    """95% interval of a pooled ratio, resampling runs (claims within a run are correlated)."""
    if not rows:
        return [None, None]
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        d = sum(den(r) for r in sample)
        if d:
            stats.append(sum(num(r) for r in sample) / d)
    stats.sort()
    return (
        [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]]
        if stats
        else [None, None]
    )


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def total(path: str) -> int:
        a, b = path.split(".") if "." in path else (path, None)
        return sum((r[a][b] if b else r[a]) for r in rows)

    kinds: dict[str, dict[str, int]] = {}
    for r in rows:
        for k, v in r["seeded"]["by_kind"].items():
            agg = kinds.setdefault(k, {"seeded": 0, "caught": 0})
            agg["seeded"] += v["seeded"]
            agg["caught"] += v["caught"]
    seeded_total = sum(v["seeded"] for v in kinds.values())
    return {
        "runs": len(rows),
        "draft_claims": total("draft.claims"),
        "draft_accuracy": _ratio(total("draft.value_ok"), total("draft.claims")),
        "draft_accuracy_ci": _bootstrap(
            rows, lambda r: r["draft"]["value_ok"], lambda r: r["draft"]["claims"]
        ),
        "draft_wording_ok": _ratio(total("draft.wording_ok"), total("draft.claims")),
        "unsourced_claims": sum(r["draft"].get("unsourced_claims", 0) for r in rows),
        "draft_unsupported_per_report": total("draft.unsupported") / len(rows) if rows else None,
        "final_claims": total("final.claims"),
        "final_accuracy": _ratio(total("final.value_ok"), total("final.claims")),
        "final_accuracy_ci": _bootstrap(
            rows, lambda r: r["final"]["value_ok"], lambda r: r["final"]["claims"]
        ),
        "final_wording_ok": _ratio(total("final.wording_ok"), total("final.claims")),
        "report_numbers_per_report": total("final.report_numbers") / len(rows) if rows else None,
        "report_unsupported_rate": _ratio(
            total("final.unsupported"), total("final.report_numbers")
        ),
        "report_mismatch_rate": _ratio(total("final.mismatched"), total("final.report_numbers")),
        "corrected": total("corrected"),
        "dropped": total("dropped"),
        "independent_sampled": total("independent.sampled"),
        "independent_covered": total("independent.covered"),
        "independent_tool_agreement": _ratio(
            total("independent.tool_agrees"), total("independent.covered")
        ),
        "independent_stated_agreement": _ratio(
            total("independent.stated_agrees"), total("independent.covered")
        ),
        "seeded_catch_rate": _ratio(sum(v["caught"] for v in kinds.values()), seeded_total),
        "seeded_by_kind": {k: _ratio(v["caught"], v["seeded"]) for k, v in sorted(kinds.items())},
        "seeded_false_positives": sum(r["seeded"]["false_positives"] for r in rows),
        "cost_per_report": total("cost_usd") / len(rows) if rows else None,
        "output_tokens_per_report": total("output_tokens") / len(rows) if rows else None,
        "tool_error_rate": _ratio(total("tool_errors"), total("tool_calls")),
    }


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _table_rows(agg: dict[str, Any]) -> dict[str, Any]:
    """One row per metric; nested dicts (catch rate per kind) become indented rows."""
    out: dict[str, Any] = {}
    for k, v in agg.items():
        if isinstance(v, dict):
            out.update({f"  {k}.{sub}": x for sub, x in v.items()})
        else:
            out[k] = v
    return out


def main() -> None:
    from nfl_scout.config import load_settings
    from nfl_scout.evaluation.snapshot import snapshot_connection
    from nfl_scout.tools.base import default_connection

    settings = load_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["full", "no_checker", "single"])
    ap.add_argument("--seeds", type=int, default=3, help="seeded-error draws per run")
    ap.add_argument("--mlflow", action="store_true", help="log each config to MLflow")
    ap.add_argument(
        "--live-db",
        action="store_true",
        help="score against data/nfl.duckdb instead of the frozen eval snapshot",
    )
    a = ap.parse_args()

    runs_dir = settings.resolve(settings.paths.runs_dir)
    # Re-runs must see the data the runs were generated with, not a later refresh.
    con = default_connection() if a.live_db else snapshot_connection()
    results = {}
    for config in a.config:
        rows = score_config(config, runs_dir, con, a.seeds)
        if not rows:
            print(f"{config}: no runs")
            continue
        results[config] = {"aggregate": aggregate(rows), "runs": rows}
        out = runs_dir / "scores" / f"{config}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results[config], indent=1))
        if a.mlflow:
            from nfl_scout.evaluation.tracking import log_config

            log_config(config, results[config], settings)
    rows = {c: _table_rows(r["aggregate"]) for c, r in results.items()}
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    print(f"{'metric':34}" + "".join(f"{c:>18}" for c in results))
    for k in keys:
        print(f"{k:34}" + "".join(f"{_fmt(rows[c].get(k)):>18}" for c in results))


if __name__ == "__main__":
    main()
