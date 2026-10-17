"""League rank for any rankable metric (backs every "Nth in the league" claim)."""

from __future__ import annotations

import inspect
from typing import Any

import duckdb

from nfl_scout.tools.base import PLAYS_CTE, TOOLS, Measure, tool


@tool(unit="rank", rankable=False)
def league_rank(
    con: duckdb.DuckDBPyConnection,
    metric: str,
    team: str,
    season: int,
    through_week: int,
    **metric_args: Any,
) -> Measure:
    """Rank of `team` among all teams on `metric` (another tool's name, called with
    `metric_args`). Rank 1 is the best value when the metric has a good direction (e.g. lowest
    EPA allowed), otherwise the highest value. Tied teams share the best rank."""
    spec = TOOLS.get(metric)
    if spec is None or not spec.rankable:
        rankable = sorted(n for n, s in TOOLS.items() if s.rankable)
        raise ValueError(f"cannot rank {metric!r}; rankable metrics: {rankable}")
    splits = sorted(
        p
        for p in inspect.signature(spec.fn).parameters
        if p not in ("team", "season", "through_week")
    )
    unknown = sorted(set(metric_args) - set(splits))
    if unknown:
        raise ValueError(
            f"{metric} accepts: {', '.join(splits) or 'no splits'}; drop {', '.join(unknown)}"
        )

    side = metric_args.get("side", "offense")
    better = spec.better.get(side) if spec.better else None
    teams = [
        t
        for (t,) in con.execute(
            f"{PLAYS_CTE} SELECT DISTINCT posteam FROM plays",
            {"season": season, "through_week": through_week},
        ).fetchall()
    ]
    results = {t: spec.fn(t, season, through_week, con=con, **metric_args) for t in teams}
    values = {t: r.value for t, r in results.items() if r.value is not None}

    meta: dict[str, Any] = {
        "metric": metric,
        "metric_args": metric_args,
        "n_teams": len(values),
        "rank_1_is": {"higher": "best (highest)", "lower": "best (lowest)"}.get(better, "highest"),
    }
    own = values.get(team)
    if own is None:
        return Measure(None, 0, {**meta, "team_value": None, "tied": False})
    ahead = sum(v < own for v in values.values()) if better == "lower" else None
    if ahead is None:
        ahead = sum(v > own for v in values.values())
    tied = sum(v == own for v in values.values()) > 1
    own_result = results[team]
    meta.update(
        team_value=own,
        tied=tied,
        weeks=own_result.query_meta.get("weeks", []),
        n_games=own_result.query_meta.get("n_games"),
    )
    return Measure(ahead + 1, own_result.n_plays, meta)
