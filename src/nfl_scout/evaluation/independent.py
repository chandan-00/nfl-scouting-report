"""Independent re-derivation of the core metrics, for the verification sample.

Written from the metric definitions (DECISIONS D2.3/D2.6), not from the tools' SQL: polars over
the raw play-by-play, no shared helpers. A disagreement with a tool means a bug in one of them.
FTN, roster and player tools aren't covered; `independent_value` returns None for those.

    uv run python -m nfl_scout.evaluation.independent   # cross-check every team and split
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import duckdb
import polars as pl

COLUMNS = [
    "season", "season_type", "week", "game_id", "posteam", "defteam", "play_type", "pass",
    "rush", "qb_dropback", "pass_attempt", "sack", "qb_hit", "two_point_attempt", "epa",
    "success", "down", "ydstogo", "yardline_100", "air_yards", "yards_gained", "run_location",
    "fixed_drive", "fixed_drive_result", "third_down_converted",
]  # fmt: skip


def load_plays(con: duckdb.DuckDBPyConnection, season: int) -> pl.DataFrame:
    """Every regular-season row of the season, unfiltered; filtering happens per call."""
    return pl.from_arrow(
        con.execute(
            f"SELECT {', '.join(COLUMNS)} FROM pbp WHERE season = ?", [season]
        ).to_arrow_table()
    )


def _scrimmage(df: pl.DataFrame, through_week: int) -> pl.DataFrame:
    return df.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("week") <= through_week)
        & pl.col("play_type").is_in(["pass", "run"])
        & (pl.col("two_point_attempt").fill_null(0) == 0)
    )


def _side(df: pl.DataFrame, team: str | None, side: str) -> pl.DataFrame:
    """The team's plays on one side; team=None keeps every team (the league as a whole)."""
    if team is None:
        return df
    return df.filter(pl.col("posteam" if side == "offense" else "defteam") == team)


def _kind(df: pl.DataFrame, play_kind: str) -> pl.DataFrame:
    if play_kind == "pass":
        return df.filter(pl.col("pass") == 1)
    if play_kind == "run":
        return df.filter(pl.col("rush") == 1)
    return df


def _mean(df: pl.DataFrame, expr: pl.Expr, scale: float = 1.0) -> tuple[float | None, int]:
    if df.height == 0:
        return None, 0
    return float(df.select(expr.cast(pl.Float64).mean()).item()) * scale, df.height


def _flag(cond: pl.Expr) -> pl.Expr:
    return pl.when(cond).then(1.0).otherwise(0.0)


def epa_per_play(df, team, side="offense", play_kind="all"):
    return _mean(_kind(_side(df, team, side), play_kind), pl.col("epa"))


def success_rate(df, team, side="offense", play_kind="all"):
    return _mean(_kind(_side(df, team, side), play_kind), pl.col("success"), 100)


def explosive_play_rate(df, team, side="offense", play_kind="all"):
    big = ((pl.col("pass") == 1) & (pl.col("yards_gained") >= 20)) | (
        (pl.col("rush") == 1) & (pl.col("yards_gained") >= 10)
    )
    return _mean(_kind(_side(df, team, side), play_kind), _flag(big), 100)


def sack_rate(df, team, side="offense"):
    drop = _side(df, team, side).filter(pl.col("qb_dropback") == 1)
    return _mean(drop, pl.col("sack"), 100)


def pressure_proxy_rate(df, team, side="offense"):
    drop = _side(df, team, side).filter(pl.col("qb_dropback") == 1)
    return _mean(drop, _flag((pl.col("sack") == 1) | (pl.col("qb_hit") == 1)), 100)


def third_down_conversion_rate(df, team, side="offense"):
    third = _side(df, team, side).filter(pl.col("down") == 3)
    return _mean(third, pl.col("third_down_converted"), 100)


def red_zone_td_rate(df, team, side="offense"):
    drives = (
        _side(df, team, side)
        .group_by("game_id", "fixed_drive")
        .agg(
            pl.col("yardline_100").min().alias("closest"),
            (pl.col("fixed_drive_result") == "Touchdown").any().alias("td"),
        )
        .filter(pl.col("closest") <= 20)
    )
    return _mean(drives, pl.col("td"), 100)


DISTANCE = {
    "short": pl.col("ydstogo") <= 3,
    "medium": pl.col("ydstogo").is_between(4, 7),
    "long": pl.col("ydstogo") >= 8,
}


def pass_rate(df, team, down=None, distance=None, early_downs_only=False):
    plays = _side(df, team, "offense")
    if down is not None:
        plays = plays.filter(pl.col("down") == down)
    if early_downs_only:
        plays = plays.filter(pl.col("down").is_in([1.0, 2.0]))
    if distance is not None:
        plays = plays.filter(DISTANCE[distance])
    return _mean(plays, pl.col("pass"), 100)


def average_depth_of_target(df, team):
    throws = _side(df, team, "offense").filter(
        (pl.col("pass_attempt") == 1) & (pl.col("sack") == 0) & pl.col("air_yards").is_not_null()
    )
    return _mean(throws, pl.col("air_yards"))


def run_direction_rate(df, team, direction):
    runs = _side(df, team, "offense").filter(
        (pl.col("rush") == 1) & pl.col("run_location").is_not_null()
    )
    return _mean(runs, _flag(pl.col("run_location") == direction), 100)


METRICS: dict[str, Callable[..., tuple[float | None, int]]] = {
    f.__name__: f
    for f in (
        epa_per_play,
        success_rate,
        explosive_play_rate,
        sack_rate,
        pressure_proxy_rate,
        third_down_conversion_rate,
        red_zone_td_rate,
        pass_rate,
        average_depth_of_target,
        run_direction_rate,
    )
}


def week_plays(df: pl.DataFrame, week: int) -> pl.DataFrame:
    """Scrimmage plays of one week only (e.g. the game a report was written for)."""
    return _scrimmage(df, week).filter(pl.col("week") == week)


def independent_value(
    plays: pl.DataFrame, tool: str, args: dict[str, Any]
) -> tuple[float | None, int] | None:
    """(value, n_plays) for a recorded tool call, or None if the tool isn't covered."""
    if tool not in METRICS:
        return None
    kwargs = {k: v for k, v in args.items() if k not in ("season", "through_week")}
    return METRICS[tool](_scrimmage(plays, args["through_week"]), **kwargs)


# Every split the cross-check exercises, per metric.
SPLITS: dict[str, list[dict[str, Any]]] = {
    "epa_per_play": [
        {"side": s, "play_kind": k} for s in ("offense", "defense") for k in ("all", "pass", "run")
    ],
    "success_rate": [
        {"side": s, "play_kind": k} for s in ("offense", "defense") for k in ("all", "pass", "run")
    ],
    "explosive_play_rate": [
        {"side": s, "play_kind": k} for s in ("offense", "defense") for k in ("all", "pass", "run")
    ],
    "sack_rate": [{"side": s} for s in ("offense", "defense")],
    "pressure_proxy_rate": [{"side": s} for s in ("offense", "defense")],
    "third_down_conversion_rate": [{"side": s} for s in ("offense", "defense")],
    "red_zone_td_rate": [{"side": s} for s in ("offense", "defense")],
    "pass_rate": [{}, {"early_downs_only": True}, {"down": 3, "distance": "long"}]
    + [{"down": d} for d in (1, 2, 3, 4)]
    + [{"distance": d} for d in ("short", "medium", "long")],
    "average_depth_of_target": [{}],
    "run_direction_rate": [{"direction": d} for d in ("left", "middle", "right")],
}


def cross_check(
    con: duckdb.DuckDBPyConnection, season: int, through_week: int, tol: float = 1e-6
) -> tuple[int, list[dict[str, Any]]]:
    """Compare every covered tool with its independent version for every team and split.
    Returns (comparisons made, disagreements)."""
    from nfl_scout.tools import TOOLS

    plays = load_plays(con, season)
    teams = [t for (t,) in con.execute("SELECT team_abbr FROM teams").fetchall()]
    teams = [t for t in teams if t in set(plays["posteam"].drop_nulls().unique())]
    n, bad = 0, []
    for tool, splits in SPLITS.items():
        for split in splits:
            for team in teams:
                args = {"team": team, "season": season, "through_week": through_week, **split}
                ours = independent_value(plays, tool, args)
                theirs = TOOLS[tool].fn(con=con, **args)
                n += 1
                same_n = ours[1] == theirs.n_plays
                both_none = ours[0] is None and theirs.value is None
                close = (
                    ours[0] is not None
                    and theirs.value is not None
                    and abs(ours[0] - theirs.value) <= tol
                )
                if not (same_n and (both_none or close)):
                    bad.append(
                        {"tool": tool, "args": args, "independent": ours,
                         "tool_value": (theirs.value, theirs.n_plays)}
                    )  # fmt: skip
    return n, bad


def main() -> None:
    from nfl_scout.config import load_settings
    from nfl_scout.tools.base import default_connection

    settings = load_settings()
    con = default_connection()
    for tw in sorted({w - 1 for w in settings.eval.weeks}):
        n, bad = cross_check(con, settings.data.eval_season, tw)
        print(f"through week {tw}: {n} comparisons, {len(bad)} disagreements")
        for b in bad[:10]:
            print("  ", b)


if __name__ == "__main__":
    main()
