"""Shared machinery for the metric tools: result type, registry, argument checks, play queries.

Leakage guard: every play-by-play query goes through `PLAYS_CTE`, which keeps only regular-season
plays from weeks <= through_week. Roster tables are filtered on week the same way.
"""

from __future__ import annotations

import functools
import inspect
import re
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import duckdb
from pydantic import BaseModel

from nfl_scout.config import load_settings

Side = Literal["offense", "defense"]
PlayKind = Literal["all", "pass", "run"]
Better = Literal["higher", "lower"]

# Scrimmage plays only: no special teams, kneels, spikes, penalties that wiped out the play,
# or two-point tries. `pass = 1` includes sacks and scrambles; `rush = 1` is designed runs.
PLAYS_CTE = """
WITH plays AS (
    SELECT * FROM pbp
    WHERE season = $season AND season_type = 'REG' AND week <= $through_week
      AND play_type IN ('pass', 'run') AND coalesce(two_point_attempt, 0) = 0
)
"""

FTN_JOIN = """
JOIN ftn_charting f
  ON f.nflverse_game_id = p.game_id AND f.nflverse_play_id = CAST(p.play_id AS INTEGER)
"""

PLAY_KIND_SQL = {"all": "TRUE", "pass": "p.pass = 1", "run": "p.rush = 1"}


class ToolResult(BaseModel):
    """What every tool returns. `value` goes into the trace, never straight into report prose."""

    value: float | int | str | list[dict[str, Any]] | None
    n_plays: int
    query_meta: dict[str, Any]


@dataclass
class Measure:
    """A tool body's output; the `tool` wrapper adds the standard metadata."""

    value: Any
    n_plays: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: Callable[..., ToolResult]
    description: str
    unit: str
    # Which direction is good, by side ("offense"/"defense"). None: a tendency, not a quality.
    better: dict[str, Better] | None
    rankable: bool


TOOLS: dict[str, ToolSpec] = {}


@functools.lru_cache(maxsize=1)
def default_connection() -> duckdb.DuckDBPyConnection:
    settings = load_settings()
    return duckdb.connect(str(settings.resolve(settings.data.duckdb_path)), read_only=True)


def _check_choice(name: str, value: Any, annotation: Any) -> None:
    """Raise ValueError if `value` is outside a Literal (or Optional Literal) annotation."""
    allowed: tuple = ()
    optional = False
    if typing.get_origin(annotation) is Literal:
        allowed = typing.get_args(annotation)
    elif typing.get_origin(annotation) in (typing.Union, types.UnionType):
        for arg in typing.get_args(annotation):
            if arg is type(None):
                optional = True
            elif typing.get_origin(arg) is Literal:
                allowed += typing.get_args(arg)
    if allowed and not (value in allowed or (optional and value is None)):
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def _check_common(con: duckdb.DuckDBPyConnection, team: str, through_week: int) -> None:
    if through_week < 1:
        raise ValueError(f"through_week must be >= 1, got {through_week}")
    if not con.execute("SELECT 1 FROM teams WHERE team_abbr = ?", [team]).fetchone():
        raise ValueError(f"unknown team {team!r}; use a team abbreviation such as 'KC'")


def tool(
    *, unit: str, better: dict[str, Better] | None = None, rankable: bool = True
) -> Callable[[Callable[..., Measure]], Callable[..., ToolResult]]:
    """Register a tool body `fn(con, team, season, through_week, ...) -> Measure`.

    The public function drops `con` from the front and takes it as an optional keyword instead
    (default: the project DuckDB file). Literal-typed arguments are checked before the query runs.
    """

    def deco(fn: Callable[..., Measure]) -> Callable[..., ToolResult]:
        sig = inspect.signature(fn)
        public_sig = sig.replace(parameters=list(sig.parameters.values())[1:])
        hints = typing.get_type_hints(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, con: duckdb.DuckDBPyConnection | None = None, **kwargs: Any):
            bound = public_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            call_args: dict[str, Any] = {}
            for name, value in bound.arguments.items():
                if sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    call_args.update(value)
                else:
                    _check_choice(name, value, hints.get(name))
                    call_args[name] = value
            con = con or default_connection()
            _check_common(con, call_args["team"], call_args["through_week"])
            m = fn(con, *bound.args, **bound.kwargs)
            meta = {"tool": fn.__name__, "args": call_args, "unit": unit, **m.meta}
            return ToolResult(value=m.value, n_plays=m.n_plays, query_meta=meta)

        wrapper.__signature__ = public_sig
        TOOLS[fn.__name__] = ToolSpec(
            name=fn.__name__,
            fn=wrapper,
            description=inspect.getdoc(fn) or "",
            unit=unit,
            better=better,
            rankable=rankable,
        )
        return wrapper

    return deco


def run_query(con: duckdb.DuckDBPyConnection, sql: str, params: dict[str, Any]) -> tuple:
    """Execute `sql`, passing only the named parameters it actually uses."""
    used = set(re.findall(r"\$(\w+)", sql))
    return con.execute(sql, {k: v for k, v in params.items() if k in used}).fetchone()


def measure_plays(
    con: duckdb.DuckDBPyConnection,
    value_sql: str,
    where: str,
    params: dict[str, Any],
    definition: str,
    *,
    ftn: bool = False,
) -> Measure:
    """Aggregate `value_sql` over the leakage-filtered plays `p` (joined to FTN as `f`)."""
    sql = f"""
        {PLAYS_CTE}
        SELECT {value_sql}, count(*), list_sort(list(DISTINCT p.week)), count(DISTINCT p.game_id)
        FROM plays p {FTN_JOIN if ftn else ""}
        WHERE {where}
    """
    value, n, weeks, n_games = run_query(con, sql, params)
    meta = {"definition": definition, "weeks": weeks or [], "n_games": n_games}
    return Measure(value=value, n_plays=n, meta=meta)


def team_col(side: Side) -> str:
    return "p.posteam" if side == "offense" else "p.defteam"
