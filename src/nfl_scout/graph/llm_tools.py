"""Expose the metric tools to the LLM: strict JSON schemas and an executor that records a trace.

The model never chooses `season` or `through_week`: they are left out of the schemas and injected
by the executor from the request, so the leakage cutoff can't be widened by the model.
"""

from __future__ import annotations

import inspect
import json
import types
import typing
from typing import Any, Literal

import duckdb

from nfl_scout.graph.state import ToolCall
from nfl_scout.tools import TOOLS
from nfl_scout.tools.base import default_connection

INJECTED = ("season", "through_week")
PY_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(annotation: Any) -> tuple[dict[str, Any], bool]:
    """JSON schema for a parameter annotation, and whether it is Optional."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        schema, _ = _json_type(args[0])
        return schema, True
    if origin is Literal:
        values = list(typing.get_args(annotation))
        return {"type": PY_TYPES[type(values[0])], "enum": values}, False
    return {"type": PY_TYPES[annotation]}, False


def _params(name: str, teams: list[str]) -> tuple[dict[str, dict], list[str]]:
    fn = TOOLS[name].fn
    hints = typing.get_type_hints(inspect.unwrap(fn))
    props: dict[str, dict] = {}
    required: list[str] = []
    for p in inspect.signature(fn).parameters.values():
        if p.name in INJECTED or p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        schema, optional = _json_type(hints[p.name])
        if p.name == "team":
            schema = {"type": "string", "enum": teams, "description": "Team abbreviation."}
        props[p.name] = schema
        if p.default is inspect.Parameter.empty and not optional:
            required.append(p.name)
    return props, required


def _league_rank_params(teams: list[str]) -> tuple[dict[str, dict], list[str]]:
    rankable = sorted(n for n, s in TOOLS.items() if s.rankable)
    props: dict[str, dict] = {
        "metric": {"type": "string", "enum": rankable, "description": "Tool to rank on."},
        "team": {"type": "string", "enum": teams, "description": "Team abbreviation."},
    }
    for name in rankable:
        for key, schema in _params(name, teams)[0].items():
            props.setdefault(
                key,
                {
                    **schema,
                    "description": "Pass only if the metric takes this split.",
                },
            )
    return props, ["metric", "team"]


def _league_rank_note(teams: list[str]) -> str:
    """Which splits each rankable metric takes, for the league_rank description."""
    lines = []
    for name in sorted(n for n, s in TOOLS.items() if s.rankable):
        splits = sorted(k for k in _params(name, teams)[0] if k != "team")
        lines.append(f"- {name}: {', '.join(splits) or 'none'}")
    return "\nSplits each metric takes (pass no others):\n" + "\n".join(lines)


def tool_schemas(teams: list[str], names: list[str] | None = None) -> list[dict[str, Any]]:
    """Strict tool definitions for `names` (default: every registered tool), in a fixed order
    (cache-stable). The API allows at most 20 strict tools per request."""
    schemas = []
    for name in sorted(names or TOOLS):
        spec = TOOLS[name]
        props, required = (
            _league_rank_params(teams) if name == "league_rank" else _params(name, teams)
        )
        description = f"{spec.description}\nUnit of value: {spec.unit}."
        if name == "league_rank":
            description += _league_rank_note(teams)
        schemas.append(
            {
                "name": name,
                "description": description,
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        )
    return schemas


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


# query_meta keys the model doesn't see: they repeat the arguments, or (team_value) would let the
# model cite a rank call for the metric's value instead of calling the metric itself.
HIDDEN_META = {"tool", "args", "metric_args", "team_value"}


class ToolExecutor:
    """Runs tool calls for one node, injecting the cutoff and numbering calls (o1, o2, ...)."""

    def __init__(
        self,
        season: int,
        through_week: int,
        prefix: str,
        node: str = "",
        con: duckdb.DuckDBPyConnection | None = None,
    ):
        self.season = season
        self.through_week = through_week
        self.prefix = prefix
        self.node = node
        # Own cursor per executor: analysts run in parallel threads, and a DuckDB connection
        # shared across threads mixes up query results.
        self.con = (con or default_connection()).cursor()
        self.calls: list[ToolCall] = []

    def run(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Execute one call. Returns (tool_result content for the model, is_error)."""
        call_id = f"{self.prefix}{len(self.calls) + 1}"
        args = {k: v for k, v in args.items() if v is not None}
        full = {**args, "season": self.season, "through_week": self.through_week}
        call = ToolCall(call_id=call_id, node=self.node, tool=name, args=full)
        self.calls.append(call)
        try:
            if name not in TOOLS:
                raise ValueError(f"unknown tool {name!r}")
            if set(INJECTED) & args.keys():
                raise ValueError("season and through_week are set by the system; don't pass them")
            result = TOOLS[name].fn(con=self.con, **full)
        except (ValueError, TypeError) as e:
            call.error = str(e)
            return json.dumps({"call_id": call_id, "error": str(e)}), True
        call.result = result.model_dump()
        meta = {k: v for k, v in result.query_meta.items() if k not in HIDDEN_META}
        payload = {"call_id": call_id, "value": result.value, "n_plays": result.n_plays, **meta}
        return json.dumps(_round(payload)), False
