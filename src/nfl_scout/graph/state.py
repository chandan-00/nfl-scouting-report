"""Typed graph state, the claim schema, and parsing of analyst output.

Claims are written by the LLM: `stated_value` is the number the model typed in its prose. The tool
output it cites lives only in `tool_calls` (the trace), where the checker and scorer can see it.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, computed_field, model_validator

DerivedOp = Literal["ratio", "difference", "sum"]


class Request(BaseModel):
    """Scout `opponent` for `team`'s game in `week`, using only data through the week before."""

    team: str
    opponent: str
    season: int
    week: int

    @computed_field
    @property
    def through_week(self) -> int:
        return self.week - 1


class Claim(BaseModel):
    claim_id: str
    text: str
    stated_value: float | str
    tool_call_id: str | None = None
    derived_from: list[str] | None = None
    op: DerivedOp | None = None

    @model_validator(mode="after")
    def _one_source(self) -> Claim:
        cited = self.tool_call_id is not None
        derived = bool(self.derived_from)
        if cited == derived:
            raise ValueError(
                f"claim {self.claim_id} needs exactly one of tool_call_id/derived_from"
            )
        if derived and self.op is None:
            raise ValueError(f"derived claim {self.claim_id} needs an op")
        return self


class Section(BaseModel):
    role: str
    prose: str
    claims: list[Claim]


class ToolCall(BaseModel):
    """One executed tool call. `args` include the injected season and through_week."""

    call_id: str
    node: str
    tool: str
    args: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class Usage(BaseModel):
    node: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    latency_s: float


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


class ScoutState(TypedDict, total=False):
    request: Request
    plan: dict[str, list[str]]
    sections: Annotated[dict[str, Section], _merge]
    tool_calls: Annotated[list[ToolCall], operator.add]
    usage: Annotated[list[Usage], operator.add]


# Structured-output schema for an analyst's final answer. Every field is required (nullable where
# optional) to keep the schema strict.
SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prose": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "text": {"type": "string"},
                    "stated_value": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                    "tool_call_id": {"type": ["string", "null"]},
                    "derived_from": {
                        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
                    },
                    "op": {
                        "anyOf": [
                            {"type": "string", "enum": ["ratio", "difference", "sum"]},
                            {"type": "null"},
                        ]
                    },
                },
                "required": [
                    "claim_id",
                    "text",
                    "stated_value",
                    "tool_call_id",
                    "derived_from",
                    "op",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["prose", "claims"],
    "additionalProperties": False,
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "offense_focus": {"type": "array", "items": {"type": "string"}},
        "defense_focus": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["offense_focus", "defense_focus"],
    "additionalProperties": False,
}

ROLE_PREFIX = {"offense_analyst": "o", "defense_analyst": "d"}


def parse_section(role: str, raw: dict[str, Any]) -> Section:
    """Validate an analyst's output and namespace its claim ids (c1 -> oc1) in claims and prose."""
    prefix = ROLE_PREFIX.get(role, role[0])
    ids = {c["claim_id"]: f"{prefix}{c['claim_id']}" for c in raw["claims"]}
    claims = []
    for c in raw["claims"]:
        derived = [ids.get(d, d) for d in c.get("derived_from") or []] or None
        claims.append(Claim(**{**c, "claim_id": ids[c["claim_id"]], "derived_from": derived}))

    def _rename(m: re.Match) -> str:
        return "[" + ", ".join(ids.get(x.strip(), x.strip()) for x in m.group(1).split(",")) + "]"

    prose = re.sub(r"\[([^\]]+)\]", _rename, raw["prose"])
    return Section(role=role, prose=prose, claims=claims)
