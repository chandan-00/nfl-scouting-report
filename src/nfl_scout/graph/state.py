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
# Which field of the cited tool result the claim states: its value, or its sample size.
Measured = Literal["value", "n_plays"]


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
    measure: Measured = "value"

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
    # Claims the model wrote with no source (or two). Kept out of `claims` so their markers
    # count as unsupported numbers, and recorded here for the scorer.
    unsourced: list[dict[str, Any]] = []


class ReportSection(BaseModel):
    heading: str
    body: str


class Report(BaseModel):
    """The editor's final report. Bodies keep the claim markers, so every number stays linked."""

    title: str
    summary: str
    sections: list[ReportSection]

    def text(self) -> str:
        """Summary and section bodies: the prose the number checks run over."""
        return "\n\n".join([self.summary, *(s.body for s in self.sections)])

    def markdown(self) -> str:
        parts = [f"# {self.title}", self.summary]
        parts += [f"## {s.heading}\n{s.body}" for s in self.sections]
        return "\n\n".join(parts)


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
    draft: dict[str, Section]  # sections before the first fact-check (scored as drafts)
    checks: list[Any]  # nfl_scout.checker.CheckResult, latest round
    check_log: Annotated[list[dict[str, Any]], operator.add]  # every failure, every round
    checks_run: int
    revise_roles: list[str]
    revise_role: str  # set on each reviser's Send
    dropped: list[dict[str, Any]]  # claims and numbers still failing after the last loop
    report: Report
    number_checks: list[Any]  # nfl_scout.numbers.NumberCheck, over the final report
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
                    "measure": {"type": "string", "enum": ["value", "n_plays"]},
                },
                "required": [
                    "claim_id",
                    "text",
                    "stated_value",
                    "tool_call_id",
                    "derived_from",
                    "op",
                    "measure",
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
        "injury_focus": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["offense_focus", "defense_focus", "injury_focus"],
    "additionalProperties": False,
}

REPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"heading": {"type": "string"}, "body": {"type": "string"}},
                "required": ["heading", "body"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "summary", "sections"],
    "additionalProperties": False,
}

# The single-agent baseline's final answer: the whole report plus one claims list.
SINGLE_SCHEMA: dict[str, Any] = {
    **REPORT_SCHEMA,
    "properties": {
        **REPORT_SCHEMA["properties"],
        "claims": SECTION_SCHEMA["properties"]["claims"],
    },
    "required": [*REPORT_SCHEMA["required"], "claims"],
}

# Prefix for each role's tool call ids (o1, ...) and claim ids (oc1, ...).
ROLE_PREFIX = {
    "single_agent": "s",
    "offense_analyst": "o",
    "defense_analyst": "d",
    "injury_analyst": "i",
    "matchup_analyst": "m",
}


def _namespace(role: str, raw_claims: list[dict]) -> tuple[list[Claim], list[dict], Any]:
    """Claims with ids prefixed per role (c1 -> oc1), the claims set aside for having no single
    source, and a function that renames markers."""
    prefix = ROLE_PREFIX.get(role, role[0])
    ids = {c["claim_id"]: f"{prefix}{c['claim_id']}" for c in raw_claims}
    claims, unsourced = [], []
    for c in raw_claims:
        derived = [ids.get(d, d) for d in c.get("derived_from") or []] or None
        renamed = {**c, "claim_id": ids[c["claim_id"]], "derived_from": derived}
        if (c.get("tool_call_id") is None) == (derived is None) or (derived and not c.get("op")):
            unsourced.append(renamed)
            continue
        claims.append(Claim(**renamed))

    def _rename(m: re.Match) -> str:
        return "[" + ", ".join(ids.get(x.strip(), x.strip()) for x in m.group(1).split(",")) + "]"

    return claims, unsourced, lambda text: re.sub(r"\[([^\]]+)\]", _rename, text)


def parse_section(role: str, raw: dict[str, Any]) -> Section:
    """Validate an analyst's output and namespace its claim ids (c1 -> oc1) in claims and prose."""
    claims, unsourced, rename = _namespace(role, raw["claims"])
    return Section(role=role, prose=rename(raw["prose"]), claims=claims, unsourced=unsourced)


def parse_single(raw: dict[str, Any]) -> tuple[Section, Report]:
    """The single agent's report, plus a section holding all its claims (prose = report text),
    so the checker and scorer treat it like any other section."""
    claims, unsourced, rename = _namespace("single_agent", raw["claims"])
    report = Report(
        title=raw["title"],
        summary=rename(raw["summary"]),
        sections=[
            ReportSection(heading=s["heading"], body=rename(s["body"])) for s in raw["sections"]
        ],
    )
    section = Section(role="single_agent", prose=report.text(), claims=claims, unsourced=unsourced)
    return section, report
