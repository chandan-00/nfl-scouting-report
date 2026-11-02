"""Dollar cost of LLM calls from their recorded token usage."""

from __future__ import annotations

from collections.abc import Iterable

from nfl_scout.config import Settings
from nfl_scout.graph.state import Usage


def usage_cost(u: Usage, settings: Settings) -> float:
    p = settings.price(u.model)
    return (
        u.input_tokens * p.input
        + u.output_tokens * p.output
        + u.cache_read_input_tokens * p.cache_read
        + u.cache_creation_input_tokens * p.cache_write
    ) / 1_000_000


def cost_by_node(usages: Iterable[Usage], settings: Settings) -> dict[str, float]:
    out: dict[str, float] = {}
    for u in usages:
        out[u.node] = out.get(u.node, 0.0) + usage_cost(u, settings)
    return out
