"""Thin wrapper over the Anthropic client: per-node model and effort, caching, usage records."""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from nfl_scout.config import Settings
from nfl_scout.graph.state import Usage

MAX_TOKENS = 16000


class LLM:
    def __init__(self, client: Any, settings: Settings):
        self.client = client
        self.settings = settings

    def call(
        self,
        node: str,
        *,
        system: str,
        messages: list[dict],
        output_schema: dict[str, Any],
        tools: list[dict] | None = None,
    ) -> tuple[Any, Usage]:
        """One Messages API request. The tools + system prefix carries the cache breakpoint; the
        top-level cache_control also caches the growing conversation in tool loops."""
        cfg = self.settings.node_model(node)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": cfg.effort,
                "format": {"type": "json_schema", "schema": output_schema},
            },
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {"type": "auto"}
        start = time.perf_counter()
        response = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - start

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"{node}: request refused ({getattr(response, 'stop_details', None)})"
            )
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{node}: hit max_tokens={MAX_TOKENS}")
        u = response.usage
        usage = Usage(
            node=node,
            model=cfg.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_input_tokens=u.cache_read_input_tokens or 0,
            cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
            latency_s=latency,
        )
        return response, usage


def final_json(response: Any) -> dict[str, Any]:
    """The structured-output JSON from a final (non-tool) response."""
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def default_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()
