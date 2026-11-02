"""Thin wrapper over the Anthropic client: per-node model and effort, caching, usage records."""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from nfl_scout.config import Settings
from nfl_scout.graph.state import Usage

MAX_TOKENS = 16000
# Above this the SDK refuses non-streaming requests (they could outlast its 10-minute timeout).
STREAM_ABOVE = 16000


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
        output_schema: dict[str, Any] | None,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = MAX_TOKENS,
    ) -> tuple[Any, Usage]:
        """One Messages API request. The tools + system prefix carries the cache breakpoint; the
        top-level cache_control also caches the growing conversation in tool loops."""
        cfg = self.settings.node_model(node)
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": cfg.effort},
            "cache_control": {"type": "ephemeral"},
        }
        if output_schema is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": output_schema}
        if tools:
            kwargs["tools"] = tools
            # "auto", or "none" to force a final answer (forced tool use 400s on 5.5 models).
            kwargs["tool_choice"] = {"type": tool_choice}
        start = time.perf_counter()
        if max_tokens > STREAM_ABOVE:
            with self.client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
        else:
            response = self.client.messages.create(**kwargs)
        latency = time.perf_counter() - start

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"{node}: request refused ({getattr(response, 'stop_details', None)})"
            )
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"{node}: hit max_tokens={max_tokens}")
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
