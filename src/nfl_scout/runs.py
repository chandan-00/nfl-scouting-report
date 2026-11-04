"""Save each graph run to runs/<config>/<matchup>.json and load it back. Scoring reads only
these files (generate once, score many times)."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from nfl_scout.config import Settings
from nfl_scout.costs import cost_by_node
from nfl_scout.graph.state import Request


def matchup_name(r: Request) -> str:
    return f"{r.season}_w{r.week:02d}_{r.team}_vs_{r.opponent}"


def run_path(runs_dir: Path, config: str, request: Request, repeat: int | None = None) -> Path:
    """runs/<config>/[r<repeat>/]<matchup>.json"""
    folder = runs_dir / config / (f"r{repeat}" if repeat is not None else "")
    return folder / f"{matchup_name(request)}.json"


def _plain(x: Any) -> Any:
    if isinstance(x, BaseModel):
        return x.model_dump(mode="json")
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {k: _plain(v) for k, v in dataclasses.asdict(x).items()}
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    return x


def save_run(
    out: dict[str, Any],
    config: str,
    settings: Settings,
    runs_dir: Path | None = None,
    repeat: int | None = None,
) -> Path:
    """Write the run (draft, final sections, report, checks, trace, usage, cost) to disk."""
    request: Request = out["request"]
    runs_dir = runs_dir or settings.resolve(settings.paths.runs_dir)
    path = run_path(runs_dir, config, request, repeat)
    path.parent.mkdir(parents=True, exist_ok=True)
    costs = cost_by_node(out.get("usage", []), settings)
    record = {
        "config": config,
        "repeat": repeat,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": {node: settings.node_model(node).model_dump() for node in settings.models.nodes},
        "max_revision_loops": settings.models.max_revision_loops,
        "request": request.model_dump(),
        "plan": out.get("plan"),
        "draft": _plain(out.get("draft", {})),
        "sections": _plain(out.get("sections", {})),
        "checks": _plain(out.get("checks", [])),
        "check_log": _plain(out.get("check_log", [])),
        "checks_run": out.get("checks_run", 0),
        "dropped": _plain(out.get("dropped", [])),
        "report": _plain(out.get("report")),
        "number_checks": _plain(out.get("number_checks", [])),
        "tool_calls": _plain(out.get("tool_calls", [])),
        "usage": _plain(out.get("usage", [])),
        "cost_usd": {"total": sum(costs.values()), "by_node": costs},
    }
    path.write_text(json.dumps(record, indent=1))
    return path


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
