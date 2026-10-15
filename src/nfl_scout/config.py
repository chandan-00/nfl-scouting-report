"""Load config/settings.yaml and resolve the model and effort for each graph node."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Environment variable that switches every node to one model for a comparison run.
MODEL_OVERRIDE_ENV = "NFL_SCOUT_MODEL"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class NodeModel(BaseModel):
    model: str
    effort: Effort


class ModelSettings(BaseModel):
    aliases: dict[str, str]
    default: str
    default_effort: Effort
    nodes: dict[str, NodeModel]
    max_revision_loops: int

    @model_validator(mode="after")
    def _check_aliases(self) -> ModelSettings:
        names = {self.default, *(n.model for n in self.nodes.values())}
        unknown = names - self.aliases.keys()
        if unknown:
            raise ValueError(f"unknown model alias(es) in settings: {sorted(unknown)}")
        return self


class DataSettings(BaseModel):
    seasons: list[int]
    eval_season: int
    duckdb_path: Path


class EvalSettings(BaseModel):
    weeks: list[int]
    n_matchups: int
    independent_check_fraction: float


class PathSettings(BaseModel):
    runs_dir: Path


class Settings(BaseModel):
    models: ModelSettings
    data: DataSettings
    eval: EvalSettings
    paths: PathSettings

    def node_model(self, node: str) -> NodeModel:
        """Return the full model ID and effort for a node.

        NFL_SCOUT_MODEL (haiku | sonnet | opus), when set, replaces the model on every node
        but keeps each node's effort.
        """
        cfg = self.models.nodes.get(
            node, NodeModel(model=self.models.default, effort=self.models.default_effort)
        )
        alias = os.environ.get(MODEL_OVERRIDE_ENV) or cfg.model
        if alias not in self.models.aliases:
            raise ValueError(
                f"{MODEL_OVERRIDE_ENV}={alias!r} is not one of {sorted(self.models.aliases)}"
            )
        return NodeModel(model=self.models.aliases[alias], effort=cfg.effort)

    def resolve(self, path: Path) -> Path:
        """Resolve a settings path relative to the project root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    with path.open() as f:
        return Settings.model_validate(yaml.safe_load(f))
