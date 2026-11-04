"""Log scored configs to MLflow: one MLflow run per config scoring, with the settings as params,
the aggregate scores as metrics and the full score file as an artifact. The store is a local
SQLite file under mlruns/ (git-ignored); browse it with

    uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence MLflow's import-time banner

import mlflow  # noqa: E402

from nfl_scout.config import PROJECT_ROOT, Settings

EXPERIMENT = "nfl-scouting-eval"
MLRUNS = PROJECT_ROOT / "mlruns"


def flatten(agg: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Numeric metrics only; intervals become _lo/_hi, nested dicts become dotted names."""
    out: dict[str, float] = {}
    for key, value in agg.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float):
            out[name] = value
        elif isinstance(value, list) and len(value) == 2:
            for suffix, v in zip(("lo", "hi"), value, strict=True):
                if isinstance(v, int | float):
                    out[f"{name}_{suffix}"] = v
        elif isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
    return out


def log_config(
    config: str,
    result: dict[str, Any],
    settings: Settings,
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
) -> str:
    """Log one config's scores; returns the MLflow run id."""
    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{MLRUNS / 'mlflow.db'}")
    artifacts = (artifact_root or MLRUNS / "artifacts").resolve()
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    params = {"config": config, "max_revision_loops": settings.models.max_revision_loops}
    for node in settings.models.nodes:
        cfg = settings.node_model(node)
        params[f"model.{node}"] = cfg.model
        params[f"effort.{node}"] = cfg.effort
    with mlflow.start_run(run_name=config) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(flatten(result["aggregate"]))
        mlflow.log_dict(result, "scores.json")
        return run.info.run_id
