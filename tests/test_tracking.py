"""MLflow logging of a config's aggregate scores (local SQLite store under tmp)."""

import mlflow

from nfl_scout.config import load_settings
from nfl_scout.evaluation.tracking import flatten, log_config

RESULT = {
    "aggregate": {
        "runs": 2,
        "draft_accuracy": 0.8,
        "draft_accuracy_ci": [0.6, 1.0],
        "final_accuracy": None,
        "seeded_by_kind": {"wrong_team": 1.0, "wrong_value": 0.5},
    },
    "runs": [{"matchup": "m1"}],
}


def test_flatten_keeps_numbers_and_names_nested_ones():
    assert flatten(RESULT["aggregate"]) == {
        "runs": 2,
        "draft_accuracy": 0.8,
        "draft_accuracy_ci_lo": 0.6,
        "draft_accuracy_ci_hi": 1.0,
        "seeded_by_kind.wrong_team": 1.0,
        "seeded_by_kind.wrong_value": 0.5,
    }


def test_log_config_records_params_metrics_and_scores(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = log_config("full", RESULT, load_settings(), tracking_uri=uri, artifact_root=tmp_path)
    client = mlflow.MlflowClient(uri)
    run = client.get_run(run_id)
    assert run.data.metrics["draft_accuracy"] == 0.8
    assert run.data.metrics["seeded_by_kind.wrong_value"] == 0.5
    assert run.data.params["config"] == "full"
    assert run.data.params["model.offense_analyst"] == "claude-haiku-5-5"
    assert "scores.json" in [a.path for a in client.list_artifacts(run_id)]
