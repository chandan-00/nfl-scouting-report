#!/usr/bin/env bash
# After the eval batch: retry failed reports, score, forward check, results write-up, grading
# packet. Every step after the retry reads saved runs only (no API calls).
#   scripts/after_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1

for c in full no_checker single; do
  # Re-runs only reports with no saved file (failures); prints the estimate first.
  uv run python -m nfl_scout.evaluation.run_eval --config "$c" --repeats 1 2 3 --yes
done
uv run python -m nfl_scout.evaluation.score --mlflow
uv run python -m nfl_scout.evaluation.forward
uv run python -m nfl_scout.evaluation.results
if [ ! -d runs/grading ]; then
  uv run python -m nfl_scout.evaluation.grading build
fi
echo "done: docs/results.md, runs/scores/, runs/grading/packet.md (MLflow: mlruns/mlflow.db)"
