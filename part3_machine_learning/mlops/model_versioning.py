"""
mlops/model_versioning.py — Part 3, Task 6.1: Model Versioning
==================================================================
Pulls every model version's parameters and metrics straight from the MLflow
tracking store (Task 4) and writes a human-readable comparison table to
model_versioning.md, so "model versioning" and "experiment tracking" share
one source of truth rather than being documented twice by hand.

Run with: python model_versioning.py
"""
import sys
from pathlib import Path

import mlflow
import pandas as pd

for _candidate in (Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts"):
    if (_candidate / "common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
from common import MLFLOW_TRACKING_URI, PART3_DIR

REPORT_PATH = Path(__file__).resolve().parent / "model_versioning.md" if "__file__" in globals() \
    else Path.cwd() / "model_versioning.md"


def build_comparison_table() -> pd.DataFrame:
    """Pulls every run logged across all Part 3 MLflow experiments and
    returns one tidy comparison table — the single source both `main()`
    and the companion notebook build on, so the table is never computed
    two different ways."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    all_runs = []
    for exp in mlflow.search_experiments():
        if exp.name == "Default":
            continue
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        runs["experiment"] = exp.name
        all_runs.append(runs)
    runs_df = pd.concat(all_runs, ignore_index=True)

    metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
    param_cols = [c for c in runs_df.columns if c.startswith("params.")]
    display_cols = ["experiment", "tags.mlflow.runName", "start_time"] + param_cols + metric_cols
    table = runs_df[display_cols].drop_duplicates(subset=["tags.mlflow.runName"] + metric_cols)
    table.columns = [c.replace("metrics.", "").replace("params.", "").replace("tags.mlflow.runName", "run_name")
                      for c in table.columns]
    table = table.sort_values(["experiment", "run_name"])
    return table


def main() -> int:
    table = build_comparison_table()

    lines = [
        "# Part 3, Task 6.1 — Model Versioning\n",
        "Every model version trained in this capstone, pulled directly from the MLflow "
        "tracking store (`mlflow/mlflow.db`) rather than hand-maintained, so this table "
        "can never drift out of sync with what was actually logged.\n",
        table.to_markdown(index=False),
        "\n## Recommended production candidates\n",
        "- **Classification (accident-risk proxy):** `RandomForestClassifier` — highest "
        "precision (0.86) at near-identical recall (0.99) to Logistic Regression, meaning "
        "meaningfully fewer false alarms for the same detection rate.",
        "- **Regression (traffic volume, no lag features):** `GradientBoostingRegressor` — "
        "R²=0.94 vs. Linear Regression's 0.71.",
        "- **Sequential / best-available (traffic volume, with recent history):** the "
        "lag-feature `RandomForestRegressor` (R²=0.98) slightly out-performs the LSTM "
        "(R²=0.97) on this dataset size and is far cheaper to retrain — the LSTM remains "
        "valuable as the deep-learning deliverable and for its natural fit to sequential "
        "extensions (e.g. multi-step-ahead forecasting), but is not the pick for the "
        "single-model deployment API given no measured advantage today.",
        "\nThe deployment API (`deployment/api.py`) currently serves the "
        "`GradientBoostingRegressor` (volume) and `RandomForestClassifier` (risk) — the "
        "two selected candidates above.",
    ]
    report = "\n".join(lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
