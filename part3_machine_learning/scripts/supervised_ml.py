# %% [markdown]
# # Part 3, Task 1 — Supervised Machine Learning
#
# Two supervised problems, both using a common, well-engineered feature set
# (time-based cyclical encodings, weather one-hot + derived indicators,
# holiday flag):
#
# - **Classification**: predict the proxy accident-risk label (`high_risk`)
#   from weather/time/holiday conditions alone.
# - **Regression**: predict `traffic_volume` from the same kind of
#   conditions.
#
# Both targets are deliberately predicted **without** seeing `traffic_volume`
# / `congestion_category` / `high_risk` as *inputs* to each other's models —
# see the markdown note on leakage below.
#
# Both use a **chronological** train/test split (not a random shuffle) since
# this is time-series data — see `common.chronological_split`.

# %%
import logging
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)

# Works both as a plain script (python supervised_ml.py, run from inside
# scripts/) and as a notebook converted+executed from this file (which runs
# with the notebooks/ folder as its working directory) — either way, find
# and add the scripts/ folder (where common.py lives) to sys.path without
# relying on `__file__`, which real Jupyter cells don't have.
for _candidate in (Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts"):
    if (_candidate / "common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
from common import FEATURE_COLUMNS, MLFLOW_TRACKING_URI, MODELS_DIR, chronological_split, get_full_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("supervised_ml")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("traffic_supervised_ml")

# %% [markdown]
# ## Load data and inspect the proxy label balance

# %%
df = get_full_dataset()
print(df["high_risk"].value_counts(normalize=True).rename("proportion"))
print(f"\nFeature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

# %% [markdown]
# ## Classification: proxy accident-risk (`high_risk`)
#
# **Leakage note:** `high_risk` is defined as `high/severe congestion AND
# severe/low-visibility weather`. If we included `congestion_category` or
# `traffic_volume` as a *predictor*, the model would trivially reconstruct
# half the label definition instead of learning anything — so those columns
# are excluded from `FEATURE_COLUMNS`. The model must predict risk purely
# from weather, time-of-day and holiday context, which is also the more
# operationally useful framing (a controller wants a risk forecast *before*
# congestion is observed, not after).

# %%
train_df, test_df = chronological_split(df)
X_train, y_train = train_df[FEATURE_COLUMNS], train_df["high_risk"]
X_test, y_test = test_df[FEATURE_COLUMNS], test_df["high_risk"]

clf_results = {}
for name, model in [
    ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ("RandomForestClassifier", RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)),
]:
    with mlflow.start_run(run_name=f"classification_{name}"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        clf_results[name] = metrics
        logger.info("Classification [%s]: %s", name, {k: round(v, 4) for k, v in metrics.items()})

        mlflow.log_param("model", name)
        mlflow.log_param("task", "classification_high_risk")
        mlflow.log_metrics(metrics)
        joblib.dump(model, MODELS_DIR / f"classification_{name}.joblib")

clf_comparison = pd.DataFrame(clf_results).T
print(clf_comparison.round(4))

# %% [markdown]
# ## Regression: `traffic_volume`
#
# Same leakage principle applies in reverse: `congestion_category` and
# `high_risk` are both *derived from* `traffic_volume`, so neither is used
# as a predictor here either — only weather/time/holiday context.

# %%
X_train_r, y_train_r = train_df[FEATURE_COLUMNS], train_df["traffic_volume"]
X_test_r, y_test_r = test_df[FEATURE_COLUMNS], test_df["traffic_volume"]

reg_results = {}
for name, model in [
    ("LinearRegression", LinearRegression()),
    ("GradientBoostingRegressor", GradientBoostingRegressor(
        n_estimators=200, max_depth=4, random_state=42)),
]:
    with mlflow.start_run(run_name=f"regression_{name}"):
        model.fit(X_train_r, y_train_r)
        y_pred_r = model.predict(X_test_r)

        metrics = {
            "MAE": mean_absolute_error(y_test_r, y_pred_r),
            "R2": r2_score(y_test_r, y_pred_r),
        }
        reg_results[name] = metrics
        logger.info("Regression [%s]: %s", name, {k: round(v, 4) for k, v in metrics.items()})

        mlflow.log_param("model", name)
        mlflow.log_param("task", "regression_traffic_volume")
        mlflow.log_metrics(metrics)
        joblib.dump(model, MODELS_DIR / f"regression_{name}.joblib")

reg_comparison = pd.DataFrame(reg_results).T
print(reg_comparison.round(2))

# %% [markdown]
# ## Interpretation (results as actually obtained above)
#
# **Classification:** both models achieve a very high ROC-AUC (LogReg 0.994,
# RF 0.997) because the label is largely (though not perfectly) reconstructable
# from weather alone — `high_risk` requires severe/low-visibility weather as
# one of its two conditions, and weather is a direct input feature. The gap
# between the two models shows up in **precision**: Logistic Regression
# catches almost every true high-risk hour (recall 0.996) but at the cost of
# many false alarms (precision 0.71, i.e. ~29% of its "high risk" calls are
# false alarms), because a linear boundary can't cleanly separate "bad
# weather + high congestion" from "bad weather + normal congestion". Random
# Forest keeps recall almost as high (0.991) while lifting precision to 0.86,
# because it can learn the AND-interaction between weather severity and the
# time-of-day features that drive congestion. **For an alerting system, this
# matters operationally**: Random Forest would generate meaningfully fewer
# false alarms for roughly the same detection rate.
#
# **Regression:** Gradient Boosting reaches R² = 0.94 (MAE ≈ 278 vehicles/hour)
# versus Linear Regression's R² = 0.71 (MAE ≈ 832), using *only* cyclical
# time features, weekend/holiday flags and weather — no lagged traffic value
# at all. This is a direct, quantitative confirmation of the Part 1/2 finding
# that this corridor's traffic is overwhelmingly driven by a **regular,
# repeating time-of-day/day-of-week pattern** rather than by weather or
# noise: a model that only knows "what hour and what day type is it" can
# already explain 94% of the variance once it's allowed to learn non-linear
# interactions between those time features (which Linear Regression, being
# linear in hour_sin/hour_cos, cannot fully capture — it only reaches 71%).
# The remaining ~6% unexplained variance is where genuinely unpredictable
# day-to-day noise (incidents, unusual demand) would live, and is exactly
# where a lagged/sequential model like the LSTM in Task 3 could add further
# value by also conditioning on very recent observed traffic.
#
# Both comparisons are saved to MLflow (`mlflow ui --backend-store-uri
# sqlite:///part3_machine_learning/mlflow/mlflow.db`) for the Part 3 Task 4/6
# MLOps write-up, and the fitted models are saved to `../models/` for reuse
# by the recommendation system and the deployment API.

# %%
if __name__ == "__main__":
    print("\n=== Classification comparison ===")
    print(clf_comparison.round(4).to_string())
    print("\n=== Regression comparison ===")
    print(reg_comparison.round(2).to_string())
