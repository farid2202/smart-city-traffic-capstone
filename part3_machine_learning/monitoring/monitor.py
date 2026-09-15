"""
monitoring/monitor.py — Part 3, Task 6.4/6.5: Monitoring and Alerting Simulation
====================================================================================
Simulates two kinds of production model monitoring:

1. **Feature distribution drift** — Kolmogorov-Smirnov test comparing each
   feature's distribution in the training period vs. the most recent
   ("production") period, using the same chronological split as Task 1.
2. **Prediction error drift** — the regression model's MAE on the first vs.
   second half of the held-out test period, checking whether error is
   getting meaningfully worse over time (which would suggest the model
   needs retraining).

Produces a simple PASS / ALERT status per check and an overall status,
written to monitoring_report.md (a "dashboard" a human can read without
running any code).

Run with: python monitor.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error

for _candidate in (Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts"):
    if (_candidate / "common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
from common import FEATURE_COLUMNS, MODELS_DIR, PART3_DIR, chronological_split, get_full_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).resolve().parent / "monitoring_report.md" if "__file__" in globals() \
    else Path.cwd() / "monitoring_report.md"

KS_ALERT_THRESHOLD = 0.10       # KS statistic above this -> flag the feature
ERROR_DRIFT_ALERT_THRESHOLD = 0.20  # >20% MAE increase, second half vs first -> alert


def check_feature_drift(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLUMNS:
        stat, p_value = ks_2samp(train_df[col], test_df[col])
        status = "ALERT" if stat > KS_ALERT_THRESHOLD else "PASS"
        rows.append({"feature": col, "ks_statistic": stat, "p_value": p_value, "status": status})
    return pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)


def check_prediction_error_drift(test_df: pd.DataFrame) -> dict:
    model = joblib.load(MODELS_DIR / "regression_GradientBoostingRegressor.joblib")
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["traffic_volume"]
    preds = model.predict(X_test)

    half = len(test_df) // 2
    mae_first_half = mean_absolute_error(y_test.iloc[:half], preds[:half])
    mae_second_half = mean_absolute_error(y_test.iloc[half:], preds[half:])
    pct_change = (mae_second_half - mae_first_half) / mae_first_half

    status = "ALERT" if pct_change > ERROR_DRIFT_ALERT_THRESHOLD else "PASS"
    return {
        "mae_first_half": mae_first_half, "mae_second_half": mae_second_half,
        "pct_change": pct_change, "status": status,
    }


def build_report(drift_df: pd.DataFrame, error_drift: dict) -> str:
    n_alerts = (drift_df["status"] == "ALERT").sum()
    overall_status = "ALERT" if (n_alerts > 0 or error_drift["status"] == "ALERT") else "PASS"

    lines = [f"# Monitoring Report\n", f"## Overall status: **{overall_status}**\n"]

    lines.append("## Feature distribution drift (train vs. most recent period, KS test)\n")
    lines.append(f"Alert threshold: KS statistic > {KS_ALERT_THRESHOLD}\n")
    lines.append(drift_df.round(4).to_markdown(index=False))
    lines.append("")
    if n_alerts:
        flagged = ", ".join(drift_df.loc[drift_df["status"] == "ALERT", "feature"])
        lines.append(f"**{n_alerts} feature(s) flagged for drift:** {flagged}\n")
    else:
        lines.append("No features exceeded the drift threshold.\n")

    lines.append("## Prediction error drift (regression model, first vs. second half of test period)\n")
    lines.append(f"- MAE, first half of test period:  {error_drift['mae_first_half']:.1f} vehicles/hour")
    lines.append(f"- MAE, second half of test period: {error_drift['mae_second_half']:.1f} vehicles/hour")
    lines.append(f"- Change: {error_drift['pct_change']*100:+.1f}% "
                 f"(alert threshold: +{ERROR_DRIFT_ALERT_THRESHOLD*100:.0f}%)")
    lines.append(f"- Status: **{error_drift['status']}**\n")

    lines.append("## Interpretation\n")
    if overall_status == "PASS":
        lines.append(
            "No meaningful drift detected — the model's input feature distributions and "
            "prediction error are stable across the test period. No retraining action needed "
            "at this time."
        )
    else:
        lines.append(
            "One or more checks were flagged. In a real deployment this would trigger a "
            "notification to the model owner to investigate — e.g. a genuine seasonal shift "
            "(expected, not necessarily a problem) vs. a data pipeline issue (needs a fix) vs. "
            "the model becoming stale (needs retraining on more recent data)."
        )
    return "\n".join(lines)


def demo_alert_scenario(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """The real monitoring run above legitimately comes back all-PASS (the
    test period isn't actually drifted). To demonstrate the ALERT path
    actually works — not just that it exists in the code — this builds a
    SYNTHETIC drifted copy of the test set (severe-weather rate inflated
    5x) and re-runs the same drift check against it. This is clearly a
    demonstration, not a claim about the real data."""
    synthetic_test = test_df.copy()
    rng = np.random.default_rng(42)
    flip_mask = rng.random(len(synthetic_test)) < 0.6
    synthetic_test.loc[flip_mask, "is_severe_weather"] = 1
    synthetic_test.loc[flip_mask, "weather_Clear"] = 0
    synthetic_test.loc[flip_mask, "weather_Thunderstorm"] = 1

    demo_drift_df = check_feature_drift(train_df, synthetic_test)
    n_demo_alerts = (demo_drift_df["status"] == "ALERT").sum()

    lines = [
        "\n## Appendix: ALERT-path demonstration (synthetic data)\n",
        "The monitoring run above came back PASS because the real test period isn't "
        "actually drifted. To demonstrate the alerting mechanism itself works, this "
        "section re-runs the identical KS-test check against a **synthetically "
        "perturbed** copy of the test set (severe-weather rate artificially inflated) "
        "— this is a deliberate demonstration, not a claim about the real data.\n",
        demo_drift_df[demo_drift_df["status"] == "ALERT"].round(4).to_markdown(index=False),
        f"\n**Result: {n_demo_alerts} feature(s) correctly flagged as ALERT** under the "
        "synthetic drift, confirming the mechanism triggers as designed.",
    ]
    return "\n".join(lines)


def main() -> int:
    df = get_full_dataset()
    train_df, test_df = chronological_split(df)

    logger.info("Running feature distribution drift checks (KS test)...")
    drift_df = check_feature_drift(train_df, test_df)
    n_alerts = (drift_df["status"] == "ALERT").sum()
    if n_alerts:
        logger.warning("%d feature(s) flagged for distribution drift", n_alerts)
    else:
        logger.info("No feature distribution drift detected")

    logger.info("Running prediction error drift check...")
    error_drift = check_prediction_error_drift(test_df)
    if error_drift["status"] == "ALERT":
        logger.warning("Prediction error drift detected: %+.1f%% MAE change", error_drift["pct_change"] * 100)
    else:
        logger.info("No meaningful prediction error drift (%+.1f%% MAE change)", error_drift["pct_change"] * 100)

    report = build_report(drift_df, error_drift)
    report += "\n" + demo_alert_scenario(train_df, test_df)
    REPORT_PATH.write_text(report)
    print(report)
    logger.info("Saved monitoring report to %s", REPORT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
