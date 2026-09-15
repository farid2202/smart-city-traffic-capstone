"""
common.py — shared feature set and proxy-label construction for Part 3
==========================================================================
Reuses the cleaned + featured dataset produced in Part 2 (extends it rather
than re-deriving it from scratch), and builds a documented proxy
accident-risk label, since no real accident dataset was available for
this project.

Every Part 3 script imports FEATURE_COLUMNS from here so all models are
compared on exactly the same, consistently-built feature set.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent   # common.py is always run as an
                                                  # imported module, so __file__
                                                  # is reliable here even though
                                                  # it is NOT inside notebooks
                                                  # converted from the other
                                                  # scripts in this folder.
PART3_DIR = SCRIPTS_DIR.parent
PART2_FEATURED_CSV = PART3_DIR.parent / "part2_python" / "featured_traffic_data.csv"
MODELS_DIR = PART3_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MLFLOW_DIR = PART3_DIR / "mlflow"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}"

# Severe/low-visibility weather already encoded as is_severe_weather /
# is_low_visibility in Part 2's feature_engineering.py — reused here directly.

# The common, well-engineered feature set used by every Part 3 model.
# NOTE: traffic_volume, congestion_category and high_risk are deliberately
# EXCLUDED from the classification feature set — they are the classification
# target or built directly from it, so including any of them as predictors
# would leak the answer into the model (see report.md, "Modelling decisions").
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
WEATHER_FEATURES = [
    "weather_Clear", "weather_Clouds", "weather_Drizzle", "weather_Fog",
    "weather_Haze", "weather_Mist", "weather_Rain", "weather_Smoke",
    "weather_Snow", "weather_Squall", "weather_Thunderstorm",
    "is_severe_weather", "is_low_visibility", "has_precipitation",
]
HOLIDAY_FEATURE = ["is_holiday"]

FEATURE_COLUMNS = TIME_FEATURES + WEATHER_FEATURES + HOLIDAY_FEATURE


def load_features(path: Path = PART2_FEATURED_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    logger.info("Loaded Part 2 featured dataset from %s: %d rows, %d columns", path, *df.shape)
    return df


def add_proxy_accident_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """A record is HIGH RISK when high/severe congestion co-occurs with
    severe or low-visibility weather. This proxy is intended only to
    demonstrate an ML classification workflow on this dataset and
    is NOT a real prediction of accident likelihood — see
    responsible_ai_report.md for the documented risk of using it as such."""
    df = df.copy()
    high_congestion = df["congestion_category"].isin(["High", "Severe"])
    risky_weather = (df["is_severe_weather"] == 1) | (df["is_low_visibility"] == 1)
    df["high_risk"] = (high_congestion & risky_weather).astype(int)
    logger.info(
        "Added proxy 'high_risk' label: %d high-risk hours out of %d (%.1f%%)",
        df["high_risk"].sum(), len(df), df["high_risk"].mean() * 100,
    )
    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Split by TIME, not randomly — this is a time series, so a random
    shuffle would let the model 'see the future' via nearby rows in the
    training set. The model is trained on the earlier ~80% of the data and
    evaluated on the most recent ~20%, which is the realistic deployment
    scenario (predicting conditions the model has not seen yet)."""
    split_idx = int(len(df) * train_frac)
    train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    logger.info(
        "Chronological train/test split: train=%d rows (%s to %s), test=%d rows (%s to %s)",
        len(train), train["date_time"].min(), train["date_time"].max(),
        len(test), test["date_time"].min(), test["date_time"].max(),
    )
    return train, test


def get_full_dataset() -> pd.DataFrame:
    df = load_features()
    df = add_proxy_accident_risk_label(df)
    return df


# ---------------------------------------------------------------------------
# Shared scenario-to-feature-row builder, used by both the recommendation
# system and the deployment API so "what a (day_type, weather, hour,
# is_holiday) scenario looks like as a feature vector" is defined in exactly
# one place.
# ---------------------------------------------------------------------------
WEATHER_OPTIONS = [
    "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
    "Rain", "Smoke", "Snow", "Squall", "Thunderstorm",
]
SEVERE_WEATHER = {"Thunderstorm", "Snow", "Squall"}
LOW_VISIBILITY_WEATHER = {"Fog", "Mist", "Haze", "Smoke"}


def build_feature_frame(hours, day_type: str, weather: str, is_holiday: bool) -> pd.DataFrame:
    """One row per hour in `hours` with every column in FEATURE_COLUMNS,
    for the given day type / weather / holiday scenario."""
    if day_type not in ("weekday", "weekend"):
        raise ValueError(f"day_type must be 'weekday' or 'weekend', got {day_type!r}")
    if weather not in WEATHER_OPTIONS:
        raise ValueError(f"weather must be one of {WEATHER_OPTIONS}, got {weather!r}")

    df = pd.DataFrame({"hour": list(hours)})
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    dow = 5 if day_type == "weekend" else 2  # representative Sat / Wed
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = 1 if day_type == "weekend" else 0
    df["is_holiday"] = int(is_holiday)

    for w in WEATHER_OPTIONS:
        df[f"weather_{w}"] = int(w == weather)
    df["is_severe_weather"] = int(weather in SEVERE_WEATHER)
    df["is_low_visibility"] = int(weather in LOW_VISIBILITY_WEATHER)
    df["has_precipitation"] = int(weather in {"Rain", "Drizzle", "Snow", "Thunderstorm"})

    return df[["hour"] + FEATURE_COLUMNS]
