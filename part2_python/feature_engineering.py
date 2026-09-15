"""
feature_engineering.py — Part 2, Task 2: Feature Engineering (NumPy and Pandas)
==================================================================================
Takes the cleaned dataset produced by pipeline.py and engineers an ML-ready
feature set: time features (including a cyclical encoding), weather encodings
and derived indicators, scaled numeric features, and a data-driven congestion
category used again in Part 3.

Run with:  python feature_engineering.py [--input PATH] [--output PATH]
(Run pipeline.py first if cleaned_traffic_data.csv doesn't exist yet.)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)  # never the bare root logger — see pipeline.py's configure_logging()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "cleaned_traffic_data.csv"
DEFAULT_OUTPUT = BASE_DIR / "featured_traffic_data.csv"

# Severe / low-visibility weather conditions, used both for the derived weather
# indicator here and for the Part 3 proxy accident-risk label.
SEVERE_WEATHER = {"Thunderstorm", "Snow", "Squall"}
LOW_VISIBILITY_WEATHER = {"Fog", "Mist", "Haze", "Smoke"}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_time"] = pd.to_datetime(df["date_time"])

    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # 0 = Monday ... 6 = Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical encoding of hour: a plain integer 0-23 implies hour 23 and hour 0
    # are 23 apart, when really they are 1 hour apart. Sine/cosine encoding
    # preserves that adjacency for any model that would otherwise be misled by
    # the discontinuity (also applied to day_of_week for the same reason).
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    logger.info(
        "Added time features: hour, day_of_week, is_weekend, and cyclical "
        "sin/cos encodings for hour and day_of_week"
    )
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # One-hot encode the categorical weather_main condition.
    weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather", dtype=int)
    df = pd.concat([df, weather_dummies], axis=1)

    # Derived indicators: two binary flags capturing "conditions bad enough to
    # plausibly affect driving", used again for the Part 3 proxy risk label.
    df["is_severe_weather"] = df["weather_main"].isin(SEVERE_WEATHER).astype(int)
    df["is_low_visibility"] = df["weather_main"].isin(LOW_VISIBILITY_WEATHER).astype(int)
    df["has_precipitation"] = ((df["rain_1h"] > 0) | (df["snow_1h"] > 0)).astype(int)

    logger.info(
        "Added weather features: %d one-hot columns for weather_main, plus "
        "derived indicators is_severe_weather, is_low_visibility, has_precipitation",
        weather_dummies.shape[1],
    )
    return df


def add_scaled_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale temp and clouds_all to [0, 1]. Implemented directly with
    NumPy/Pandas (rather than importing sklearn here) so the scaling logic is
    transparent; Part 3 uses sklearn's StandardScaler/pipelines directly where
    a fitted, reusable scaler object is actually needed for train/test splits."""
    df = df.copy()
    for col in ["temp", "clouds_all"]:
        col_min, col_max = df[col].min(), df[col].max()
        scaled_col = f"{col}_scaled"
        df[scaled_col] = (df[col] - col_min) / (col_max - col_min)
        logger.debug("Scaling %s: min=%.3f, max=%.3f -> %s in [0, 1]", col, col_min, col_max, scaled_col)

    logger.info("Added min-max scaled versions of: temp, clouds_all")
    return df


def add_congestion_category(df: pd.DataFrame) -> pd.DataFrame:
    """Data-driven congestion category based on quartiles of traffic_volume
    (matches the definition used again in Part 3 for the proxy accident-risk
    label, so the two parts stay consistent):
        <= Q1            -> Low
        Q1 < v <= Q2      -> Medium
        Q2 < v <= Q3      -> High
        > Q3             -> Severe
    Using data-driven quartiles (rather than fixed thresholds) means the
    category boundaries adapt automatically if the traffic pattern on this
    corridor changes over time.
    """
    df = df.copy()
    q1, q2, q3 = df["traffic_volume"].quantile([0.25, 0.5, 0.75]).values
    logger.debug("Congestion category thresholds — Q1=%.1f, Q2 (median)=%.1f, Q3=%.1f", q1, q2, q3)

    def bucket(v: float) -> str:
        if v <= q1:
            return "Low"
        elif v <= q2:
            return "Medium"
        elif v <= q3:
            return "High"
        return "Severe"

    df["congestion_category"] = df["traffic_volume"].apply(bucket)
    counts = df["congestion_category"].value_counts().to_dict()
    logger.info("Added data-driven 'congestion_category' (quartile-based): %s", counts)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Feature engineering starting — input shape: %d rows x %d columns", *df.shape)
    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_scaled_numeric_features(df)
    df = add_congestion_category(df)
    logger.info("Feature engineering complete — output shape: %d rows x %d columns", *df.shape)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Engineer ML-ready features for the traffic dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not logging.getLogger().handlers:
        # allows this script to also be run standalone (not just imported after
        # pipeline.py has configured logging) — see pipeline.configure_logging
        from pipeline import configure_logging
        configure_logging("INFO")

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        logger.error(
            "Cleaned input file not found at %s — run pipeline.py first", args.input, exc_info=True,
        )
        return 1

    df = engineer_features(df)
    df.to_csv(args.output, index=False)
    logger.info("Saved featured dataset to %s (%d rows, %d columns)", args.output, *df.shape)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
