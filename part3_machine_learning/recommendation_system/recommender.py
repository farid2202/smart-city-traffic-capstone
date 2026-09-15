"""
recommender.py — Part 3, Task 5: Traffic Recommendation System
==================================================================
Transforms the Part 3 Task 1 regression model's OUTPUTS into a practical
travel-timing recommendation system. Because the dataset represents a single
corridor (westbound I-94), this focuses on WHEN to travel, not which route
to take.

For a given day type, weather condition and holiday status, the trained
GradientBoostingRegressor model predicts traffic volume for every hour of
the day, and the lowest-predicted-volume hours (within a realistic 06:00-
22:00 travel window, matching the Part 2 CLI app's rationale) become the
recommended departure windows.

Run with:  python recommender.py --day-type weekday --weather Clear
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

for _candidate in (Path.cwd(), Path.cwd() / "scripts", Path.cwd().parent / "scripts"):
    if (_candidate / "common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
from common import FEATURE_COLUMNS, MODELS_DIR, WEATHER_OPTIONS, build_feature_frame

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "regression_GradientBoostingRegressor.joblib"


def _load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        logger.error(
            "Regression model not found at %s — run supervised_ml.py first", MODEL_PATH, exc_info=True,
        )
        raise SystemExit(1)


def recommend(day_type: str, weather: str = "Clear", is_holiday: bool = False,
              n_windows: int = 3, allow_overnight: bool = False) -> dict:
    model = _load_model()
    scenario = build_feature_frame(range(24), day_type, weather, is_holiday)
    scenario["predicted_volume"] = model.predict(scenario[FEATURE_COLUMNS])

    candidates = scenario if allow_overnight else scenario[(scenario["hour"] >= 6) & (scenario["hour"] <= 22)]
    best = candidates.nsmallest(n_windows, "predicted_volume")

    windows = [
        {"hour": int(r.hour), "predicted_volume": round(float(r.predicted_volume), 0)}
        for r in best.itertuples()
    ]
    best_hour = windows[0]["hour"]

    holiday_note = " (public holiday)" if is_holiday else ""
    message = (
        f"For a {day_type} journey{holiday_note} in {weather.lower()} weather, consider travelling "
        f"between {best_hour:02d}:00 and {(best_hour+1)%24:02d}:00, when model-predicted traffic "
        f"volumes are typically lower (~{windows[0]['predicted_volume']:.0f} vehicles/hour)."
    )
    logger.info(
        "Recommendation generated: day_type=%s weather=%s is_holiday=%s -> best_hour=%02d:00",
        day_type, weather, is_holiday, best_hour,
    )
    return {"day_type": day_type, "weather": weather, "is_holiday": is_holiday,
            "windows": windows, "message": message}


def main() -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Recommend low-traffic travel windows.")
    parser.add_argument("--day-type", required=True, choices=["weekday", "weekend"])
    parser.add_argument("--weather", default="Clear", choices=WEATHER_OPTIONS)
    parser.add_argument("--holiday", action="store_true")
    parser.add_argument("--allow-overnight", action="store_true")
    args = parser.parse_args()

    try:
        result = recommend(args.day_type, args.weather, args.holiday, allow_overnight=args.allow_overnight)
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        print(f"Error: {exc}")
        return 1

    print(f"\nRecommended low-traffic windows ({args.day_type}, {args.weather}"
          f"{', holiday' if args.holiday else ''}):")
    for w in result["windows"]:
        print(f"  {w['hour']:02d}:00-{(w['hour']+1)%24:02d}:00  ->  ~{w['predicted_volume']:.0f} vehicles/hour")
    print(f"\n{result['message']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
