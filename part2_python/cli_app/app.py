"""
cli_app/app.py — Part 2, Task 4: Mini Traffic Analytics Application
========================================================================
A small command-line application for querying the processed traffic dataset.
print() is used ONLY for the final answer shown to the end user, never for
internal status/progress — that all goes through logging.

Commands:
  query-datetime          Traffic + weather conditions at (or nearest to) a given date/time
  high-traffic            The N highest-traffic hours-of-day, on average
  compare-weekday-weekend Weekday vs. weekend traffic comparison
  recommend-travel        Lowest-traffic recommended travel windows for a day type

Examples:
  python app.py query-datetime --datetime "2017-07-04 08:00:00"
  python app.py high-traffic --top 5
  python app.py compare-weekday-weekend
  python app.py recommend-travel --day-type weekend
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "featured_traffic_data.csv"


def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        logger.error(
            "Processed data file not found at %s — run pipeline.py then "
            "feature_engineering.py first", DATA_PATH, exc_info=True,
        )
        raise SystemExit(1)
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df


def cmd_query_datetime(df: pd.DataFrame, args: argparse.Namespace) -> None:
    try:
        target = pd.to_datetime(args.datetime)
    except (ValueError, TypeError):
        logger.error("Invalid --datetime value received: %r (expected e.g. '2017-07-04 08:00:00')", args.datetime)
        print(f"Error: '{args.datetime}' is not a valid date/time. Expected format: YYYY-MM-DD HH:MM:SS")
        return

    df = df.assign(_diff=(df["date_time"] - target).abs())
    row = df.loc[df["_diff"].idxmin()]

    if row["_diff"] > pd.Timedelta(hours=1):
        print(f"No record exactly at {target}; showing the nearest available record: {row['date_time']}")
    print(f"Date/time      : {row['date_time']}")
    print(f"Traffic volume : {row['traffic_volume']:.0f} vehicles/hour ({row['congestion_category']})")
    print(f"Weather        : {row['weather_main']} ({row['weather_description']})")
    print(f"Temperature    : {row['temp']-273.15:.1f}°C")
    print(f"Day type       : {'Weekend' if row['is_weekend'] else 'Weekday'}")


def cmd_high_traffic(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if args.top <= 0:
        logger.error("Invalid --top value received: %s (must be a positive integer)", args.top)
        print("Error: --top must be a positive integer.")
        return

    hourly_avg = df.groupby("hour")["traffic_volume"].mean().sort_values(ascending=False)
    top_n = hourly_avg.head(args.top)
    print(f"Top {args.top} highest-traffic hours of day (by average volume):")
    for hour, vol in top_n.items():
        print(f"  {hour:02d}:00  ->  {vol:.0f} vehicles/hour")


def cmd_compare_weekday_weekend(df: pd.DataFrame, args: argparse.Namespace) -> None:
    weekday_avg = df.loc[df["is_weekend"] == 0, "traffic_volume"].mean()
    weekend_avg = df.loc[df["is_weekend"] == 1, "traffic_volume"].mean()
    diff = weekday_avg - weekend_avg
    pct = diff / weekend_avg * 100
    print(f"Average weekday traffic : {weekday_avg:.0f} vehicles/hour")
    print(f"Average weekend traffic : {weekend_avg:.0f} vehicles/hour")
    print(f"Difference              : {diff:+.0f} vehicles/hour ({pct:+.1f}% vs. weekend)")


def cmd_recommend_travel(df: pd.DataFrame, args: argparse.Namespace) -> None:
    day_type = args.day_type.lower()
    if day_type not in ("weekday", "weekend"):
        logger.error("Invalid --day-type value received: %r (expected 'weekday' or 'weekend')", args.day_type)
        print("Error: --day-type must be 'weekday' or 'weekend'.")
        return

    is_weekend_flag = 1 if day_type == "weekend" else 0
    subset = df[df["is_weekend"] == is_weekend_flag]
    if not args.allow_overnight:
        # Restrict to a realistic travel window (06:00-22:00) — a technically-
        # correct "travel at 2am" recommendation isn't a useful one for a
        # commuter; --allow-overnight opts back into the full 24 hours.
        subset = subset[(subset["hour"] >= 6) & (subset["hour"] <= 22)]
    hourly_avg = subset.groupby("hour")["traffic_volume"].mean().sort_values()
    best_hours = hourly_avg.head(3)

    print(f"Recommended low-traffic travel windows for a {day_type} journey:")
    for hour, vol in best_hours.items():
        print(f"  {hour:02d}:00-{(hour+1)%24:02d}:00  ->  ~{vol:.0f} vehicles/hour (historically light)")
    best_hour = best_hours.index[0]
    print(
        f"\nFor a {day_type} journey, consider travelling between {best_hour:02d}:00 and "
        f"{(best_hour+1)%24:02d}:00, when historical traffic volumes are typically lower."
    )


COMMANDS = {
    "query-datetime": cmd_query_datetime,
    "high-traffic": cmd_high_traffic,
    "compare-weekday-weekend": cmd_compare_weekday_weekend,
    "recommend-travel": cmd_recommend_travel,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini Traffic Analytics Application")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("query-datetime", help="Traffic + weather at a given date/time")
    p1.add_argument("--datetime", required=True, help="e.g. '2017-07-04 08:00:00'")

    p2 = sub.add_parser("high-traffic", help="Top N highest-traffic hours of day")
    p2.add_argument("--top", type=int, default=5)

    sub.add_parser("compare-weekday-weekend", help="Weekday vs. weekend traffic comparison")

    p4 = sub.add_parser("recommend-travel", help="Recommended low-traffic travel windows")
    p4.add_argument("--day-type", default="weekday", help="'weekday' or 'weekend'")
    p4.add_argument("--allow-overnight", action="store_true",
                     help="Include 22:00-06:00 hours (excluded by default as impractical commute windows)")

    return parser


def main() -> int:
    if not logging.getLogger().handlers:
        sys.path.insert(0, str(BASE_DIR.parent))
        from pipeline import configure_logging
        configure_logging("INFO")

    parser = build_parser()
    args = parser.parse_args()

    logger.info("Command invoked: %s | args: %s", args.command, vars(args))

    try:
        df = load_data()
        COMMANDS[args.command](df, args)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Command '%s' failed: %s", args.command, exc, exc_info=True)
        print(f"Error: could not complete '{args.command}'. See pipeline.log for details.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
