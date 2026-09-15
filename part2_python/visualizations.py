"""
visualizations.py — Part 2, Task 3: Visualise Traffic Patterns
==================================================================
Produces 4 Matplotlib visualisations from the featured dataset and saves them
to figures/. Each save is logged at INFO level with its file path (this is
the confirmation trail that all expected output files were produced).

Run with:  python visualizations.py [--input PATH] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed to save PNGs
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "featured_traffic_data.csv"
DEFAULT_FIG_DIR = BASE_DIR / "figures"

INTERPRETATIONS = {}  # collected as each figure is built, written out to figure_interpretations.md


def _save(fig, path: Path, interpretation: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    INTERPRETATIONS[path.name] = interpretation
    logger.info("Saved figure to %s", path)


def plot_traffic_by_hour(df: pd.DataFrame, out_dir: Path) -> None:
    hourly = df.groupby("hour")["traffic_volume"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hourly.index, hourly.values, marker="o", color="#2b6cb0")
    ax.set_title("Average Traffic Volume by Hour of Day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average traffic volume (vehicles/hour)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    peak_hour = hourly.idxmax()
    _save(fig, out_dir / "01_traffic_by_hour.png", (
        f"Traffic follows a clear bimodal commute pattern, peaking around "
        f"{peak_hour}:00. Volume is lowest overnight (roughly 00:00-04:00), "
        "consistent with a corridor dominated by commuter rather than "
        "overnight freight/leisure traffic."
    ))


def plot_weekday_vs_weekend(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.groupby(["hour", "is_weekend"])["traffic_volume"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pivot.index, pivot[0], marker="o", label="Weekday", color="#2b6cb0")
    ax.plot(pivot.index, pivot[1], marker="o", label="Weekend", color="#dd6b20")
    ax.set_title("Weekday vs. Weekend Traffic by Hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average traffic volume (vehicles/hour)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out_dir / "02_weekday_vs_weekend.png", (
        "Weekday traffic shows two sharp commute peaks (morning and evening) "
        "that weekend traffic does not: weekends instead have a single, "
        "flatter midday bulge. This confirms the corridor's traffic is "
        "driven primarily by work-commute patterns rather than general "
        "leisure travel, and is one of the strongest predictive signals "
        "available for the Part 3 models."
    ))


def plot_traffic_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["traffic_volume"], bins=40, color="#2b6cb0", edgecolor="white")
    ax.axvline(df["traffic_volume"].mean(), color="#dd6b20", linestyle="--", label=f"Mean = {df['traffic_volume'].mean():.0f}")
    ax.axvline(df["traffic_volume"].median(), color="#38a169", linestyle="--", label=f"Median = {df['traffic_volume'].median():.0f}")
    ax.set_title("Distribution of Traffic Volume")
    ax.set_xlabel("Traffic volume (vehicles/hour)")
    ax.set_ylabel("Frequency (hours)")
    ax.legend()
    _save(fig, out_dir / "03_traffic_distribution.png", (
        "The distribution is bimodal rather than a simple bell curve: a "
        "cluster of low-traffic overnight hours, and a separate cluster of "
        "moderate-to-high daytime hours. This bimodality is exactly why the "
        "mean/median alone (Part 1, Task 2) understate how variable traffic "
        "really is — a single 'typical volume' number blends two very "
        "different regimes (night vs. day) into one figure."
    ))


def plot_weather_impact(df: pd.DataFrame, out_dir: Path) -> None:
    weather_avg = df.groupby("weather_main")["traffic_volume"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(weather_avg.index, weather_avg.values, color="#2b6cb0")
    ax.set_title("Average Traffic Volume by Weather Condition")
    ax.set_xlabel("Weather condition (weather_main)")
    ax.set_ylabel("Average traffic volume (vehicles/hour)")
    ax.tick_params(axis="x", rotation=45)
    highest, lowest = weather_avg.index[0], weather_avg.index[-1]
    diff = weather_avg.iloc[0] - weather_avg.iloc[-1]
    _save(fig, out_dir / "04_weather_impact.png", (
        f"'{highest}' has the highest average traffic volume and '{lowest}' the "
        f"lowest, a difference of {diff:.0f} vehicles/hour. The overall spread "
        "across weather categories is modest relative to the hour-of-day "
        "effect seen in Figure 1 — consistent with the weak temperature-"
        "traffic correlation (r ≈ 0.14) and near-chance congestion/weather "
        "independence found in the Part 1 probability analysis: weather "
        "shifts traffic volume only at the margins."
    ))


def build_all_visualizations(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_traffic_by_hour(df, out_dir)
    plot_weekday_vs_weekend(df, out_dir)
    plot_traffic_distribution(df, out_dir)
    plot_weather_impact(df, out_dir)

    md_path = out_dir / "figure_interpretations.md"
    lines = ["# Figure Interpretations\n"]
    for fname, note in INTERPRETATIONS.items():
        lines.append(f"### `{fname}`\n{note}\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved figure interpretations to %s", md_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate traffic pattern visualisations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    if not logging.getLogger().handlers:
        from pipeline import configure_logging
        configure_logging("INFO")

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        logger.error(
            "Featured input file not found at %s — run pipeline.py then "
            "feature_engineering.py first", args.input, exc_info=True,
        )
        return 1

    build_all_visualizations(df, args.output_dir)
    logger.info("All visualisations generated successfully.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
