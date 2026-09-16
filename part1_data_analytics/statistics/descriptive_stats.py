"""
Part 1, Task 2 — Descriptive Statistics and Correlation
==========================================================
Computes traffic-volume descriptive statistics and the temperature <-> traffic
volume correlation, using the de-duplicated hourly dataset (one row per hour)
built in run_sql_analysis.py so results are consistent with Task 1.

Run with: python descriptive_stats.py
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "sql" / "traffic.db"
CSV_PATH = BASE_DIR.parent.parent / "Metro_Interstate_Traffic_Volume.csv"
REPORT_PATH = BASE_DIR / "descriptive_stats_results.md"


def load_hourly() -> pd.DataFrame:
    """Reuse the de-duplicated hourly view from the SQLite DB if it exists
    (built by run_sql_analysis.py); otherwise fall back to de-duplicating the
    raw CSV directly on date_time so this script also runs standalone."""
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM traffic_hourly", conn)
            df["date_time"] = pd.to_datetime(df["date_time"])
            return df
        finally:
            conn.close()
    df = pd.read_csv(CSV_PATH)
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df.sort_values("date_time").drop_duplicates(subset="date_time", keep="first")


def descriptive_statistics(df: pd.DataFrame) -> dict:
    v = df["traffic_volume"]
    return {
        "mean": v.mean(),
        "median": v.median(),
        "std": v.std(ddof=1),
        "variance": v.var(ddof=1),
        "range": v.max() - v.min(),
        "min": v.min(),
        "max": v.max(),
        "count": v.shape[0],
    }


def correlation_analysis(df: pd.DataFrame) -> dict:
    # temp == 0 Kelvin is a known impossible sensor error (see Part 2 cleaning) —
    # excluded here so it doesn't distort the correlation coefficient.
    clean = df[df["temp"] > 0]
    r, p_value = stats.pearsonr(clean["temp"], clean["traffic_volume"])
    return {
        "r": r,
        "p_value": p_value,
        "n": len(clean),
        "n_excluded_zero_temp": len(df) - len(clean),
    }


def interpret_strength(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    if a < 0.7:
        return "strong"
    return "very strong"


def build_report(stats_d: dict, corr: dict) -> str:
    lines = []
    lines.append("# Part 1, Task 2 — Descriptive Statistics and Correlation: Results\n")

    lines.append("## 2.1 Traffic Volume Statistics\n")
    lines.append(f"- **N (hours):** {stats_d['count']:,}")
    lines.append(f"- **Mean:** {stats_d['mean']:.1f} vehicles/hour")
    lines.append(f"- **Median:** {stats_d['median']:.1f} vehicles/hour")
    lines.append(f"- **Standard deviation:** {stats_d['std']:.1f}")
    lines.append(f"- **Variance:** {stats_d['variance']:.1f}")
    lines.append(f"- **Range:** {stats_d['range']:.0f} (min {stats_d['min']:.0f}, max {stats_d['max']:.0f})\n")

    lines.append("**Interpretation:**\n")
    skew_note = (
        "The mean is noticeably higher than the median, which indicates a **right-skewed "
        "distribution** — most hours (overnight, off-peak) carry low-to-moderate volume, while "
        "a smaller number of peak-hour observations pull the mean upward."
        if stats_d["mean"] > stats_d["median"] * 1.05
        else "The mean and median are close, suggesting a fairly symmetric distribution."
    )
    lines.append(
        f"{skew_note} The standard deviation ({stats_d['std']:.0f}) is very large relative to the "
        f"mean ({stats_d['mean']:.0f}) — a coefficient of variation of "
        f"{stats_d['std']/stats_d['mean']*100:.0f}% — which tells us traffic volume swings "
        "dramatically within a day (near-zero overnight vs. several thousand vehicles/hour at "
        "peak), rather than sitting in a narrow band. The full range "
        f"({stats_d['min']:.0f}–{stats_d['max']:.0f}) confirms the corridor experiences "
        "everything from essentially empty roads to near-capacity flow, so any single "
        "'typical volume' figure understates how much hour-of-day and day-of-week matter — "
        "single summary statistics should always be read alongside the hourly/weekly pattern, "
        "not instead of it.\n"
    )

    lines.append("## 2.2 Correlation Analysis: Temperature vs. Traffic Volume\n")
    lines.append(
        f"- **Pearson correlation coefficient (r):** {corr['r']:.4f} "
        f"(p = {corr['p_value']:.3g}, n = {corr['n']:,})"
    )
    lines.append(
        f"- {corr['n_excluded_zero_temp']} rows with an impossible **0 Kelvin** temperature "
        "reading (a known sensor error, corrected properly in Part 2) were excluded from this "
        "calculation so they don't distort the coefficient.\n"
    )
    direction = "positive" if corr["r"] > 0 else "negative"
    lines.append("**Interpretation:**\n")
    lines.append(
        f"- **Direction:** The relationship is **{direction}** — "
        f"{'warmer temperatures are associated with slightly higher traffic volume' if corr['r']>0 else 'warmer temperatures are associated with slightly lower traffic volume'}."
    )
    lines.append(
        f"- **Strength:** r = {corr['r']:.3f} is a **{interpret_strength(corr['r'])}** correlation "
        "in conventional terms (|r| well below 0.3). Temperature explains only "
        f"{corr['r']**2*100:.1f}% of the variance in traffic volume (R²), so temperature alone "
        "is a poor predictor of how busy the road will be — hour-of-day and day-of-week almost "
        "certainly matter far more (confirmed by the wide swings seen in the descriptive "
        "statistics above, and explored further with Python feature engineering in Part 2)."
    )
    lines.append(
        "- **Correlation vs. causation:** even if the correlation were stronger, this figure "
        "alone would not establish that temperature *causes* traffic changes. Both variables "
        "are driven by shared underlying factors — for example, season and time-of-day jointly "
        "affect both typical temperature and typical commuter behaviour (e.g. school terms, "
        "daylight hours, holiday timing). A cold snap and a traffic dip could both simply be "
        "consequences of 'it's a winter holiday period' rather than one causing the other. "
        "Establishing causation would require controlling for these confounders (or a "
        "controlled/quasi-experimental design), which a simple correlation coefficient cannot do.\n"
    )
    return "\n".join(lines)


def main():
    df = load_hourly()
    stats_d = descriptive_statistics(df)
    corr = correlation_analysis(df)
    report = build_report(stats_d, corr)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
