"""
Part 1, Task 3 — Probability and Congestion Analysis
=======================================================
Congestion is defined as traffic_volume > 5,500 vehicles/hour.
High temperature is defined as temp > 292K.

Run with: python probability_analysis.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "sql" / "traffic.db"
CSV_PATH = BASE_DIR.parent.parent / "Metro_Interstate_Traffic_Volume.csv"
REPORT_PATH = BASE_DIR / "probability_analysis_results.md"

CONGESTION_THRESHOLD = 5500
HIGH_TEMP_THRESHOLD = 292


def load_hourly() -> pd.DataFrame:
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql("SELECT * FROM traffic_hourly", conn)
            return df
        finally:
            conn.close()
    df = pd.read_csv(CSV_PATH)
    return df.sort_values("date_time").drop_duplicates(subset="date_time", keep="first")


def main():
    df = load_hourly()
    df = df[df["temp"] > 0]  # drop the known 0K sensor-error rows (see Part 2 cleaning)
    n = len(df)

    congestion = df["traffic_volume"] > CONGESTION_THRESHOLD
    clear = df["weather_main"] == "Clear"
    cloudy = df["weather_main"] == "Clouds"
    high_temp = df["temp"] > HIGH_TEMP_THRESHOLD

    p_congestion = congestion.mean()
    p_clear = clear.mean()
    p_congestion_and_clear = (congestion & clear).mean()

    p_clear_given_congestion = (congestion & clear).sum() / congestion.sum()
    p_hightemp_given_congestion = (congestion & high_temp).sum() / congestion.sum()

    independent_expected = p_congestion * p_clear
    independent_actual = p_congestion_and_clear
    independence_gap = independent_actual - independent_expected

    # Odds ratio: congestion in clear vs. cloudy weather
    n_clear = clear.sum()
    n_cloudy = cloudy.sum()
    congestion_in_clear = (congestion & clear).sum()
    congestion_in_cloudy = (congestion & cloudy).sum()
    odds_clear = congestion_in_clear / (n_clear - congestion_in_clear)
    odds_cloudy = congestion_in_cloudy / (n_cloudy - congestion_in_cloudy)
    odds_ratio = odds_clear / odds_cloudy

    lines = []
    lines.append("# Part 1, Task 3 — Probability and Congestion Analysis: Results\n")
    lines.append(f"Definitions: **Congestion** = traffic_volume > {CONGESTION_THRESHOLD}; "
                  f"**High temperature** = temp > {HIGH_TEMP_THRESHOLD}K "
                  f"({HIGH_TEMP_THRESHOLD-273.15:.1f}°C). N = {n:,} hourly records "
                  "(0K sensor-error rows excluded).\n")

    lines.append("## 3.1 Basic Probability\n")
    lines.append(f"- **P(Congestion)** = {congestion.sum():,} / {n:,} = **{p_congestion:.4f}** "
                  f"({p_congestion*100:.1f}% of hours are congested)")
    lines.append(f"- **P(Clear Weather)** = {clear.sum():,} / {n:,} = **{p_clear:.4f}** "
                  f"({p_clear*100:.1f}% of hours are clear)")
    lines.append(f"- **P(Congestion AND Clear Weather)** = {(congestion & clear).sum():,} / {n:,} = "
                  f"**{p_congestion_and_clear:.4f}**\n")

    lines.append("## 3.2 Conditional Probability\n")
    lines.append(f"- **P(Clear Weather | Congestion)** = {(congestion & clear).sum():,} / "
                  f"{congestion.sum():,} = **{p_clear_given_congestion:.4f}**")
    lines.append(f"- **P(High Temperature | Congestion)** = {(congestion & high_temp).sum():,} / "
                  f"{congestion.sum():,} = **{p_hightemp_given_congestion:.4f}**\n")

    lines.append("**Independence check** — P(A ∩ B) vs. P(A) × P(B), for A = Congestion, B = Clear Weather:")
    lines.append(f"- P(Congestion) × P(Clear) = {p_congestion:.4f} × {p_clear:.4f} = **{independent_expected:.4f}**")
    lines.append(f"- Observed P(Congestion ∩ Clear) = **{independent_actual:.4f}**")
    lines.append(f"- Gap = {independence_gap:+.4f} "
                 f"({'events are NOT independent' if abs(independence_gap) > 0.001 else 'events appear independent'}) — "
                 f"{'observed joint probability exceeds the independence expectation, so congestion and clear weather co-occur slightly more than chance would predict' if independence_gap > 0 else 'observed joint probability is below the independence expectation, so congestion is somewhat less likely than chance during clear weather'}.\n")

    lines.append(f"**Odds ratio (congestion: clear vs. cloudy weather)**")
    lines.append(f"- Odds of congestion given Clear = {congestion_in_clear:,}/{n_clear-congestion_in_clear:,} = {odds_clear:.4f}")
    lines.append(f"- Odds of congestion given Clouds = {congestion_in_cloudy:,}/{n_cloudy-congestion_in_cloudy:,} = {odds_cloudy:.4f}")
    lines.append(f"- **Odds ratio = {odds_ratio:.3f}**\n")

    lines.append("## Conclusion\n")
    lines.append(
        f"Congestion occurs in about {p_congestion*100:.0f}% of recorded hours overall, and "
        f"clear weather in about {p_clear*100:.0f}%. The gap between the observed joint "
        f"probability ({independent_actual:.4f}) and the independence expectation "
        f"({independent_expected:.4f}) shows that weather and congestion are "
        f"**{'not strictly independent' if abs(independence_gap) > 0.001 else 'close to independent'}**, "
        f"though the odds ratio of {odds_ratio:.2f} indicates the practical effect size is "
        f"{'modest' if 0.7 < odds_ratio < 1.4 else 'meaningful'} — clear weather is "
        f"{'associated with somewhat ' + ('higher' if odds_ratio>1 else 'lower') + ' odds of congestion than cloudy weather' if abs(odds_ratio-1)>0.05 else 'associated with roughly the same odds of congestion as cloudy weather'}. "
        "In practical terms, weather condition alone is a weak lever for predicting congestion "
        "on this corridor — consistent with the weak temperature-traffic correlation found in "
        "Task 2 — and time-of-day/day-of-week factors (explored quantitatively in the Python "
        "pipeline in Part 2) are far stronger drivers of whether the road is congested."
    )

    report = "\n".join(lines)
    REPORT_PATH.write_text(report)
    print(report)
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
