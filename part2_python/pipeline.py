"""
pipeline.py — Part 2, Task 1: Data Pipeline Construction (with mandatory logging)
====================================================================================
Loads the raw Metro Interstate Traffic Volume CSV, validates its schema, cleans
it (standardises categorical values, parses/validates datetimes, removes
duplicates, detects and imputes outliers), and writes a cleaned CSV that
feature_engineering.py, visualizations.py and cli_app/app.py all build on.

Run with:  python pipeline.py [--input PATH] [--output PATH] [--log-level DEBUG]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger. Never use the bare root logger inside library code —
# logging.getLogger(__name__) gives each module its own named logger
# ("pipeline", "feature_engineering", ...) that all share the handlers
# configured once, below, in configure_logging().
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR.parent / "Metro_Interstate_Traffic_Volume.csv"
DEFAULT_OUTPUT = BASE_DIR / "cleaned_traffic_data.csv"
LOG_FILE = BASE_DIR / "pipeline.log"

EXPECTED_COLUMNS = [
    "holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
    "weather_main", "weather_description", "date_time", "traffic_volume",
]

# Physically-implausible sentinel values, confirmed by inspecting the raw data
# (see Part 1 SQL analysis): a 0 Kelvin reading and a >9,000mm hourly rainfall
# reading cannot occur in reality and must be sensor/logging errors.
IMPOSSIBLE_TEMP_K = 0
MAX_PLAUSIBLE_RAIN_MM = 9000


class SchemaValidationError(Exception):
    """Raised when the raw CSV does not match the expected traffic-data schema."""


def configure_logging(log_level: str = "INFO") -> None:
    """Configure logging ONCE, here, in the entry-point script. Every other
    module in this project only calls logging.getLogger(__name__) and never
    attaches its own handlers, so log records from pipeline.py,
    feature_engineering.py, visualizations.py and cli_app/app.py all flow
    through this single configuration (console + file, with DEBUG only
    reaching the log file so normal runs stay readable on screen)."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setLevel(logging.DEBUG)  # the file keeps everything, even if console is INFO
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Silence noisy third-party libraries (matplotlib's font manager in
    # particular emits hundreds of DEBUG lines) so pipeline.log stays a
    # readable trail of THIS project's events, not library internals.
    for noisy_logger in ("matplotlib", "PIL", "fontTools", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV with explicit file-handling error handling.
    Never uses a bare `except:` — each failure mode is named so the log
    message actually says what went wrong."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Raw data file not found at %s", path, exc_info=True)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Raw data file at %s is empty", path, exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Raw data file at %s could not be parsed as CSV", path, exc_info=True)
        raise
    except UnicodeDecodeError:
        logger.error("Raw data file at %s has an unexpected encoding", path, exc_info=True)
        raise

    logger.info(
        "Successfully loaded raw data from %s: %d rows, %d columns",
        path.name, df.shape[0], df.shape[1],
    )
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the schema BEFORE any other processing happens. Raises
    SchemaValidationError (rather than continuing silently) if any expected
    column is missing, since every downstream step assumes these columns exist."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Schema validation failed — missing columns: %s", missing)
        raise SchemaValidationError(f"Missing expected columns: {missing}")
    logger.info("Schema validation passed — all %d expected columns present", len(EXPECTED_COLUMNS))


def standardise_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent casing found in weather_description (e.g. 'Sky is
    Clear' vs 'sky is clear', and an all-caps 'SQUALLS' outlier) so the same
    real-world condition isn't split into multiple categories."""
    df = df.copy()
    before = df["weather_description"].copy()
    df["weather_description"] = df["weather_description"].str.strip().str.lower()
    changed = (before != df["weather_description"]).sum()
    if changed:
        logger.warning(
            "Standardised casing/whitespace on %d rows in 'weather_description' "
            "(reason: inconsistent capitalisation, e.g. 'Sky is Clear' vs 'sky is clear')",
            changed,
        )

    # holiday: 'None' is read by pandas as an actual NaN (it's in pandas' default
    # na_values list), which is the correct representation for "not a holiday" —
    # made explicit here rather than left implicit, and holiday names are
    # stripped of stray whitespace for consistency.
    holiday_before_na = df["holiday"].isna().sum()
    df["holiday"] = df["holiday"].astype("string").str.strip()
    df["is_holiday"] = df["holiday"].notna()
    logger.info(
        "Standardised 'holiday' column: %d rows flagged as a public holiday, "
        "%d rows are ordinary days (added boolean 'is_holiday' column)",
        int(df["is_holiday"].sum()), holiday_before_na,
    )
    return df


def parse_and_validate_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_time to a real datetime dtype and drop rows that fail to parse."""
    df = df.copy()
    before_rows = len(df)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    invalid_mask = df["date_time"].isna()
    n_invalid = int(invalid_mask.sum())
    if n_invalid:
        df = df.loc[~invalid_mask].copy()
        logger.warning(
            "Dropped %d rows with an unparseable date_time value (reason: "
            "invalid/malformed timestamp string)",
            n_invalid,
        )
    logger.info(
        "Parsed date_time to datetime dtype for %d rows (from %d originally)",
        len(df), before_rows,
    )
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows, then collapse duplicate timestamps (the
    same hour logged more than once, typically because the weather API
    returned more than one simultaneous condition for that hour) down to a
    single row per hour, since Part 3's time-series modelling needs one
    observation per timestamp."""
    df = df.copy()

    n_before = len(df)
    df = df.drop_duplicates()
    n_exact_dupes = n_before - len(df)
    if n_exact_dupes:
        logger.warning(
            "Removed %d fully duplicate rows (reason: identical row appeared "
            "more than once in the raw file)",
            n_exact_dupes,
        )

    n_before_ts = len(df)
    df = df.sort_values("date_time").drop_duplicates(subset="date_time", keep="first")
    n_ts_dupes = n_before_ts - len(df)
    if n_ts_dupes:
        logger.warning(
            "Collapsed %d rows that shared a timestamp with another row down "
            "to one row per hour (reason: the weather API logged more than "
            "one simultaneous condition for the same hour; the first "
            "reported condition for each hour was kept)",
            n_ts_dupes,
        )
    return df.reset_index(drop=True)


def _monthly_median_impute(df: pd.DataFrame, column: str, invalid_mask: pd.Series) -> pd.DataFrame:
    """Impute `column` values flagged by `invalid_mask` using that month's
    median of the VALID values — computed and applied explicitly month by
    month with a loop, rather than a single global average, because traffic
    and weather conditions vary a great deal by season (e.g. imputing a
    winter month's missing temperature with the whole dataset's average
    would badly overstate it)."""
    df = df.copy()
    month = df["date_time"].dt.month
    n_imputed_total = 0

    for m in range(1, 13):  # explicit loop over each calendar month
        month_mask = month == m
        valid_this_month = df.loc[month_mask & ~invalid_mask, column]
        if valid_this_month.empty:
            continue  # no valid readings this month to compute a median from
        month_median = valid_this_month.median()

        rows_to_fix = month_mask & invalid_mask
        n_this_month = int(rows_to_fix.sum())
        if n_this_month > 0:
            df.loc[rows_to_fix, column] = month_median
            n_imputed_total += n_this_month

    return df, n_imputed_total


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and impute physically-impossible sensor readings using the
    monthly-median strategy above. Each outlier type gets its own log
    message (never a single generic 'cleaning completed')."""
    df = df.copy()

    # --- temp: 0 Kelvin is impossible (absolute zero) -----------------------
    temp_invalid = df["temp"] <= IMPOSSIBLE_TEMP_K
    n_temp_invalid = int(temp_invalid.sum())
    if n_temp_invalid:
        df, n_imputed = _monthly_median_impute(df, "temp", temp_invalid)
        logger.warning(
            "Imputed %d rows with an impossible temperature (<= %sK) using "
            "that month's median temperature (reason: 0K/absolute-zero "
            "sensor error)",
            n_imputed, IMPOSSIBLE_TEMP_K,
        )

    # --- rain_1h: values above the physically plausible range ---------------
    rain_invalid = df["rain_1h"] > MAX_PLAUSIBLE_RAIN_MM
    n_rain_invalid = int(rain_invalid.sum())
    if n_rain_invalid:
        df, n_imputed = _monthly_median_impute(df, "rain_1h", rain_invalid)
        logger.warning(
            "Imputed %d rows with an implausible rain_1h value (> %smm in "
            "one hour) using that month's median rainfall (reason: sensor "
            "error — no real hourly rainfall reaches this level)",
            n_imputed, MAX_PLAUSIBLE_RAIN_MM,
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step in sequence. Each step logs its own outcome —
    see standardise_categorical_values / parse_and_validate_datetime /
    remove_duplicate_rows / handle_outliers above."""
    df = standardise_categorical_values(df)
    df = parse_and_validate_datetime(df)
    df = remove_duplicate_rows(df)
    df = handle_outliers(df)
    return df


def save_cleaned_data(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    logger.info("Saved cleaned dataset to %s (%d rows, %d columns)", path, df.shape[0], df.shape[1])


def run_pipeline(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    df = load_raw_data(input_path)
    validate_schema(df)
    df = clean_data(df)
    save_cleaned_data(df, output_path)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean the Metro Interstate Traffic Volume dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    configure_logging(args.log_level)

    try:
        run_pipeline(args.input, args.output)
    except (SchemaValidationError, FileNotFoundError, pd.errors.ParserError,
            pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        logger.error("Pipeline could not complete: %s", exc, exc_info=True)
        return 1
    except Exception as exc:  # last-resort safety net so the program always exits gracefully
        logger.error("Pipeline failed with an unexpected error: %s", exc, exc_info=True)
        return 1

    logger.info("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
