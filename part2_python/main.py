"""
main.py — Part 2 entry point: runs the full pipeline end-to-end in one process
==================================================================================
load/clean (pipeline.py) -> feature engineering (feature_engineering.py) ->
visualisations (visualizations.py), all under one shared logging configuration
so pipeline.log contains one continuous, correctly-ordered trail of the whole
run (running the three scripts separately each starts its own fresh log file
instead — see README.md).

Run with:  python main.py [--log-level DEBUG]
"""
from __future__ import annotations

import argparse
import logging
import sys

from pipeline import configure_logging, run_pipeline, DEFAULT_INPUT, DEFAULT_OUTPUT
from feature_engineering import engineer_features, DEFAULT_OUTPUT as FEATURED_OUTPUT
from visualizations import build_all_visualizations, DEFAULT_FIG_DIR

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full Part 2 traffic analytics pipeline.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger.info("=== Starting full Part 2 pipeline run (clean -> feature-engineer -> visualise) ===")

    try:
        df_clean = run_pipeline(DEFAULT_INPUT, DEFAULT_OUTPUT)
        df_features = engineer_features(df_clean)
        df_features.to_csv(FEATURED_OUTPUT, index=False)
        logger.info(
            "Saved featured dataset to %s (%d rows, %d columns)",
            FEATURED_OUTPUT, *df_features.shape,
        )
        build_all_visualizations(df_features, DEFAULT_FIG_DIR)
    except Exception as exc:
        logger.error("Full pipeline run failed: %s", exc, exc_info=True)
        return 1

    logger.info("=== Full pipeline run complete — see cli_app/app.py for the query application ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
