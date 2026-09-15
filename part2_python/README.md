# Part 2 — Reproducible Traffic Analytics Pipeline

## Project structure
```
part2_python/
├── pipeline.py                # Task 1: load, validate schema, clean the raw CSV
├── pipeline.ipynb              # Notebook walkthrough of pipeline.py (same functions, step by step)
├── feature_engineering.py     # Task 2: time/weather/scaled/congestion features
├── feature_engineering.ipynb   # Notebook walkthrough of feature_engineering.py
├── visualizations.py          # Task 3: 4 Matplotlib charts -> figures/
├── visualizations.ipynb        # Notebook walkthrough of visualizations.py, with figures shown inline
├── main.py                    # Runs all three of the above end-to-end, one shared log
├── cli_app/
│   └── app.py                 # Task 4: mini command-line analytics application
├── figures/                   # Generated PNGs + figure_interpretations.md
├── cleaned_traffic_data.csv   # Output of pipeline.py
├── featured_traffic_data.csv  # Output of feature_engineering.py
├── pipeline.log               # Sample log from a full main.py run
├── report.md                  # Methodology and findings (1-2 pages)
└── README.md                  # This file
```

## How to run

Install dependencies once (from the repo root):
```bash
pip install -r requirements.txt
```

**Recommended — run the whole pipeline in one go** (produces one combined,
correctly-ordered `pipeline.log`):
```bash
cd part2_python
python main.py                  # INFO-level console output, full DEBUG trail in pipeline.log
python main.py --log-level DEBUG  # also prints DEBUG-level detail to the console
```

**Or run each stage individually** (each run configures its own logging and
starts a fresh `pipeline.log`, so if you run these separately you'll only see
the log for whichever script ran last — use `main.py` if you want one combined
log covering the whole pipeline):
```bash
python pipeline.py                 # -> cleaned_traffic_data.csv
python feature_engineering.py      # -> featured_traffic_data.csv
python visualizations.py           # -> figures/*.png
```

**Mini application** (run after the pipeline has produced `featured_traffic_data.csv`):
```bash
cd cli_app
python app.py query-datetime --datetime "2017-07-04 08:00:00"
python app.py high-traffic --top 5
python app.py compare-weekday-weekend
python app.py recommend-travel --day-type weekend
```

**Notebook versions** (`pipeline.ipynb`, `feature_engineering.ipynb`,
`visualizations.ipynb`) are optional companions for viewing each stage
interactively in Jupyter — they import and call the exact same functions as
the `.py` files (nothing is re-implemented), already executed with outputs
inline. `main.py` and `cli_app/app.py` still import from the `.py` files
directly, so those remain the scripts that actually run this pipeline
end-to-end; the notebooks are for walking through *how* each stage works.

## Logging configuration

All logging is configured in exactly one place: `configure_logging()` in
`pipeline.py`. Every other module (`feature_engineering`, `visualizations`,
`cli_app/app`) only calls `logging.getLogger(__name__)` — none of them attach
their own handlers, so every log record ultimately flows through the same
two handlers set up once at the entry point:

- **Console handler** — prints at the level you choose with `--log-level`
  (default `INFO`), so normal runs aren't cluttered with debug detail.
- **File handler** (`pipeline.log`) — always captures everything from `DEBUG`
  upward, so the full trail is available for review even if the console was
  quieter.

Both handlers share one formatter: `timestamp | LEVEL | module | message`.

**What each level means in this project:**
| Level | Used for |
|---|---|
| `DEBUG` | Internal intermediate values not part of the final output — e.g. the exact min/max used to scale a column, or the quartile thresholds used to build `congestion_category`. Only visible in `pipeline.log`, or on the console with `--log-level DEBUG`. |
| `INFO` | Normal pipeline milestones — data loaded, a cleaning/feature step completed, a figure saved, a CLI command invoked. |
| `WARNING` | Recoverable but noteworthy data issues — rows dropped, values imputed, categories standardised. Always includes the row count affected and the reason. |
| `ERROR` | A failure that stops the current step (bad file path, broken schema, invalid CLI input) — always logged with `exc_info=True` (or, for the CLI, with a clear plain-language error) so the program exits gracefully instead of crashing with a raw traceback. |

Third-party libraries that log very verbosely at `DEBUG` (matplotlib's font
manager, in particular) are explicitly set to `WARNING` in
`configure_logging()` so `pipeline.log` stays a readable trail of *this
project's* events rather than library internals.

`print()` is never used for internal status — only `cli_app/app.py` uses it,
and only for the actual answer to a user's command (e.g. printing the
requested traffic figures), which is the appropriate use of `print()` in a
CLI tool: it's user-facing output, not an internal log event.

## Data cleaning summary (see `pipeline.log` / `report.md` for full detail)
- 1,730 rows had inconsistent `weather_description` casing, standardised to lowercase.
- 17 fully duplicate rows removed; 7,612 rows sharing a timestamp with another row collapsed to one row per hour.
- 10 rows with an impossible 0K temperature and 1 row with an implausible >9,000mm hourly rainfall value were imputed using that month's median (not a single global average).
