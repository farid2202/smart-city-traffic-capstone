# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

Capstone project — AI, ML and Data Science programme. Three connected parts
analysing and building AI solutions on the Metro Interstate Traffic Volume
dataset (westbound I-94, Minneapolis-St Paul).

## Start here
- **`final_capstone_report.md`** — methodology and findings across all three parts.

## Structure
```
smart-city-traffic-capstone/
├── Metro_Interstate_Traffic_Volume.csv
├── requirements.txt
├── part1_data_analytics/    # SQL, statistics, probability, Power BI dashboard, insights report
├── part2_python/            # cleaning pipeline, feature engineering, viz, CLI app
├── part3_machine_learning/  # supervised/unsupervised ML, deep learning, MLOps, recommender
└── final_capstone_report.md
```

## Setup
Requires **Python 3.10–3.12**. TensorFlow (used in Part 3's LSTM model) does
not yet publish wheels for Python 3.13+, so creating the environment with a
newer default Python will fail partway through `pip install -r requirements.txt`
when it reaches `tensorflow`. Run `py -0p` (Windows) to see which Python
versions are installed and pick a 3.10–3.12 one if your default is newer.

```bash
python -m venv venv              # or: py -3.11 -m venv venv
source venv/bin/activate         # macOS/Linux
venv\Scripts\activate            # Windows (cmd or PowerShell)
pip install -r requirements.txt
```

## Running each part
- **Part 1:** `python part1_data_analytics/sql/run_sql_analysis.py`, then
  `descriptive_stats.py` and `probability_analysis.py` in
  `part1_data_analytics/statistics/`. Power BI dashboard:
  `part1_data_analytics/powerbi/traffic_dashboard.pbix`.
- **Part 2:** `cd part2_python && python main.py` (runs the full pipeline
  end-to-end). CLI app: `cd cli_app && python app.py --help`.
- **Part 3:** see `part3_machine_learning/README.md` for the full run order.

## Git history
This repository was built with incremental, descriptively-named commits per
task (`git log --oneline`) rather than one final commit.
