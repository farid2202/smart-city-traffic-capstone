# Part 3 — Machine Learning and AI: Building an Intelligent Mobility Solution

## Accident dataset statement
**No real accident dataset was sourced or used.** As instructed in the
capstone brief, a **documented proxy accident-risk label** (`high_risk`) was
constructed from `congestion_category` (data-driven quartiles of
`traffic_volume`) combined with severe/low-visibility weather — see
`scripts/common.py::add_proxy_accident_risk_label` for the exact logic, and
`responsible_ai_report.md` for the risks of over-interpreting this proxy as
a real accident predictor.

## Project structure
```
part3_machine_learning/
├── scripts/
│   ├── common.py                       # shared feature set, proxy label, scenario builder
│   ├── supervised_ml.py                # Task 1: classification + regression
│   ├── unsupervised_ml.py              # Task 2: k-means + association rules
│   └── deep_learning_explainability.py # Task 3: LSTM + SHAP
├── notebooks/                          # executed .ipynb versions of the 3 scripts above, + figures/
├── models/                             # saved .joblib / .keras model files
├── mlflow/mlflow.db                    # MLflow tracking store (Task 4)
├── advanced_ai_technique.md            # Task 4 write-up
├── recommendation_system/recommender.py  # Task 5
├── mlops/                              # Task 6.1-6.2: model_versioning.py/.md, README.md
├── deployment/api.py                   # Task 6.3: FastAPI deployment simulation
├── monitoring/monitor.py               # Task 6.4-6.5: drift monitoring + alerting
└── responsible_ai_report.md            # Task 7
```

## How to run everything, in order
```bash
cd part3_machine_learning
python scripts/supervised_ml.py
python scripts/unsupervised_ml.py
python scripts/deep_learning_explainability.py
python mlops/model_versioning.py
python monitoring/monitor.py
python recommendation_system/recommender.py --day-type weekday --weather Clear
cd deployment && uvicorn api:app --reload --port 8000   # then POST to /predict/volume or /predict/risk
```
Or open the pre-executed notebooks in `notebooks/` to see all outputs without
re-running anything (each has its outputs saved from the last run).

## Reused from earlier parts
Every script here loads `part2_python/featured_traffic_data.csv` (Part 2's
cleaned + feature-engineered output) rather than re-deriving features from
raw data — the congestion category, cyclical time encodings, and weather
indicators used throughout Part 3 are exactly the ones built in Part 2,
Task 2.
