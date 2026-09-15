# Part 3, Task 6 — MLOps and Deployment Simulation

| Sub-task | Where it lives |
|---|---|
| 6.1 Model versioning | `model_versioning.py` / `model_versioning.md` — generated directly from MLflow run history |
| 6.2 Experiment tracking | MLflow (`../mlflow/mlflow.db`), populated by `scripts/supervised_ml.py` and `scripts/deep_learning_explainability.py` — see `../advanced_ai_technique.md` |
| 6.3 Deployment | `../deployment/api.py` — FastAPI service serving the regression (volume) and classification (risk) models |
| 6.4 Monitoring | `../monitoring/monitor.py` / `monitoring_report.md` — feature distribution drift (KS test) + prediction error drift |
| 6.5 Alerting | Same `monitor.py` — PASS/ALERT status per check and overall, with a synthetic-data appendix demonstrating the ALERT path actually triggers |

## How to run the full MLOps simulation
```bash
cd part3_machine_learning
python scripts/supervised_ml.py                 # trains models, logs to MLflow
python scripts/deep_learning_explainability.py   # trains LSTM + explainability RF, logs to MLflow
python mlops/model_versioning.py                 # -> mlops/model_versioning.md
python monitoring/monitor.py                     # -> monitoring/monitoring_report.md
cd deployment && uvicorn api:app --reload --port 8000   # -> http://127.0.0.1:8000/docs
```

## Limitations (see also `../advanced_ai_technique.md` and `../responsible_ai_report.md`)
- Single-machine simulation: no real load balancing, authentication, or
  horizontal scaling — `deployment/api.py` demonstrates the *shape* of a
  deployment (a model behind an HTTP prediction endpoint), not a
  production-hardened service.
- Monitoring runs on-demand rather than continuously/scheduled; a real
  deployment would run these checks on a schedule (e.g. daily) and alert a
  human via email/Slack/PagerDuty rather than writing a markdown file.
- Drift thresholds (KS statistic > 0.10, MAE increase > 20%) are reasonable
  starting points, not tuned against real operational cost/consequence data
  — a real deployment should calibrate these against how costly a false
  alarm vs. a missed drift event actually is for the mobility team.
