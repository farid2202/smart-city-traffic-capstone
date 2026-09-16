# Monitoring Report

## Overall status: **PASS**

## Feature distribution drift (train vs. most recent period, KS test)

Alert threshold: KS statistic > 0.1

| feature              |   ks_statistic |   p_value | status   |
|:---------------------|---------------:|----------:|:---------|
| weather_Clouds       |         0.0742 |    0      | PASS     |
| is_severe_weather    |         0.0411 |    0      | PASS     |
| weather_Snow         |         0.0328 |    0      | PASS     |
| weather_Clear        |         0.0271 |    0.0001 | PASS     |
| is_low_visibility    |         0.0177 |    0.0331 | PASS     |
| weather_Mist         |         0.0121 |    0.2948 | PASS     |
| weather_Rain         |         0.0104 |    0.4828 | PASS     |
| weather_Thunderstorm |         0.0083 |    0.7528 | PASS     |
| hour_cos             |         0.0058 |    0.9795 | PASS     |
| weather_Haze         |         0.0056 |    0.9853 | PASS     |
| is_weekend           |         0.0032 |    1      | PASS     |
| dow_sin              |         0.0032 |    1      | PASS     |
| has_precipitation    |         0.0024 |    1      | PASS     |
| hour_sin             |         0.0023 |    1      | PASS     |
| dow_cos              |         0.0016 |    1      | PASS     |
| weather_Drizzle      |         0.0014 |    1      | PASS     |
| is_holiday           |         0.0001 |    1      | PASS     |
| weather_Smoke        |         0.0001 |    1      | PASS     |
| weather_Fog          |         0.0001 |    1      | PASS     |
| weather_Squall       |         0      |    1      | PASS     |

No features exceeded the drift threshold.

## Prediction error drift (regression model, first vs. second half of test period)

- MAE, first half of test period:  317.6 vehicles/hour
- MAE, second half of test period: 239.1 vehicles/hour
- Change: -24.7% (alert threshold: +20%)
- Status: **PASS**

## Interpretation

No meaningful drift detected — the model's input feature distributions and prediction error are stable across the test period. No retraining action needed at this time.

## Appendix: ALERT-path demonstration (synthetic data)

The monitoring run above came back PASS because the real test period isn't actually drifted. To demonstrate the alerting mechanism itself works, this section re-runs the identical KS-test check against a **synthetically perturbed** copy of the test set (severe-weather rate artificially inflated) — this is a deliberate demonstration, not a claim about the real data.

| feature              |   ks_statistic |   p_value | status   |
|:---------------------|---------------:|----------:|:---------|
| weather_Thunderstorm |         0.604  |         0 | ALERT    |
| is_severe_weather    |         0.5887 |         0 | ALERT    |
| weather_Clear        |         0.1846 |         0 | ALERT    |

**Result: 3 feature(s) correctly flagged as ALERT** under the synthetic drift, confirming the mechanism triggers as designed.