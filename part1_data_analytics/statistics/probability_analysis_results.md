# Part 1, Task 3 — Probability and Congestion Analysis: Results

Definitions: **Congestion** = traffic_volume > 5500; **High temperature** = temp > 292K (18.9°C). N = 40,565 hourly records (0K sensor-error rows excluded).

## 3.1 Basic Probability

- **P(Congestion)** = 6,107 / 40,565 = **0.1505** (15.1% of hours are congested)
- **P(Clear Weather)** = 13,361 / 40,565 = **0.3294** (32.9% of hours are clear)
- **P(Congestion AND Clear Weather)** = 1,763 / 40,565 = **0.0435**

## 3.2 Conditional Probability

- **P(Clear Weather | Congestion)** = 1,763 / 6,107 = **0.2887**
- **P(High Temperature | Congestion)** = 1,699 / 6,107 = **0.2782**

**Independence check** — P(A ∩ B) vs. P(A) × P(B), for A = Congestion, B = Clear Weather:
- P(Congestion) × P(Clear) = 0.1505 × 0.3294 = **0.0496**
- Observed P(Congestion ∩ Clear) = **0.0435**
- Gap = -0.0061 (events are NOT independent) — observed joint probability is below the independence expectation, so congestion is somewhat less likely than chance during clear weather.

**Odds ratio (congestion: clear vs. cloudy weather)**
- Odds of congestion given Clear = 1,763/11,598 = 0.1520
- Odds of congestion given Clouds = 2,586/12,534 = 0.2063
- **Odds ratio = 0.737**

## Conclusion

Congestion occurs in about 15% of recorded hours overall, and clear weather in about 33%. The gap between the observed joint probability (0.0435) and the independence expectation (0.0496) shows that weather and congestion are **not strictly independent**, though the odds ratio of 0.74 indicates the practical effect size is modest — clear weather is associated with somewhat lower odds of congestion than cloudy weather. In practical terms, weather condition alone is a weak lever for predicting congestion on this corridor — consistent with the weak temperature-traffic correlation found in Task 2 — and time-of-day/day-of-week factors (explored quantitatively in the Python pipeline in Part 2) are far stronger drivers of whether the road is congested.