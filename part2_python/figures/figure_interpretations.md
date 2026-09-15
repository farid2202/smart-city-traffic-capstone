# Figure Interpretations

### `01_traffic_by_hour.png`
Traffic follows a clear bimodal commute pattern, peaking around 16:00. Volume is lowest overnight (roughly 00:00-04:00), consistent with a corridor dominated by commuter rather than overnight freight/leisure traffic.

### `02_weekday_vs_weekend.png`
Weekday traffic shows two sharp commute peaks (morning and evening) that weekend traffic does not: weekends instead have a single, flatter midday bulge. This confirms the corridor's traffic is driven primarily by work-commute patterns rather than general leisure travel, and is one of the strongest predictive signals available for the Part 3 models.

### `03_traffic_distribution.png`
The distribution is bimodal rather than a simple bell curve: a cluster of low-traffic overnight hours, and a separate cluster of moderate-to-high daytime hours. This bimodality is exactly why the mean/median alone (Part 1, Task 2) understate how variable traffic really is — a single 'typical volume' number blends two very different regimes (night vs. day) into one figure.

### `04_weather_impact.png`
'Clouds' has the highest average traffic volume and 'Squall' the lowest, a difference of 3197 vehicles/hour. The overall spread across weather categories is modest relative to the hour-of-day effect seen in Figure 1 — consistent with the weak temperature-traffic correlation (r ≈ 0.14) and near-chance congestion/weather independence found in the Part 1 probability analysis: weather shifts traffic volume only at the margins.
