# Part 1, Task 2 — Descriptive Statistics and Correlation: Results

## 2.1 Traffic Volume Statistics

- **N (hours):** 40,575
- **Mean:** 3290.7 vehicles/hour
- **Median:** 3427.0 vehicles/hour
- **Standard deviation:** 1984.8
- **Variance:** 3939323.5
- **Range:** 7280 (min 0, max 7280)

**Interpretation:**

The mean and median are close, suggesting a fairly symmetric distribution. The standard deviation (1985) is very large relative to the mean (3291) — a coefficient of variation of 60% — which tells us traffic volume swings dramatically within a day (near-zero overnight vs. several thousand vehicles/hour at peak), rather than sitting in a narrow band. The full range (0–7280) confirms the corridor experiences everything from essentially empty roads to near-capacity flow, so any single 'typical volume' figure understates how much hour-of-day and day-of-week matter — single summary statistics should always be read alongside the hourly/weekly pattern, not instead of it.

## 2.2 Correlation Analysis: Temperature vs. Traffic Volume

- **Pearson correlation coefficient (r):** 0.1392 (p = 1.54e-174, n = 40,565)
- 10 rows with an impossible **0 Kelvin** temperature reading (a known sensor error, corrected properly in Part 2) were excluded from this calculation so they don't distort the coefficient.

**Interpretation:**

- **Direction:** The relationship is **positive** — warmer temperatures are associated with slightly higher traffic volume.
- **Strength:** r = 0.139 is a **weak** correlation in conventional terms (|r| well below 0.3). Temperature explains only 1.9% of the variance in traffic volume (R²), so temperature alone is a poor predictor of how busy the road will be — hour-of-day and day-of-week almost certainly matter far more (confirmed by the wide swings seen in the descriptive statistics above, and explored further with Python feature engineering in Part 2).
- **Correlation vs. causation:** even if the correlation were stronger, this figure alone would not establish that temperature *causes* traffic changes. Both variables are driven by shared underlying factors — for example, season and time-of-day jointly affect both typical temperature and typical commuter behaviour (e.g. school terms, daylight hours, holiday timing). A cold snap and a traffic dip could both simply be consequences of 'it's a winter holiday period' rather than one causing the other. Establishing causation would require controlling for these confounders (or a controlled/quasi-experimental design), which a simple correlation coefficient cannot do.
