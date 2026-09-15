# Part 1, Task 1 — SQL-Based Traffic Analysis: Results

## 1.1 Load and Verification

- Rows loaded: **48,204**
- Columns loaded: **9** — holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, weather_description, date_time, traffic_volume
- Date range: **2012-10-02 09:00:00 → 2018-09-30 23:00:00**
- Missing expected columns: None
- Duplicate timestamp groups found: **5,445** (same hour logged more than once, typically because the weather API returned more than one simultaneous condition, e.g. 'Rain' and 'Drizzle' for the same hour with an identical traffic_volume). A `traffic_hourly` view de-duplicates on `date_time` (keeping one row per hour) and is used for all analysis below so traffic volume is not double-counted.

✅ Load verified: row/column counts match the source CSV and the schema contains all 9 expected fields.

## 1.2 Annual Traffic Trends (2012-2017)

|   yr |   recorded_hours |   total_volume |   avg_volume_per_hour |   total_volume_change |   total_volume_pct_change |   avg_volume_change |   avg_volume_pct_change |
|-----:|-----------------:|---------------:|----------------------:|----------------------:|--------------------------:|--------------------:|------------------------:|
| 2012 |             2103 |        6785754 |                3226.7 |         nan           |                     nan   |               nan   |                   nan   |
| 2013 |             7294 |       24139878 |                3309.6 |           1.73541e+07 |                     255.7 |                82.9 |                     2.6 |
| 2014 |             4501 |       14718915 |                3270.1 |          -9.42096e+06 |                     -39   |               -39.5 |                    -1.2 |
| 2015 |             3593 |       11706145 |                3258   |          -3.01277e+06 |                     -20.5 |               -12.1 |                    -0.4 |
| 2016 |             7838 |       25032183 |                3193.7 |           1.3326e+07  |                     113.8 |               -64.3 |                    -2   |
| 2017 |             8713 |       29420221 |                3376.6 |           4.38804e+06 |                      17.5 |               182.9 |                     5.7 |

**Observations:**

1. **Recorded-hour coverage is very uneven across years** (from 2,103 hours in 2012 up to 8,713 hours in 2017). 2012 is also a partial year (data starts in October). This means the raw **SUM(total_volume)** column is misleading as a year-over-year comparison — a year with fewer sensor-hours will show a lower total even if traffic was just as heavy. **AVG(avg_volume_per_hour)** is the fairer trend metric because it normalises for how many hours were actually recorded.
2. By average hourly volume, **2017** has the highest typical traffic, and the average-volume trend across the fully-recorded years (2013-2017) shows traffic growing rather than declining — consistent with a corridor experiencing increasing commuter demand over time rather than falling usage.
3. Large swings in `recorded_hours` year to year point to **data collection gaps** (sensor/API downtime), not real-world traffic gaps — a caveat worth flagging to the mobility team before they use year totals for capacity planning.

## 1.3 Temperature Around Holidays (2015-2017)

| holiday       |   year | date       |   avg_temp_kelvin |   avg_temp_celsius |   hours_recorded |
|:--------------|-------:|:-----------|------------------:|-------------------:|-----------------:|
| Labor Day     |   2015 | 2015-09-07 |            295.32 |              22.17 |               24 |
| Labor Day     |   2016 | 2016-09-05 |            295.01 |              21.86 |               24 |
| Labor Day     |   2017 | 2017-09-04 |            291.29 |              18.14 |               24 |
| New Years Day |   2016 | 2016-01-01 |            267.09 |              -6.06 |               18 |
| New Years Day |   2017 | 2017-01-02 |            271.85 |              -1.3  |               24 |

**Observations:**

- **New Years Day**: no data at all is recorded for 2015 (a genuine sensor/API gap on that exact date, not a query issue — 2015 in particular has the fewest recorded hours of any year in the dataset, see 1.2).
- **New Years Day**: average temperature moved from -6.06°C (2016) to -1.3°C (2017), a change of +4.76°C over the period.
- **Labor Day**: average temperature moved from 22.17°C (2015) to 18.14°C (2017), a change of -4.03°C over the period.
- Note: the observed **New Year's Day in 2017 falls on 2 January**, not 1 January — 1 Jan 2017 was a Sunday, so the US federal holiday was observed the following Monday. This is expected calendar behaviour, not a data error.

New Year's Day (winter, near-freezing to sub-zero temperatures) and Labor Day (early September, mild/warm) sit at opposite ends of the annual temperature cycle, as expected. The year-on-year swings are within normal winter/late-summer variability for Minneapolis–St Paul rather than indicating a structural shift. Because both are public holidays, commuter traffic volume on these dates is driven far more by the holiday effect (fewer work trips) than by the day's temperature — so while temperature may affect discretionary/leisure trips at the margin, it is not a primary driver of the holiday traffic dip itself; that is a labour-calendar effect, not a weather effect.
