-- =====================================================================
-- Smart City Traffic Intelligence — Part 1, Task 1: SQL-Based Traffic Analysis
-- Database: traffic.db (SQLite)  |  Source: Metro_Interstate_Traffic_Volume.csv
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1.1 Load the dataset / verify it loaded correctly
-- ---------------------------------------------------------------------

-- Row / column count check
SELECT COUNT(*) AS row_count FROM traffic_raw;
PRAGMA table_info(traffic_raw);

-- Spot-check the first and last few rows (ordering matches source file order)
SELECT * FROM traffic_raw LIMIT 5;

-- Sanity checks used to confirm a correct load
SELECT MIN(date_time) AS earliest_ts, MAX(date_time) AS latest_ts FROM traffic_raw;
SELECT COUNT(*) AS duplicate_timestamp_rows
FROM (
    SELECT date_time FROM traffic_raw GROUP BY date_time HAVING COUNT(*) > 1
);

-- ---------------------------------------------------------------------
-- 1.2 Analyse annual traffic trends (2012-2017)
-- ---------------------------------------------------------------------

DROP VIEW IF EXISTS traffic_hourly;
CREATE VIEW traffic_hourly AS
SELECT date_time,
       MIN(traffic_volume) AS traffic_volume,   -- identical across duplicate rows for a given hour
       MIN(temp)           AS temp,
       MIN(holiday)        AS holiday,
       MIN(weather_main)   AS weather_main
FROM traffic_raw
GROUP BY date_time;

-- Yearly totals and averages, raw vs. de-duplicated, with recorded-hour counts
SELECT strftime('%Y', date_time)                        AS yr,
       COUNT(*)                                          AS recorded_hours,
       SUM(traffic_volume)                                AS total_volume_dedup,
       ROUND(AVG(traffic_volume), 1)                      AS avg_volume_per_hour
FROM traffic_hourly
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY yr
ORDER BY yr;

WITH yearly AS (
    SELECT strftime('%Y', date_time)     AS yr,
           COUNT(*)                       AS recorded_hours,
           SUM(traffic_volume)            AS total_volume,
           ROUND(AVG(traffic_volume), 1)  AS avg_volume_per_hour
    FROM traffic_hourly
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY yr
)
SELECT yr,
       recorded_hours,
       total_volume,
       avg_volume_per_hour,
       total_volume - LAG(total_volume) OVER (ORDER BY yr)              AS total_volume_change,
       ROUND(100.0 * (total_volume - LAG(total_volume) OVER (ORDER BY yr))
             / LAG(total_volume) OVER (ORDER BY yr), 1)                 AS total_volume_pct_change,
       avg_volume_per_hour - LAG(avg_volume_per_hour) OVER (ORDER BY yr) AS avg_volume_change,
       ROUND(100.0 * (avg_volume_per_hour - LAG(avg_volume_per_hour) OVER (ORDER BY yr))
             / LAG(avg_volume_per_hour) OVER (ORDER BY yr), 1)          AS avg_volume_pct_change
FROM yearly
ORDER BY yr;

-- ---------------------------------------------------------------------
-- 1.3 Analyse temperature around holidays (New Year's Day, Labor Day: 2015-2017)
-- ---------------------------------------------------------------------

-- Step A: find the exact calendar date each named holiday fell on, per year
SELECT strftime('%Y', date_time) AS yr, holiday, DATE(date_time) AS holiday_date
FROM traffic_raw
WHERE holiday IN ('New Years Day', 'Labor Day')
  AND strftime('%Y', date_time) BETWEEN '2015' AND '2017'
GROUP BY yr, holiday;

-- Step B: average temperature (in Kelvin, converted to Celsius) recorded on
-- each of those calendar dates
SELECT DATE(date_time)                              AS holiday_date,
       ROUND(AVG(temp), 2)                          AS avg_temp_kelvin,
       ROUND(AVG(temp) - 273.15, 2)                 AS avg_temp_celsius,
       COUNT(*)                                     AS hours_recorded
FROM traffic_hourly
WHERE DATE(date_time) IN (
    SELECT DATE(date_time) FROM traffic_raw
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND strftime('%Y', date_time) BETWEEN '2015' AND '2017'
)
GROUP BY holiday_date
ORDER BY holiday_date;
