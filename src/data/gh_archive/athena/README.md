# queries ran in athena

## database creation

```sql
-- 1) Create database (run once)
CREATE DATABASE IF NOT EXISTS impermanent;

-- 2) Create versioned external table
CREATE EXTERNAL TABLE IF NOT EXISTS impermanent.v0_1_0_gh_archive_metrics_raw_events (
  event_id string,
  created_at string,
  event_type string,
  repo_id bigint,
  repo_name string,
  actor_id bigint,
  actor_login string,
  action string,
  push_distinct_size int
)
PARTITIONED BY (
  year int,
  month int,
  day int,
  hour int
)
STORED AS PARQUET
LOCATION 's3://impermanent-benchmark/v0.1.0/gh-archive/raw-events/';

-- 3) Enable partition projection (this makes it effectively "auto-updating")
ALTER TABLE impermanent.v0_1_0_gh_archive_metrics_raw_events SET TBLPROPERTIES (
  'projection.enabled'='true',

  'projection.year.type'='integer',
  'projection.year.range'='2020,2030',

  'projection.month.type'='integer',
  'projection.month.range'='1,12',
  'projection.month.digits'='2',

  'projection.day.type'='integer',
  'projection.day.range'='1,31',
  'projection.day.digits'='2',

  'projection.hour.type'='integer',
  'projection.hour.range'='0,23',
  'projection.hour.digits'='2',

  'storage.location.template'='s3://impermanent-benchmark/v0.1.0/gh-archive/raw-events/year=${year}/month=${month}/day=${day}/hour=${hour}/'
);
```


## repo selection

```sql
CREATE TABLE impermanent.v0_1_0_gh_archive_metrics_repos
WITH (
  format = 'PARQUET',
  parquet_compression = 'ZSTD'
) AS

WITH params AS (
  SELECT
    200 AS n_large,
    100 AS n_medium,
    100 AS n_low,
    'v0.1.0' AS salt
),

base_2025 AS (
  SELECT
    repo_id,
    repo_name,
    event_type,
    action,
    from_iso8601_timestamp(created_at) AS ts
  FROM impermanent.v0_1_0_gh_archive_metrics_raw_events
  WHERE year = 2025
    AND event_type IN ('WatchEvent','PullRequestEvent','IssuesEvent','PushEvent')
    AND repo_id IS NOT NULL
    AND repo_name IS NOT NULL
),

repo_stats_2025 AS (
  SELECT
    repo_id,
    max(repo_name) AS repo_name,
    count_if(event_type = 'WatchEvent') AS stars_2025,
    count_if(event_type = 'PullRequestEvent' AND action = 'opened') AS prs_opened_2025,
    count_if(event_type = 'IssuesEvent' AND action = 'opened') AS issues_opened_2025,
    count_if(event_type = 'PushEvent') AS pushes_2025,
    count(*) AS total_events_2025
  FROM base_2025
  GROUP BY repo_id
),

active_months_jul_dec AS (
  SELECT
    repo_id,
    count(DISTINCT date_trunc('month', ts)) AS active_months_jul_dec
  FROM base_2025
  WHERE ts >= timestamp '2025-07-01 00:00:00 UTC'
    AND ts <  timestamp '2026-01-01 00:00:00 UTC'
  GROUP BY repo_id
),

eligible AS (
  SELECT
    s.*,
    m.active_months_jul_dec
  FROM repo_stats_2025 s
  JOIN active_months_jul_dec m
    ON s.repo_id = m.repo_id
  WHERE m.active_months_jul_dec = 6
    AND s.stars_2025 > 0
),

rank_all AS (
  SELECT
    *,
    percent_rank() OVER (ORDER BY stars_2025) AS star_prank_all
  FROM eligible
),

upper_half AS (
  SELECT *
  FROM rank_all
  WHERE star_prank_all >= 0.99
),

rank_upper AS (
  SELECT
    *,
    percent_rank() OVER (ORDER BY stars_2025) AS star_prank_upper
  FROM upper_half
),

bucketed AS (
  SELECT
    *,
    CASE
      WHEN star_prank_upper < 0.20 THEN 'low'                                   -- bottom 20% of upper half
      WHEN star_prank_upper >= 0.40 AND star_prank_upper < 0.60 THEN 'medium'   -- around median
      WHEN star_prank_upper >= 0.90 THEN 'large'                                -- top 10% of upper half
      ELSE NULL
    END AS size_bucket,

    xxhash64(
      to_utf8(cast(repo_id AS varchar) || ':' || (SELECT salt FROM params))
    ) AS sample_key
  FROM rank_upper
),

filtered AS (
  SELECT *
  FROM bucketed
  WHERE size_bucket IS NOT NULL
),

ranked AS (
  SELECT
    *,
    row_number() OVER (PARTITION BY size_bucket ORDER BY sample_key) AS rn
  FROM filtered
)

SELECT
  repo_id,
  repo_name,
  size_bucket,
  stars_2025,
  prs_opened_2025,
  issues_opened_2025,
  pushes_2025,
  total_events_2025,
  active_months_jul_dec,
  star_prank_all,
  star_prank_upper
FROM ranked
CROSS JOIN params
WHERE (size_bucket = 'large'  AND rn <= n_large)
   OR (size_bucket = 'medium' AND rn <= n_medium)
   OR (size_bucket = 'low'    AND rn <= n_low);
```

```sql
UNLOAD (SELECT * FROM impermanent.v0_1_0_gh_archive_metrics_repos)
TO 's3://impermanent-benchmark/v0.1.0/gh-archive/meta/repos/'
WITH (format='PARQUET', compression='ZSTD');
```

## first event per repo

Find the earliest hour each benchmark repo was observed in the raw events.
This is used by the transform step to only include a repo starting from its
first observed event (instead of backfilling zeros for all prior hours).

```sql
UNLOAD (
  SELECT
    ev.repo_id,
    MIN(date_parse(
      cast(ev.year AS varchar) || '-' ||
      cast(ev.month AS varchar) || '-' ||
      cast(ev.day AS varchar) || ' ' ||
      cast(ev.hour AS varchar) || ':00:00',
      '%Y-%c-%e %k:%i:%s'
    )) AS first_hour
  FROM impermanent.v0_1_0_gh_archive_metrics_raw_events ev
  INNER JOIN impermanent.v0_1_0_gh_archive_metrics_repos r
    ON ev.repo_id = r.repo_id
  GROUP BY ev.repo_id
)
TO 's3://impermanent-benchmark/v0.1.0/gh-archive/meta/first_events/'
WITH (format='PARQUET', compression='ZSTD');
```

## processed daily series (for Athena)

Parquet layout matches the pipeline: `unique_id = '{repo_id}:{metric}'`, integer counts `y`,
Hive partitions `year/month/day` under `processed-events/daily/`.

```sql
-- Run once: external table over aggregated daily series
CREATE EXTERNAL TABLE IF NOT EXISTS impermanent.v0_1_0_gh_archive_processed_daily_series (
  repo_id bigint,
  repo_name string,
  metric string,
  unique_id string,
  ds timestamp,
  y bigint
)
PARTITIONED BY (
  year int,
  month int,
  day int
)
STORED AS PARQUET
LOCATION 's3://impermanent-benchmark/v0.1.0/gh-archive/processed-events/daily/';

ALTER TABLE impermanent.v0_1_0_gh_archive_processed_daily_series SET TBLPROPERTIES (
  'projection.enabled'='true',
  'projection.year.type'='integer',
  'projection.year.range'='2020,2030',
  'projection.month.type'='integer',
  'projection.month.range'='1,12',
  'projection.month.digits'='2',
  'projection.day.type'='integer',
  'projection.day.range'='1,31',
  'projection.day.digits'='2',
  'storage.location.template'='s3://impermanent-benchmark/v0.1.0/gh-archive/processed-events/daily/year=${year}/month=${month}/day=${day}/'
);
```

## series sparsity (per `unique_id`, **within each metric**)

**Sparsity** is summarized by the **zero rate**: share of daily observations with `y = 0`
(no activity that day for that metric). Values are split into **three tertiles separately
for each metric kind** so you compare repos to peers with the same signal:

| `metric` (from processed daily series) | Series kind |
|----------------------------------------|-------------|
| `stars` | Watch / star events |
| `pushes` | Push events |
| `prs_opened` | PRs opened |
| `issues_opened` | Issues opened |

Tertiles use ``NTILE(3) OVER (PARTITION BY metric ORDER BY zero_rate)`` — **low / medium /
high are relative within that metric**, not across all metrics (stars-heavy series are not
mixed with pushes-only sparsity).

| `sparsity_level` | Meaning (within the same `metric`) |
|------------------|-------------------------------------|
| `low` | Lowest third of zero rates (densest series) |
| `medium` | Middle third |
| `high` | Highest third of zero rates (sparsest series) |

Restrict partitions (example: calendar year 2025) so the query stays bounded. Join to
evaluation rows on ``unique_id`` (``{repo_id}:{metric}``); ``metric`` is included in the
unload for checks and group-bys.


```sql
UNLOAD (
  WITH daily AS (
    SELECT unique_id, metric, y
    FROM impermanent.v0_1_0_gh_archive_processed_daily_series
    WHERE year = 2025
      AND metric IN ('stars', 'pushes', 'prs_opened', 'issues_opened')
  ),
  per_series AS (
    SELECT
      unique_id,
      max(metric) AS metric,
      COUNT(*) AS n_obs,
      SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END) AS n_zero,
      CAST(SUM(CASE WHEN y = 0 THEN 1 ELSE 0 END) AS double) / CAST(COUNT(*) AS double) AS zero_rate
    FROM daily
    GROUP BY unique_id
  ),
  tertiled AS (
    SELECT
      unique_id,
      metric,
      n_obs,
      n_zero,
      zero_rate,
      NTILE(3) OVER (PARTITION BY metric ORDER BY zero_rate) AS sparsity_tertile
    FROM per_series
    WHERE n_obs > 0
  )
  SELECT
    unique_id,
    metric,
    n_obs,
    n_zero,
    zero_rate,
    CASE sparsity_tertile
      WHEN 1 THEN 'low'
      WHEN 2 THEN 'medium'
      WHEN 3 THEN 'high'
    END AS sparsity_level
  FROM tertiled
)
TO 's3://impermanent-benchmark/v0.1.0/gh-archive/meta/series_sparsity/'
WITH (format='PARQUET', compression='ZSTD');
```

Athena writes **opaque filenames** (no ``.parquet`` suffix). When loading with DuckDB, use a glob like
``/path/to/series_sparsity/*``, not ``*.parquet`` or a bare directory path.