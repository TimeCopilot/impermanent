import datetime as dt
from pathlib import Path

import duckdb

from ..extract.core import RawGHArchiveHourExtractor


class HourTransformer(RawGHArchiveHourExtractor):
    """
    Aggregate one raw GH Archive hour into processed format:

      version, repo_id, repo_name, metric, unique_id, ds, y

    where:
      unique_id = "{repo_id}:{metric}"
      ds        = hour_start_utc
      y         = integer count for that metric during the hour

    """

    def __init__(
        self,
        raw_events_dir: str | Path,
        processed_events_dir: str | Path,
        repos_dir: str | Path,
        first_events_path: str | Path,
        output_filename: str = "series.parquet",
    ):
        self.raw_events_dir = Path(raw_events_dir)
        self.processed_events_dir = Path(processed_events_dir)
        self.repos_dir = Path(repos_dir)
        self.first_events_path = Path(first_events_path)
        self.output_filename = output_filename

    def hour_local_path(self, dt_hour: dt.datetime) -> Path:
        path = (
            self.processed_events_dir
            / f"year={dt_hour.year:04d}"
            / f"month={dt_hour.month:02d}"
            / f"day={dt_hour.day:02d}"
            / f"hour={dt_hour.hour:02d}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path / self.output_filename

    def hour_exists(self, str_hour: str) -> bool:
        dt_hour = RawGHArchiveHourExtractor.str_hour_to_dt(str_hour)
        local_path = self.hour_local_path(dt_hour)
        return local_path.exists()

    def aggregate_hour(self, str_hour: str) -> Path:
        """
        Aggregates exactly one hour.

        Only repos whose first observed event (per first_events_path)
        is at or before this hour are included.
        """
        extractor = RawGHArchiveHourExtractor(output_dir=self.raw_events_dir)
        dt_hour = extractor.str_hour_to_dt(str_hour)

        in_path = extractor.hour_local_path(dt_hour)
        out_path = self.hour_local_path(dt_hour)
        if not in_path.exists():
            raise FileNotFoundError(f"Missing raw hour parquet: {in_path}")

        repos_glob = str(self.repos_dir / "*")
        fe_glob = str(self.first_events_path / "*")

        con = duckdb.connect(database=":memory:")
        con.execute("INSTALL parquet; LOAD parquet;")

        # selected repos (the benchmark set)
        con.execute(
            f"""
            CREATE TEMP VIEW repos AS
            SELECT
              repo_id::BIGINT AS repo_id,
              repo_name
            FROM read_parquet('{repos_glob}');
        """
        )

        # raw events for this hour (minimal columns)
        con.execute(
            f"""
            CREATE TEMP VIEW ev AS
            SELECT
              repo_id::BIGINT AS repo_id,
              event_type,
              action
            FROM read_parquet('{in_path.as_posix()}');
        """
        )

        # first observed event per repo
        con.execute(
            f"""
            CREATE TEMP VIEW first_events AS
            SELECT repo_id::BIGINT AS repo_id, first_hour
            FROM read_parquet('{fe_glob}');
        """
        )

        # Only repos with first_hour <= current hour
        ds_str = dt_hour.strftime("%Y-%m-%d %H:%M:%S")
        # fmt: off
        con.execute(f"""
            CREATE TEMP VIEW agg AS
            SELECT
              r.repo_id,
              r.repo_name,
              SUM(CASE WHEN ev.event_type = 'WatchEvent'
                  THEN 1 ELSE 0 END) AS stars,
              SUM(CASE WHEN ev.event_type = 'PullRequestEvent'
                  AND ev.action = 'opened'
                  THEN 1 ELSE 0 END) AS prs_opened,
              SUM(CASE WHEN ev.event_type = 'IssuesEvent'
                  AND ev.action = 'opened'
                  THEN 1 ELSE 0 END) AS issues_opened,
              SUM(CASE WHEN ev.event_type = 'PushEvent'
                  THEN 1 ELSE 0 END) AS pushes
            FROM repos r
            INNER JOIN first_events fe
              ON fe.repo_id = r.repo_id
            LEFT JOIN ev ON ev.repo_id = r.repo_id
            WHERE fe.first_hour <= TIMESTAMP '{ds_str}'
            GROUP BY 1,2;
        """)
        # fmt: on

        # unpivot to long format (unique_id, ds, y)
        con.execute(
            f"""
            COPY (
              SELECT
                repo_id,
                repo_name,
                metric,
                (cast(repo_id AS varchar) || ':' || metric) AS unique_id,
                TIMESTAMP '{ds_str}' AS ds,
                y::BIGINT AS y
              FROM (
                SELECT repo_id, repo_name, 'stars' AS metric, stars AS y FROM agg
                UNION ALL
                SELECT 
                    repo_id, repo_name, 'prs_opened' AS metric, prs_opened AS y FROM agg
                UNION ALL
                SELECT 
                    repo_id, repo_name, 'issues_opened' AS metric, issues_opened AS y 
                FROM agg
                UNION ALL
                SELECT repo_id, repo_name, 'pushes' AS metric, pushes AS y FROM agg
              )
            )
            TO '{out_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        )

        return out_path
