import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd

from ..transform.core import HourTransformer


def get_output_dir_from_freq(
    base_path: str | Path,
    date: dt.datetime,
    freq: str,
) -> Path:
    base_path = Path(base_path)
    if freq.upper().startswith("H"):
        path = (
            base_path
            / f"year={date.year:04d}"
            / f"month={date.month:02d}"
            / f"day={date.day:02d}"
            / f"hour={date.hour:02d}"
        )
    elif freq.upper().startswith("D"):
        path = (
            base_path
            / f"year={date.year:04d}"
            / f"month={date.month:02d}"
            / f"day={date.day:02d}"
        )
    elif freq.upper().startswith("W"):
        iso_year, iso_week, _ = date.isocalendar()
        path = base_path / f"year={iso_year:04d}" / f"week={iso_week:02d}"
    elif freq.upper().startswith("M"):
        path = base_path / f"year={date.year:04d}" / f"month={date.month:02d}"
    else:
        path = (
            base_path
            / f"year={date.year:04d}"
            / f"month={date.month:02d}"
            / f"day={date.day:02d}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


class PeriodAggregator:
    """
    Aggregate hourly data into longer periods (daily, weekly, monthly).

    Given a frequency and a date, aggregates the previous period up to that date.
    For example:
    - If frequency is 'D' (daily) and date is 2026-01-01-01, aggregates December 31
    - If frequency is 'M' (monthly) and date is 2026-01-01, aggregates December

    Before aggregating, checks if hourly data is complete at a configurable
    percentage threshold (differentiated by frequency).

    The `ds` column in aggregated output uses period_start, which is aligned to
    pandas frequency boundaries. This ensures compatibility with pandas resampling:
    - Daily 'D': ds = YYYY-MM-DD 00:00:00 (start of day)
    - Weekly 'W': ds = start of week 00:00:00
    - Monthly 'M': ds = YYYY-MM-01 00:00:00 (first day of month)
    - Monthly 'MS': ds = YYYY-MM-01 00:00:00 (month start)

    All timestamps are in UTC and compatible with pandas Period and DatetimeIndex.
    """

    # Default completeness thresholds by frequency type
    DEFAULT_THRESHOLDS = {
        "daily": 0.90,  # 90% of hours in a day
        "weekly": 0.95,  # 95% of hours in a week (more lenient)
        "monthly": 0.99,  # 99% of hours in a month (most lenient)
    }

    def __init__(
        self,
        hourly_processed_dir: str | Path,
        aggregated_events_dir: str | Path,
        repos_dir: str | Path,
        first_events_path: str | Path,
        output_filename: str = "series.parquet",
        completeness_thresholds: dict[str, float] | None = None,
    ):
        self.hourly_processed_dir = Path(hourly_processed_dir)
        self.aggregated_events_dir = Path(aggregated_events_dir)
        self.repos_dir = Path(repos_dir)
        self.first_events_path = Path(first_events_path)
        self.output_filename = output_filename
        self.completeness_thresholds = (
            completeness_thresholds or self.DEFAULT_THRESHOLDS.copy()
        )

    @staticmethod
    def get_frequency_type(freq: str) -> str:
        """Determine frequency type from pandas frequency string."""
        freq_upper = freq.upper()
        if freq_upper.startswith("H"):
            return "hourly"
        elif freq_upper.startswith("D"):
            return "daily"
        elif freq_upper.startswith("W"):
            return "weekly"
        elif freq_upper.startswith("M") or freq_upper.startswith("MS"):
            return "monthly"
        else:
            raise ValueError(f"Unknown frequency: {freq}")

    def _get_completeness_threshold(self, freq: str) -> float:
        """Get completeness threshold for a given frequency."""
        freq_type = self.get_frequency_type(freq)
        return self.completeness_thresholds.get(freq_type, 0.90)

    @staticmethod
    def get_previous_period(
        date: dt.datetime,
        freq: str,
    ) -> tuple[dt.datetime, dt.datetime, dt.datetime]:
        """
        Get the previous period's start and end times based on frequency.

        Args:
            date: The reference date (in UTC)
            freq: Pandas frequency string (e.g., 'D', 'W', 'M')

        Returns:
            Tuple of (period_start, period_end, period_ds) in UTC.
            period_ds is the start timestamp of the aggregated period,
            correctly aligned to the frequency boundary for use as ds.
        """
        # Ensure date is in UTC
        if date.tzinfo is None:
            date = date.replace(tzinfo=dt.timezone.utc)
        else:
            date = date.astimezone(dt.timezone.utc)

        # Create a pandas Period to handle frequency logic
        # We want to aggregate the period that ends just before the given date
        period_end = date.replace(minute=0, second=0, microsecond=0)

        # Get the period to aggregate using pandas Timestamp and offsets
        try:
            # Convert to pandas Timestamp for easier manipulation
            # Ensure it's timezone-aware in UTC
            if period_end.tzinfo is None:
                ts = pd.Timestamp(period_end, tz="UTC")
            else:
                ts = pd.Timestamp(period_end).tz_convert("UTC")

            # Find the period containing the date
            # For weekly frequencies, use to_period to match pandas behavior
            if freq.upper().startswith("W"):
                # Convert to naive timestamp for to_period (pandas requirement)
                ts_naive = ts.tz_localize(None) if ts.tz is not None else ts
                current_period = ts_naive.to_period(freq)

                # Get previous period using Period arithmetic
                previous_period = current_period - 1

                # For W-SUN, the period label is the Sunday that ends the week
                # So start_time gives us the Monday of that week
                # We need to get the actual week boundaries
                current_period_start_naive = current_period.start_time
                prev_period_start_naive = previous_period.start_time

                # Convert to UTC-aware timestamps
                current_period_start_ts = pd.Timestamp(
                    current_period_start_naive, tz="UTC"
                )
                prev_period_start_ts = pd.Timestamp(prev_period_start_naive, tz="UTC")
            else:
                # For other frequencies (daily, monthly), use to_period
                # Convert to naive timestamp for to_period (pandas requirement)
                ts_naive = ts.tz_localize(None) if ts.tz is not None else ts

                # Map offset frequencies to period frequencies for to_period()
                # MS (Month Start) -> M (Month period)
                period_freq = "M" if freq.upper() == "MS" else freq
                current_period = ts_naive.to_period(period_freq)

                # Get previous period using Period arithmetic
                previous_period = current_period - 1

                # Get start times (these are timezone-naive Timestamps)
                current_period_start_naive = current_period.start_time
                prev_period_start_naive = previous_period.start_time

                # Convert to UTC-aware timestamps
                current_period_start_ts = pd.Timestamp(
                    current_period_start_naive, tz="UTC"
                )
                prev_period_start_ts = pd.Timestamp(prev_period_start_naive, tz="UTC")

            # Helper function to normalize timestamp to UTC and hour boundary
            def normalize_period_ts(period_ts):
                # Ensure it's UTC-aware
                if period_ts.tz is None:
                    period_ts = period_ts.tz_localize("UTC")
                else:
                    period_ts = period_ts.tz_convert("UTC")
                # Normalize to hour boundary (00:00:00)
                period_ts = period_ts.normalize()
                if period_ts.hour != 0 or period_ts.minute != 0:
                    period_ts = period_ts.replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                return period_ts

            # Normalize both timestamps
            current_period_start_ts = normalize_period_ts(current_period_start_ts)
            prev_period_start_ts = normalize_period_ts(prev_period_start_ts)

            period_start = prev_period_start_ts.to_pydatetime()
            # Period end is exclusive (start of current period)
            period_end_exclusive = current_period_start_ts.to_pydatetime()

            # period_ds depends on frequency:
            # - For weekly (W-*): use current period start minus 1 day
            #   (W-SUN period.start_time is Monday, but ds should be Sunday)
            # - For other frequencies: use previous period start
            if freq.upper().startswith("W"):
                # For weekly, ds represents the week containing the date
                # Subtract 1 day because W-SUN period.start_time is Monday,
                # but we want Sunday
                period_ds_ts = current_period_start_ts - pd.Timedelta(days=1)
            else:
                # For daily, monthly, etc., ds represents the previous period
                period_ds_ts = prev_period_start_ts

            # period_ds_ts is already normalized from normalize_period_ts
            period_ds = period_ds_ts.to_pydatetime()

            # Ensure period_end_exclusive is in UTC first
            if period_end_exclusive.tzinfo is None:
                period_end_exclusive = period_end_exclusive.replace(
                    tzinfo=dt.timezone.utc
                )
            else:
                period_end_exclusive = period_end_exclusive.astimezone(dt.timezone.utc)

            # For hourly aggregation, we want the last hour (23:00:00)
            if period_end_exclusive.hour == 0 and period_end_exclusive.minute == 0:
                period_end = period_end_exclusive - pd.Timedelta(hours=1)
            else:
                period_end = period_end_exclusive.replace(
                    minute=0, second=0, microsecond=0
                )

            # Normalize to hour boundary
            period_end = period_end.replace(minute=0, second=0, microsecond=0)

            # Ensure both are timezone-aware in UTC
            if period_start.tzinfo is None:
                period_start = period_start.replace(tzinfo=dt.timezone.utc)
            else:
                period_start = period_start.astimezone(dt.timezone.utc)

            period_start = period_start.replace(minute=0, second=0, microsecond=0)

            # Ensure period_ds is normalized
            if period_ds.tzinfo is None:
                period_ds = period_ds.replace(tzinfo=dt.timezone.utc)
            else:
                period_ds = period_ds.astimezone(dt.timezone.utc)
            period_ds = period_ds.replace(minute=0, second=0, microsecond=0)

            return period_start, period_end, period_ds
        except Exception as e:
            raise ValueError(f"Invalid frequency '{freq}': {e}") from e

    @staticmethod
    def _get_expected_hours(
        period_start: dt.datetime,
        period_end: dt.datetime,
    ) -> int:
        """Calculate expected number of hours in a period."""
        delta = period_end - period_start
        # Add 1 hour because period_end is inclusive of the last hour
        expected_hours = int(delta.total_seconds() / 3600) + 1
        return expected_hours

    def _get_hour_paths_in_period(
        self, period_start: dt.datetime, period_end: dt.datetime
    ) -> list[Path]:
        """Get all hourly file paths that should exist in the period."""
        transformer = HourTransformer(
            raw_events_dir=self.hourly_processed_dir,
            processed_events_dir=self.hourly_processed_dir,
            repos_dir=self.repos_dir,
            first_events_path=self.first_events_path,
        )
        return [
            transformer.hour_local_path(ts.to_pydatetime())
            for ts in pd.date_range(period_start, period_end, freq="h")
        ]

    def _check_completeness(
        self,
        period_start: dt.datetime,
        period_end: dt.datetime,
        threshold: float,
    ) -> tuple[bool, float, int, int]:
        """
        Check if hourly data is complete enough for aggregation.

        Returns:
            Tuple of (is_complete, completeness_ratio, found_hours,
            expected_hours)
        """
        expected_hours = self._get_expected_hours(period_start, period_end)
        hour_paths = self._get_hour_paths_in_period(period_start, period_end)
        found_hours = sum(1 for path in hour_paths if path.exists())
        completeness_ratio = found_hours / expected_hours if expected_hours > 0 else 0.0
        is_complete = completeness_ratio >= threshold
        return is_complete, completeness_ratio, found_hours, expected_hours

    def get_output_path(self, period_start: dt.datetime, freq: str) -> Path:
        """Generate output path for aggregated period."""
        output_dir = get_output_dir_from_freq(
            self.aggregated_events_dir, period_start, freq
        )
        return output_dir / self.output_filename

    def aggregate_period(
        self, date: dt.datetime, freq: str, overwrite: bool = False
    ) -> tuple[Path, dict]:
        """
        Aggregate the previous period up to the given date.

        Args:
            date: Reference date (will be converted to UTC if not already)
            freq: Pandas frequency string (e.g., 'D', 'W', 'M', 'MS')

        Returns:
            Tuple of (output_path, metadata_dict)

        Raises:
            ValueError: If data completeness is below threshold
        """
        period_start, period_end, period_ds = self.get_previous_period(date, freq)

        threshold = self._get_completeness_threshold(freq)
        (
            is_complete,
            completeness_ratio,
            found_hours,
            expected_hours,
        ) = self._check_completeness(period_start, period_end, threshold)

        if not is_complete:
            raise ValueError(
                f"Data completeness ({completeness_ratio:.2%}) below "
                f"threshold ({threshold:.2%}). "
                f"Found {found_hours}/{expected_hours} hours "
                f"for period {period_start} to {period_end}"
            )

        output_path = self.get_output_path(period_start, freq)

        if output_path.exists() and not overwrite:
            return output_path, {
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "completeness_ratio": completeness_ratio,
                "found_hours": found_hours,
                "expected_hours": expected_hours,
                "already_exists": True,
            }

        # Aggregate using DuckDB
        hour_paths = self._get_hour_paths_in_period(period_start, period_end)
        existing_paths = [p for p in hour_paths if p.exists()]

        if not existing_paths:
            raise ValueError(
                f"No hourly data found for period {period_start} to {period_end}"
            )

        con = duckdb.connect(database=":memory:")
        con.execute("INSTALL parquet; LOAD parquet;")

        paths_str = ", ".join(f"'{p.as_posix()}'" for p in existing_paths)
        con.execute(
            f"""
            CREATE TEMP VIEW hourly_data AS
            SELECT repo_id, repo_name, metric, unique_id, ds, y
            FROM read_parquet([{paths_str}]);
        """
        )

        period_ds_normalized = period_ds.replace(minute=0, second=0, microsecond=0)
        period_ds_str = period_ds_normalized.strftime("%Y-%m-%d %H:%M:%S")

        con.execute(
            f"""
            COPY (
              SELECT
                repo_id,
                repo_name,
                metric,
                unique_id,
                TIMESTAMP '{period_ds_str}' AS ds,
                SUM(y)::BIGINT AS y
              FROM hourly_data
              GROUP BY repo_id, repo_name, metric, unique_id
            )
            TO '{output_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        )

        metadata = {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "completeness_ratio": completeness_ratio,
            "found_hours": found_hours,
            "expected_hours": expected_hours,
            "already_exists": False,
        }

        return output_path, metadata
