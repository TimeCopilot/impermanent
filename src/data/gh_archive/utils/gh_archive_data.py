"""GHArchiveData: retrieve and query GH Archive time series from S3 or local."""

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import boto3
import duckdb
import pandas as pd
import pyarrow.parquet as pq
import typer

from .parquet_validator import ParquetValidator, ValidationReport

logger = logging.getLogger(__name__)

FrequencyType = Literal["hourly", "daily", "weekly", "monthly"]

# S3 bucket configuration
S3_BUCKET = "impermanent-benchmark"
S3_PREFIX = "v0.1.0/gh-archive/processed-events"


class GHArchiveData:
    """Retrieve aggregated GitHub Archive time series for forecasting.

    Data is stored in Hive-style partitions.  This class uses
    ``pd.date_range`` to compute the exact set of partition paths needed
    for a given *cutoff* + *max_window_size* and reads them with DuckDB.

    Attributes:
        frequency: The frequency type ("hourly", "daily", "weekly", "monthly").
        h: Forecast horizon for the given frequency.
        freq: Pandas frequency string for the given frequency.

    Example:
        >>> # From S3 (default)
        >>> data = (
        ...     GHArchiveData(
        ...         "daily"
        ...     )
        ... )
        >>> df = data.get_df(
        ...     cutoff="2026-01-15",
        ...     max_window_size=30,
        ... )

        >>> # From local filesystem (e.g., Modal volume)
        >>> data = GHArchiveData(
        ...     "daily",
        ...     base_path="/s3-bucket/v0.1.0/gh-archive-metrics/processed-events",
        ... )
        >>> df = data.get_df(
        ...     cutoff="2026-01-15",
        ...     max_window_size=30,
        ... )
    """

    FREQ_CONFIG: dict[str | FrequencyType, dict] = {
        "hourly": {"h": 24, "freq": "h", "max_window_size": 1_024},
        "daily": {"h": 7, "freq": "D", "max_window_size": 512},
        "weekly": {"h": 1, "freq": "W-SUN", "max_window_size": 114},
        "monthly": {"h": 1, "freq": "MS", "max_window_size": 24},
    }

    def __init__(
        self,
        frequency: str | FrequencyType,
        base_path: str | Path | None = None,
    ):
        """Initialize GHArchiveData with a specific frequency.

        Args:
            frequency: One of "hourly", "daily", "weekly", "monthly".
            base_path: Optional base path for local filesystem access.
                If None, reads from S3. If provided, reads from local
                filesystem (e.g., Modal volume mount).

        Raises:
            ValueError: If frequency is not a valid option.
        """
        if frequency not in self.FREQ_CONFIG:
            valid = ", ".join(self.FREQ_CONFIG.keys())
            raise ValueError(
                f"Invalid frequency '{frequency}'. Must be one of: {valid}"
            )

        self.frequency = frequency
        self.h: int = self.FREQ_CONFIG[frequency]["h"]
        self.freq: str = self.FREQ_CONFIG[frequency]["freq"]
        self.max_window_size: int = self.FREQ_CONFIG[frequency]["max_window_size"]

        self._use_s3 = base_path is None
        self._base_path: Path | None = (
            Path(base_path) / frequency if base_path else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_df(
        self,
        cutoff: str | dt.datetime,
        max_window_size: int | None = None,
        sort: bool = True,
    ) -> pd.DataFrame:
        """Get DataFrame with the last *max_window_size* periods up to *cutoff*.

        Uses ``pd.date_range`` to compute which partition files to read,
        then queries them with DuckDB.

        Args:
            cutoff: The last timestamp to include.  Accepts
                "YYYY-MM-DD-HH", "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD",
                or a datetime object.
            max_window_size: Number of periods to return per unique_id.
                Defaults to ``self.max_window_size``.
            sort: If ``True`` (default), sort the result by
                ``unique_id, ds``.  Set to ``False`` to skip the sort
                when the caller does not need ordered output, which is
                significantly faster for large datasets.

        Returns:
            DataFrame with columns: unique_id, ds, y.

        Raises:
            ValueError: If no parquet files are found for the requested range.
        """
        if max_window_size is None:
            max_window_size = self.max_window_size
        cutoff_dt = self._parse_cutoff(cutoff)
        dates = pd.date_range(end=cutoff_dt, periods=max_window_size, freq=self.freq)
        # check_exists=False on mount avoids N stat() calls (very slow over FUSE)
        paths = self._resolve_paths(dates, check_exists=False)

        if not paths:
            raise ValueError(
                f"No parquet files found for frequency '{self.frequency}' "
                f"up to {cutoff_dt.isoformat()}"
            )

        return self._read_parquet_paths(
            paths,
            sort=sort,
            columns=("unique_id", "ds", "y"),
        )

    def get_actuals(
        self,
        cutoff: str | dt.datetime,
    ) -> pd.DataFrame:
        """Get the next *h* periods after *cutoff* (for evaluation).

        Args:
            cutoff: The cutoff timestamp (same formats as ``get_df``).

        Returns:
            DataFrame with columns: unique_id, ds, y.

        Raises:
            ValueError: If no parquet files are found after the cutoff.
        """
        cutoff_dt = self._parse_cutoff(cutoff)
        # Skip the cutoff period itself; take the next h periods.
        dates = pd.date_range(start=cutoff_dt, periods=self.h + 1, freq=self.freq)[1:]
        paths = self._resolve_paths(dates, check_exists=False)

        if not paths:
            raise ValueError(
                f"No parquet files found for frequency '{self.frequency}' "
                f"after {cutoff_dt.isoformat()}"
            )

        return self._read_parquet_paths(
            paths, sort=True, columns=("unique_id", "ds", "y")
        )

    def validate_files(
        self,
        start_date: str | dt.datetime,
        end_date: str | dt.datetime | None = None,
    ) -> ValidationReport:
        """Validate parquet files in a date range.

        Generates all expected partition paths between *start_date* and
        *end_date* and delegates to ``ParquetValidator``.

        Args:
            start_date: Start of the date range.
            end_date: End of the date range (inclusive).  Defaults to now.

        Returns:
            A ``ValidationReport`` with per-file results.
        """
        start_dt = self._parse_cutoff(start_date)
        end_dt = (
            dt.datetime.now(dt.timezone.utc)
            if end_date is None
            else self._parse_cutoff(end_date)
        )

        if self._use_s3:
            raise NotImplementedError(
                "S3 validation not yet supported. Use base_path for local validation."
            )

        dates = pd.date_range(start_dt, end_dt, freq=self.freq)
        paths = [
            self._base_path / self._date_to_partition(d) / "series.parquet"  # type: ignore
            for d in dates
        ]

        validator = ParquetValidator()
        return validator.validate_files(paths)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_cutoff(cutoff: str | dt.datetime) -> dt.datetime:
        """Parse *cutoff* to a UTC datetime.

        Args:
            cutoff: Cutoff date as string or datetime.

        Returns:
            Cutoff as datetime with UTC timezone.
        """
        if isinstance(cutoff, str):
            for fmt in ["%Y-%m-%d-%H", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    parsed = dt.datetime.strptime(cutoff, fmt)
                    return parsed.replace(tzinfo=dt.timezone.utc)
                except ValueError:
                    continue
            raise ValueError(
                f"Could not parse cutoff '{cutoff}'. "
                "Expected format: YYYY-MM-DD-HH, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD"
            )

        if cutoff.tzinfo is None:
            return cutoff.replace(tzinfo=dt.timezone.utc)
        return cutoff.astimezone(dt.timezone.utc)

    def _date_to_partition(self, date: dt.datetime | pd.Timestamp) -> str:
        """Map a date to its Hive-style partition path (relative)."""
        if self.frequency == "hourly":
            return (
                f"year={date.year:04d}/month={date.month:02d}"
                f"/day={date.day:02d}/hour={date.hour:02d}"
            )
        if self.frequency == "daily":
            return f"year={date.year:04d}/month={date.month:02d}/day={date.day:02d}"
        if self.frequency == "weekly":
            iso_year, iso_week, _ = date.isocalendar()
            return f"year={iso_year:04d}/week={iso_week:02d}"
        if self.frequency == "monthly":
            return f"year={date.year:04d}/month={date.month:02d}"
        raise ValueError(f"Unknown frequency: {self.frequency}")

    def _resolve_paths(
        self, dates: pd.DatetimeIndex, check_exists: bool = True
    ) -> list[str]:
        """Convert dates to file paths (S3 URIs or local paths).

        Args:
            dates: Timestamps to resolve.
            check_exists: When ``True`` (default) and using local paths,
                only return files that actually exist on disk.  Set to
                ``False`` to skip the per-file ``stat()`` call, which is
                much faster on FUSE / network mounts.
        """
        partitions = [self._date_to_partition(d) for d in dates]

        if self._use_s3:
            return [
                f"s3://{S3_BUCKET}/{S3_PREFIX}/{self.frequency}/{p}/series.parquet"
                for p in partitions
            ]

        if check_exists:
            paths = []
            for p in partitions:
                full = self._base_path / p / "series.parquet"  # type: ignore
                if full.exists():
                    paths.append(str(full))
            return paths

        return [
            str(self._base_path / p / "series.parquet")  # type: ignore
            for p in partitions
        ]

    def _read_parquet_paths(
        self,
        paths: list[str],
        *,
        sort: bool = True,
        columns: tuple[str, ...] = ("unique_id", "ds", "y"),
    ) -> pd.DataFrame:
        """Read multiple parquet files and return a single DataFrame.

        Uses parallel PyArrow reads for local/mount paths (faster on FUSE);
        uses DuckDB for S3 (httpfs).
        """
        if self._use_s3:
            con = self._make_connection()
            paths_str = ", ".join(f"'{p}'" for p in paths)
            col_list = ", ".join(columns)
            query = f"SELECT {col_list} FROM read_parquet([{paths_str}])"
            if sort:
                query += " ORDER BY unique_id, ds"
            return con.execute(query).fetchdf()

        # Local/mount: parallel reads with PyArrow to saturate FUSE/network
        max_workers = min(32, len(paths))

        def _read_one(path: str):
            t = pq.read_table(path, columns=list(columns))
            return t

        tables = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_read_one, p): p for p in paths}
            for future in as_completed(futures):
                tables.append(future.result())

        if not tables:
            return pd.DataFrame(columns=list(columns))

        import pyarrow as pa

        combined = pa.concat_tables(tables)
        df = combined.to_pandas()
        if sort and len(df) > 0:
            df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        return df

    def _make_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a DuckDB connection, configured for S3 when needed."""
        con = duckdb.connect(":memory:")
        if self._use_s3:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            con.execute("SET s3_region='us-east-1';")
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials:
                frozen = credentials.get_frozen_credentials()
                con.execute(f"SET s3_access_key_id='{frozen.access_key}';")
                con.execute(f"SET s3_secret_access_key='{frozen.secret_key}';")
                if frozen.token:
                    con.execute(f"SET s3_session_token='{frozen.token}';")
        return con


app = typer.Typer()


@app.command()
def preview(
    frequency: str = typer.Option(
        ...,
        "--frequency",
        "-f",
        help="Frequency: hourly, daily, weekly, or monthly",
    ),
    cutoff: str = typer.Option(
        ...,
        "--cutoff",
        "-c",
        help="Cutoff date (YYYY-MM-DD-HH, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD)",
    ),
    max_window_size: int = typer.Option(
        ...,
        "--max-window-size",
        "-w",
        help="Number of periods to return per unique_id",
    ),
    head: int = typer.Option(
        20,
        "--head",
        "-n",
        help="Number of rows to display (default: 20)",
    ),
) -> None:
    """Fetch GH Archive data and display a preview."""
    try:
        data = GHArchiveData(frequency=frequency)
        print(f"Frequency: {data.frequency}")
        print(f"Forecast horizon (h): {data.h}")
        print(f"Pandas freq: {data.freq}")
        print(f"Cutoff: {cutoff}")
        print(f"Max window size: {max_window_size}")
        print("-" * 60)

        df = data.get_df(cutoff=cutoff, max_window_size=max_window_size)

        print(f"Shape: {df.shape}")
        print(f"Unique series: {df['unique_id'].nunique()}")
        print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
        print("-" * 60)
        print(df.head(head).to_string())

    except ValueError as e:
        print(f"Error: {e}")
        raise typer.Exit(1) from e


@app.command()
def validate(
    frequency: str = typer.Option(
        ...,
        "--frequency",
        "-f",
        help="Frequency: hourly, daily, weekly, or monthly",
    ),
    start_date: str = typer.Option(
        ...,
        "--start-date",
        "-s",
        help="Start date (YYYY-MM-DD-HH, YYYY-MM-DD HH:MM:SS, or YYYY-MM-DD)",
    ),
    end_date: str = typer.Option(
        None,
        "--end-date",
        "-e",
        help="End date (defaults to now UTC)",
    ),
    base_path: str = typer.Option(
        None,
        "--base-path",
        "-b",
        help="Local base path (if omitted, reads from S3)",
    ),
) -> None:
    """Validate that all parquet files in a date range are readable and non-empty."""
    try:
        data = GHArchiveData(frequency=frequency, base_path=base_path)
        source = base_path if base_path else f"s3://{S3_BUCKET}/{S3_PREFIX}"
        print(f"Validating {frequency} files from {source}")
        print(f"  start_date: {start_date}")
        print(f"  end_date:   {end_date or 'now (UTC)'}")
        print("-" * 60)

        report = data.validate_files(
            start_date=start_date,
            end_date=end_date,
        )

        print(report.summary())

        if not report.all_ok:
            raise typer.Exit(1)

    except ValueError as e:
        print(f"Error: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
