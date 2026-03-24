"""GHArchiveForecaster: forecast path management & orchestration on GH Archive data."""

import datetime as dt
import logging
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import pandas as pd

from src.data.gh_archive.utils.gh_archive_data import GHArchiveData

logger = logging.getLogger(__name__)

# Key separator for composite (model, cutoff) keys used in validation.
KEY_SEP = "::"
S3_FCST_PATH = "s3://impermanent-benchmark/v0.1.0/gh-archive/forecasts"


class GHArchiveForecaster(GHArchiveData):
    """Extends :class:`GHArchiveData` with forecast path management.

    Inherits data loading (``get_df``, ``get_actuals``) from the parent
    and adds helpers to compute output paths, check existence, generate
    cutoff ranges, and run+save forecasts.

    Args:
        frequency: One of ``"hourly"``, ``"daily"``, ``"weekly"``, ``"monthly"``.
        base_path: Base path for reading processed-events data (local or S3).
        forecasts_dir: Root directory where forecast parquet files are stored.

    Example:
        >>> f = GHArchiveForecaster(
        ...     "daily",
        ...     base_path="/s3-bucket/v0.1.0/gh-archive/processed-events",
        ...     forecasts_dir="/s3-bucket/v0.1.0/gh-archive/forecasts",
        ... )
        >>> f.forecast_path(
        ...     "2026-01-15",
        ...     "auto_arima",
        ... )
        '/s3-bucket/.../daily/auto_arima/year=2026/month=01/day=15/forecasts.parquet'
    """

    def __init__(
        self,
        frequency: str,
        base_path: str | Path | None = None,
        forecasts_dir: str | Path | None = None,
    ):
        super().__init__(frequency, base_path)
        if forecasts_dir is None:
            self._forecasts_dir: str | Path = S3_FCST_PATH
        elif isinstance(forecasts_dir, str) and forecasts_dir.startswith("s3://"):
            self._forecasts_dir = forecasts_dir
        else:
            self._forecasts_dir = Path(forecasts_dir)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def forecast_path(
        self,
        cutoff: str | dt.datetime,
        model_name: str,
        filename: str = "forecasts.parquet",
        mkdir: bool = False,
    ) -> str:
        """Full path for a forecast file.

        Reuses ``_date_to_partition`` from :class:`GHArchiveData` so path
        layout is consistent with the processed-events data. Uses string
        joining for S3 URIs so that ``s3://`` is preserved (pathlib would
        collapse it to ``s3:/``).
        """
        cutoff_dt = self._parse_cutoff(cutoff)
        partition = self._date_to_partition(cutoff_dt)
        if isinstance(self._forecasts_dir, str) and self._forecasts_dir.startswith(
            "s3://"
        ):
            base = self._forecasts_dir.rstrip("/")
            return f"{base}/{self.frequency}/{model_name}/{partition}/{filename}"
        path = self._forecasts_dir / self.frequency / model_name / partition  # type: ignore
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return str(path / filename)

    def forecast_exists(
        self,
        cutoff: str | dt.datetime,
        model_name: str,
        filename: str = "forecasts.parquet",
    ) -> bool:
        """Check whether a forecast file already exists (local disk or S3)."""
        path = self.forecast_path(cutoff, model_name, filename)
        if path.startswith("s3://"):
            return self._s3_object_exists(path)
        return Path(path).exists()

    def _s3_object_exists(self, s3_uri: str) -> bool:
        """Return True if the S3 object exists."""
        parsed = urlparse(s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            return False
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        try:
            import boto3

            client = boto3.client("s3")
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Cutoff range generation
    # ------------------------------------------------------------------

    def cutoff_range(self, start_date: str | dt.datetime) -> list[str]:
        """Generate cutoff date strings spaced by the forecast horizon.

        The step between consecutive cutoffs equals ``h × base_freq``
        (e.g., 7D for daily, 24h for hourly).

        Args:
            start_date: First cutoff.  Required when *last_cutoffs* is None.
            last_cutoffs: Number of most-recent cutoffs to generate
                (counted backwards from now).

        Returns:
            Sorted list of cutoff strings in ``"YYYY-MM-DD-HH"`` format.
        """
        base_freq = self.freq.split("-")[0]  # "W-SUN" → "W"
        cutoff_step = f"{self.h}{base_freq}"

        end = dt.datetime.now(dt.timezone.utc)

        kwargs: dict = {"freq": cutoff_step, "end": end}
        kwargs["start"] = self._parse_cutoff(start_date)

        dates = pd.date_range(**kwargs)
        if self.frequency in ["monthly", "weekly"]:
            dates = dates[:-1]  # exclude the current month or week
        cutoff_range = [d.strftime("%Y-%m-%d-%H") for d in dates]
        logger.info(f"Generated cutoff range: {cutoff_range}")
        return cutoff_range

    def run_forecast(
        self,
        model_name: str,
        cutoff: str,
        max_window_size: int | None = None,
        overwrite: bool = False,
    ) -> str | None:
        """Load data, generate a forecast, and save to *forecasts_dir*.

        Args:
            model_name: Key in the model registry.
            cutoff: Cutoff date string.
            max_window_size: Number of historical periods for training.
                Defaults to ``self.max_window_size``.
            overwrite: If ``False`` and the forecast already exists, skip.

        Returns:
            Path to saved forecast parquet, or ``None`` if skipped / failed.
        """
        from src.forecast.forecast import generate_forecast

        if max_window_size is None:
            max_window_size = self.max_window_size

        if not overwrite and self.forecast_exists(cutoff, model_name):
            logger.info(
                f"Forecast exists for {model_name}/{self.frequency}/"
                f"{cutoff}, skipping",
            )
            return None

        logger.info(f"Loading data for cutoff {cutoff}")
        df = self.get_df(
            cutoff=cutoff,
            max_window_size=max_window_size,
            sort=False,
        )
        max_ds = df.groupby("unique_id")["ds"].max().unique()
        if len(max_ds) > 1:
            raise ValueError(f"Multiple max ds for unique_id: {max_ds}")
        parsed_cutoff = self._parse_cutoff(cutoff).replace(tzinfo=None)
        if max_ds[0] != parsed_cutoff:
            raise ValueError(
                f"Actual cutoff {max_ds[0]} != requested cutoff {parsed_cutoff}"
            )

        logger.info(
            f"Loaded {len(df)} rows, {df['unique_id'].nunique()} series",
        )

        try:
            forecasts = generate_forecast(model_name, df, h=self.h, freq=self.freq)
            forecasts["cutoff"] = cutoff

            output_path = self.forecast_path(cutoff, model_name, mkdir=True)

            con = duckdb.connect(":memory:")
            con.execute("INSTALL parquet; LOAD parquet;")
            con.register("forecasts_df", forecasts)
            con.execute(
                f"COPY forecasts_df TO '{output_path}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD);"
            )

            logger.info(f"Saved forecasts to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Model {model_name} failed for cutoff {cutoff}: {e}")
            return None

    @staticmethod
    def make_key(model_name: str, cutoff: str) -> str:
        """Encode ``(model_name, cutoff)`` as a single string key."""
        return f"{model_name}{KEY_SEP}{cutoff}"

    @staticmethod
    def split_key(key: str) -> tuple[str, str]:
        """Decode a composite key back into ``(model_name, cutoff)``."""
        model_name, cutoff = key.split(KEY_SEP, 1)
        return model_name, cutoff
