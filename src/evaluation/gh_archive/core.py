"""GHArchiveEvaluator: evaluation orchestration on GH Archive forecasts."""

import datetime as dt
import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.forecast.gh_archive.core import GHArchiveForecaster

logger = logging.getLogger(__name__)


class GHArchiveEvaluator(GHArchiveForecaster):
    """Extends :class:`GHArchiveForecaster` with evaluation path management.

    Inherits data loading (``get_df``, ``get_actuals``) and forecast path
    helpers from the parent, and adds evaluation output paths, existence
    checks, and a ``run_evaluation`` method.

    Args:
        frequency: One of ``"hourly"``, ``"daily"``, ``"weekly"``, ``"monthly"``.
        base_path: Base path for reading processed-events data (local or S3).
        forecasts_dir: Root directory where forecast parquet files are stored.
        eval_dir: Root directory where evaluation metrics are saved.
    """

    def __init__(
        self,
        frequency: str,
        base_path: str | Path | None = None,
        forecasts_dir: str | Path | None = None,
        eval_dir: str | Path | None = None,
    ):
        super().__init__(frequency, base_path, forecasts_dir)
        self._eval_dir: Path | None = Path(eval_dir) if eval_dir else None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def eval_path(
        self,
        cutoff: str | dt.datetime,
        model_name: str,
        filename: str = "metrics.parquet",
        mkdir: bool = False,
    ) -> str:
        """Full path for an evaluation metrics file.

        Layout mirrors forecast paths:
        ``{eval_dir}/{frequency}/{model_name}/{partition}/{filename}``.
        """
        if self._eval_dir is None:
            raise ValueError("eval_dir was not set")
        cutoff_dt = self._parse_cutoff(cutoff)
        partition = self._date_to_partition(cutoff_dt)
        path = self._eval_dir / self.frequency / model_name / partition
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return str(path / filename)

    def eval_exists(
        self,
        cutoff: str | dt.datetime,
        model_name: str,
        filename: str = "metrics.parquet",
    ) -> bool:
        """Check whether an evaluation metrics file already exists."""
        return Path(self.eval_path(cutoff, model_name, filename)).exists()

    # ------------------------------------------------------------------
    # Evaluation execution
    # ------------------------------------------------------------------
    def get_forecast(
        self,
        cutoff: str | dt.datetime,
        model_name: str,
    ) -> pd.DataFrame:
        """Get the forecast for a given cutoff and model name.

        Uses DuckDB so that both local and S3 paths are supported
        (same pattern as :class:`GHArchiveData` for processed-events).
        """
        forecast_path = self.forecast_path(cutoff, model_name)
        con = self._make_connection()
        try:
            forecast_df = con.execute(
                "SELECT * FROM read_parquet(?)", [forecast_path]
            ).fetchdf()
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg or "Not Found" in err_msg:
                from src.forecast.forecast import MODELS

                available = list(MODELS.keys())
                raise FileNotFoundError(
                    f"Forecast file not found at {forecast_path}. "
                    "Check that (1) the forecast was run for this cutoff and model, "
                    "and (2) model_name matches the registry"
                    "(e.g. auto_arima not auto-arima). "
                    f"Available model names: {available}"
                ) from e
            raise
        return forecast_df

    def run_evaluation(
        self,
        model_name: str,
        cutoff: str,
        max_window_size: int | None = None,
        overwrite: bool = False,
    ) -> str | None:
        """Load forecast + data, evaluate, and save metrics.

        Writes one row per series (``unique_id``) with ``mase`` and ``scaled_crps``.
        The leaderboard aggregates these with per-subdataset medians.

        Args:
            model_name: Key in the model registry.
            cutoff: Cutoff date string.
            max_window_size: Number of historical periods for MASE scaling.
                Defaults to ``self.max_window_size``.
            overwrite: If ``False`` and evaluation already exists, skip.

        Returns:
            Path to saved metrics parquet, or ``None`` if skipped / failed.
        """
        from src.evaluation.evaluate import evaluate_forecast

        if max_window_size is None:
            max_window_size = self.max_window_size

        if not overwrite and self.eval_exists(cutoff, model_name):
            logger.info(
                f"Evaluation exists for {model_name}/{self.frequency}/"
                f"{cutoff}, skipping",
            )
            return None

        if not self.forecast_exists(cutoff, model_name):
            logger.warning(
                f"Forecast not found for {model_name}/{self.frequency}/"
                f"{cutoff}, skipping",
            )
            return None

        forecast_path = self.forecast_path(cutoff, model_name)
        forecast_df = self.get_forecast(cutoff, model_name)
        min_ds = pd.to_datetime(forecast_df["ds"].min()).replace(tzinfo=None)
        cutoff_dt = self._parse_cutoff(forecast_df["cutoff"].iloc[0]).replace(
            tzinfo=None
        )
        if min_ds < cutoff_dt:
            raise ValueError(
                "Forecast contains data before the cutoff "
                f"for model {model_name} and cutoff {cutoff}. "
                f"Initial forecast ds: {min_ds}, "
                f"Actual cutoff: {cutoff}"
            )
        logger.info(
            f"Loaded forecast: {len(forecast_df)} rows from {forecast_path}",
        )

        train_df = self.get_df(
            cutoff=cutoff, max_window_size=max_window_size, sort=False
        )
        logger.info(f"Loaded training data: {len(train_df)} rows")

        actuals_df = self.get_actuals(cutoff=cutoff)
        logger.info(f"Loaded actuals: {len(actuals_df)} rows")

        if len(actuals_df) == 0:
            logger.warning(f"No actuals available for cutoff {cutoff}, skipping")
            return None

        try:
            per_uid_df, model_alias = evaluate_forecast(
                forecast_df=forecast_df,
                actuals_df=actuals_df,
                train_df=train_df,
            )

            per_uid_df["subdataset"] = (
                per_uid_df["unique_id"].str.rsplit(":", n=1).str[1]
            )

            if per_uid_df[["mase", "scaled_crps"]].isna().any().any():
                raise ValueError(
                    f"Metrics contain NaNs for model {model_name} and cutoff {cutoff}"
                )

            per_uid_df["model"] = model_name
            per_uid_df["model_alias"] = model_alias
            per_uid_df["cutoff"] = cutoff
            per_uid_df["frequency"] = self.frequency
            metrics_df = per_uid_df[
                [
                    "unique_id",
                    "model",
                    "model_alias",
                    "cutoff",
                    "frequency",
                    "subdataset",
                    "mase",
                    "scaled_crps",
                ]
            ]
            logger.info(f"Per-series metrics: {len(metrics_df)} rows")

            output_path = self.eval_path(cutoff, model_name, mkdir=True)

            con = duckdb.connect(":memory:")
            con.execute("INSTALL parquet; LOAD parquet;")
            con.register("metrics_df", metrics_df)
            con.execute(
                f"COPY metrics_df TO '{output_path}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD);"
            )

            logger.info(
                f"Saved {len(metrics_df)} per-series metrics to {output_path} "
                f"{metrics_df.to_markdown()}",
            )
            return output_path

        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}/{cutoff}: {e}")
            raise e
