"""Fracast-0 adapter for the Impermanent forecasting registry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.forecast.constants import QUANTILES


class Fracast:
    """Forecast GH-activity series with the released Fracast-0 model."""

    alias = "fracast-0"

    def __init__(
        self,
        model_id: str = "ztxtech/fracast-0",
        weights: str = "fp32",
        batch_size: int = 64,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.device = device
        if model is None:
            from fracast import FracastModel

            model = FracastModel.from_pretrained(
                model_id, weights=weights, device=device
            )
        self._model = model

    def _infer_freq(self, df: pd.DataFrame, freq: str | None) -> str:
        if freq is not None:
            return freq
        first_id = df["unique_id"].iloc[0]
        timestamps = df.loc[df["unique_id"] == first_id, "ds"].sort_values()
        inferred = pd.infer_freq(timestamps)
        if inferred is None:
            raise ValueError("could not infer a regular forecast frequency")
        return inferred

    def _select_quantiles(
        self, forecasts: np.ndarray, quantiles: list[float]
    ) -> np.ndarray:
        model_quantiles = np.asarray(self._model.quantiles, dtype=np.float32)
        if np.allclose(quantiles, model_quantiles):
            return forecasts
        selected = np.empty((*forecasts.shape[:2], len(quantiles)), dtype=np.float32)
        for index, quantile in enumerate(quantiles):
            selected[:, :, index] = np.stack(
                [
                    np.interp(quantile, model_quantiles, forecasts[row, :, :])
                    for row in range(forecasts.shape[0])
                ]
            )
        return selected

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """Return point and quantile forecasts for every input series."""
        if level is not None:
            raise ValueError("Fracast supports quantiles but not intervals")
        if not isinstance(h, (int, np.integer)) or h < 1:
            raise ValueError("h must be a positive integer")
        requested = QUANTILES if quantiles is None else list(quantiles)
        if len(requested) == 0:
            raise ValueError("quantiles must not be empty")

        work = df.loc[:, ["unique_id", "ds", "y"]].copy()
        work["ds"] = pd.to_datetime(work["ds"])
        max_ds = work["ds"].max()
        same_end = work.groupby("unique_id", observed=True)["ds"].max()
        if not (same_end == max_ds).all():
            raise ValueError("every series must end at the same cutoff")

        grouped = work.groupby("unique_id", sort=True, observed=True)
        ids: list[str] = []
        contexts: list[np.ndarray] = []
        for unique_id, group in grouped:
            values = group.sort_values("ds")["y"].to_numpy(dtype=np.float32)
            values[~np.isfinite(values)] = np.nan
            ids.append(str(unique_id))
            contexts.append(values)

        width = max(len(values) for values in contexts)
        context = np.full((len(contexts), width), np.nan, dtype=np.float32)
        for row, values in enumerate(contexts):
            context[row, -len(values) :] = values

        forecasts = np.concatenate(
            [
                self._model.forecast_batch(
                    context[start : start + self.batch_size], horizon=int(h)
                )
                for start in range(0, len(context), self.batch_size)
            ],
            axis=0,
        )
        forecasts = self._select_quantiles(forecasts, requested)
        median = int(np.argmin(np.abs(np.asarray(requested) - 0.5)))
        future = pd.date_range(
            start=max_ds,
            periods=int(h) + 1,
            freq=self._infer_freq(work, freq),
        )[1:]

        result = pd.DataFrame(
            {
                "unique_id": np.repeat(np.asarray(ids), int(h)),
                "ds": np.tile(future, len(ids)),
                self.alias: forecasts[:, :, median].reshape(-1),
            }
        )
        for index, quantile in enumerate(requested):
            result[f"{self.alias}-q-{int(round(quantile * 100))}"] = forecasts[
                :, :, index
            ].reshape(-1)
        return result
