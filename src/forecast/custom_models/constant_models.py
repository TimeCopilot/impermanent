"""Minimal forecasters for tests and as integration examples.

These mirror the column layout expected by
:func:`src.forecast.forecast.generate_forecast` and
:func:`src.evaluation.evaluate.evaluate_forecast` (point column + ``{alias}-q-{pp}``).
"""

import numpy as np
import pandas as pd
from timecopilot.models.utils.forecaster import Forecaster

from src.forecast.constants import QUANTILES


class ConstantForecastModel(Forecaster):
    """Point and quantile forecasts equal a fixed value (default ``1.0``).

    Subclasses :class:`timecopilot.models.utils.forecaster.Forecaster`. Use as a
    sanity-check baseline and a minimal template for adding models: ``forecast``
    returns ``unique_id``, ``ds``, ``{alias}``, and ``{alias}-q-10`` … ``{alias}-q-90``.
    """

    def __init__(self, value: float = 1.0, alias: str = "constant_one") -> None:
        self.value = float(value)
        self.alias = alias

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        if level is not None:
            raise NotImplementedError(
                "ConstantForecastModel does not support ``level``"
            )
        inferred = self._maybe_infer_freq(df, freq)
        if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
            df = df.copy()
            df["ds"] = pd.to_datetime(df["ds"])

        # Last observed timestamp is shared across series in this benchmark loader.
        max_ds = df["ds"].max()
        future = pd.date_range(
            start=max_ds,
            periods=h + 1,
            freq=inferred,
        )[1:]

        qs = QUANTILES if quantiles is None else quantiles
        uids = df["unique_id"].unique()
        n_u, n_f = len(uids), len(future)
        # Vectorized (series × horizon); no per-row Python loop.
        out = pd.DataFrame(
            {
                "unique_id": np.repeat(uids, n_f),
                "ds": np.tile(np.asarray(future, dtype="datetime64[ns]"), n_u),
                self.alias: self.value,
            }
        )
        # Same constant for each quantile column; names match evaluate.py.
        q_block = {
            f"{self.alias}-q-{int(round(q * 100))}": np.full(n_u * n_f, self.value)
            for q in qs
        }
        return pd.concat([out, pd.DataFrame(q_block)], axis=1)
