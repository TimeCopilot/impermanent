import numpy as np
import pandas as pd

from src.forecast.custom_models.fracast import Fracast


class FakeFracastModel:
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def forecast_batch(self, context, *, horizon):
        assert context.shape == (3, 12)
        assert horizon == 3
        rows = np.arange(context.shape[0], dtype=np.float32)[:, None, None]
        horizons = np.arange(horizon, dtype=np.float32)[None, :, None]
        quantiles = np.arange(len(self.quantiles), dtype=np.float32)[None, None, :]
        return rows + horizons + quantiles


def test_fracast_forecast_layout():
    timestamps = pd.date_range("2026-01-01", periods=12, freq="D")
    rows = []
    for series in range(3):
        rows.extend(
            {
                "unique_id": f"repo:{series}",
                "ds": timestamp,
                "y": float(series + timestamp.dayofyear),
            }
            for timestamp in timestamps
        )
    model = Fracast(model=FakeFracastModel(), device="cpu")

    forecast = model.forecast(pd.DataFrame(rows), h=3, freq="D")

    assert len(forecast) == 9
    assert list(forecast.columns) == [
        "unique_id",
        "ds",
        "fracast-0",
        *[
            f"fracast-0-q-{quantile}"
            for quantile in (10, 20, 30, 40, 50, 60, 70, 80, 90)
        ],
    ]
    assert not forecast.isna().any().any()
    first = forecast[forecast["unique_id"] == "repo:0"].iloc[0]
    assert first["fracast-0"] == 4.0
    assert first["fracast-0-q-90"] == 8.0
