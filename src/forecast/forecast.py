"""Model registry and forecast generation logic."""

from collections.abc import Callable

import pandas as pd
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.moirai import Moirai
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.tirex import TiRex
from timecopilot.models.prophet import Prophet
from timecopilot.models.stats import (
    AutoARIMA,
    AutoCES,
    AutoETS,
    DynamicOptimizedTheta,
    HistoricAverage,
    SeasonalNaive,
    ZeroModel,
)

from src.forecast.constant_models import ConstantForecastModel
from src.forecast.constants import QUANTILES

# Model categories for infrastructure selection
# CPU models: Statistical and ML models that don't need GPU
CPU_MODELS: dict[str, Callable] = {
    # Baseline models
    "zero_model": lambda: ZeroModel(),
    "historic_average": lambda: HistoricAverage(),
    "seasonal_naive": lambda: SeasonalNaive(),
    # Example
    "constant_one": lambda: ConstantForecastModel(1.0, alias="constant_one"),
    # Statistical models
    "auto_arima": lambda: AutoARIMA(),
    "auto_ets": lambda: AutoETS(),
    "auto_ces": lambda: AutoCES(),
    "dynamic_optimized_theta": lambda: DynamicOptimizedTheta(),
    "prophet": lambda: Prophet(),
}

# GPU models: Neural and Foundation models that benefit from GPU
GPU_MODELS: dict[str, Callable] = {
    # Foundation models
    "chronos-2": lambda: Chronos(repo_id="amazon/chronos-2", batch_size=64),
    "moirai": lambda: Moirai(batch_size=64),
    "timesfm": lambda: TimesFM(
        repo_id="google/timesfm-2.5-200m-pytorch",
        batch_size=64,
    ),
    "tirex": lambda: TiRex(batch_size=64),
}

# Combined registry
MODELS: dict[str, Callable] = {**CPU_MODELS, **GPU_MODELS}


def get_model(name: str):
    """Get a model instance by name.

    Args:
        name: Model name (must be a key in MODELS registry).

    Returns:
        Instantiated model object.

    Raises:
        ValueError: If model name is not in registry.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name]()


def is_gpu_model(name: str) -> bool:
    """Check if a model requires GPU.

    Args:
        name: Model name.

    Returns:
        True if model requires GPU, False otherwise.
    """
    return name in GPU_MODELS


def generate_forecast(
    model_name: str,
    df: pd.DataFrame,
    h: int,
    freq: str,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """Generate forecasts using specified model's forecast method.

    Args:
        model_name: Name of the model to use (from MODELS registry).
        df: Input DataFrame with columns: unique_id, ds, y.
        h: Forecast horizon (number of periods to forecast).
        freq: Pandas frequency string (e.g., "D", "h", "W-SUN").
        quantiles: List of quantiles to compute (e.g., [0.1, 0.2, ..., 0.9]).
            If None, uses QUANTILES constant.

    Returns:
        DataFrame with forecast results including quantile columns.
    """
    if quantiles is None:
        quantiles = QUANTILES

    model = get_model(model_name)

    # Some models (AutoLGBM, AutoNHITS, AutoTFT) don't support quantiles yet
    # For those, we just return point forecasts
    try:
        return model.forecast(df=df, h=h, freq=freq, quantiles=quantiles)
    except (ValueError, TypeError):
        # Fallback to point forecast only
        return model.forecast(df=df, h=h, freq=freq)
