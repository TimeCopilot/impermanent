"""Forecast generation module for GitHub time series data."""

from .forecast import (
    CPU_MODELS,
    GPU_MODELS,
    MODELS,
    QUANTILES,
    generate_forecast,
    get_model,
    is_gpu_model,
)

__all__ = [
    "CPU_MODELS",
    "GPU_MODELS",
    "MODELS",
    "QUANTILES",
    "generate_forecast",
    "get_model",
    "is_gpu_model",
]
