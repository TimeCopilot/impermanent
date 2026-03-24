"""Evaluation module for forecast accuracy metrics."""

from .evaluate import (
    QUANTILES,
    detect_model_alias,
    evaluate_forecast,
)

__all__ = [
    "QUANTILES",
    "detect_model_alias",
    "evaluate_forecast",
]
