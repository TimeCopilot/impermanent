"""Shared forecast defaults (single source of truth for quantile levels)."""

# Used by :mod:`src.forecast.forecast`, :mod:`src.forecast.constant_models`, and
# must stay aligned with evaluation in :mod:`src.evaluation.evaluate`.
QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
