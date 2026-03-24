"""Core evaluation logic using utilsforecast."""

import re

import numpy as np
import pandas as pd
from utilsforecast.losses import mae


def mase(
    df: pd.DataFrame,
    models: list[str],
    seasonality: int,
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    mean_abs_err = mae(df, models, id_col, target_col)
    mean_abs_err = mean_abs_err.set_index(id_col)
    # assume train_df is sorted
    lagged = train_df.groupby(id_col, observed=True)[target_col].shift(seasonality)
    scale = train_df[target_col].sub(lagged).abs()
    scale = scale.groupby(train_df[id_col], observed=True).mean()
    eps = 1e-2
    scale = scale.clip(lower=eps)
    res = mean_abs_err.div(scale, axis=0)
    res.index.name = id_col
    res = res.reset_index()
    return res


def quantile_loss(
    df: pd.DataFrame,
    models: list[str],
    q: float,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    # pinball / quantile loss
    delta = df[models].sub(df[target_col], axis=0)
    res = np.maximum(q * delta, (q - 1) * delta)
    res[id_col] = df[id_col].values
    res = res.groupby(id_col, observed=True).sum()
    res.index.name = id_col
    return res.reset_index()


def scaled_crps(
    df: pd.DataFrame,
    models: list[str],
    quantiles: list[float],
    train_df: pd.DataFrame,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    eval_prob_dfs = []
    for q in quantiles:
        prob_cols = [f"{model}-q-{int(100*q)}" for model in models]
        eval_q_df = quantile_loss(df, models=prob_cols, q=q, target_col=target_col)
        eval_q_df = eval_q_df.rename(columns=dict(zip(prob_cols, models, strict=False)))
        eval_q_df["quantile"] = q
        eval_prob_dfs.append(eval_q_df)
    eval_prob_df = pd.concat(eval_prob_dfs, ignore_index=True)
    eval_prob_df = eval_prob_df.groupby(id_col, observed=True)[models].mean()
    eval_prob_df = 2.0 * eval_prob_df
    scale = train_df.groupby(id_col, observed=True)[target_col].apply(
        lambda x: np.abs(x).mean()
    )
    eps = 1.0
    scale = scale.clip(lower=eps)
    res = eval_prob_df.div(scale, axis=0)
    res.index.name = id_col
    res = res.reset_index()
    return res


# Quantiles used for evaluation (must match forecast generation)
QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def detect_model_alias(df: pd.DataFrame) -> str | None:
    """Detect the model alias from forecast DataFrame columns.

    Looks for columns matching ``{alias}-q-{percentile}``.

    Returns:
        Model alias string, or ``None`` if not detected.
    """
    pattern = re.compile(r"^(.+)-q-(\d+)$")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            return match.group(1)
    return None


def evaluate_forecast(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    train_df: pd.DataFrame,
    seasonality: int = 1,
) -> tuple[pd.DataFrame, str]:
    """Evaluate a forecast and return per-unique_id metrics.

    Uses ``utilsforecast.evaluation.evaluate`` for MASE and calls
    ``scaled_crps`` directly (our forecasts use ``q-XX`` columns which
    the ``evaluate`` helper does not map automatically).

    Args:
        forecast_df: Forecasts with ``unique_id``, ``ds``, and quantile columns.
        actuals_df: Actuals with ``unique_id``, ``ds``, ``y``.
        train_df: Training data for MASE scaling (``unique_id``, ``ds``, ``y``).
        model_name: Model name (for logging / metadata).
        seasonality: Seasonality period for MASE (e.g. 24 for hourly).

    Returns:
        Tuple of ``(metrics_df, model_alias)`` where *metrics_df* has
        columns ``unique_id``, ``mase``, ``scaled_crps`` (one row per
        unique_id, unaggregated).

    Note:
        We default to a seasonality of 1,
        to prevent errors when series are too short
        for seasonal MASE calculation.
        In this case, the errors are scaled by the one step
        naive method.
    """
    model_alias = detect_model_alias(forecast_df)
    if model_alias is None:
        raise ValueError(
            f"Could not detect model alias from forecast columns. "
            f"Available columns: {list(forecast_df.columns)}"
        )

    # Merge forecasts with actuals
    fcst_cols = [col for col in forecast_df.columns if col.startswith(model_alias)]
    merged = actuals_df[["unique_id", "ds", "y"]].merge(
        forecast_df[["unique_id", "ds"] + fcst_cols],
        on=["unique_id", "ds"],
        how="left",
    )

    if len(merged) == 0:
        raise ValueError(
            "No matching rows after merging forecasts with actuals. "
            "Check that unique_id and ds match."
        )

    # Optional fill: when mean is present but quantiles are NA (e.g. short history),
    # replace quantile NAs with the mean for evaluation only,
    # if it affects < 5% of rows.
    q_pattern = re.compile(rf"^{re.escape(model_alias)}-q-(\d+)$")
    q_cols = sorted(
        (col for col in merged.columns if q_pattern.match(col)),
        key=lambda c: int(q_pattern.match(c).group(1)),  # type: ignore[union-attr]
    )
    if q_cols and model_alias in merged.columns:
        mean_ok = merged[model_alias].notna()
        any_q_na = merged[q_cols].isna().any(axis=1)
        fill_candidate = mean_ok & any_q_na
        n_fill = fill_candidate.sum()
        if n_fill > 0:
            pct = n_fill / len(merged)
            if pct < 0.05:
                for c in q_cols:
                    merged[c] = merged[c].fillna(merged[model_alias])
            # else: leave NAs so the check below will raise

    na_counts = merged.isna().sum()
    cols_with_na = na_counts[na_counts > 0]
    if not cols_with_na.empty:
        raise ValueError(
            f"Merged DataFrame contains NAs in columns: "
            f"{dict(cols_with_na)}. "
            "This likely means some (unique_id, ds) pairs in actuals "
            "have no matching forecast."
        )

    # --- MASE via utilsforecast.evaluate (per unique_id) ---
    mase_df = mase(merged, [model_alias], seasonality, train_df)
    # mase_df has columns: unique_id, metric, {model_alias}
    mase_per_uid = mase_df[["unique_id", model_alias]].rename(
        columns={model_alias: "mase"}
    )

    # --- Scaled CRPS per unique_id ---
    quantiles = np.array([int(q_pattern.match(c).group(1)) / 100 for c in q_cols])  # type: ignore[union-attr]

    crps_df = scaled_crps(
        df=merged,
        models=[model_alias],
        quantiles=quantiles,
        train_df=train_df,
    )
    crps_per_uid = crps_df[["unique_id", model_alias]].rename(
        columns={model_alias: "scaled_crps"}
    )

    # Join per-unique_id metrics
    result = mase_per_uid.merge(crps_per_uid, on="unique_id")
    print(result)
    return result, model_alias
