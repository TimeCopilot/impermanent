from __future__ import annotations

import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return (y_true - y_pred).abs().mean()


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return ((y_true - y_pred) ** 2).mean() ** 0.5


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = (y_true.abs() + y_pred.abs()) / 2
    out = (y_true - y_pred).abs() / denom
    out = out.where(denom != 0, 0.0)
    return 100 * out.mean()


def evaluate_forecasts(merged: pd.DataFrame) -> pd.DataFrame:
    per_uid = (
        merged.groupby("unique_id", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "mae": mae(g["y"], g["y_hat"]),
                    "rmse": rmse(g["y"], g["y_hat"]),
                    "smape": smape(g["y"], g["y_hat"]),
                }
            )
        )
        .reset_index(drop=True)
    )

    overall = pd.DataFrame(
        [
            {
                "unique_id": "__overall__",
                "mae": mae(merged["y"], merged["y_hat"]),
                "rmse": rmse(merged["y"], merged["y_hat"]),
                "smape": smape(merged["y"], merged["y_hat"]),
            }
        ]
    )

    return pd.concat([per_uid, overall], ignore_index=True)
