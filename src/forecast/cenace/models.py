from __future__ import annotations

import pandas as pd

from src.forecast.forecast import generate_forecast


def naive_last_value(train_df: pd.DataFrame, cutoff: str, h: int = 24) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)

    last_values = (
        train_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id", as_index=False)
        .tail(1)[["unique_id", "y"]]
        .rename(columns={"y": "y_hat"})
    )

    future_ds = pd.date_range(
        cutoff + pd.Timedelta(hours=1),
        periods=h,
        freq="h",
    )

    out = []
    for ds in future_ds:
        tmp = last_values.copy()
        tmp["ds"] = ds
        out.append(tmp)

    fcst = pd.concat(out, ignore_index=True)
    return (
        fcst[["unique_id", "ds", "y_hat"]]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def seasonal_naive_24(train_df: pd.DataFrame, cutoff: str, h: int = 24) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)

    expected_start = cutoff - pd.Timedelta(hours=23)
    last_day = train_df.loc[
        (train_df["ds"] >= expected_start) & (train_df["ds"] <= cutoff),
        ["unique_id", "ds", "y"],
    ].copy()

    counts = last_day.groupby("unique_id")["ds"].count()
    bad_ids = counts[counts != 24]
    if not bad_ids.empty:
        raise ValueError(
            "seasonal_naive_24 needs exactly 24 hourly observations in the last day "
            f"for every series. Bad series count: {len(bad_ids)}"
        )

    last_day["hour_ahead"] = (
        (last_day["ds"] - expected_start) / pd.Timedelta(hours=1)
    ).astype(int)
    profile = last_day[["unique_id", "hour_ahead", "y"]].rename(columns={"y": "y_hat"})

    unique_ids = profile["unique_id"].drop_duplicates().sort_values().tolist()
    future_ds = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=h, freq="h")

    future_index = pd.DataFrame(
        [(uid, ds, i % 24) for i, ds in enumerate(future_ds) for uid in unique_ids],
        columns=["unique_id", "ds", "hour_ahead"],
    )

    fcst = (
        future_index.merge(profile, on=["unique_id", "hour_ahead"], how="left")[
            ["unique_id", "ds", "y_hat"]
        ]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )

    if fcst["y_hat"].isna().any():
        raise ValueError(
            "seasonal_naive_24" + " produced missing forecasts after profile merge."
        )

    return fcst


def auto_arima(train_df: pd.DataFrame, cutoff: str, h: int = 24) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)

    raw = generate_forecast(
        model_name="auto_arima",
        df=train_df,
        h=h,
        freq="h",
    ).copy()

    point_candidates = [
        col
        for col in raw.columns
        if col not in {"unique_id", "ds"} and "-q-" not in col
    ]
    if "auto_arima" in point_candidates:
        point_col = "auto_arima"
    elif len(point_candidates) == 1:
        point_col = point_candidates[0]
    else:
        raise ValueError(
            "Could not detect AutoARIMA point forecast column. "
            f"Candidates found: {point_candidates}"
        )

    fcst = (
        raw[["unique_id", "ds", point_col]]
        .rename(columns={point_col: "y_hat"})
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return fcst


MODEL_REGISTRY = {
    "naive_last_value": naive_last_value,
    "seasonal_naive_24": seasonal_naive_24,
    "auto_arima": auto_arima,
}
