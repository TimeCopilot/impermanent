from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.cenace.config import (
    EVALUATIONS_HOURLY_DIR,
    FORECASTS_HOURLY_DIR,
    PROCESSED_EVENTS_HOURLY_DIR,
)
from src.data.cenace.utils.cenace_data import CENACEData
from src.evaluation.evaluate import evaluate_forecast


def cutoff_partition(root: Path, cutoff: pd.Timestamp) -> Path:
    return (
        root
        / f"year={cutoff.year:04d}"
        / f"month={cutoff.month:02d}"
        / f"day={cutoff.day:02d}"
    )


def run_evaluation(
    cutoff: str | pd.Timestamp,
    model: str,
    h: int = 24,
    max_window_size: int = 48,
) -> Path:
    cutoff = pd.Timestamp(cutoff)

    data = CENACEData(
        base_path=PROCESSED_EVENTS_HOURLY_DIR,
        freq="hourly",
        h=h,
        max_window_size=max_window_size,
    )

    forecast_path = (
        FORECASTS_HOURLY_DIR
        / model
        / f"year={cutoff.year:04d}"
        / f"month={cutoff.month:02d}"
        / f"day={cutoff.day:02d}"
        / "forecasts.parquet"
    )

    train = data.get_df(cutoff, max_window_size=max_window_size)
    actuals = data.get_actuals(cutoff, h=h)
    forecasts = pd.read_parquet(forecast_path)

    metrics, _ = evaluate_forecast(
        forecast_df=forecasts,
        actuals_df=actuals,
        train_df=train,
        seasonality=24,
    )

    eval_root = EVALUATIONS_HOURLY_DIR / model
    out_dir = cutoff_partition(eval_root, cutoff)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.parquet"

    metrics.to_parquet(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--max-window-size", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = run_evaluation(
        cutoff=args.cutoff,
        model=args.model,
        h=args.h,
        max_window_size=args.max_window_size,
    )
    metrics = pd.read_parquet(out_path)
    print(f"Saved metrics: {out_path}")
    print(metrics.head())
    print(metrics.shape)


if __name__ == "__main__":
    main()
