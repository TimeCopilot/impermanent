from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.cenace.config import FORECASTS_HOURLY_DIR, PROCESSED_EVENTS_HOURLY_DIR
from src.data.cenace.utils.cenace_data import CENACEData
from src.forecast.cenace.models import MODEL_REGISTRY


def cutoff_partition(root: Path, cutoff: pd.Timestamp) -> Path:
    return (
        root
        / f"year={cutoff.year:04d}"
        / f"month={cutoff.month:02d}"
        / f"day={cutoff.day:02d}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--max-window-size", type=int, default=48)
    return parser.parse_args()


def run_forecast(
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

    train = data.get_df(cutoff, max_window_size=max_window_size)

    if model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown CENACE model: {model}. " f"Available: {sorted(MODEL_REGISTRY)}"
        )

    model_fn = MODEL_REGISTRY[model]
    forecasts = model_fn(train, cutoff=cutoff, h=h)

    forecast_root = FORECASTS_HOURLY_DIR / model
    out_dir = cutoff_partition(forecast_root, cutoff)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forecasts.parquet"

    forecasts.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()
    cutoff = pd.Timestamp(args.cutoff)

    data = CENACEData(
        base_path=PROCESSED_EVENTS_HOURLY_DIR,
        freq="hourly",
        h=args.h,
        max_window_size=args.max_window_size,
    )

    train = data.get_df(cutoff, max_window_size=args.max_window_size)
    model_fn = MODEL_REGISTRY[args.model]
    forecasts = model_fn(train, cutoff=cutoff, h=args.h)

    forecast_root = FORECASTS_HOURLY_DIR / args.model
    out_dir = cutoff_partition(forecast_root, cutoff)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forecasts.parquet"

    forecasts.to_parquet(out_path, index=False)

    print(f"Saved forecasts: {out_path}")
    print(forecasts.head())
    print(forecasts.shape)


if __name__ == "__main__":
    main()
