from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.data.caiso.config import FORECASTS_HOURLY_DIR, PROCESSED_EVENTS_HOURLY_DIR
from src.data.caiso.utils.caiso_data import CAISOData
from src.forecast.forecast import generate_forecast


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
    parser.add_argument("--model", required=True)
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

    data = CAISOData(
        base_path=PROCESSED_EVENTS_HOURLY_DIR,
        freq="hourly",
        h=h,
        max_window_size=max_window_size,
    )

    train = data.get_df(cutoff, max_window_size=max_window_size)

    model_name = "seasonal_naive" if model == "seasonal_naive_24" else model

    forecasts = generate_forecast(
        model_name=model_name,
        df=train,
        h=h,
        freq="h",
    )

    if "y_hat" in forecasts.columns:
        forecasts = forecasts.rename(columns={"y_hat": model})

    forecast_root = FORECASTS_HOURLY_DIR / model
    out_dir = cutoff_partition(forecast_root, cutoff)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forecasts.parquet"

    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        forecasts.to_parquet(tmp.name, index=False)
        shutil.copyfile(tmp.name, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    out_path = run_forecast(
        cutoff=args.cutoff,
        model=args.model,
        h=args.h,
        max_window_size=args.max_window_size,
    )
    print(f"Forecasts saved to: {out_path}")


if __name__ == "__main__":
    main()
