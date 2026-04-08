from __future__ import annotations

import argparse

import pandas as pd

from src.data.cenace.aggregate.core import build_hourly_partitions
from src.evaluation.cenace.core import run_evaluation
from src.forecast.cenace.core import run_forecast


def run_cenace_pipeline(
    cutoff: str,
    model: str,
    h: int = 24,
    max_window_size: int = 48,
    skip_aggregate: bool = False,
) -> tuple[str, str]:
    try:
        cutoff_ts = pd.Timestamp(cutoff)
    except Exception as exc:
        raise ValueError(f"Invalid cutoff timestamp: {cutoff}") from exc

    if not skip_aggregate:
        n_written = build_hourly_partitions()
        print(f"Aggregated {n_written} partitions")

    forecast_path = run_forecast(
        cutoff=cutoff_ts,
        model=model,
        h=h,
        max_window_size=max_window_size,
    )
    print(f"Forecasts saved to: {forecast_path}")

    eval_path = run_evaluation(
        cutoff=cutoff_ts,
        model=model,
        h=h,
        max_window_size=max_window_size,
    )
    print(f"Metrics saved to: {eval_path}")

    return str(forecast_path), str(eval_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--max-window-size", type=int, default=48)
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cenace_pipeline(
        cutoff=args.cutoff,
        model=args.model,
        h=args.h,
        max_window_size=args.max_window_size,
        skip_aggregate=args.skip_aggregate,
    )


if __name__ == "__main__":
    main()
