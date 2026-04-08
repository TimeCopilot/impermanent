from __future__ import annotations

import argparse

from src.dataset_registry import PIPELINE_DATASET_CHOICES, run_dataset_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=PIPELINE_DATASET_CHOICES)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--max-window-size", type=int, default=48)
    parser.add_argument("--skip-aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dataset_pipeline(
        dataset=args.dataset,
        cutoff=args.cutoff,
        model=args.model,
        h=args.h,
        max_window_size=args.max_window_size,
        skip_aggregate=args.skip_aggregate,
    )


if __name__ == "__main__":
    main()
