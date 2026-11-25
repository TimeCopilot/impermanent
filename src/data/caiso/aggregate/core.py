from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.data.caiso.config import PROCESSED_CSV, PROCESSED_EVENTS_HOURLY_DIR

INPUT_CSV = PROCESSED_CSV
OUTPUT_ROOT = PROCESSED_EVENTS_HOURLY_DIR


def write_hourly_partitions(df: pd.DataFrame, output_root: Path = OUTPUT_ROOT) -> int:
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna(subset=["unique_id", "ds", "y"]).copy()
    df = df.sort_values(["unique_id", "ds"]).drop_duplicates(["unique_id", "ds"])

    df["year"] = df["ds"].dt.year
    df["month"] = df["ds"].dt.month
    df["day"] = df["ds"].dt.day

    output_root.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for (year, month, day), part in df.groupby(["year", "month", "day"], sort=True):
        part_dir = (
            output_root / f"year={year:04d}" / f"month={month:02d}" / f"day={day:02d}"
        )
        part_dir.mkdir(parents=True, exist_ok=True)

        out_path = part_dir / "series.parquet"
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            part[["unique_id", "ds", "y"]].to_parquet(tmp.name, index=False)
            shutil.copyfile(tmp.name, out_path)

        print(f"Saved: {out_path}")
        n_written += 1

    return n_written


def build_hourly_partitions(
    input_csv: Path = INPUT_CSV,
    output_root: Path = OUTPUT_ROOT,
) -> int:
    df = pd.read_csv(input_csv)
    return write_hourly_partitions(df, output_root)


def main() -> None:
    n_written = build_hourly_partitions()
    print(f"\nDone. Wrote {n_written} daily partitions.")


if __name__ == "__main__":
    main()
