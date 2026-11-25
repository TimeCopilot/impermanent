from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.data.caiso.config import NODES, PROCESSED_CSV, RAW_DIR


def transform_caiso(
    raw_dir: Path = RAW_DIR,
    output_path: Path = PROCESSED_CSV,
) -> Path:
    frames: list[pd.DataFrame] = []

    for zip_path in sorted(raw_dir.glob("CAISO_3_NODES_*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for filename in archive.namelist():
                if not filename.endswith(".csv"):
                    continue

                with archive.open(filename) as file:
                    frame = pd.read_csv(file)

                frame = frame[
                    (frame["LMP_TYPE"] == "LMP")
                    & (frame["XML_DATA_ITEM"] == "LMP_PRC")
                    & (frame["NODE"].isin(NODES))
                ]

                frames.append(frame[["NODE", "INTERVALSTARTTIME_GMT", "MW"]])

    if not frames:
        raise RuntimeError(f"No CAISO ZIP files found in {raw_dir}")

    result = pd.concat(frames, ignore_index=True)
    result = result.rename(
        columns={
            "NODE": "unique_id",
            "INTERVALSTARTTIME_GMT": "ds",
            "MW": "y",
        }
    )

    result["ds"] = pd.to_datetime(result["ds"], utc=True)
    result["y"] = pd.to_numeric(result["y"], errors="coerce")

    result = (
        result.dropna(subset=["unique_id", "ds", "y"])
        .drop_duplicates(["unique_id", "ds"])
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(result.groupby("unique_id").size())
    print(f"Saved {len(result)} rows to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_CSV)
    args = parser.parse_args()

    transform_caiso(args.raw_dir, args.output)
