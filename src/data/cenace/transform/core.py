from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.data.cenace.config import PROCESSED_CSV, TMP_DIR

SELECTED_NODES = (
    "01CMC-230",
    "01PAC-85",
    "02PRI-115",
    "02VRU-115",
    "03GDU-230",
    "03QRO-115",
    "03SLP-115",
    "04HLU-115",
    "04MZD-230",
    "05CUN-115",
    "05TRR-115",
    "06ALT-115",
    "06ESC-115",
    "08CTE-115",
    "08MDA-115",
)


def read_clean_file(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="latin-1").splitlines()

    header_idx = next(
        i for i, line in enumerate(lines) if "Hora" in line and "Clave del nodo" in line
    )

    return pd.read_csv(
        path,
        encoding="latin-1",
        skiprows=header_idx,
        sep=",",
        quotechar='"',
    )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        for c in df.columns
    ]

    col_map = {}
    for col in df.columns:
        if "hora" in col:
            col_map[col] = "hour"
        elif "clave" in col and "nodo" in col:
            col_map[col] = "unique_id"
        elif "precio marginal" in col:
            col_map[col] = "y"

    return df.rename(columns=col_map)


def build_timestamp(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    match = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
    if not match:
        raise ValueError(f"Date not found in filename: {path.name}")

    result = df.copy()
    base_date = pd.to_datetime(match.group(0))

    result["hour"] = pd.to_numeric(result["hour"], errors="coerce")
    result = result.dropna(subset=["hour"]).copy()
    result["hour"] = result["hour"].astype(int)
    result["ds"] = base_date + pd.to_timedelta(result["hour"] - 1, unit="h")

    return result


def transform_raw_files(tmp_dir: Path = TMP_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in sorted(tmp_dir.glob("*.csv")):
        if path.name.startswith(".~lock"):
            continue

        frame = read_clean_file(path)
        frame = clean_columns(frame)
        frame = build_timestamp(frame, path)
        frame = frame[["unique_id", "ds", "y"]]
        frame = frame[frame["unique_id"].isin(SELECTED_NODES)]
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No valid CENACE CSV files parsed from {tmp_dir}")

    result = pd.concat(frames, ignore_index=True)
    result["y"] = pd.to_numeric(result["y"], errors="coerce")

    return (
        result.dropna(subset=["unique_id", "ds", "y"])
        .drop_duplicates(["unique_id", "ds"])
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


def transform_cenace(
    tmp_dir: Path = TMP_DIR,
    output_path: Path = PROCESSED_CSV,
) -> Path:
    result = transform_raw_files(tmp_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(result.groupby("unique_id").size())
    print(f"Saved {len(result)} rows to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmp-dir", type=Path, default=TMP_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_CSV)
    args = parser.parse_args()

    transform_cenace(args.tmp_dir, args.output)
