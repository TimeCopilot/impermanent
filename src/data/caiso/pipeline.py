from __future__ import annotations

import shutil
import tempfile
from datetime import date
from pathlib import Path

from src.data.caiso.aggregate.core import build_hourly_partitions
from src.data.caiso.config import DATA_ROOT, RAW_DIR
from src.data.caiso.extract.core import extract_caiso
from src.data.caiso.transform.core import transform_caiso


def update_date_range(
    start: date,
    end: date,
    data_root: Path = DATA_ROOT,
    raw_dir: Path = RAW_DIR,
) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        tmp_raw = tmp_root / "raw"
        tmp_processed = tmp_root / "processed"
        tmp_csv = tmp_processed / "caiso.csv"

        extract_caiso(start=start, end=end, output_dir=tmp_raw)

        raw_dir.mkdir(parents=True, exist_ok=True)
        for source_zip in sorted(tmp_raw.glob("*.zip")):
            shutil.copyfile(source_zip, raw_dir / source_zip.name)

        transform_caiso(raw_dir=tmp_raw, output_path=tmp_csv)

        output_root = data_root / "processed-events" / "hourly"
        return build_hourly_partitions(input_csv=tmp_csv, output_root=output_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    n_written = update_date_range(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
    print(f"Done. Wrote {n_written} CAISO daily partitions.")
