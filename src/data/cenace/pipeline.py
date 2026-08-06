from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from src.data.cenace.aggregate.core import write_hourly_partitions
from src.data.cenace.config import DATA_ROOT, PROCESSED_EVENTS_HOURLY_DIR, RAW_DIR
from src.data.cenace.extract.core import run_execution_date, target_date_for_execution
from src.data.cenace.transform.core import transform_raw_files


def update_execution_date(
    execution_date: datetime,
    data_root: Path = DATA_ROOT,
    raw_dir: Path = RAW_DIR,
    output_root: Path = PROCESSED_EVENTS_HOURLY_DIR,
) -> int:
    target_date = target_date_for_execution(execution_date)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        tmp_raw_dir = tmp_root / "raw"
        tmp_extract_dir = tmp_root / "tmp"

        ok = run_execution_date(
            execution_date=execution_date,
            base_dir=tmp_root,
        )
        if not ok:
            target = f"{target_date:%Y-%m-%d}"
            raise RuntimeError(f"CENACE download failed for target date {target}")

        raw_dir.mkdir(parents=True, exist_ok=True)
        source_zip = tmp_raw_dir / f"{target_date:%Y%m%d}.zip"
        if source_zip.exists():
            shutil.copyfile(source_zip, raw_dir / source_zip.name)

        df = transform_raw_files(tmp_extract_dir)
        return write_hourly_partitions(df, output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-date", required=True)
    args = parser.parse_args()

    n_written = update_execution_date(datetime.fromisoformat(args.execution_date))
    print(f"Done. Wrote {n_written} CENACE daily partitions.")


if __name__ == "__main__":
    main()
