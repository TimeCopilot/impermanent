from __future__ import annotations

import argparse
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from src.data.caiso.config import MARKET_RUN_ID, NODES, RAW_DIR, START_DATE

OASIS_URL = "https://oasis.caiso.com/oasisapi/SingleZip"
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = ZoneInfo("UTC")


def oasis_datetime(value: date) -> str:
    local_midnight = datetime.combine(value, datetime.min.time(), PACIFIC)
    return local_midnight.astimezone(UTC).strftime("%Y%m%dT%H:%M-0000")


def download_node(
    node: str,
    start: date,
    end: date,
    output_dir: Path = RAW_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"{node}_{start:%Y%m%d}_{end:%Y%m%d}.zip"

    if output.exists() and zipfile.is_zipfile(output):
        print(f"Skipping existing: {output}")
        return output

    params = {
        "queryname": "PRC_LMP",
        "startdatetime": oasis_datetime(start),
        "enddatetime": oasis_datetime(end),
        "version": "12",
        "market_run_id": MARKET_RUN_ID,
        # "node": node,
        "node": ",".join(NODES),
        "resultformat": "6",
    }

    response = requests.get(OASIS_URL, params=params, timeout=120)
    response.raise_for_status()
    output.write_bytes(response.content)

    if not zipfile.is_zipfile(output):
        output.unlink(missing_ok=True)
        raise RuntimeError(f"CAISO returned a non-ZIP response for {node}")

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if names and names[0].endswith(".xml"):
            with archive.open(names[0]) as file:
                text = file.read().decode("utf-8", errors="replace")
            if "<m:ERROR>" in text or "<ERR_DESC>" in text:
                output.unlink(missing_ok=True)
                message = f"CAISO returned an error response for {node}: {text}"
                raise RuntimeError(message)

    print(f"Downloaded: {output}")
    return output


def extract_caiso(
    start: date,
    end: date,
    output_dir: Path = RAW_DIR,
) -> None:
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=31), end)

        download_node("CAISO_3_NODES", current, chunk_end, output_dir=output_dir)

        current = chunk_end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extract_caiso(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
    )
