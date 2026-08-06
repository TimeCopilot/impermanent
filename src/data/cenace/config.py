from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = Path(os.environ.get("CENACE_DATA_ROOT", ROOT / "data" / "cenace"))

RAW_DIR = DATA_ROOT / "raw"
TMP_DIR = DATA_ROOT / "tmp"

PROCESSED_DIR = DATA_ROOT / "processed"
PROCESSED_CSV = PROCESSED_DIR / "cenace.csv"

PROCESSED_EVENTS_HOURLY_DIR = DATA_ROOT / "processed-events" / "hourly"
FORECASTS_HOURLY_DIR = DATA_ROOT / "forecasts" / "hourly"
EVALUATIONS_HOURLY_DIR = DATA_ROOT / "evaluations" / "hourly"
