from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dateutil.relativedelta import relativedelta

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("CAISO_DATA_ROOT", ROOT / "data" / "caiso"))

RAW_DIR = DATA_ROOT / "raw"
TMP_DIR = DATA_ROOT / "tmp"
PROCESSED_DIR = DATA_ROOT / "processed"
PROCESSED_CSV = PROCESSED_DIR / "caiso.csv"

PROCESSED_EVENTS_HOURLY_DIR = DATA_ROOT / "processed-events" / "hourly"
FORECASTS_HOURLY_DIR = DATA_ROOT / "forecasts" / "hourly"
EVALUATIONS_HOURLY_DIR = DATA_ROOT / "evaluations" / "hourly"

START_DATE = (date.today() - relativedelta(months=39)).isoformat()
MARKET_RUN_ID = "DAM"

NODES = (
    "TH_NP15_GEN-APND",
    "TH_SP15_GEN-APND",
    "TH_ZP26_GEN-APND",
)
