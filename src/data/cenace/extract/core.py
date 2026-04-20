from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import zipfile

import requests
from bs4 import BeautifulSoup

URL = "https://www.cenace.gob.mx/Paginas/SIM/Reportes/PreEnerServConMTR.aspx"

session = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": URL,
    "Origin": "https://www.cenace.gob.mx",
    "Content-Type": "application/x-www-form-urlencoded",
}

# repo root = impermanent/
ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_BASE_DIR = ROOT_DIR / "data" / "cenace"


def target_date_for_execution(execution_date: datetime) -> datetime:
    return execution_date + timedelta(days=1)


def raw_zip_path(date: datetime, raw_dir: Path) -> Path:
    return raw_dir / f"{date.strftime('%Y%m%d')}.zip"


def get_form_state() -> dict[str, str]:
    r = session.get(URL, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    def get_value(name: str) -> str:
        el = soup.find("input", {"name": name})
        return el.get("value") if el else ""

    return {
        "__VIEWSTATE": get_value("__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": get_value("__VIEWSTATEGENERATOR"),
        "__VIEWSTATEENCRYPTED": get_value("__VIEWSTATEENCRYPTED"),
        "__EVENTVALIDATION": get_value("__EVENTVALIDATION"),
    }


def download_and_extract(date: datetime, raw_dir: Path, tmp_dir: Path) -> bool:
    date_str = date.strftime("%d/%m/%Y")
    period_str = f"{date_str} - {date_str}"

    state = get_form_state()

    payload = {
        "ctl00$ContentPlaceHolder1$ddlReporte": "362,325",
        "ctl00$ContentPlaceHolder1$ddlPeriodicidad": "D",
        "ctl00$ContentPlaceHolder1$ddlSistema": "SIN",
        "ctl00$ContentPlaceHolder1$txtPeriodo": period_str,
        "ctl00$ContentPlaceHolder1$hdfStartDateSelected": date_str,
        "ctl00$ContentPlaceHolder1$hdfEndDateSelected": date_str,
        "ctl00$ContentPlaceHolder1$btnDescargarZIP": "Descargar ZIP",
        "__VIEWSTATE": state["__VIEWSTATE"],
        "__VIEWSTATEGENERATOR": state["__VIEWSTATEGENERATOR"],
        "__VIEWSTATEENCRYPTED": state["__VIEWSTATEENCRYPTED"],
        "__EVENTVALIDATION": state["__EVENTVALIDATION"],
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
    }

    r = session.post(URL, data=payload, headers=HEADERS)
    r.raise_for_status()

    size = len(r.content)
    print(f"{date_str} | {size} bytes")

    if size < 10000:
        print(f"Skipping {date_str}: file not published or response too small")
        return False

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_zip_path(date, raw_dir)

    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)

    return True


def backfill_missing(start_date: datetime, end_date: datetime, base_dir: Path) -> None:
    raw_dir = base_dir / "raw"
    tmp_dir = base_dir / "tmp"

    current = start_date
    while current <= end_date:
        zip_path = raw_zip_path(current, raw_dir)
        if zip_path.exists():
            print(f"Already have {current.strftime('%Y-%m-%d')}, skipping")
        else:
            try:
                ok = download_and_extract(current, raw_dir, tmp_dir)
                if not ok:
                    print(f"Stopping at {current.strftime('%Y-%m-%d')}")
                    break
            except Exception as e:
                print(f"Error on {current.strftime('%Y-%m-%d')}: {e}")
                break
        current += timedelta(days=1)


def run_execution_date(execution_date: datetime, base_dir: Path) -> bool:
    raw_dir = base_dir / "raw"
    tmp_dir = base_dir / "tmp"
    target_date = target_date_for_execution(execution_date)

    zip_path = raw_zip_path(target_date, raw_dir)
    if zip_path.exists():
        print(f"Already have {target_date.strftime('%Y-%m-%d')}, skipping")
        return True

    return download_and_extract(target_date, raw_dir, tmp_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-date", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--out", default=str(DEFAULT_BASE_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.out).resolve()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        elif args.execution_date:
            end_date = target_date_for_execution(
                datetime.strptime(args.execution_date, "%Y-%m-%d")
            )
        else:
            end_date = datetime.today()
        backfill_missing(start_date=start_date, end_date=end_date, base_dir=base_dir)
        return

    if args.execution_date:
        execution_date = datetime.strptime(args.execution_date, "%Y-%m-%d")
        ok = run_execution_date(execution_date=execution_date, base_dir=base_dir)
        if not ok:
            print("No new CENACE publication detected; stopping cleanly")
        return

    raise ValueError("Provide either --start-date or --execution-date")


if __name__ == "__main__":
    main()
