import argparse
from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import zipfile

URL = "https://www.cenace.gob.mx/Paginas/SIM/Reportes/PreEnerServConMTR.aspx"

session = requests.Session()

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": URL,
    "Origin": "https://www.cenace.gob.mx",
    "Content-Type": "application/x-www-form-urlencoded",
}

# repo root = imper/
ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_BASE_DIR = ROOT_DIR / "data" / "cenace"


def get_form_state():
    r = session.get(URL, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")

    def get_value(name):
        el = soup.find("input", {"name": name})
        return el.get("value") if el else ""

    return {
        "__VIEWSTATE": get_value("__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": get_value("__VIEWSTATEGENERATOR"),
        "__VIEWSTATEENCRYPTED": get_value("__VIEWSTATEENCRYPTED"),
        "__EVENTVALIDATION": get_value("__EVENTVALIDATION"),
    }


def download_and_extract(date, raw_dir, tmp_dir):
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

    size = len(r.content)
    print(f"{date_str} | {size} bytes")

    if size < 10000:
        print(f"Skipping {date_str}")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / f"{date.strftime('%Y%m%d')}.zip"

    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmp_dir)


def run(start_date, end_date, base_dir):
    raw_dir = base_dir / "raw"
    tmp_dir = base_dir / "tmp"

    current = start_date
    while current <= end_date:
        try:
            download_and_extract(current, raw_dir, tmp_dir)
        except Exception as e:
            print(f"Error on {current.strftime('%Y-%m-%d')}: {e}")
        current += timedelta(days=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--out", default=str(DEFAULT_BASE_DIR))

    args = parser.parse_args()

    run(
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d"),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d"),
        base_dir=Path(args.out).resolve(),
    )
