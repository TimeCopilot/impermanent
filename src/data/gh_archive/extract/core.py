import datetime as dt
import gzip
import io
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

DEFAULT_KEEP_EVENT_TYPES = {
    "WatchEvent",
    "PullRequestEvent",
    "IssuesEvent",
    "PushEvent",
}


class RawGHArchiveHourExtractor:
    def __init__(
        self,
        output_dir: str | Path,
        keep_event_types: set[str] = DEFAULT_KEEP_EVENT_TYPES,
        output_filename: str = "events.parquet",
    ):
        self.output_dir = Path(output_dir)
        self.keep_event_types = keep_event_types
        self.output_filename = output_filename

    def hour_local_path(self, dt_hour: dt.datetime) -> Path:
        path = (
            self.output_dir
            / f"year={dt_hour.year:04d}"
            / f"month={dt_hour.month:02d}"
            / f"day={dt_hour.day:02d}"
            / f"hour={dt_hour.hour:02d}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path / self.output_filename

    def iter_events_from_gz_bytes(self, gz_bytes: bytes) -> Iterable[dict]:
        with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes), mode="rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def extract_action(self, payload: dict) -> str | None:
        # Only some event types carry "action"
        if not isinstance(payload, dict):
            return None
        a = payload.get("action")
        return a if isinstance(a, str) else None

    def extract_push_distinct_size(self, payload: dict) -> int | None:
        if not isinstance(payload, dict):
            return None
        v = payload.get("distinct_size")
        return int(v) if isinstance(v, int | float) else None

    def build_table(
        self,
        events: Iterable[dict],
    ) -> pa.Table:
        event_id = []
        created_at = []
        event_type = []
        repo_id = []
        repo_name = []
        actor_id = []
        actor_login = []
        action = []
        push_distinct_size = []

        for e in events:
            t = e.get("type")
            if t not in self.keep_event_types:
                continue

            repo = e.get("repo") or {}
            rname = repo.get("name")
            if not isinstance(rname, str):
                continue

            payload = e.get("payload") or {}

            event_id.append(str(e.get("id")) if e.get("id") is not None else None)
            created_at.append(e.get("created_at"))  # keep as string; can cast later
            event_type.append(t)

            rid = repo.get("id")
            repo_id.append(int(rid) if isinstance(rid, int | float) else None)
            repo_name.append(rname)

            actor = e.get("actor") or {}
            aid = actor.get("id")
            actor_id.append(int(aid) if isinstance(aid, int | float) else None)
            alog = actor.get("login")
            actor_login.append(alog if isinstance(alog, str) else None)

            action.append(self.extract_action(payload))
            push_distinct_size.append(self.extract_push_distinct_size(payload))

        schema = pa.schema(
            [
                ("event_id", pa.string()),
                (
                    "created_at",
                    pa.string(),
                ),  # store raw; you can parse to timestamp in Gold
                ("event_type", pa.string()),
                ("repo_id", pa.int64()),
                ("repo_name", pa.string()),
                ("actor_id", pa.int64()),
                ("actor_login", pa.string()),
                ("action", pa.string()),
                ("push_distinct_size", pa.int32()),
            ]
        )

        arrays = [
            pa.array(event_id, type=pa.string()),
            pa.array(created_at, type=pa.string()),
            pa.array(event_type, type=pa.string()),
            pa.array(repo_id, type=pa.int64()),
            pa.array(repo_name, type=pa.string()),
            pa.array(actor_id, type=pa.int64()),
            pa.array(actor_login, type=pa.string()),
            pa.array(action, type=pa.string()),
            pa.array(push_distinct_size, type=pa.int32()),
        ]
        return pa.Table.from_arrays(arrays, schema=schema)

    def hour_stamp(self, dt_hour: dt.datetime) -> str:
        dt_hour = dt_hour.astimezone(dt.timezone.utc)
        return (
            f"{dt_hour.year:04d}-{dt_hour.month:02d}-{dt_hour.day:02d}-{dt_hour.hour:d}"
        )

    def download_hour_gz(self, dt_hour: dt.datetime, timeout_s: int = 60) -> bytes:
        stamp = self.hour_stamp(dt_hour)
        url = f"https://data.gharchive.org/{stamp}.json.gz"
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def str_hour_to_dt(str_hour: str) -> dt.datetime:
        y, m, d, hh = str_hour.split("-")
        return dt.datetime(int(y), int(m), int(d), int(hh), tzinfo=dt.timezone.utc)

    def hour_exists(self, str_hour: str) -> bool:
        dt_hour = self.str_hour_to_dt(str_hour)
        local_path = self.hour_local_path(dt_hour)
        return local_path.exists()

    def extract_hour(self, str_hour: str) -> Path:
        dt_hour = self.str_hour_to_dt(str_hour)
        gz_bytes = self.download_hour_gz(dt_hour)
        events = self.iter_events_from_gz_bytes(gz_bytes)
        table = self.build_table(events)
        local_path = self.hour_local_path(dt_hour)
        pq.write_table(table, str(local_path), compression="zstd")
        return local_path

    @staticmethod
    def last_cutoff_str_hour() -> str:
        now = dt.datetime.now(dt.timezone.utc)
        last_date = pd.date_range(
            start=RawGHArchiveHourExtractor.str_hour_to_dt("2023-01-01-00"),
            end=now,
            freq="24h",
        )[-1].strftime("%Y-%m-%d-%H")
        return last_date

    @staticmethod
    def generate_str_hours(start_str_hour: str, end_str_hour: str) -> list[str]:
        start_ts = RawGHArchiveHourExtractor.str_hour_to_dt(start_str_hour)
        end_ts = RawGHArchiveHourExtractor.str_hour_to_dt(end_str_hour)
        return [
            ts.strftime("%Y-%m-%d-%H")
            for ts in pd.date_range(start_ts, end_ts, freq="h")
        ]

    @staticmethod
    def generate_previous_str_hours(last_hours: int) -> list[str]:
        """Generate the previous N hours in the format "YYYY-MM-DD-HH".
        without including the current hour."""
        now = dt.datetime.now(dt.timezone.utc)
        now = now - dt.timedelta(hours=1)
        start_str_hour = now - dt.timedelta(hours=last_hours - 1)
        return RawGHArchiveHourExtractor.generate_str_hours(
            start_str_hour=start_str_hour.strftime("%Y-%m-%d-%H"),
            end_str_hour=now.strftime("%Y-%m-%d-%H"),
        )
