from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass
class CENACEData:
    base_path: Path
    freq: str = "hourly"
    h: int = 24
    max_window_size: int = 24 * 90

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path)

    def _date_to_partition(self, d: pd.Timestamp) -> Path:
        return (
            self.base_path
            / f"year={d.year:04d}"
            / f"month={d.month:02d}"
            / f"day={d.day:02d}"
            / "series.parquet"
        )

    def _paths_for_range(self, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
        days = pd.date_range(start.normalize(), end.normalize(), freq="D")
        paths = [self._date_to_partition(d) for d in days]
        existing = [str(p) for p in paths if p.exists()]
        if not existing:
            raise FileNotFoundError(
                f"No parquet files found between {start} and \
                {end} under {self.base_path}"
            )
        return existing

    def get_df(
        self,
        cutoff: str | pd.Timestamp,
        max_window_size: int | None = None,
        sort: bool = True,
    ) -> pd.DataFrame:
        cutoff = pd.Timestamp(cutoff)
        window = max_window_size or self.max_window_size
        start = cutoff - pd.Timedelta(hours=window - 1)

        paths = self._paths_for_range(start, cutoff)

        query = f"""
        SELECT unique_id, ds, y
        FROM read_parquet({paths})
        WHERE ds >= TIMESTAMP '{start}'
          AND ds <= TIMESTAMP '{cutoff}'
        """

        df = duckdb.sql(query).df()
        df["ds"] = pd.to_datetime(df["ds"])

        if sort:
            df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        return df

    def get_actuals(
        self, cutoff: str | pd.Timestamp, h: int | None = None
    ) -> pd.DataFrame:
        cutoff = pd.Timestamp(cutoff)
        horizon = h or self.h

        start = cutoff + pd.Timedelta(hours=1)
        end = cutoff + pd.Timedelta(hours=horizon)

        paths = self._paths_for_range(start, end)

        query = f"""
        SELECT unique_id, ds, y
        FROM read_parquet({paths})
        WHERE ds >= TIMESTAMP '{start}'
          AND ds <= TIMESTAMP '{end}'
        """

        df = duckdb.sql(query).df()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        return df
